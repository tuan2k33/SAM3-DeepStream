"""
EfficientSAM3 Specialize — ONNX + TRT (bbox-only or instance segmentation)
--------------------------------------------------------------------------
Same conventions as SAM3Fixed/export.py, but for the EfficientSAM3 EV-M checkpoint
(EfficientViT-b1 vision encoder + MobileCLIP-S0-variant text encoder, 4-layer
"mct" architecture -- confirmed by inspecting the checkpoint's actual keys;
the repo's own README/table says "MobileCLIP-S1" but that maps to a 12-layer
"base" variant that does NOT match this checkpoint's weights).
Source: https://github.com/SimonZeng7108/efficientsam3 (Apache 2.0).

--mode det:
    output:  detections [B, N_cls*200, 6]          x1 y1 x2 y2 score cls_id (pixel space)
    DS blobs: detections
    parser:  NvDsInferParseSAM3Det

--mode seg (default):
    output:  detections [B, N_cls*200, 6]          x1 y1 x2 y2 score cls_id (pixel space)
             masks      [B, N_cls*200, Hm, Wm]     probability masks [0, 1]
    DS blobs: detections;masks
    parser:  NvDsInferParseSAM3Full

Output dir: efficientsam3_1008_{mode}_{classes}/

imgsz is HARD-LOCKED at 1008 (unlike SAM3Fixed/export.py for the original SAM3,
which still accepts a custom --imgsz): EfficientSAM3 was trained at 1008 and
degrades sharply when lowered -- not a gradual drop but MISSED detections
outright. Measured on bus.jpg (class bus+person):

    1008: bus 0.85 | 4 people, each with 1 tight box (0.77 0.76 0.69 0.49)
    504 : bus 0.79 | the middle person SPLIT INTO 2 BOXES (0.33 upper body +
                     0.40 lower body), the person at the far left edge is
                     COMPLETELY MISSED, score drops to ~0.2

This is NOT an export bug: ONNX was cross-checked against PyTorch at both
resolutions, with only 0.047px (1008) and 0.016px (504) discrepancy on boxes
scoring >0.5 -- ONNX faithfully reproduces PyTorch; the model itself already
produces bad boxes at 504. The cause is that the feature map is half the size
in each dimension (126 instead of 252 at the stride-4 stage), so a distant or
partially-occluded person no longer has enough pixels for the decoder to
group into a single object.

(Ruled out the pos-emb caching hypothesis: PositionEmbeddingSine caches keyed
on the FEATURE MAP size, and pos-emb is a pure function of that size,
normalized to [0,2pi] -- so a cache hit always returns the correct value for
that size.)

Usage:
    python3 specialize_efficientsam3.py --classes adult child phone --mode det --device cuda
    python3 specialize_efficientsam3.py --classes adult child phone --mode seg --device cuda
"""
import sys
import os
import time
import argparse
import urllib.request

SAM3_DEPLOY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EFFICIENTSAM3_ROOT = os.path.join(SAM3_DEPLOY_ROOT, '..', 'efficientsam3')
sys.path.insert(0, EFFICIENTSAM3_ROOT)
sys.path.insert(0, os.path.join(SAM3_DEPLOY_ROOT, 'SAM3Fixed'))  # for export.py (original SAM3)

import torch
import torch.nn as nn

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.data_misc import FindStage, interpolate
from sam3.model import box_ops
from sam3.model.decoder import TransformerDecoder

from export import generate_ds_config
from build_mixed_precision_engine import build as build_mixed_precision_engine


# ── Fix for the _get_coords bug (same root cause as SAM3Fixed/export.py) ──
#
# TransformerDecoder._get_coords (vendored copy of the same upstream SAM3
# decoder code, in efficientsam3/sam3/sam3/model/decoder.py) generates a
# coordinate grid arange(0,H)/H -> range [0,(H-1)/H], NEVER reaching 1.0.
# Same bug as SAM3Fixed/export.py's _get_coords_denom_fix (see that file for
# the full writeup). Traced empirically: the decoder's box-refinement
# cross-attention here runs on a 72x72 feature map at imgsz=1008 (identical
# to SAM3's own 1008/14=72 grid), so the error is ~1.4% here too, same order
# of magnitude as SAM3Fixed at 1008 -- not the dominant cause of the
# missed/split detections documented below at imgsz<1008 (that's the coarser
# backbone feature map), but a real, free-to-fix parallel bug regardless of
# imgsz.
#
# UNLIKE SAM3Fixed, the grid here is capped below 1.0 rather than a full
# 1.0. Reason: this checkpoint was never retrained against a corrected
# grid, so pushing the max all the way to 1.0 is a bigger train/inference
# distribution shift than the network was calibrated for -- measured on
# bus.jpg (imgsz 420/700/1008), a full 1.0 cap shifts the predicted box by
# up to ~10-11px toward the far corner and pushes it past the frame edge
# (x2 up to 1.0069, needing the clamp below) at BOTH 700 and 1008.
#
# cap=0.987654321 (rather than a round 0.99) is carried over from
# SAM3Fixed/export.py's much more thorough investigation on the original
# SAM3 -- a 71-point sweep (imgsz 420-1400) against a groundtruth proxy
# found it beats both a full 1.0 cap and cap=0.99 on 3 of 4 spot-check
# sizes (see SAM3-DeepStream/SAM3_test/GETCOORDS_INVESTIGATION.md for the
# full writeup). That investigation was run entirely on SAM3Fixed's
# Sam3DetrDecoder, not independently re-verified against this vendored
# EfficientSAM3 decoder -- carried over as the best available estimate
# since it's the same root-cause bug and the same reasoning applies.
_COORDS_CAP = 0.987654321


def _get_coords_denom_fix(H, W, device):
    coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / (H - 1) * _COORDS_CAP
    coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / (W - 1) * _COORDS_CAP
    return coords_h, coords_w


TransformerDecoder._get_coords = staticmethod(_get_coords_denom_fix)

CHECKPOINT_URL = ('https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/'
                   'efficientsam3_ft/efficientsam3_efficientvit.pt')
CHECKPOINT_PATH = os.path.join(EFFICIENTSAM3_ROOT, 'efficientsam3_efficientvit.pt')
IMGSZ = 1008
NUM_TOP_QUERIES = 200


def ensure_checkpoint(path=CHECKPOINT_PATH, url=CHECKPOINT_URL):
    if os.path.exists(path):
        return path
    print(f'[Download] {url} -> {path}')
    token = os.environ.get('HF_TOKEN')
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'} if token else {})
    with urllib.request.urlopen(req) as resp, open(path, 'wb') as f:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f'\r  {downloaded / total * 100:.1f}%', end='')
    print('\n[Download] done')
    return path


def build_ev_m(checkpoint_path=CHECKPOINT_PATH, device='cpu'):
    ensure_checkpoint(checkpoint_path)
    return build_efficientsam3_image_model(
        checkpoint_path=checkpoint_path,
        backbone_type='efficientvit',
        model_name='b1',
        text_encoder_type='MobileCLIP-S0',  # matches checkpoint's actual 4-layer "mct" arch, not README's claimed S1
        text_encoder_context_length=16,
        load_from_HF=False,
        device=device,
    )


def normalize_class(cls):
    """Only used for the engine folder name (paths can't contain spaces):
    'blue bus' and 'blue-bus' both map to 'blue-bus'. The prompt sent to the
    model and labels.txt keep the original text as typed (raw arg, may
    contain spaces)."""
    return '-'.join(cls.strip().split())


def encode_concepts(model, classes, device):
    print(f'[Encode] {len(classes)} class(es): {classes}')
    encoded = []
    for cls in classes:
        text_out = model.backbone.forward_text([cls], device=device)
        encoded.append(text_out)
        print(f"  [+] '{cls}'  language_features {tuple(text_out['language_features'].shape)}")
    return encoded


class EfficientSam3Wrapper(nn.Module):
    """
    Single-input wrapper. Text features for each class are pre-baked as buffers
    (ONNX constants once exported), mirroring SAM3Fixed/export.py's Sam3Wrapper.

    with_mask=False (default):
        pixel_values [B,3,H,W]  ->  detections [B, N_cls*Q, 6]
        columns: x1 y1 x2 y2 score cls_id  (pixel space, score=sigmoid(logit)*presence)

    with_mask=True:
        pixel_values [B,3,H,W]  ->  detections [B, N_cls*Q, 6]
                                    masks      [B, N_cls*Q, Hm, Wm]
        cls_id for query i: i // Q  (Q=200)
    """

    def __init__(self, model, classes, device, imgsz=IMGSZ, with_mask=False, num_top_queries=NUM_TOP_QUERIES):
        super().__init__()
        self.model = model
        self.num_classes = len(classes)
        self.with_mask = with_mask
        self.num_top_queries = num_top_queries

        encoded = encode_concepts(model, classes, device)
        for i, text_out in enumerate(encoded):
            self.register_buffer(f'text_features_{i}', text_out['language_features'])
            self.register_buffer(f'text_mask_{i}', text_out['language_mask'])
            self.register_buffer(f'text_embeds_{i}', text_out['language_embeds'])

        self.register_buffer('scale', torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32))

    def forward(self, pixel_values: torch.Tensor):
        B = pixel_values.shape[0]
        backbone_out = self.model.backbone.forward_image(pixel_values)
        geometric_prompt = self.model._get_dummy_prompt(num_prompts=B)

        # img_ids/text_ids must have length = B, one "find query" per image in
        # the batch (paired with a copy of text already .expand(-1,B,-1)'d
        # below). This used to be hardcoded to torch.tensor([0]) (length 1) in
        # __init__, so _get_img_feats() always picked only the first image
        # (index 0) from backbone_fpn regardless of the actual batch size ->
        # the detections output stayed "pinned" to batch=1 even when
        # pixel_values had batch>1 (the remaining images in the batch were
        # silently ignored -- not an export/ONNX bug).
        find_ids = torch.arange(B, device=pixel_values.device, dtype=torch.long)
        find_stage = FindStage(
            img_ids=find_ids, text_ids=find_ids,
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

        det_chunks, mask_chunks = [], []
        for i in range(self.num_classes):
            backbone_out = dict(backbone_out)
            # language_features/embeds are (Seq, Batch, Dim); language_mask is (Batch, Seq)
            backbone_out['language_features'] = getattr(self, f'text_features_{i}').expand(-1, B, -1)
            backbone_out['language_mask'] = getattr(self, f'text_mask_{i}').expand(B, -1)
            backbone_out['language_embeds'] = getattr(self, f'text_embeds_{i}').expand(-1, B, -1)

            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )
            # The _get_coords denom fix (module top) can push an
            # already-near-edge box slightly outside the frame (e.g.
            # x2=1.01) -- clip back into [0,1] before scaling to pixels.
            out_bbox = outputs['pred_boxes'].clamp(0.0, 1.0)
            out_logits = outputs['pred_logits']
            presence = outputs['presence_logit_dec'].sigmoid().unsqueeze(1)
            scores = (out_logits.sigmoid() * presence).squeeze(-1)  # [B, Q]

            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) * self.scale
            cls_col = torch.full_like(scores, float(i)).unsqueeze(-1)
            det_chunks.append(torch.cat([boxes, scores.unsqueeze(-1), cls_col], dim=-1))

            if self.with_mask:
                m = interpolate(outputs['pred_masks'], pixel_values.shape[-2:],
                                 mode='bilinear', align_corners=False).sigmoid()
                mask_chunks.append(m)

        detections = torch.cat(det_chunks, dim=1).half()  # [B, N_cls*Q, 6]
        if self.with_mask:
            return detections, torch.cat(mask_chunks, dim=1).half()  # [B, N_cls*Q, Hm, Wm]
        return detections


def specialize_and_export(
    classes, output_path, checkpoint_path=CHECKPOINT_PATH,
    opset=17, device='cpu',
    min_batch=1, opt_batch=4, max_batch=16,
    skip_trt=False, with_mask=False,
):
    imgsz = IMGSZ  # hard-locked, see the docstring at the top of the file
    t0 = time.perf_counter()
    model = build_ev_m(checkpoint_path, device)
    print(f'[Load] {time.perf_counter() - t0:.1f}s')

    print(f'[Build] EfficientSam3Wrapper (with_mask={with_mask}) ...')
    wrapper = EfficientSam3Wrapper(model, classes, device, imgsz=imgsz, with_mask=with_mask).to(device).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    print('[Check] Sanity forward pass ...')
    with torch.no_grad():
        out = wrapper(dummy)

    nc = len(classes)
    if with_mask:
        detections, masks = out
        Q = detections.shape[1] // nc
        Hm, Wm = masks.shape[2], masks.shape[3]
        print(f'[Check] detections {tuple(detections.shape)}  expect [1,{nc * Q},6]')
        print(f'[Check] masks      {tuple(masks.shape)}  expect [1,{nc * Q},{Hm},{Wm}]')
        output_names = ['detections', 'masks']
        dynamic_axes = {'pixel_values': {0: 'batch'}, 'detections': {0: 'batch'}, 'masks': {0: 'batch'}}
    else:
        detections = out
        Q = detections.shape[1] // nc
        Hm, Wm = 0, 0
        print(f'[Check] detections {tuple(detections.shape)}  expect [1,{nc * Q},6]')
        output_names = ['detections']
        dynamic_axes = {'pixel_values': {0: 'batch'}, 'detections': {0: 'batch'}}

    print(f'[Check] boxes  [{detections[0, :, 0].min():.1f}, {detections[0, :, 2].max():.1f}]')
    print(f'[Check] scores [{detections[0, :, 4].min():.3f}, {detections[0, :, 4].max():.3f}]')

    print(f'[Export] {output_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, output_path,
            input_names=['pixel_values'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'[Export] Saved: {output_path}  ({size_mb:.0f} MB)')

    engine_path = output_path.replace('.onnx', '.engine')
    if not skip_trt:
        build_mixed_precision_engine(output_path, engine_path, imgsz, min_batch, opt_batch, max_batch)

    generate_ds_config(classes, engine_path, Q=Q, with_mask=with_mask, Hm=Hm, Wm=Wm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    p.add_argument('--classes', nargs='+', required=True)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--min-batch', type=int, default=1)
    p.add_argument('--opt-batch', type=int, default=4)
    p.add_argument('--max-batch', type=int, default=16)
    p.add_argument('--skip-trt', action='store_true')
    p.add_argument('--mode', default='seg', choices=['det', 'seg'],
                    help='det=bbox only  seg=det+masks[B,N,Hm,Wm] fp16 (default)')
    return p.parse_args()


def main():
    args = parse_args()
    suffix = '_'.join(normalize_class(c) for c in args.classes)
    config_dir = os.path.join(os.path.dirname(__file__), f'efficientsam3_{IMGSZ}_{args.mode}_{suffix}')
    os.makedirs(config_dir, exist_ok=True)
    output_path = os.path.join(config_dir, 'efficientsam3.onnx')
    specialize_and_export(
        classes=args.classes,
        output_path=output_path,
        checkpoint_path=args.checkpoint,
        opset=args.opset,
        device=args.device,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        skip_trt=args.skip_trt,
        with_mask=(args.mode == 'seg'),
    )


if __name__ == '__main__':
    main()
