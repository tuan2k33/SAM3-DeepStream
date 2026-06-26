"""
SAM3 Specialize — ONNX + TRT (bbox-only or instance segmentation)
------------------------------------------------------------------
Decomposes SAM3 into ONNX-traceable sub-modules for dynamic batch.

--mask false (default):
    output:  detections [B, N_cls*200, 6]          x1 y1 x2 y2 score cls_id (pixel space)
    DS blobs: detections
    parser:  NvDsInferParseSAM3Det

--mask true:
    output:  detections [B, N_cls*200, 6]          x1 y1 x2 y2 score cls_id (pixel space)
             masks      [B, N_cls*200, Hm, Wm]     probability masks [0, 1]
    DS blobs: detections;masks
    parser:  NvDsInferParseSAM3Full

Usage:
    CUDA_VISIBLE_DEVICES=1 python specialize.py --classes adult child phone --device cuda
    CUDA_VISIBLE_DEVICES=1 python specialize.py --classes adult child phone --mask --device cuda
"""

import argparse
import math
import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding


# ── Sinusoidal pos enc ────────────────────────────────────────────────────────

def _sine_pos_enc(B, H, W, device, dtype, num_pos_feats=128, temp=10000):
    scale = 2 * math.pi
    y = torch.arange(1, H+1, dtype=dtype, device=device).view(1,H,1).expand(B,H,W)
    x = torch.arange(1, W+1, dtype=dtype, device=device).view(1,1,W).expand(B,H,W)
    y = y / (H + 1e-6) * scale
    x = x / (W + 1e-6) * scale
    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temp ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x[..., None] / dim_t
    pos_y = y[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ── Vision encoders ───────────────────────────────────────────────────────────

class _VisionEncoder(nn.Module):
    """ViT backbone + FPN neck.

    forward → (fpn_feat_2, fpn_pos_2)          bbox-only path
    Subclass _VisionEncoderFull additionally returns fpn0/fpn1 for the mask decoder.
    """

    def __init__(self, sam3: Sam3Model, device: str, imgsz: int):
        super().__init__()
        bb = sam3.vision_encoder.backbone
        self.patch_embed = bb.embeddings.patch_embeddings
        self.dropout     = bb.embeddings.dropout
        self.layer_norm  = bb.layer_norm
        self.layers      = bb.layers
        self.neck        = sam3.vision_encoder.neck

        ps = bb.config.patch_size
        Hp = Wp = imgsz // ps
        self.Hp, self.Wp = Hp, Wp
        D = bb.config.hidden_size

        for layer in self.layers:
            if getattr(layer, "window_size", 0) != 0:
                continue
            cfg = getattr(layer, "config", bb.config)
            scale = cfg.window_size / Hp
            layer.rotary_emb = Sam3ViTRotaryEmbedding(
                cfg, end_x=Hp, end_y=Wp, scale=scale
            ).to(device)

        orig = bb.embeddings.position_embeddings.data
        n0   = int(orig.shape[1] ** 0.5)
        pe   = orig.reshape(1, n0, n0, D).permute(0, 3, 1, 2)
        rh   = Hp // n0 + 1
        rw   = Wp // n0 + 1
        pe   = pe.tile([1, 1, rh, rw])[:, :, :Hp, :Wp]
        pe   = pe.permute(0, 2, 3, 1).reshape(1, Hp * Wp, D)
        self.register_buffer("vit_pos_embed", pe.to(device))

        npf  = sam3.vision_encoder.neck.config.fpn_hidden_size // 2
        pos2 = _sine_pos_enc(1, Hp, Wp, torch.device(device), torch.float32, npf)
        self.register_buffer("fpn_pos_2", pos2)

    def _encode(self, images: torch.Tensor):
        B   = images.shape[0]
        emb = self.patch_embed(images) + self.vit_pos_embed
        emb = self.dropout(emb)
        hs  = self.layer_norm(emb.view(B, self.Hp, self.Wp, -1))
        for layer in self.layers:
            hs = layer(hs)
        spatial = (hs.view(B, self.Hp * self.Wp, -1)
                     .view(B, self.Hp, self.Wp, -1)
                     .permute(0, 3, 1, 2))
        fpn, _ = self.neck(spatial)
        return fpn, B

    def forward(self, images: torch.Tensor):
        fpn, B = self._encode(images)
        return fpn[2], self.fpn_pos_2.expand(B, -1, -1, -1)


class _VisionEncoderFull(_VisionEncoder):
    """Also returns fpn0 (finest 288×288) and fpn1 (mid 144×144) for the mask decoder."""

    def forward(self, images: torch.Tensor):
        fpn, B = self._encode(images)
        # fpn[0]=finest 288×288, fpn[1]=mid 144×144, fpn[2]=coarse 72×72
        return fpn[2], self.fpn_pos_2.expand(B, -1, -1, -1), fpn[0], fpn[1]


# ── Decoders ──────────────────────────────────────────────────────────────────

class _Decoder(nn.Module):
    """DETR encoder + decoder + scoring.

    forward → (boxes, logits)                   bbox-only path
    Subclass _DecoderFull additionally returns dec_queries/enc_hidden for the mask head.
    """

    def __init__(self, sam3: Sam3Model):
        super().__init__()
        self.detr_encoder        = sam3.detr_encoder
        self.detr_decoder        = sam3.detr_decoder
        self.dot_product_scoring = sam3.dot_product_scoring
        self.box_head            = sam3.detr_decoder.box_head

    @staticmethod
    def _inv_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def _cxcywh_to_xyxy(x):
        cx, cy, w, h = x.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    def _decode(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        enc = self.detr_encoder(
            vision_features   = [fpn_feat_2],
            text_features     = text_features,
            vision_pos_embeds = [fpn_pos_2],
            text_mask         = text_mask,
        )
        dec = self.detr_decoder(
            vision_features     = enc.last_hidden_state,
            text_features       = enc.text_features,
            vision_pos_encoding = enc.pos_embeds_flattened,
            text_mask           = text_mask,
            spatial_shapes      = enc.spatial_shapes,
        )
        offsets    = self.box_head(dec.intermediate_hidden_states)
        ref_inv    = self._inv_sigmoid(dec.reference_boxes)
        all_boxes  = self._cxcywh_to_xyxy((ref_inv + offsets).sigmoid())
        all_logits = self.dot_product_scoring(
            decoder_hidden_states = dec.intermediate_hidden_states,
            text_features         = enc.text_features,
            text_mask             = text_mask,
        ).squeeze(-1)
        return (
            all_boxes[-1],                          # [B, Q, 4]  xyxy normalized [0,1]
            all_logits[-1],                         # [B, Q]     logits (pre-sigmoid)
            dec.intermediate_hidden_states[-1],     # [B, Q, D]  for mask coefficients
            enc.last_hidden_state,                  # [B, Hp*Wp+text, D] for prototypes
        )

    def forward(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        boxes, logits, _, _ = self._decode(fpn_feat_2, fpn_pos_2, text_features, text_mask)
        return boxes, logits


class _DecoderFull(_Decoder):
    """Also returns dec_queries and enc_hidden for the mask head."""

    def forward(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        return self._decode(fpn_feat_2, fpn_pos_2, text_features, text_mask)


# ── Unified wrapper ───────────────────────────────────────────────────────────

class Sam3Wrapper(nn.Module):
    """
    Single-input wrapper. Text features for each class are pre-baked as ONNX constants.

    with_mask=False (default):
        pixel_values [B,3,H,W]  →  detections [B, N_cls*Q, 6]
        columns: x1 y1 x2 y2 score cls_id  (pixel space, score=sigmoid(logit))

    with_mask=True:
        pixel_values [B,3,H,W]  →  detections [B, N_cls*Q, 6]
                                    masks      [B, N_cls*Q, Hm, Wm]
        cls_id for query i: i // Q  (Q=200)
    """

    def __init__(self, sam3: Sam3Model, text_embeds_all: torch.Tensor,
                 attn_masks_all: torch.Tensor, imgsz: int = 1008, with_mask: bool = False):
        super().__init__()
        device = str(next(sam3.parameters()).device)
        self.with_mask   = with_mask
        self.num_classes = text_embeds_all.shape[0]

        if with_mask:
            self.vision_enc    = _VisionEncoderFull(sam3, device, imgsz)
            self.decoder       = _DecoderFull(sam3)
            self.mask_embedder = sam3.mask_decoder.mask_embedder
            self.pixel_decoder = sam3.mask_decoder.pixel_decoder
            self.instance_proj = sam3.mask_decoder.instance_projection
            ps      = sam3.vision_encoder.backbone.config.patch_size
            self.Hp = imgsz // ps
            self.Wp = imgsz // ps
        else:
            self.vision_enc = _VisionEncoder(sam3, device, imgsz)
            self.decoder    = _Decoder(sam3)

        for i in range(self.num_classes):
            self.register_buffer(f"text_feat_{i}", text_embeds_all[i:i+1])
            self.register_buffer(f"text_mask_{i}", attn_masks_all[i:i+1] > 0)

        self.register_buffer(
            "scale",
            torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32),
        )

    def _prototypes(self, fpn0, fpn1, enc_hidden):
        B = fpn0.shape[0]
        enc_vis = (
            enc_hidden[:, :self.Hp * self.Wp, :]
            .transpose(1, 2)
            .reshape(B, -1, self.Hp, self.Wp)
        )
        pixel_embed = self.pixel_decoder([fpn0, fpn1, enc_vis])
        return self.instance_proj(pixel_embed)  # [B, K, 288, 288]

    def forward(self, pixel_values: torch.Tensor):
        B = pixel_values.shape[0]

        if self.with_mask:
            fpn2, fpn_pos, fpn0, fpn1 = self.vision_enc(pixel_values)
        else:
            fpn2, fpn_pos = self.vision_enc(pixel_values)

        det_chunks  = []
        mask_chunks = []

        for i in range(self.num_classes):
            text_feat = getattr(self, f"text_feat_{i}").expand(B, -1, -1)
            text_mask = getattr(self, f"text_mask_{i}").expand(B, -1)

            if self.with_mask:
                pred_boxes, pred_logits, dec_queries, enc_hidden = self.decoder(
                    fpn2, fpn_pos, text_feat, text_mask
                )
                protos = self._prototypes(fpn0, fpn1, enc_hidden)
                coeffs = self.mask_embedder(dec_queries)
                masks  = torch.einsum("bqk,bkhw->bqhw", coeffs, protos).sigmoid()
                masks  = F.interpolate(masks, (36, 36), mode="bilinear", align_corners=False).half()
                mask_chunks.append(masks)
            else:
                pred_boxes, pred_logits = self.decoder(fpn2, fpn_pos, text_feat, text_mask)

            boxes_px = pred_boxes * self.scale
            scores   = pred_logits.sigmoid().unsqueeze(-1)
            cls_id   = torch.full(
                (B, pred_boxes.shape[1], 1), float(i),
                dtype=pred_boxes.dtype, device=pred_boxes.device,
            )
            det_chunks.append(torch.cat([boxes_px, scores, cls_id], dim=-1))

        detections = torch.cat(det_chunks, dim=1).half()  # [B, N_cls*Q, 6] fp16

        if self.with_mask:
            return detections, torch.cat(mask_chunks, dim=1)  # [B, N_cls*Q, Hm, Wm] fp16
        return detections


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_imgsz(model, imgsz, device="cpu"):
    print(f"[Validate] imgsz={imgsz} ...")
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    try:
        with torch.no_grad():
            out = model.get_vision_features(pixel_values=dummy)
        print(f"[Validate] OK — {tuple(out.last_hidden_state.shape)}")
        return True
    except Exception as e:
        print(f"[Validate] FAILED: {str(e)[:200]}")
        return False


def encode_concepts(model, processor, classes):
    print(f"[Encode] {len(classes)} class(es): {classes}")
    embeds, masks = [], []
    dev = next(model.parameters()).device
    for cls in classes:
        inp = processor(text=cls, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            out = model.get_text_features(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )
        embeds.append(out.pooler_output)
        masks.append(inp["attention_mask"])
        print(f"  [+] '{cls}'  {tuple(out.pooler_output.shape)}")
    return torch.cat(embeds, dim=0), torch.cat(masks, dim=0)


def run_trtexec(onnx_path, engine_path, imgsz, min_batch, opt_batch, max_batch):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=pixel_values:{min_batch}x3x{imgsz}x{imgsz}",
        f"--optShapes=pixel_values:{opt_batch}x3x{imgsz}x{imgsz}",
        f"--maxShapes=pixel_values:{max_batch}x3x{imgsz}x{imgsz}",
    ]
    print(f"\n[TRT] {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    r  = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"trtexec failed (exit {r.returncode})")
    print(f"\n[TRT] Saved: {engine_path}  ({(time.perf_counter()-t0)/60:.1f} min)")


def generate_ds_config(classes, engine_path, Q=200, with_mask=False, Hm=288, Wm=288):
    config_dir = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(config_dir, exist_ok=True)

    nc = len(classes)
    N  = nc * Q

    labels_path = os.path.join(config_dir, "labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(classes) + "\n")

    lib_path    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnvdsinfer_sam3.so")
    class_attrs = "\n".join(
        f"[class-attrs-{i}]\npre-cluster-threshold=0.3" for i in range(nc)
    )

    if with_mask:
        output_blob_names = "detections;masks"
        blob_comment      = f"# detections [{N},6] fp16  masks [{N},{Hm},{Wm}] fp16"
        parser_line       = f"parse-bbox-instance-mask-func-name=NvDsInferParseSAM3Full"
        network_type      = 3
    else:
        output_blob_names = "detections"
        blob_comment      = f"# detections [{N},6] fp16  x1 y1 x2 y2 score cls_id"
        parser_line       = f"parse-bbox-func-name=NvDsInferParseSAM3Det"
        network_type      = 0

    config_path = os.path.join(config_dir, "config_infer.txt")
    with open(config_path, "w") as f:
        f.write(f"""\
[property]
gpu-id=0

model-engine-file={os.path.abspath(engine_path)}
labelfile-path={labels_path}
num-detected-classes={nc}
batch-size=4

net-scale-factor=0.00784313725490196
offsets=127.5;127.5;127.5
model-color-format=0

network-mode=1
process-mode=1
network-type={network_type}
{"segmentation-threshold=0.5" if with_mask else ""}
cluster-mode=4
gie-unique-id=3

{blob_comment}
output-blob-names={output_blob_names}

{parser_line}
custom-lib-path={lib_path}

[class-attrs-all]
topk={N}

{class_attrs}
""")
    print(f"[Config] {config_dir}/config_infer.txt  ({nc} classes, N={N}, mask={with_mask})")


# ── Main ──────────────────────────────────────────────────────────────────────

def specialize_and_export(
    checkpoint, classes, output_path,
    imgsz=1008, opset=17, device="cpu",
    min_batch=1, opt_batch=4, max_batch=16,
    skip_trt=False, with_mask=False,
):
    token = os.environ.get("HF_TOKEN")
    print(f"[Load] {checkpoint} ...")
    t0        = time.perf_counter()
    model     = Sam3Model.from_pretrained(checkpoint, token=token).to(device).eval()
    processor = Sam3Processor.from_pretrained(checkpoint, token=token)
    print(f"[Load] {time.perf_counter()-t0:.1f}s")

    if not validate_imgsz(model, imgsz, device):
        raise ValueError(f"imgsz={imgsz} not compatible")

    text_embeds, attn_masks = encode_concepts(model, processor, classes)

    print(f"[Build] Sam3Wrapper (with_mask={with_mask}) ...")
    wrapper = Sam3Wrapper(
        model, text_embeds, attn_masks, imgsz, with_mask=with_mask,
    ).to(device).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    print("[Check] Sanity forward pass ...")
    with torch.no_grad():
        out = wrapper(dummy)

    nc = len(classes)
    if with_mask:
        detections, masks = out
        Q       = detections.shape[1] // nc
        Hm, Wm  = masks.shape[2], masks.shape[3]
        print(f"[Check] detections {tuple(detections.shape)}  expect [1,{nc*Q},6]")
        print(f"[Check] masks      {tuple(masks.shape)}  expect [1,{nc*Q},{Hm},{Wm}]")
        print(f"[Check] boxes  [{detections[0,:,0].min():.1f}, {detections[0,:,2].max():.1f}]")
        print(f"[Check] scores [{detections[0,:,4].min():.3f}, {detections[0,:,4].max():.3f}]")
        print(f"[Check] masks  [{masks.min():.3f}, {masks.max():.3f}]")
        output_names = ["detections", "masks"]
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "detections":   {0: "batch"},
            "masks":        {0: "batch"},
        }
    else:
        detections = out
        Q       = detections.shape[1] // nc
        Hm, Wm  = 0, 0
        print(f"[Check] detections {tuple(detections.shape)}  expect [1,{nc*Q},6]")
        print(f"[Check] boxes  [{detections[0,:,0].min():.1f}, {detections[0,:,2].max():.1f}]")
        print(f"[Check] scores [{detections[0,:,4].min():.3f}, {detections[0,:,4].max():.3f}]")
        output_names = ["detections"]
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "detections":   {0: "batch"},
        }

    print(f"[Export] {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, output_path,
            input_names  = ["pixel_values"],
            output_names = output_names,
            dynamic_axes = dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[Export] Saved: {output_path}  ({size_mb:.0f} MB)")

    engine_path = output_path.replace(".onnx", ".engine")
    if not skip_trt:
        run_trtexec(output_path, engine_path, imgsz, min_batch, opt_batch, max_batch)

    generate_ds_config(classes, engine_path, Q=Q, with_mask=with_mask, Hm=Hm, Wm=Wm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="facebook/sam3")
    p.add_argument("--classes",    nargs="+", required=True)
    p.add_argument("--imgsz",      type=int, default=1008)
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--min-batch",  type=int, default=1)
    p.add_argument("--opt-batch",  type=int, default=4)
    p.add_argument("--max-batch",  type=int, default=16)
    p.add_argument("--skip-trt",   action="store_true")
    p.add_argument("--mask",       type=int, default=1, choices=[0, 1],
                   help="0=det only  1=det+masks[B,N,36,36] fp16 (default)")
    return p.parse_args()


def main():
    args        = parse_args()
    suffix      = "_".join(args.classes)
    config_dir  = os.path.join(os.path.dirname(__file__), f"sam3_{args.mask}_{suffix}")
    os.makedirs(config_dir, exist_ok=True)
    output_path = os.path.join(config_dir, "sam3.onnx")
    specialize_and_export(
        checkpoint=args.checkpoint,
        classes=args.classes,
        output_path=output_path,
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        skip_trt=args.skip_trt,
        with_mask=bool(args.mask),
    )


if __name__ == "__main__":
    main()
