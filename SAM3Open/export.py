#!/usr/bin/env python3
"""
Dynamic-prompt ONNX export for the ORIGINAL SAM3 (facebook/sam3, 840M params),
mirroring efficientsam3's export_onnx_open_vocab_paired.py: prompts are
encoded and passed in at RUNTIME (not baked as buffers like Sam3Wrapper in
SAM3Fixed/export.py), so no re-export is needed to change classes.

Two graphs:
  1. text_encoder.onnx  : input_ids [B,32] + attention_mask [B,32]
                              -> text_features [B,32,256], text_mask [B,32]
  2. vision_decoder.onnx: pixel_values [B,3,H,W] + text_features [B,32,256]
                              + text_mask [B,32]  -> detections [B,200,5]

Unlike efficientsam3, SAM3's detr_encoder/detr_decoder consume text_features
per BATCH ROW directly (no FindStage.text_ids indexing) -- so image[i] is
automatically paired with text_features[i] (row i), as long as the caller
puts them in matching order. Any batch of images with ANY same-size batch of
prompts works out of the box; no separate "paired" vs "multiclass" variant is
needed the way it was for efficientsam3 (see build_dynamic_engines.py for the
"every image x every class" pattern using this same graph, just called once
per class with the SAME image batch).

Usage:
    python3 export.py --imgsz 1008   # -> ../weight/sam3_1008_open/
"""
import os
import sys
import time
import argparse

import torch
import torch.nn as nn

SAM3_DEPLOY_ROOT = os.path.dirname(os.path.abspath(__file__))
# _VisionEncoder/_Decoder live in the sibling SAM3Fixed/export.py, not here
# -- both architectures share the same ONNX-traceable submodules.
sys.path.insert(0, os.path.join(os.path.dirname(SAM3_DEPLOY_ROOT), "SAM3Fixed"))

from transformers.models.sam3 import Sam3Model, Sam3Processor
from export import _VisionEncoder, _Decoder

IMGSZ_DEFAULT = 1008
CTX_LEN = 32  # SAM3 text tokenizer context length (see encode_concepts() output shape)


class TextEncoderWrapper(nn.Module):
    """Only holds text_encoder + text_projection (what get_text_features()
    actually calls) -- NOT a reference to the full sam3 model, otherwise
    ONNX export serializes the whole vision backbone as unused initializers
    (bloated the first attempt's text_encoder.onnx to 1.4GB)."""

    def __init__(self, sam3):
        super().__init__()
        self.text_encoder = sam3.text_encoder
        self.text_projection = sam3.text_projection

    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooler_output = self.text_projection(text_outputs.last_hidden_state)
        text_mask = attention_mask > 0
        return pooler_output, text_mask


class DynamicVisionDecoderWrapper(nn.Module):
    """pixel_values [B,3,H,W] + text_features [B,32,256] + text_mask [B,32]
    -> detections [B,200,5]. image[i] paired with text[i] (row i) -- SAM3's
    detr_encoder/decoder process per-batch-row directly, no gather needed."""

    def __init__(self, sam3, device, imgsz=IMGSZ_DEFAULT):
        super().__init__()
        self.vision_enc = _VisionEncoder(sam3, device, imgsz)
        self.decoder = _Decoder(sam3)
        self.register_buffer('scale', torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32))

    def forward(self, pixel_values, text_features, text_mask):
        fpn_feat_2, fpn_pos_2 = self.vision_enc(pixel_values)
        boxes, logits, presence = self.decoder(fpn_feat_2, fpn_pos_2, text_features, text_mask)
        boxes_px = boxes * self.scale
        scores = logits.sigmoid() * presence.sigmoid()
        return torch.cat([boxes_px, scores.unsqueeze(-1)], dim=-1)  # [B,Q,5]


def export(args):
    device = args.device
    t0 = time.perf_counter()
    print('[Load] facebook/sam3 ...')
    model = Sam3Model.from_pretrained('facebook/sam3').to(device).eval()
    processor = Sam3Processor.from_pretrained('facebook/sam3')
    print(f'[Load] {time.perf_counter() - t0:.1f}s')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- text encoder ---
    text_wrapper = TextEncoderWrapper(model).to(device).eval()
    dummy_prompt = processor(text="dog", return_tensors="pt")
    dummy_ids = dummy_prompt["input_ids"].to(device)
    dummy_mask = dummy_prompt["attention_mask"].to(device)
    print(f'[Info] tokenized shape: {tuple(dummy_ids.shape)} (CTX_LEN={CTX_LEN})')

    with torch.no_grad():
        text_features, text_mask = text_wrapper(dummy_ids, dummy_mask)
    print(f'[Check] text_features {tuple(text_features.shape)} text_mask {tuple(text_mask.shape)}')

    text_onnx_path = os.path.join(args.output_dir, 'text_encoder.onnx')
    print(f'[Export] {text_onnx_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            text_wrapper, (dummy_ids, dummy_mask), text_onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_features', 'text_mask'],
            dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'},
                           'text_features': {0: 'batch'}, 'text_mask': {0: 'batch'}},
            opset_version=args.opset, dynamo=False,
        )
    print(f'[Export] Saved: {text_onnx_path}')

    # --- vision + decoder (paired by batch row) ---
    vision_wrapper = DynamicVisionDecoderWrapper(model, device, imgsz=args.imgsz).to(device).eval()
    dummy_pixels = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    print('[Check] Vision+decoder sanity forward pass ...')
    with torch.no_grad():
        detections = vision_wrapper(dummy_pixels, text_features, text_mask)
    print(f'[Check] detections {tuple(detections.shape)}')

    vision_onnx_path = os.path.join(args.output_dir, 'vision_decoder.onnx')
    print(f'[Export] {vision_onnx_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            vision_wrapper, (dummy_pixels, text_features, text_mask), vision_onnx_path,
            input_names=['pixel_values', 'text_features', 'text_mask'],
            output_names=['detections'],
            dynamic_axes={'pixel_values': {0: 'batch'}, 'text_features': {0: 'batch'},
                           'text_mask': {0: 'batch'}, 'detections': {0: 'batch'}},
            opset_version=args.opset, dynamo=False,
        )
    print(f'[Export] Saved: {vision_onnx_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--imgsz', type=int, default=IMGSZ_DEFAULT)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default=None,
                    help='default: ../weight/sam3_{imgsz}_open/ (repo-root weight/ folder, shared with SAM3Fixed)')
    args = p.parse_args()
    if args.output_dir is None:
        weight_dir = os.path.join(os.path.dirname(SAM3_DEPLOY_ROOT), 'weight')
        args.output_dir = os.path.join(weight_dir, f'sam3_{args.imgsz}_open')
    return args


if __name__ == '__main__':
    export(parse_args())
