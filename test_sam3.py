"""
Standalone SAM3 image inference — no DeepStream, no TensorRT.
Uses the Sam3Wrapper defined in sam3-deploy/specialize.py directly in eager PyTorch.

Usage:
    python3 test_sam3.py image.jpg --classes person car bus --mask 1 --conf 0.3
"""
import sys
import os
import argparse

SAM3_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAM3_ROOT)

import torch
import cv2
import numpy as np
from specialize import Sam3Wrapper, encode_concepts
from transformers.models.sam3 import Sam3Model, Sam3Processor

CHECKPOINT = 'facebook/sam3'
IMGSZ = 1008


def preprocess(bgr, imgsz=IMGSZ):
    resized = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - 127.5) / 127.5  # matches config_infer.txt: net-scale-factor=1/127.5, offsets=127.5
    chw = normalized.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def load_wrapper(classes, device='cuda', with_mask=False):
    token = os.environ.get('HF_TOKEN')
    print(f'[Load] {CHECKPOINT} ...')
    model = Sam3Model.from_pretrained(CHECKPOINT, token=token).to(device).eval()
    processor = Sam3Processor.from_pretrained(CHECKPOINT, token=token)
    text_embeds, attn_masks = encode_concepts(model, processor, classes)
    wrapper = Sam3Wrapper(model, text_embeds, attn_masks, IMGSZ, with_mask=with_mask).to(device).eval()
    return wrapper


def draw_detections(bgr, det, classes, orig_w, orig_h, imgsz=IMGSZ):
    scale_x, scale_y = orig_w / imgsz, orig_h / imgsz
    for x1, y1, x2, y2, score, cls_id in det.tolist():
        cls_id = int(cls_id)
        name = classes[cls_id]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(bgr, f'{name} {score:.2f}', (int(x1), max(int(y1) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--mask', type=int, default=0, choices=[0, 1])
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='sam3_output.jpg')
    args = parser.parse_args()

    wrapper = load_wrapper(args.classes, args.device, with_mask=bool(args.mask))

    bgr = cv2.imread(args.image)
    h, w = bgr.shape[:2]
    x = preprocess(bgr).to(args.device)

    with torch.no_grad():
        out = wrapper(x)
    detections = out[0] if args.mask else out
    det = detections[0].float()
    det = det[det[:, 4] > args.conf]

    print(f'Detections (conf > {args.conf}): {det.shape[0]}')
    for x1, y1, x2, y2, score, cls_id in det.tolist():
        print(f'  class={args.classes[int(cls_id)]} conf={score:.2f} bbox=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]')

    draw_detections(bgr, det, args.classes, w, h)
    cv2.imwrite(args.output, bgr)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
