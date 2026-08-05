"""Quick TensorRT-engine inference test for a specialize_efficientsam3.py export.

Usage:
    python3 infer_engine_efficientsam3.py <engine_dir> <image.jpg> [--conf 0.2]
    python3 infer_engine_efficientsam3.py efficientsam3_0_person bus.jpg --conf 0.2
"""
import sys
import os
import argparse

import cv2
import numpy as np
import torch
import tensorrt as trt

IMGSZ = 1008

_PALETTE_HEX = (
    "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD", "FF444F", "CCED00",
    "00F344", "BD00FF", "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3", "7D24FF",
    "7B0068", "FF1B6C", "FC6D2F", "A2FF0B",
)
PALETTE = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in _PALETTE_HEX]


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(f.read())


def preprocess(bgr, imgsz=IMGSZ):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5) -> [-1, 1]
    chw = x.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('engine_dir')
    p.add_argument('image')
    p.add_argument('--conf', type=float, default=0.2)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    engine_path = os.path.join(args.engine_dir, 'efficientsam3.engine')
    labels_path = os.path.join(args.engine_dir, 'labels.txt')
    classes = [l.strip() for l in open(labels_path) if l.strip()]

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    bgr = cv2.imread(args.image)
    h, w = bgr.shape[:2]
    inp = preprocess(bgr).cuda()

    context.set_input_shape('pixel_values', tuple(inp.shape))
    out_shape = tuple(context.get_tensor_shape('detections'))
    out = torch.empty(out_shape, dtype=torch.float16, device='cuda')

    context.set_tensor_address('pixel_values', inp.data_ptr())
    context.set_tensor_address('detections', out.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    det = out[0].float().cpu()
    det = det[det[:, 4] > args.conf]
    print(f'Detections (conf > {args.conf}): {det.shape[0]}')

    sx, sy = w / IMGSZ, h / IMGSZ
    for x1, y1, x2, y2, sc, cls in det.tolist():
        cls = int(cls)
        name = classes[cls] if cls < len(classes) else str(cls)
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        print(f'  class={name} conf={sc:.3f} bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]')
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f'{name} {sc:.2f}'
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(bgr, (int(x1), int(y1) - lh - 6), (int(x1) + lw + 2, int(y1)), color, -1)
        cv2.putText(bgr, label, (int(x1) + 1, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    output = args.output or f'{os.path.basename(args.engine_dir)}_infer.jpg'
    cv2.imwrite(output, bgr)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
