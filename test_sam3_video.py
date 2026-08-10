"""
Standalone SAM3 video inference — no DeepStream, no TensorRT.
Runs pure-PyTorch model, writes a temp mp4v video, then converts to H.264 and
deletes the temp file.

Usage:
    python3 test_sam3_video.py video.mp4 --classes person car bus --mode det --conf 0.3
"""
import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
from test_sam3 import load_wrapper, preprocess, draw_detections, IMGSZ

TMP_PATH = 'sam3_video_tmp.mp4'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--mode', default='det', choices=['det', 'seg'])
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='sam3_video_out.mp4')
    parser.add_argument('--max-frames', type=int, default=0, help='0 = whole video')
    args = parser.parse_args()

    wrapper = load_wrapper(args.classes, args.device, with_mask=(args.mode == 'seg'))

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(TMP_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # warmup
    ret, frame = cap.read()
    if ret:
        with torch.no_grad():
            wrapper(preprocess(frame).to(args.device))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = preprocess(frame).to(args.device)
        with torch.no_grad():
            out = wrapper(x)
        detections = out[0] if args.mode == 'seg' else out
        det = detections[0].float()
        det = det[det[:, 4] > args.conf]

        draw_detections(frame, det, args.classes, w, h)
        writer.write(frame)

        count += 1
        if args.max_frames and count >= args.max_frames:
            break

    cap.release()
    writer.release()

    subprocess.run(
        ['ffmpeg', '-y', '-i', TMP_PATH, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', args.output],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(TMP_PATH)
    print(f'Processed {count} frames -> {args.output}')


if __name__ == '__main__':
    main()
