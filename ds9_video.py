#!/usr/bin/env python3
"""
Restream a local video file as RTSP, then run ds9_rtsp.py on it.

Reads model/pipeline settings from config_video.txt.
The video is looped via ffmpeg → mediamtx (localhost:8554) and fed into
the same DeepStream RTSP pipeline used for live cameras.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 ds9_video.py <video.mp4>
"""

import argparse
import configparser
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).parent

_MEDIAMTX_BIN = str(_HERE / ".mediamtx" / "mediamtx")
_MEDIAMTX_URL = (
    "https://github.com/bluenviron/mediamtx/releases/download/"
    "v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz"
)
RTSP_URL = "rtsp://localhost:8554/live"


def _ensure_mediamtx():
    if os.path.exists(_MEDIAMTX_BIN):
        return
    print("[restream] Downloading mediamtx ...")
    os.makedirs(Path(_MEDIAMTX_BIN).parent, exist_ok=True)
    tmp = "/tmp/mediamtx.tar.gz"
    urllib.request.urlretrieve(_MEDIAMTX_URL, tmp)
    with tarfile.open(tmp) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("mediamtx"))
        member.name = "mediamtx"
        tar.extract(member, path=str(Path(_MEDIAMTX_BIN).parent))
    os.chmod(_MEDIAMTX_BIN, 0o755)
    print(f"[restream] mediamtx ready: {_MEDIAMTX_BIN}")


def _video_duration(video: Path) -> int:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ], text=True).strip()
    return max(1, int(float(out)) + 1)  # round up


def _make_temp_config(video: Path) -> str:
    cfg = configparser.RawConfigParser()
    cfg.read(_HERE / "config_video.txt")
    if "sources" not in cfg:
        cfg["sources"] = {}
    cfg["sources"]["urls"] = f"\n    {RTSP_URL}"
    if "pipeline" not in cfg:
        cfg["pipeline"] = {}
    cfg["pipeline"]["run_seconds"] = str(_video_duration(video))
    classes = [c.strip() for c in cfg.get("model", "classes").split(",")]
    mode = cfg.get("model", "mode", fallback="seg")
    default_out = f"output/vis_{mode}_{'_'.join(classes)}_{video.stem}.mp4"
    cfg["pipeline"].setdefault("output_file", default_out)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
    cfg.write(f)
    f.close()
    return f.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video file (.mp4)")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"ERROR: {video} not found"); sys.exit(1)

    _ensure_mediamtx()
    tmp_cfg = _make_temp_config(video)
    procs = []

    try:
        mtx_cfg = str(Path(_MEDIAMTX_BIN).parent / "mediamtx.yml")
        mtx = subprocess.Popen(
            [_MEDIAMTX_BIN, mtx_cfg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(mtx)
        time.sleep(1.0)

        ff = subprocess.Popen([
            "ffmpeg", "-loglevel", "error",
            "-re", "-stream_loop", "-1",
            "-i", str(video),
            "-c:v", "copy", "-an",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            RTSP_URL,
        ])
        procs.append(ff)
        time.sleep(1.5)

        dur = _video_duration(video)
        print(f"[ds9_video] {video.name} → {RTSP_URL}  ({dur}s)")
        subprocess.run(
            [sys.executable, str(_HERE / "ds9_rtsp.py")],
            env={**os.environ, "DS_CONFIG": tmp_cfg},
        )

    finally:
        for p in reversed(procs):
            p.terminate()
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired: p.kill()
        os.unlink(tmp_cfg)
        print("[ds9_video] Done.")


if __name__ == "__main__":
    main()
