#!/usr/bin/env python3
"""
Production DeepStream 9 RTSP pipeline for the SAM3Open (dynamic-prompt)
architecture -- each camera gets its own free-text prompt, ONE shared
nvinfer element (vision_decoder.engine) scores every camera's frame
against its own prompt in the SAME batched forward pass: the text
encoding for camera b's prompt is injected into batch row b as a
non-image input layer (see nvdsparsebbox_sam3.cpp + sam3open_text.py), so
there is no per-class loop and no engine rebuild when a prompt changes.

Prompts are hot-reloadable: edit config.txt while running and the pipeline
re-encodes any new prompt (../weight/sam3_1008_open/text_encoder.engine,
via sam3open_text.py), rewrites the snapshot files, and reloads nvinfer's
context via config-file-path set() (SAME path -- this re-triggers
NvDsInferInitializeInputLayers, verified empirically) -- no GStreamer
restart, and unlike SAM3Fixed's hot-reload, no TensorRT engine rebuild at
all, since the engine itself never changes.

Limitation: one prompt per camera (not a class list) -- vision_decoder.onnx
pairs one image with one encoded prompt per batch row. For multiple
classes per camera, use ../SAM3Fixed/ds9_rtsp.py instead (baked union-class
engine, cost scales with class count).

Reads all settings from config.txt (or $DS_CONFIG override):
  [sources]  urls          -- one RTSP url per line, in camera-index order
  [camN]     prompt         -- that camera's open-vocab prompt (required)
  [model]    conf           -- score threshold
  [pipeline] width, height, infer_interval, run_seconds, output_file

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py
"""

import configparser
import math
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from pyservicemaker import Pipeline, BatchMetadataOperator, Probe

import sam3open_text

TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
CONFIG_PATH = Path(os.environ.get("DS_CONFIG", str(_HERE / "config.txt")))
IMGSZ = 1008  # vision_decoder.engine was only built for this size
OPEN_DIR = sam3open_text.open_dir_for(IMGSZ)  # ../weight/sam3_{imgsz}_open/
VISION_ENGINE = os.path.join(OPEN_DIR, "vision_decoder.engine")
LIB_PATH = str(_HERE / "libnvdsinfer_sam3.so")
INFER_CONFIG_PATH = os.path.join(OPEN_DIR, "config_infer.txt")


def tracker_available():
    import ctypes
    if not os.path.exists(TRACKER_LIB):
        return False
    try:
        ctypes.CDLL(TRACKER_LIB)
        return True
    except OSError:
        return False


def read_config():
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    sources = [s.strip() for s in cfg.get("sources", "urls").strip().splitlines() if s.strip()]
    if len(sources) > sam3open_text.MAX_SOURCES:
        sys.exit(f"ERROR: {len(sources)} cameras > MAX_SOURCES={sam3open_text.MAX_SOURCES} "
                  f"(vision_decoder.engine's max batch -- rebuild with a bigger --max-batch to raise this)")
    conf = cfg.getfloat("model", "conf", fallback=0.3)

    camera_prompts = {}
    for i in range(len(sources)):
        sec = f"cam{i}"
        if not cfg.has_section(sec) or not cfg.get(sec, "prompt", fallback=""):
            sys.exit(f"ERROR: config.txt is missing [{sec}]/prompt")
        camera_prompts[i] = cfg.get(sec, "prompt").strip()

    pcfg = dict(
        width=cfg.getint("pipeline", "width", fallback=1280),
        height=cfg.getint("pipeline", "height", fallback=720),
        infer_interval=cfg.getint("pipeline", "infer_interval", fallback=5),
        run_seconds=cfg.getint("pipeline", "run_seconds", fallback=0),
        output_file=cfg.get("pipeline", "output_file", fallback=""),
    )
    return sources, conf, camera_prompts, pcfg


def write_infer_config(batch_size, conf):
    """Static config for vision_decoder.engine -- unlike SAM3Fixed there's no
    per-run class baking, so this is regenerated fresh each run (not just
    patched) but its content never changes except batch-size/threshold."""
    os.makedirs(os.path.dirname(INFER_CONFIG_PATH), exist_ok=True)
    Path(INFER_CONFIG_PATH).write_text(f"""\
[property]
gpu-id=0

model-engine-file={VISION_ENGINE}
num-detected-classes=1
batch-size={batch_size}

net-scale-factor=0.00784313725490196
offsets=127.5;127.5;127.5
model-color-format=0

network-mode=1
process-mode=1
network-type=0
infer-dims=3;{IMGSZ};{IMGSZ}

cluster-mode=4
gie-unique-id=2

# detections [200,5] float32  x1 y1 x2 y2 score (pixel units, no cls_id --
# one prompt per camera row, see nvdsparsebbox_sam3.cpp)
output-blob-names=detections

parse-bbox-func-name=NvDsInferParseSam3OpenDet
custom-lib-path={LIB_PATH}

[class-attrs-all]
pre-cluster-threshold={conf}
""")
    return INFER_CONFIG_PATH


class PromptLabeler(BatchMetadataOperator):
    """classId is always 0 (open-vocab has no fixed label vocabulary) -- sets
    the display label from the camera's own current prompt string instead.
    prompts can be updated LIVE by the watcher thread, guarded by a Lock
    since handle_metadata() runs on GStreamer's own thread."""

    def __init__(self, camera_prompts):
        super().__init__()
        self.lock = threading.Lock()
        self.prompts = dict(camera_prompts)

    def update(self, camera_prompts):
        with self.lock:
            self.prompts = dict(camera_prompts)

    def handle_metadata(self, batch_meta):
        with self.lock:
            prompts = self.prompts
        for frame_meta in batch_meta.frame_items:
            cam = frame_meta.pad_index
            name = prompts.get(cam, "?")
            confs = []
            for obj in frame_meta.object_items:
                conf = float(obj.confidence)
                obj.text_params.display_text = f"{name} {conf:.2f}" if conf > 0.05 else name
                confs.append(conf)
            if confs:
                print(f"  cam{cam} '{name}' #{frame_meta.frame_number}: "
                      f"×{len(confs)} avg={sum(confs)/len(confs):.2f} max={max(confs):.2f}", flush=True)


def build_pipeline(sources, width, height, infer_interval, infer_config, lbl, output_file):
    p = Pipeline("sam3open-ds")
    names = [f"cam{i}" for i in range(len(sources))]

    for uri, name in zip(sources, names):
        p.add("nvurisrcbin", name, {"uri": uri, "latency": 500, "drop-on-latency": True})

    p.add("nvstreammux", "mux", {
        "batch-size":           len(sources),
        "width":                width,
        "height":               height,
        "live-source":          True,
        "batched-push-timeout": 40000,
        "compute-hw":           1,
    })
    for name in names:
        p.link((name, "mux"), ("vsrc_%u", ""))

    interval = infer_interval if tracker_available() else 0
    p.add("nvinfer", "infer", {"config-file-path": infer_config, "interval": interval})
    p.link("mux", "infer")

    last = "infer"
    if tracker_available():
        p.add("nvtracker", "tracker", {
            "ll-lib-file":    TRACKER_LIB,
            "ll-config-file": str(_HERE / "config_tracker_NvSORT_i5.yml"),
            "tracker-width":  640,
            "tracker-height": 384,
            "gpu-id":         0,
        })
        p.link("infer", "tracker")
        last = "tracker"
    else:
        print("[open] WARNING: nvtracker not usable (missing libmosquitto.so.1 -> "
              "`sudo apt install -y libmosquitto1`). Running without a tracker, interval=0.")
    p.attach(last, Probe("prompt_labeler", lbl))

    n = len(sources)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    p.add("nvmultistreamtiler", "tiler", {"rows": rows, "columns": cols, "width": width, "height": height})
    p.link(last, "tiler")

    p.add("nvvideoconvert", "conv1", {})
    p.link("tiler", "conv1")
    p.add("nvdsosd", "osd", {"process-mode": 0, "display-text": True})
    p.link("conv1", "osd")

    # Same encoder gotchas as SAM3Fixed/ds9_rtsp.py -- see CLAUDE.md.
    p.add("nvvideoconvert", "conv2", {})
    p.link("osd", "conv2")
    p.add("capsfilter", "caps_sys", {"caps": "video/x-raw(memory:NVMM),format=NV12"})
    p.link("conv2", "caps_sys")
    p.add("nvv4l2h264enc", "enc", {"bitrate": 8000000, "iframeinterval": 25, "idrinterval": 25})
    p.link("caps_sys", "enc")

    p.add("h264parse", "parse", {})
    p.link("enc", "parse")
    p.add("mp4mux", "mp4mux", {})
    p.link("parse", "mp4mux")
    p.add("filesink", "sink", {"location": output_file, "sync": False, "async": False})
    p.link("mp4mux", "sink")

    return p


def watcher_thread(encoder, conf, pipeline, lbl, active_prompts_box, stop_event):
    """Polls config.txt's mtime. On change: re-reads per-camera
    prompts; if any differ from the active set, re-encodes them (cached
    per prompt string in `encoder`), rewrites the snapshot files, and
    forces nvinfer to reload its context via a same-path config-file-path
    set() -- no engine rebuild, just a snapshot rewrite + context reload."""
    last_mtime = CONFIG_PATH.stat().st_mtime

    while not stop_event.is_set():
        time.sleep(1.0)
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime == last_mtime:
            continue
        last_mtime = mtime
        time.sleep(0.3)

        try:
            _, _, camera_prompts, _ = read_config()
        except SystemExit as e:
            print(f"[watcher] error reading config: {e}", flush=True)
            continue

        if camera_prompts == active_prompts_box[0]:
            continue

        print(f"\n[watcher] config.txt change -> prompts={camera_prompts}", flush=True)
        t0 = time.time()
        sam3open_text.write_snapshot(encoder, camera_prompts)
        lbl.update(camera_prompts)
        try:
            pipeline["infer"].set({"config-file-path": INFER_CONFIG_PATH})
            print(f"[watcher] triggered context reload ({time.time()-t0:.2f}s)", flush=True)
        except Exception as e:
            print(f"[watcher] set() RAISED: {e!r}", flush=True)
        active_prompts_box[0] = camera_prompts


def main():
    sources, conf, camera_prompts, pcfg = read_config()
    if not sources:
        sys.exit("ERROR: config.txt [sources]/urls is empty")

    print(f"[open] {len(sources)} cams  prompts={camera_prompts}")

    encoder = sam3open_text.TextEncoder()
    sam3open_text.write_snapshot(encoder, camera_prompts)
    infer_config = write_infer_config(len(sources), conf)
    print(f"[open] snapshot written, infer config: {infer_config}")

    default_out = f"output/open_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_file = str(_HERE / (pcfg["output_file"] or default_out))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lbl = PromptLabeler(camera_prompts)
    pipeline = build_pipeline(sources, pcfg["width"], pcfg["height"], pcfg["infer_interval"],
                               infer_config, lbl, output_file)

    pipeline_done = threading.Event()
    stop_called   = threading.Event()
    stop_watcher  = threading.Event()
    active_prompts_box = [camera_prompts]

    def _run():
        pipeline.start().wait()
        pipeline_done.set()

    def _shutdown(sig, frame):
        if not stop_called.is_set():
            stop_called.set()
            stop_watcher.set()
            print("\n[open] Flushing 10s before stop...")
            time.sleep(10)
            pipeline.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    run_seconds = pcfg["run_seconds"]
    dur = f"{run_seconds}s" if run_seconds else "∞"
    print(f"[open] interval={pcfg['infer_interval']}  run={dur}  → {output_file}")
    print(f"[open] edit {CONFIG_PATH} while running to hot-swap prompts. Ctrl+C to stop.")

    threading.Thread(target=_run, daemon=True).start()
    watcher = threading.Thread(target=watcher_thread, daemon=True, args=(
        encoder, conf, pipeline, lbl, active_prompts_box, stop_watcher))
    watcher.start()

    t0 = time.time()

    def _timer():
        if not run_seconds:
            return
        for elapsed in range(1, run_seconds + 1):
            if pipeline_done.is_set() or stop_called.is_set():
                return
            time.sleep(1)
            if elapsed % 10 == 0 or elapsed == run_seconds:
                print(f"[open] {elapsed}/{run_seconds}s", flush=True)
        if not stop_called.is_set():
            stop_called.set()
            stop_watcher.set()
            print("[open] Time up, flushing 10s before stop...")
            time.sleep(10)
            pipeline.stop()

    threading.Thread(target=_timer, daemon=True).start()
    pipeline_done.wait()

    size_mb = os.path.getsize(output_file) / 1e6 if os.path.exists(output_file) else 0
    print(f"[open] Done → {output_file}  ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")

    recovered = output_file.replace(".mp4", "_fixed.mp4")
    ret = os.system(
        f"ffmpeg -y -fflags +discardcorrupt+igndts -i {output_file} "
        f"-c:v libx264 -preset fast -crf 23 -movflags +faststart {recovered} 2>/dev/null"
    )
    if ret == 0 and os.path.exists(recovered):
        os.replace(recovered, output_file)
        print(f"[open] Re-encoded → {output_file}  ({os.path.getsize(output_file)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
