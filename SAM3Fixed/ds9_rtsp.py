#!/usr/bin/env python3
"""
Production DeepStream 9 RTSP pipeline for the SAM3Fixed (baked-classes)
architecture -- multi-camera, each camera can have its own prompt, all
sharing ONE engine baked with the union of every camera's classes
(spec_cache.py), post-filtered per camera by cls_id (PerCameraFilter).

Prompts are hot-reloadable: edit config.txt while the pipeline is running
and it recomputes the union, builds a new engine if one doesn't exist yet
(spec_cache.ensure_spec), and swaps nvinfer onto it live via
config-file-path -- no GStreamer restart.

Reads all settings from config.txt (or $DS_CONFIG override):
  [sources]  urls           -- one RTSP url per line, in camera-index order
  [model]    mode, imgsz    -- shared by every camera (one engine)
             classes, conf  -- fallback prompt/threshold for cameras with
                                no own [camN] section
  [camN]     classes         -- optional per-camera prompt override
  [pipeline] width, height, infer_interval, run_seconds, output_file

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py          # live cameras (config.txt)

To test against a video file instead of a live camera, restream it as fake
RTSP yourself (e.g. via mediamtx + ffmpeg) and point a `urls` entry at that
local stream.
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

import spec_cache

TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
CONFIG_PATH = Path(os.environ.get("DS_CONFIG", str(_HERE / "config.txt")))


def tracker_available():
    """NvSORT needs libmosquitto.so.1 (from the libmosquitto1 package). If
    it's missing, nvtracker fails at init and takes the whole pipeline down
    with it -- check for it up front."""
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
    mode = cfg.get("model", "mode", fallback="seg")
    imgsz = cfg.getint("model", "imgsz", fallback=1008)
    conf = cfg.getfloat("model", "conf", fallback=0.4)
    default_classes = [c.strip() for c in cfg.get("model", "classes", fallback="object").split(",") if c.strip()]

    camera_prompts = {}
    for i in range(len(sources)):
        sec = f"cam{i}"
        if cfg.has_section(sec) and cfg.get(sec, "classes", fallback=""):
            camera_prompts[i] = [c.strip() for c in cfg.get(sec, "classes").split(",") if c.strip()]
        else:
            camera_prompts[i] = default_classes

    pcfg = dict(
        width=cfg.getint("pipeline", "width", fallback=1920),
        height=cfg.getint("pipeline", "height", fallback=1080),
        infer_interval=cfg.getint("pipeline", "infer_interval", fallback=5),
        run_seconds=cfg.getint("pipeline", "run_seconds", fallback=0),  # 0 = run until Ctrl+C
        output_file=cfg.get("pipeline", "output_file", fallback=""),
    )
    return sources, mode, imgsz, conf, camera_prompts, pcfg


def patch_infer_config(engine_dir, batch_size, conf, num_classes):
    """Fix up batch-size to match the camera count + set a threshold for
    EVERY class in the union. (export.py only emits [class-attrs-0];
    missing entries fall back to threshVec[0] in the parser -- declare them
    explicitly to be safe.)"""
    path = os.path.join(engine_dir, "config_infer.txt")
    text = Path(path).read_text()
    text = re.sub(r"^batch-size=.*$", f"batch-size={batch_size}", text, flags=re.M)
    text = re.sub(r"^num-detected-classes=.*$", f"num-detected-classes={num_classes}", text, flags=re.M)
    text = re.sub(r"\n\[class-attrs-\d+\][^\[]*", "\n", text).rstrip() + "\n"
    for c in range(num_classes):
        text += f"\n[class-attrs-{c}]\npre-cluster-threshold={conf}\n"
    Path(path).write_text(text)
    return path


class PerCameraFilter(BatchMetadataOperator):
    """Filters detections by each camera's cls_id whitelist, and logs a
    per-frame summary. union/allow can be updated LIVE by the watcher
    thread when config.txt changes -- guarded by a Lock since
    handle_metadata() runs on GStreamer's own thread, different from the
    watcher's. A few "transitional" frames right at swap time are accepted
    (the new union is already set but nvinfer is still on the old engine)."""

    def __init__(self, union_classes, cam_to_cls_ids):
        super().__init__()
        self.lock = threading.Lock()
        self.union = union_classes
        self.allow = cam_to_cls_ids

    def update(self, union_classes, cam_to_cls_ids):
        with self.lock:
            self.union = union_classes
            self.allow = cam_to_cls_ids

    def handle_metadata(self, batch_meta):
        with self.lock:
            union, allow = self.union, self.allow
        for frame_meta in batch_meta.frame_items:
            cam = frame_meta.pad_index
            allowed = set(allow.get(cam, []))
            cls_conf = {}
            for obj in frame_meta.object_items:
                cls = int(obj.class_id)
                name = union[cls] if cls < len(union) else str(cls)
                if cls in allowed:
                    conf = float(obj.confidence)
                    obj.text_params.display_text = f"{name} {conf:.2f}" if conf > 0.05 else name
                    cls_conf.setdefault(name, []).append(conf)
                else:
                    obj.rect_params.border_width = 0
                    obj.rect_params.has_bg_color = 0
                    obj.text_params.display_text = ""
                    # clearing the text alone still leaves the label
                    # background (a black box) -- must turn that off too
                    obj.text_params.set_bg_clr = 0
            if cls_conf:
                parts = [f"{n}×{len(cs)} avg={sum(cs)/len(cs):.2f} max={max(cs):.2f}"
                         for n, cs in cls_conf.items()]
                print(f"  cam{cam} #{frame_meta.frame_number}: {' | '.join(parts)}", flush=True)


def build_pipeline(sources, width, height, infer_interval, infer_config, mode, flt, output_file):
    p = Pipeline("sam3fixed-ds")
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
        # NvSORT: cascaded/greedy score matching, no separate visual tracker
        # like NvDCF -> much cheaper, tolerates a high interval. Uses a
        # custom config (probationAge=1): the stock probationAge=5 requires
        # 5 consecutive matches (~1s at interval=5) before a track "matures"
        # and emits a bbox -- objects can cross the frame faster than that,
        # so a box would never be emitted (verified: interval=0 gives boxes,
        # interval=5 with stock config gives zero detections).
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
        print("[ds9] WARNING: nvtracker not usable (missing libmosquitto.so.1 -> "
              "`sudo apt install -y libmosquitto1`). Running without a tracker, interval=0.")
    # filter AFTER the tracker so boxes get propagated through frames skipped by interval
    p.attach(last, Probe("percam_filter", flt))

    n = len(sources)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    p.add("nvmultistreamtiler", "tiler", {"rows": rows, "columns": cols, "width": width, "height": height})
    p.link(last, "tiler")

    p.add("nvvideoconvert", "conv1", {})
    p.link("tiler", "conv1")

    p.add("nvdsosd", "osd", {"process-mode": 0, "display-text": True, "display-mask": mode == "seg"})
    p.link("conv1", "osd")

    # Encoder -- measured on this hardware (RTX 50-series, driver 610.x,
    # DS 9.1). nvv4l2h264enc needs an explicit NV12 capsfilter: without it,
    # GStreamer negotiates I420 (listed first in the sink template) and the
    # encoder misreads chroma, corrupting the image (bitstream stays valid,
    # so "ffmpeg -v error" reports 0 errors -- only visual inspection
    # catches it). idrinterval is also required explicitly; iframeinterval
    # alone gets ignored. nvh264enc/nvcudah264enc don't work at all with a
    # DeepStream hardware-decode source on this GPU/driver -- see CLAUDE.md.
    p.add("nvvideoconvert", "conv2", {})
    p.link("osd", "conv2")
    p.add("capsfilter", "caps_sys", {"caps": "video/x-raw(memory:NVMM),format=NV12"})
    p.link("conv2", "caps_sys")
    p.add("nvv4l2h264enc", "enc", {
        "bitrate":        8000000,
        "iframeinterval": 25,
        "idrinterval":    25,
    })
    p.link("caps_sys", "enc")

    p.add("h264parse", "parse", {})
    p.link("enc", "parse")

    p.add("mp4mux", "mp4mux", {})
    p.link("parse", "mp4mux")

    p.add("filesink", "sink", {"location": output_file, "sync": False, "async": False})
    p.link("mp4mux", "sink")

    return p


def watcher_thread(mode, imgsz, conf, n_cams, pipeline, flt, active_union_box, stop_event):
    """Polls config.txt's mtime. On change: re-reads per-camera prompts,
    recomputes the union; if it DIFFERS from the active union -> ensure_spec
    (build if missing) -> patch config_infer.txt -> nvinfer.set({"config-
    file-path": ...}) -> update the filter. If the union hasn't changed
    (only per-camera whitelists moved), just update the filter, no engine
    swap needed."""
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
        time.sleep(0.3)  # let the editor finish writing (avoid reading a half-written file)

        try:
            _, _, _, _, camera_prompts, _ = read_config()
        except Exception as e:
            print(f"[watcher] error reading config: {e}", flush=True)
            continue

        union, cam_map = spec_cache.union_classes(camera_prompts)
        print(f"\n[watcher] config.txt change -> prompts={camera_prompts}", flush=True)
        print(f"[watcher] new union = {union}", flush=True)

        if union == active_union_box[0]:
            print("[watcher] union unchanged -> only updating the whitelist, NOT swapping the engine", flush=True)
            flt.update(union, cam_map)
            continue

        t0 = time.time()
        try:
            engine_dir = spec_cache.ensure_spec(union, mode=mode, imgsz=imgsz)
        except Exception as e:
            print(f"[watcher] engine build failed: {e}", flush=True)
            continue
        infer_config = patch_infer_config(engine_dir, n_cams, conf, len(union))
        print(f"[watcher] engine ready ({time.time()-t0:.1f}s) -> {engine_dir}", flush=True)

        flt.update(union, cam_map)
        try:
            pipeline["infer"].set({"config-file-path": infer_config})
            print(f"[watcher] triggered OTF swap -> {infer_config}", flush=True)
        except Exception as e:
            print(f"[watcher] set() RAISED: {e!r}", flush=True)
        active_union_box[0] = union


def main():
    sources, mode, imgsz, conf, camera_prompts, pcfg = read_config()
    if not sources:
        sys.exit("ERROR: config.txt [sources]/urls is empty")

    union, cam_map = spec_cache.union_classes(camera_prompts)
    print(f"[ds9] {len(sources)} cams  prompts={camera_prompts}")
    print(f"[ds9] union (shared engine): {union}  cam->cls_id: {cam_map}")

    engine_dir = spec_cache.ensure_spec(union, mode=mode, imgsz=imgsz)
    infer_config = patch_infer_config(engine_dir, len(sources), conf, len(union))
    print(f"[ds9] engine: {engine_dir}")

    default_out = f"output/vis_{mode}_{'_'.join(union)}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_file = str(_HERE / (pcfg["output_file"] or default_out))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    flt = PerCameraFilter(union, cam_map)
    pipeline = build_pipeline(sources, pcfg["width"], pcfg["height"], pcfg["infer_interval"],
                               infer_config, mode, flt, output_file)

    pipeline_done = threading.Event()
    stop_called   = threading.Event()
    stop_watcher  = threading.Event()
    active_union_box = [union]

    def _run():
        pipeline.start().wait()
        pipeline_done.set()

    def _shutdown(sig, frame):
        if not stop_called.is_set():
            stop_called.set()
            stop_watcher.set()
            print("\n[ds9] Flushing 10s before stop...")
            time.sleep(10)
            pipeline.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    run_seconds = pcfg["run_seconds"]
    dur = f"{run_seconds}s" if run_seconds else "∞"
    print(f"[ds9] interval={pcfg['infer_interval']}  run={dur}  → {output_file}")
    print(f"[ds9] edit {CONFIG_PATH} while running to hot-swap prompts. Ctrl+C to stop.")

    threading.Thread(target=_run, daemon=True).start()
    watcher = threading.Thread(target=watcher_thread, daemon=True, args=(
        mode, imgsz, conf, len(sources), pipeline, flt, active_union_box, stop_watcher))
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
                print(f"[ds9] {elapsed}/{run_seconds}s", flush=True)
        if not stop_called.is_set():
            stop_called.set()
            stop_watcher.set()
            print("[ds9] Time up, flushing 10s before stop...")
            time.sleep(10)
            pipeline.stop()

    threading.Thread(target=_timer, daemon=True).start()
    pipeline_done.wait()

    size_mb = os.path.getsize(output_file) / 1e6 if os.path.exists(output_file) else 0
    print(f"[ds9] Done → {output_file}  ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")

    # If mp4 moov is missing (pipeline killed without EOS), recover with ffmpeg
    recovered = output_file.replace(".mp4", "_fixed.mp4")
    ret = os.system(
        f"ffmpeg -y -fflags +discardcorrupt+igndts -i {output_file} "
        f"-c:v libx264 -preset fast -crf 23 -movflags +faststart {recovered} 2>/dev/null"
    )
    if ret == 0 and os.path.exists(recovered):
        os.replace(recovered, output_file)
        print(f"[ds9] Re-encoded → {output_file}  ({os.path.getsize(output_file)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
