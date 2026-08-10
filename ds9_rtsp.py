#!/usr/bin/env python3
"""
Multi-camera DeepStream 9 RTSP pipeline → annotated mp4.

Reads all settings from config.txt (or $DS_CONFIG override).
Called directly for live RTSP cameras, or via ds9_video.py for local video files.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py          # live cameras (config.txt)
    CUDA_VISIBLE_DEVICES=1 python3 ds9_video.py <file>  # local video (config_video.txt)
"""

import configparser
import os
import signal
import threading
import time
from pathlib import Path

from pyservicemaker import Pipeline, BatchMetadataOperator, Probe

_HERE = Path(__file__).parent          

_cfg = configparser.ConfigParser()
_cfg.read(os.environ.get("DS_CONFIG", str(_HERE / "config.txt")))

SOURCES      = [s.strip() for s in _cfg.get("sources", "urls").strip().splitlines() if s.strip()]
SOURCE_NAMES = [f"cam{i}" for i in range(len(SOURCES))]
CLASSES      = [c.strip() for c in _cfg.get("model", "classes").split(",")]
MODE         = _cfg.get("model", "mode", fallback="seg")
IMGSZ        = _cfg.getint("model", "imgsz", fallback=1008)
INFER_CONFIG = str(_HERE / f"sam3_{IMGSZ}_{MODE}_{'_'.join(CLASSES)}" / "config_infer.txt")
WIDTH          = _cfg.getint("pipeline", "width",  fallback=1920)
HEIGHT         = _cfg.getint("pipeline", "height", fallback=1080)
INFER_INTERVAL = _cfg.getint("pipeline", "infer_interval")
RUN_SECONDS  = _cfg.getint("pipeline", "run_seconds", fallback=0)  # 0 = run until Ctrl+C

_default_out = f"output/vis_{MODE}_{'_'.join(CLASSES)}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
OUTPUT_FILE  = str(_HERE / _cfg.get("pipeline", "output_file", fallback=_default_out))


class DetectionLogger(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self._score_cache = {}  # (cam, track_id) → last known confidence

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            cam   = frame_meta.pad_index
            n_det = 0
            for obj in frame_meta.object_items:
                n_det += 1
                try:
                    cls      = int(obj.class_id)
                    conf     = float(obj.confidence)
                    track_id = getattr(obj, "tracking_id",
                               getattr(obj, "object_id", None))
                    name = CLASSES[cls] if cls < len(CLASSES) else str(cls)

                    # Cache score from inference frames; reuse on tracked frames
                    key = (cam, track_id)
                    if conf > 0.05:
                        self._score_cache[key] = conf
                    else:
                        conf = self._score_cache.get(key, 0.0)

                    label = f"{name} {conf:.2f}" if conf > 0 else name
                    obj.text_params.display_text = label
                except Exception:
                    pass
            if n_det:
                cls_conf: dict[str, list[float]] = {}
                for obj in frame_meta.object_items:
                    try:
                        cls  = int(obj.class_id)
                        name = CLASSES[cls] if cls < len(CLASSES) else str(cls)
                        key  = (cam, getattr(obj, "tracking_id",
                                getattr(obj, "object_id", None)))
                        conf = self._score_cache.get(key, 0.0)
                        cls_conf.setdefault(name, []).append(conf)
                    except Exception:
                        pass
                parts = [
                    f"{name}×{len(cs)} avg={sum(cs)/len(cs):.2f} max={max(cs):.2f}"
                    for name, cs in cls_conf.items()
                ]
                print(f"  cam{cam} #{frame_meta.frame_number}: {' | '.join(parts)}",
                      flush=True)


def build_pipeline() -> Pipeline:
    pipeline = Pipeline("4cam-vis")

    for uri, name in zip(SOURCES, SOURCE_NAMES):
        pipeline.add("nvurisrcbin", name, {
            "uri":             uri,
            "latency":         500,
            "drop-on-latency": True,
        })

    pipeline.add("nvstreammux", "mux", {
        "batch-size":           len(SOURCES),
        "width":                WIDTH,
        "height":               HEIGHT,
        "live-source":          True,
        "batched-push-timeout": 40000,
        "compute-hw":           1,
    })
    for name in SOURCE_NAMES:
        pipeline.link((name, "mux"), ("vsrc_%u", ""))

    pipeline.add("nvinfer", "infer", {
        "config-file-path": INFER_CONFIG,
        "interval":         INFER_INTERVAL,
    })
    pipeline.link("mux", "infer")

    # Tracker propagates detections to all frames skipped by interval
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file":          "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file":       "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        "tracker-width":        640,
        "tracker-height":       384,
        "gpu-id":               0,
    })
    pipeline.link("infer", "tracker")
    pipeline.attach("tracker", Probe("logger", DetectionLogger()))

    import math
    n = len(SOURCES)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows":    rows,
        "columns": cols,
        "width":   WIDTH,
        "height":  HEIGHT,
    })
    pipeline.link("tracker", "tiler")

    pipeline.add("nvvideoconvert", "conv1", {})
    pipeline.link("tiler", "conv1")

    pipeline.add("nvdsosd", "osd", {
        "process-mode": 0,
        "display-text": True,
        "display-mask":  MODE == "seg",
    })
    pipeline.link("conv1", "osd")

    pipeline.add("nvvideoconvert", "conv2", {})
    pipeline.link("osd", "conv2")

    pipeline.add("nvh264enc", "enc", {
        "bitrate":  8000,
        "gop-size": 25,
    })
    pipeline.link("conv2", "enc")

    pipeline.add("h264parse", "parse", {})
    pipeline.link("enc", "parse")

    pipeline.add("mp4mux", "mp4mux", {})
    pipeline.link("parse", "mp4mux")

    pipeline.add("filesink", "sink", {
        "location": OUTPUT_FILE,
        "sync":     False,
        "async":    False,
    })
    pipeline.link("mp4mux", "sink")

    return pipeline


def main():
    pipeline      = build_pipeline()
    pipeline_done = threading.Event()
    _stop_called  = threading.Event()

    def _run():
        pipeline.start().wait()
        pipeline_done.set()

    def _shutdown(sig, frame):
        if not _stop_called.is_set():
            _stop_called.set()
            print("\n[vis] Flushing 10s before stop...")
            time.sleep(10)
            pipeline.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    dur = f"{RUN_SECONDS}s" if RUN_SECONDS else "∞"
    print(f"[vis] {len(SOURCES)} cams  interval={INFER_INTERVAL}  run={dur}  → {OUTPUT_FILE}")
    print(f"[vis] Ctrl+C to stop")

    threading.Thread(target=_run, daemon=True).start()
    t0 = time.time()

    def _timer():
        if not RUN_SECONDS:
            return
        for elapsed in range(1, RUN_SECONDS + 1):
            if pipeline_done.is_set() or _stop_called.is_set():
                return
            time.sleep(1)
            if elapsed % 10 == 0 or elapsed == RUN_SECONDS:
                print(f"[vis] {elapsed}/{RUN_SECONDS}s", flush=True)
        if not _stop_called.is_set():
            _stop_called.set()
            print("[vis] Time up, flushing 10s before stop...")
            time.sleep(10)
            pipeline.stop()

    threading.Thread(target=_timer, daemon=True).start()
    pipeline_done.wait()

    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6 if os.path.exists(OUTPUT_FILE) else 0
    print(f"[vis] Done → {OUTPUT_FILE}  ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")

    # If mp4 moov is missing (pipeline killed without EOS), recover with ffmpeg
    recovered = OUTPUT_FILE.replace(".mp4", "_fixed.mp4")
    ret = os.system(
        f"ffmpeg -y -fflags +discardcorrupt+igndts -i {OUTPUT_FILE} "
        f"-c:v libx264 -preset fast -crf 23 -movflags +faststart {recovered} 2>/dev/null"
    )
    if ret == 0 and os.path.exists(recovered):
        os.replace(recovered, OUTPUT_FILE)
        print(f"[vis] Re-encoded → {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
