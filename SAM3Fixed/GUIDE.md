# SAM3Fixed

Baked-classes architecture: classes are compiled into the ONNX graph as constants at export time. The resulting engine takes only an image as input and runs natively in `nvinfer` exactly like a YOLO model. Multiple classes per camera are supported (cost scales with class count). Changing classes requires re-exporting (or hot-swapping to a different already-built engine — see the union-prompt pattern in the main [README](../README.md#how-it-works)).

---

## Quick start — live RTSP cameras

**1. Write `config.txt`**

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[model]
mode = seg
imgsz = 1008
classes = adult, child, phone   # fallback prompt for cameras with no [camN] section
conf = 0.4

# optional per-camera prompt override -- all cameras share ONE engine baked
# with the union of every camera's classes; hot-editable while running
[cam0]
classes = adult, phone

[cam1]
classes = child

[pipeline]
width = 1920
height = 1080
infer_interval = 4
run_seconds = 60
```

| Field | Description |
|-------|-------------|
| `urls` | RTSP sources, one per line (any number) |
| `mode` | `det` = detection only · `seg` = detection + instance mask (default) |
| `imgsz` | SAM3 input size, must be a 14-multiple, square; default `1008` |
| `classes` | Fallback comma-separated class names for cameras with no `[camN]` section |
| `conf` | Score threshold |
| `[camN]/classes` | Optional per-camera prompt override; edit live to hot-swap (see below) |
| `width` / `height` | Pipeline resolution (all sources scaled to this) |
| `infer_interval` | Run inference every N+1 frames; tracker fills the rest |
| `run_seconds` | Stop after N seconds; `0` or omit = run until Ctrl+C |
| `output_file` | Output path; omit = auto-named `output/vis_{mode}_{classes}_{timestamp}.mp4` |

> `config.txt` contains RTSP credentials — it is in `.gitignore` and never committed.

**2. Run**

```bash
CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py
```

The engine (baked with the union of every camera's classes) is built automatically into `../weight/` on first run if it doesn't exist yet, then reused. **Editing `config.txt` while the pipeline is running hot-swaps prompts live** — if the union of classes changes, a new engine is built (if needed) and `nvinfer` is swapped onto it without restarting GStreamer; if only a camera's whitelist changes within the same union, no engine swap is needed at all.

---

## Manual steps

**Export an engine only** — either call `export.py` directly here, or use
the unified `../export.py` dispatcher from the repo root:

```bash
CUDA_VISIBLE_DEVICES=1 python3 export.py \
    --classes adult child phone --mode seg --imgsz 1008 --device cuda
# → ../weight/sam3_1008_seg_adult_child_phone/sam3.onnx + sam3.engine + config_infer.txt + labels.txt

# equivalent, from the repo root:
python3 ../export.py --prompt adult child phone --mode seg --imgsz 1008 --device cuda
```

Add `--skip-trt` to stop after ONNX export.

**Recompile the custom DeepStream parser** after editing `nvdsparsebbox_sam3.cpp`:

```bash
g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda-13.2/targets/x86_64-linux/include -std=c++14
```

---

## Note on ONNX/engine resolution

Both the ONNX graph and the TRT engine have `imgsz` (H and W) hardcoded — only
`batch` is a dynamic axis. This isn't just a build-time engine-profile choice:
`_VisionEncoder.__init__` rebuilds each ViT layer's `rotary_emb` and re-tiles
the absolute position embedding for a specific `Hp,Wp` grid *at trace time*
(not inside `forward()`), so a different resolution needs a full re-export
(`export.py --imgsz <N>`), not just a new TensorRT optimization profile on the
existing ONNX.
