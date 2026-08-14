# SAM3Open

Open-vocabulary architecture: the prompt is encoded into a tensor at runtime instead of being baked into the graph. Never requires re-exporting to change the prompt. Each camera gets exactly **one** free-text prompt (not a class list — see the main [README](../README.md#how-it-works) for why). For multiple classes on the same camera, use [SAM3Fixed](../SAM3Fixed/GUIDE.md) instead.

---

## Quick start — live RTSP cameras

**1. Write `config.txt`**

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[cam0]
prompt = forklift

[cam1]
prompt = spilled liquid on the floor

[model]
conf = 0.3

[pipeline]
width = 1280
height = 720
infer_interval = 5
run_seconds = 60
```

Each camera gets **one free-text prompt** — never requires re-exporting, and up to `MAX_SOURCES=4` cameras (`sam3open_text.py`, matching `vision_decoder.engine`'s max batch; rebuild the engine with a bigger `--max-batch` to raise this).

> `config.txt` contains RTSP credentials — it is in `.gitignore` and never committed.

**2. Run**

```bash
CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py
```

**Editing `config.txt` while running hot-swaps prompts live** — the changed prompt is re-encoded and `nvinfer`'s context is reloaded; there is no engine rebuild at all (the engine itself never changes, only the injected text tensor does).

---

## Manual steps

**Export the engines only** (built once, ever — prompts are supplied at
runtime, never baked in, so there's normally no need to re-export):

```bash
CUDA_VISIBLE_DEVICES=1 python3 export.py --imgsz 1008
# → ../weight/sam3_1008_open/{text_encoder,vision_decoder}.{onnx,engine}

# equivalent, from the repo root:
python3 ../export.py --imgsz 1008
```

**Recompile the custom DeepStream parser** after editing `nvdsparsebbox_sam3.cpp`:

```bash
g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda-13.2/targets/x86_64-linux/include -std=c++14
```

---

## Note on ONNX/engine resolution

Both the ONNX graphs and the TRT engines have `imgsz` (H and W) hardcoded at
1008x1008 — only `batch` is a dynamic axis. This isn't just a build-time
engine-profile choice: `_VisionEncoder.__init__` (imported from
`SAM3Fixed/export.py`) rebuilds each ViT layer's `rotary_emb` and re-tiles the
absolute position embedding for a specific `Hp,Wp` grid *at trace time* (not
inside `forward()`), so a different resolution needs a full re-export
(`export.py --imgsz <N>`), not just a new TensorRT optimization profile on the
existing ONNX.
