# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9. Two independent, self-contained architectures for getting a prompt into SAM3 live side by side as top-level folders — `SAM3Fixed/` (classes baked at export time) and `SAM3Open/` (open-vocabulary, prompt supplied at runtime). Each folder holds everything for that architecture: the export script, the custom `nvinfer` plugin, and the DeepStream pipeline script(s) + configs. Generated engines from both go into a shared `weight/` folder at the repo root.

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam3.git
```

Also requires (not pip-installable): DeepStream 9 (`pyservicemaker`, `pyds`), TensorRT 10.x (`trtexec`), CUDA 13.0, `ffmpeg`. All `gpu-id` properties in DeepStream configs must stay `0`; select the physical GPU via `CUDA_VISIBLE_DEVICES=N` instead.

## Common commands

Export an engine (dispatches to the right architecture based on `--prompt`):
```bash
python3 export.py --prompt truck car --mode det --imgsz 420   # baked -> weight/sam3_420_det_car_truck/
python3 export.py --imgsz 1008                                 # open-vocab -> weight/sam3_1008_open/ (once, ever)
```

Run the SAM3Fixed pipeline (baked classes, multi-class per camera, engine auto-built on first run):
```bash
cd SAM3Fixed && python3 ds9_rtsp.py    # reads config.txt; edit it live to hot-swap prompts
```

Run the SAM3Open pipeline (open-vocab, one free-text prompt per camera, up to 4 cameras):
```bash
cd SAM3Open && python3 ds9_rtsp.py   # reads config.txt; edit it live to hot-swap prompts
```

Recompile a custom DeepStream parser after editing its `.cpp` (each folder is self-contained):
```bash
cd SAM3Fixed && g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda-13.0/targets/x86_64-linux/include -std=c++14

cd SAM3Open && g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda-13.0/targets/x86_64-linux/include -std=c++14
```

There is no automated test suite — the two production pipeline scripts above are the verification path (run them against real or fake-RTSP sources and check the detections/output video).

Per-stage throughput (decode fps, text encoder, vision decoder, baked engine), from the repo root:
```bash
python3 bench_stages.py --imgsz 420 --batch 1 2 4
```

## Architecture

### Two self-contained folders, one shared weight/ folder

`SAM3Fixed/` and `SAM3Open/` each hold their own export script, custom `nvinfer` plugin (`nvdsparsebbox_sam3.cpp`/`libnvdsinfer_sam3.so` — same filenames in both folders, no collision since they're in separate directories), and DeepStream pipeline script(s) + `config.txt`. Nothing about one architecture reaches into the other except: (a) `SAM3Open/export.py` imports `_VisionEncoder`/`_Decoder` from `SAM3Fixed/export.py` (both architectures trace the same ONNX submodules), and (b) both write their generated engines into the shared `weight/` folder at the repo root (`weight/sam3_{imgsz}_{mode}_{classes}/` for Fixed, `weight/sam3_{imgsz}_open/` for Open) rather than next to themselves.

### Two ways to give SAM3 a prompt

**`SAM3Fixed/`** — classes are baked into the ONNX graph at export time as constants (`export.py`, formerly `specialize.py`). The resulting engine takes only an image as input and outputs detections directly, running natively in `nvinfer` exactly like a YOLO model. Changing classes requires re-exporting (or hot-swapping to a different already-built engine, see Multi-camera union-prompt pattern below). Multiple classes per camera are supported (cost scales with class count — see `Sam3Wrapper` below).

**`SAM3Open/`** — open-vocabulary: the prompt is encoded into a tensor at runtime instead of being baked in (`export.py`, formerly `export_onnx_dynamic.py`, produces a separate `text_encoder.engine` + `vision_decoder.engine`, built once, never re-exported). `ds9_rtsp.py` is a real, working camera pipeline: each camera gets exactly ONE free-text prompt (not a class list — `vision_decoder.engine` pairs one image with one encoded prompt per batch row), and up to `MAX_SOURCES=4` cameras (matching the engine's max profile batch — `sam3open_text.py`) are scored in the SAME `nvinfer` forward pass, each against its own prompt. For multiple classes on one camera, use SAM3Fixed instead.

`export.py` at the repo root is a thin dispatcher over both: `--prompt <classes>` calls `SAM3Fixed/export.py`; no `--prompt` calls `SAM3Open/export.py`.

### `export.py`'s export pattern (SAM3Fixed)

SAM3 is decomposed into ONNX-traceable submodules (`_VisionEncoder`/`_VisionEncoderFull`, `_Decoder`/`_DecoderFull`) rather than exporting the HuggingFace model directly, because the stock model hard-fails at any resolution other than 1008 (missing dynamic RoPE/position-embedding resize logic). `_VisionEncoder.__init__` rebuilds each ViT layer's `rotary_emb` for the target `Hp,Wp` and re-tiles the absolute position embedding before tracing — this is what makes `--imgsz` other than 1008 possible at all (must still be a multiple of `patch_size`, square). `SAM3Open/export.py` reuses `_VisionEncoder`/`_Decoder` directly (imported cross-folder, see above) but is only ever exported at `imgsz=1008`.

`Sam3Wrapper` bakes N classes as ONNX constants and loops the DETR decoder once per class, concatenating results — so a SAM3Fixed engine's cost scales with the number of baked classes, not just image size.

### Multi-camera union-prompt pattern (SAM3Fixed)

`SAM3Fixed/spec_cache.py` lets cameras with different prompts share one engine: it computes the sorted union of every camera's requested classes, builds (or reuses) one engine in `weight/` baked with that union via `export.py`, and `PerCameraFilter` (in `ds9_rtsp.py`) filters detections back down per-camera by `cls_id` afterward. Engine folders are named `sam3_{imgsz}_{mode}_{sorted-union-classes}/` so the same class set always resolves to the same folder regardless of which camera declared it first.

Because `nvinfer`'s `config-file-path` property is live-swappable while the pipeline runs (as long as input resolution is unchanged — verified experimentally), this same union-prompt engine can be hot-swapped at runtime when a camera's prompt changes, without restarting GStreamer. `ds9_rtsp.py`'s watcher thread implements this: it polls `config.txt` for changes, recomputes the union, builds a new engine if that union hasn't been seen before, and calls `nvinfer.set({"config-file-path": ...})`.

### Per-camera prompt injection (SAM3Open)

`vision_decoder.engine` takes `pixel_values`/`text_features`/`text_mask` as inputs (batch dim = camera count, matching `nvstreammux`). `nvinfer` handles `pixel_values` (the image) automatically like any detector; `text_features`/`text_mask` are non-image input layers, injected by `SAM3Open/nvdsparsebbox_sam3.cpp`'s `NvDsInferInitializeInputLayers` — a fixed symbol name `nvinfer` looks up via `dlsym` (a custom name is silently never called; verified empirically). It reads two flat snapshot files written by `SAM3Open/sam3open_text.py`:
```
/dev/shm/sam3open_text_features.bin   MAX_SOURCES x 32 x 256  float32
/dev/shm/sam3open_text_mask.bin       MAX_SOURCES x 32        uint8   (TRT reports BOOL; DeepStream maps it to UINT8)
```
row *b* = camera *b*'s current prompt encoding (from `text_encoder.engine`, run directly via TensorRT in Python — not through `nvinfer`). This function is called once per `nvinfer` context (re)load, not per frame — so a prompt change needs a snapshot rewrite followed by `nvinfer.set({"config-file-path": <SAME path>})`, which reloads the context and re-triggers it (verified: reloading the *same* config path re-reads the snapshot files). No TensorRT engine rebuild is ever needed for a prompt change. `SAM3Open/ds9_rtsp.py`'s watcher thread implements this the same way `SAM3Fixed/ds9_rtsp.py`'s does for the baked engine (see Multi-camera union-prompt pattern above).

`SAM3Open/nvdsparsebbox_sam3.cpp`'s output parser reads `detections [200,5]` float32 (`x1 y1 x2 y2 score`, pixel units) — no `cls_id` column, since there's one prompt per camera row; `classId` is always 0, and `SAM3Open/ds9_rtsp.py`'s `PromptLabeler` sets the display label from the camera's own prompt string instead of a fixed `labels.txt`.

### The `_get_coords` box-accuracy fix

`Sam3DetrDecoder._get_coords` (upstream `transformers`) generates a coordinate grid `arange(0,height)/height` that never reaches 1.0 (max value is `(height-1)/height`), which biases the decoder's cross-attention (used for box refinement at every decoder layer) to undershoot boxes toward the far corner (x2,y2) — worse at smaller `imgsz`, negligible at the native 1008. `SAM3Fixed/export.py` monkeypatches `Sam3DetrDecoder._get_coords` at **module import time** (before any model is constructed) to divide by `(height-1)`/`(width-1)` instead, so the grid actually reaches 1.0 — no flag, no opt-in, baked into every exported engine automatically.

An earlier version of this fix used an empirically-calibrated additive offset (0 at 1008 → 0.3 at 280) plus a box-dilation post-process instead of touching the denominator; the `/(height-1)` form was adopted after `probe_get_coords.py`-style A/B/C comparisons (bus/person/cat/remote, imgsz 1008/504/280) showed it has consistently lower total box-coordinate error. Trade-off: unlike the offset patch, it is not an exact no-op at 1008 (~0.5-1% shift can appear even at native resolution), and it can push an already-near-edge box slightly past the frame (e.g. `x2=1.01`) — `Sam3Wrapper.forward()` clamps the final box to `[0,1]` to handle this.

### DeepStream pipeline gotchas (measured on this hardware — RTX 50-series, driver 610.x, DS 9.1)

- **Video encoder**: use `nvv4l2h264enc`, and force `video/x-raw(memory:NVMM),format=NV12` via a `capsfilter` immediately before it — without the explicit format, GStreamer negotiates I420 (listed first in the sink template) and the encoder misreads chroma, corrupting the image (bitstream stays valid, so `ffmpeg -v error` reports zero errors — only visual inspection catches it). Also set `idrinterval` explicitly; `iframeinterval` alone gets ignored and produces too few real seek points. `nvh264enc` and `nvcudah264enc` do not work at all with a DeepStream hardware-decode source on this GPU/driver combination.
- **Tracker**: `NvSORT` (cascaded/greedy matching, no visual tracker) is cheaper than `NvDCF` and tolerates a higher `interval`, but its stock `probationAge`/`minTrackerConfidence` assume near-every-frame detection — at `interval=5` they need lowering (`probationAge=1`, `minTrackerConfidence≈0.1`) or tracks never "mature" and zero boxes get emitted (see `config_tracker_NvSORT_i5.yml`, present in both `SAM3Fixed/` and `SAM3Open/`).
- Both custom parsers read detection count and layer shapes from the actual output tensor at runtime rather than assuming a fixed layout, so they stay correct across different baked class counts (SAM3Fixed) or camera counts (SAM3Open) without recompiling.
