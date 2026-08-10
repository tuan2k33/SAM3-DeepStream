# sam3-deploy

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9.

**Workflow:** define classes in a config file → `run.sh` exports a TRT FP16 engine (once) → runs a DeepStream pipeline → saves annotated mp4.

---

## Requirements

- NVIDIA GPU 50x series
- DeepStream 9 + pyservicemaker, CUDA 13, TensorRT
- SAM3 checkpoint at HF

---

## Updates

<details open>
<summary><strong>2026-08-10</strong></summary>

- WxH other than 1008x1008 works now with conditions: W and H must be equal and must be 14-multiple. Smaller input size leads to less accurate bounding boxes (see /test).

  | imgsz | latency | qps | quality |
  |-------|---------|-----|---------|
  | 1008  | 179ms   | 5.6 | clean |
  | 504   | 55ms    | 18.2| clean |
  | 420   | 47.5ms  | 20.9| clean |
  | 350   | 41ms    | 24.0| clean |
  | 280   | 19ms    | 51.4| bad boxes |

- `specialize.py` output folder renamed `sam3_{mode}_{classes}` → `sam3_{imgsz}_{mode}_{classes}`; `--mask 0/1` arg replaced by `--mode det/seg` (`config.txt`/`config_video.txt` field renamed `mask_mode` → `mode`, plus new `imgsz` field).

</details>

<details>
<summary><strong>2026-08-06</strong></summary>

- EfficientSAM3 works now.

</details>

<details>
<summary><strong>2026-07-20</strong></summary>

- Fixed detection score calculation: SAM3's final per-query score must be gated by the scene-level presence token (matching the HF `transformers` reference implementation).

</details>

<details>
<summary><strong>2026-06-26</strong></summary>

- Dynamic batch works now.

</details>

<details>
<summary><strong>2026-06-25</strong></summary>

- Initial commit and first update to the repo.

</details>

## Notes

- All `gpu-id` properties in DeepStream must be `0`; use `CUDA_VISIBLE_DEVICES=N` to select the physical GPU.
- SAM3.1 (`weights/sam3.1_multiplex.pt`) uses a different architecture and is not yet supported by `specialize.py`.

---

## Quick start — live RTSP cameras

### 1. Write `config.txt`

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[model]
classes = adult, child, phone
mode = seg
imgsz = 1008

[pipeline]
width = 1920
height = 1080
infer_interval = 4
run_seconds = 60
```

| Field | Description |
|-------|-------------|
| `urls` | RTSP sources, one per line (any number) |
| `classes` | Comma-separated class names to detect |
| `mode` | `det` = detection only · `seg` = detection + instance mask (default) |
| `imgsz` | SAM3 input size, must be a 14-multiple, square; default `1008` |
| `width` / `height` | Pipeline resolution (all sources scaled to this) |
| `infer_interval` | Run inference every N+1 frames; tracker fills the rest |
| `run_seconds` | Stop after N seconds; `0` or omit = run until Ctrl+C |
| `output_file` | Output path; omit = auto-named `output/vis_{mode}_{classes}_{timestamp}.mp4` |

> `config.txt` contains RTSP credentials — it is in `.gitignore` and never committed.

### 2. Run

```bash
./run.sh
```

`run.sh` reads `config.txt`, exports the engine if needed, then starts the pipeline.

To force re-export after changing classes:
```bash
rm -rf sam3_1008_seg_adult_child_phone/
./run.sh
```

---

## Quick start — local video file

### 1. Write `config_video.txt`

```ini
[model]
classes = adult, child, phone
mode = seg
imgsz = 1008

[pipeline]
width = 1920
height = 1080
infer_interval = 4
```

No `[sources]` needed — the video file is the source.
No `run_seconds` needed — duration is read from the video automatically.
`output_file` is auto-named from the video filename if omitted.

`config_video.txt` is committed to the repo (no credentials).

### 2. Run

```bash
CUDA_VISIBLE_DEVICES=1 python3 ds9_video.py video_test/my_video.mp4
```

Internally this:
1. Downloads `mediamtx` once to `.mediamtx/` (single binary, ~30 MB)
2. Starts a local RTSP server at `rtsp://localhost:8554/live`
3. Loops the video via `ffmpeg` into that server
4. Runs `ds9_rtsp.py` pointed at `localhost:8554/live`
5. Saves output to `output/vis_{mode}_{classes}_{video_stem}.mp4`

Make sure the engine exists first — run `./run.sh` or `specialize.py` once with the same classes/mode/imgsz as in `config_video.txt`.

---

## Manual steps

### Export engine only

```bash
CUDA_VISIBLE_DEVICES=1 python3 specialize.py \
    --classes adult child phone --mode seg --imgsz 1008 --device cuda
# → sam3_1008_seg_adult_child_phone/sam3.onnx + sam3.engine + config_infer.txt + labels.txt
```

Add `--skip-trt` to stop after ONNX export.

### Run RTSP pipeline only

```bash
CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py
```

### Recompile C++ parser (after editing `nvdsparsebbox_sam3.cpp`)

```bash
g++ -shared -fPIC -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
    -std=c++14
```

---

## How it works

```
specialize.py  (run once per class set)
  SAM3 ViT backbone + FPN + DETR decoder
    └─ text class embeddings baked as ONNX constants
    └─ mask head output: 36×36 fp16 per object  (mode=seg)
  → sam3_{imgsz}_{mode}_{classes}/sam3.engine  (TRT FP16)

ds9_rtsp.py  (runtime — live cameras or via ds9_video.py)
  N× nvurisrcbin (RTSP)
    → nvstreammux → nvinfer (SAM3 engine, interval=4)
        → nvtracker (NvDCF, propagates boxes to skipped frames)
            → nvmultistreamtiler → nvdsosd (boxes + masks)
                → nvh264enc → mp4mux → output/

ds9_video.py  (video file wrapper)
  ffmpeg loop → mediamtx (localhost:8554) → ds9_rtsp.py
```

---

## File structure

```
sam3-deploy/
├── run.sh                    # one-shot: config.txt → engine → pipeline
├── specialize.py             # export SAM3 → ONNX → TRT engine
├── ds9_rtsp.py               # multi-cam DeepStream RTSP pipeline
├── ds9_video.py              # restream local video → run ds9_rtsp.py
├── nvdsparsebbox_sam3.cpp    # DeepStream custom bbox/mask parser
├── libnvdsinfer_sam3.so      # compiled parser
├── config_video.txt          # model/pipeline settings for video inference
├── config.txt                # sources + settings for live cameras
├── weights/                  # sam3.pt
├── sam3_{imgsz}_{mode}_{classes}/  # generated: onnx + engine + config_infer.txt
├── .mediamtx/                # auto-downloaded mediamtx binary
└── output/                   # annotated mp4 outputs
```

---

## Disclaimer

This is a personal project. It is not affiliated with, endorsed by, or connected to any company or organization. All work here represents individual research and experimentation.

---

## License

Derivative work of SAM3 by Meta. Distributed under the [SAM License](LICENSE).
