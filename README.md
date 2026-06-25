# sam3-deploy

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9.

**Workflow:** define classes in `config.txt` → `run.sh` exports a TRT FP16 engine (once) → runs a DeepStream pipeline → saves annotated mp4.

---

## Requirements

- NVIDIA GPU, CUDA 13.0, TensorRT 10.x
- DeepStream 9 + pyservicemaker
- Python packages: `torch`, `transformers`, `onnx`, `opencv-python`
- SAM3 checkpoint at `weights/sam3.pt`

---

## Quick start

### 1. Configure `config.txt`

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[model]
classes = person, car
mask_mode = 1

[pipeline]
width = 1920
height = 1080
infer_interval = 4
run_seconds = 60
```

| Field | Description |
|-------|-------------|
| `urls` | RTSP sources, one per line |
| `classes` | Comma-separated class names to detect |
| `mask_mode` | `0` = detection only · `1` = detection + instance mask |
| `width` / `height` | Pipeline resolution (all sources scaled to this) |
| `infer_interval` | Run inference every N+1 frames; tracker fills the rest |
| `run_seconds` | Stop after N seconds; `0` or omit = run until Ctrl+C |
| `output_file` | Output path; omit = auto-named `output/vis_{mode}_{classes}_{timestamp}.mp4` |

### 2. Run

```bash
./run.sh
```

`run.sh` will:
1. Read `classes` and `mask_mode` from `config.txt`
2. Export engine → `sam3_{mode}_{classes}/sam3.engine` (skipped if already exists)
3. Start the DeepStream pipeline → save to `output/`

To force re-export (e.g. after changing classes):
```bash
rm -rf sam3_1_person_car/
./run.sh
```

---

## Manual steps

### Export engine only

```bash
CUDA_VISIBLE_DEVICES=1 python3 specialize.py \
    --classes person car --mask 1 --device cuda
# → sam3_1_person_car/sam3.onnx + sam3.engine + config_infer.txt + labels.txt
```

Add `--skip-trt` to stop after ONNX export.

### Run pipeline only

```bash
CUDA_VISIBLE_DEVICES=1 python3 ds_vis_psm.py
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
    └─ mask head output: 36×36 fp16 per object  (mask_mode=1)
  → sam3_{mode}_{classes}/sam3.engine  (TRT FP16)

ds_vis_psm.py  (runtime)
  N× nvurisrcbin (RTSP)
    → nvstreammux → nvinfer (SAM3 engine, interval=4)
        → nvtracker (NvDCF, propagates boxes to skipped frames)
            → nvmultistreamtiler → nvdsosd (boxes + masks)
                → nvh264enc → mp4mux → output/
```

The vision encoder runs **once per batch** regardless of class count.

---

## File structure

```
sam3-deploy/
├── run.sh                    # one-shot: config → engine → pipeline
├── specialize.py             # export SAM3 → ONNX → TRT engine
├── ds_vis_psm.py             # multi-cam DeepStream pipeline
├── ds_video_infer.py         # single video inference (Gst direct)
├── nvdsparsebbox_sam3.cpp    # DeepStream custom bbox/mask parser
├── libnvdsinfer_sam3.so      # compiled parser
├── config.txt                # sources, classes, pipeline settings
├── weights/                  # sam3.pt, sam3.1_multiplex.pt
├── exemplar/                 # exemplar crops (adult/child/phone)
├── sam3_{mode}_{classes}/    # generated: onnx + engine + config
└── output/                   # annotated mp4 outputs
```

---

## Notes

- `imgsz=1008` is the only validated resolution for the SAM3 ViT backbone.
- All `gpu-id` properties in DeepStream must be `0`; use `CUDA_VISIBLE_DEVICES=N` to select physical GPU.
- SAM3.1 (`weights/sam3.1_multiplex.pt`) uses a different architecture and is not yet supported by `specialize.py`.
- `config.txt` contains RTSP credentials — add it to `.gitignore` before pushing.

---

## Disclaimer

This is a personal project. It is not affiliated with, endorsed by, or connected to any company or organization. All work here represents individual research and experimentation.

---

## License

Derivative work of SAM3 by Meta. Distributed under the [SAM License](LICENSE).
