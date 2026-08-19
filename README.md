# SAM3-DeepStream

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9.

**Workflow:** write a `config.txt` → run the pipeline script for the architecture you want (`SAM3Fixed/ds9_rtsp.py` or `SAM3Open/ds9_open_rtsp.py`) → saves annotated mp4. Setup steps live in [SAM3Fixed/GUIDE.md](SAM3Fixed/GUIDE.md) and [SAM3Open/GUIDE.md](SAM3Open/GUIDE.md).

---

## Tested on

- NVIDIA GPU 50x series
- DeepStream 9.1 + pyservicemaker, CUDA 13.2, TensorRT 10.16

---

## Updates

<details open>
<summary><strong>2026-08-13</strong></summary>

- **Fix small-imgsz model degrading (>280 should work).** 
  Tiny-imgsz model like 280 on small object will still have high-error boxes
  due to feature loss while downsizing an image, so not recommend below 280.
- **Repository architecture reconstructed.** 
  See `SAM3Fixed/` and `SAM3Open/`.
- **All generated engines now live in one shared `weight/` folder** 
  `weight/sam3_{imgsz}_{mode}_{classes}/` for SAM3Fixed,
  `weight/sam3_{imgsz}_open/` for SAM3Open (det only).
- **SAM3 DeepStream pipeline OTF model update**
  Can choose between:
   + Live prompt change -> text tokenizer re-encode -> nvinfer reload -> DeepStream
   + Specialize another SAM3Detector -> change path -> nvinfer reload -> DeepStream


</details>

<details>
<summary><strong>2026-08-10</strong></summary>

- WxH other than 1008x1008 works now with conditions: W and H must be equal and must be 14-multiple. Smaller input size leads to less accurate bounding boxes.

  | imgsz | latency | qps | quality |
  |-------|---------|-----|---------|
  | 1008  | 179ms   | 5.6 | clean |
  | 504   | 55ms    | 18.2| clean |
  | 420   | 47.5ms  | 20.9| clean |
  | 350   | 41ms    | 24.0| clean |
  | 280   | 19.3ms  | 51.8| clean at conf≥0.6 (default conf bumped 0.5→0.6, --pad bumped 0.05→0.10) |

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
- SAM3.1 (`weights/sam3.1_multiplex.pt`) uses a different architecture and is not yet supported by `export.py`.

---

## Which one do I want?

| | [`SAM3Fixed/`](SAM3Fixed/GUIDE.md) | [`SAM3Open/`](SAM3Open/GUIDE.md) |
|---|---|---|
| Prompt | Baked into the engine at export time | Free text, supplied at runtime |
| Classes per camera | Multiple | One/Few |
| Changing the prompt | Re-export (or hot-swap to another already-built engine) | Just edit `config.txt` — no re-export ever |

Setup, `config.txt` reference, and manual export/recompile steps for each are in their own guide: **[SAM3Fixed/GUIDE.md](SAM3Fixed/GUIDE.md)** and **[SAM3Open/GUIDE.md](SAM3Open/GUIDE.md)**.

---

## How it works

`SAM3Fixed/` and `SAM3Open/` are two fully self-contained folders, one per
architecture: each has its own export script, custom `nvinfer` plugin, and
DeepStream pipeline script(s) + `config.txt`. Both write their generated
engines into a shared `weight/` folder at the repo root instead of next to
themselves. The only cross-folder dependency is `SAM3Open/export.py`
importing `_VisionEncoder`/`_Decoder` from `SAM3Fixed/export.py` (both
architectures trace the same ONNX-traceable submodules).

There are two ways to give SAM3 a prompt:

**`SAM3Fixed/`** — classes are baked into the graph at export time as ONNX
constants. The resulting engine takes only an image as input and outputs
detections directly, running natively in `nvinfer` exactly like a YOLO model.
Changing classes requires re-exporting (or swapping to a different
already-exported engine on the fly, see `spec_cache.py`).

```
export.py  (run once per class set)
  SAM3 ViT backbone + FPN + DETR decoder
    └─ text class embeddings baked as ONNX constants
    └─ mask head output: 36×36 fp16 per object  (mode=seg)
  → weight/sam3_{imgsz}_{mode}_{classes}/sam3.engine  (TRT FP16)

ds9_rtsp.py  (runtime — live RTSP cameras)
  N× nvurisrcbin (RTSP)
    → nvstreammux → nvinfer (weight/ engine, interval=4)
        → nvtracker (propagates boxes to skipped frames)
            → PerCameraFilter (filters union detections down to each camera's own classes)
                → nvmultistreamtiler → nvdsosd (boxes + masks)
                    → nvv4l2h264enc → mp4mux → output/
```

`spec_cache.py` adds a union-prompt layer on top: multiple cameras with
different prompts all share ONE engine baked (into `weight/`) with the
union of every class needed, with `PerCameraFilter` filtering detections back
down to what each camera actually asked for by `cls_id`. Because `nvinfer`'s
`config-file-path` property is live-swappable at runtime as long as input
resolution stays the same, this also lets `ds9_rtsp.py` hot-reload a running
pipeline onto a newly-built engine without a restart, by watching `config.txt`
for changes.

**`SAM3Open/`** — open-vocabulary: the text prompt is encoded into a
tensor at runtime (`export.py` produces a separate text-encoder engine +
vision-decoder engine) instead of being baked into the graph. Never requires
re-exporting to change the prompt.

```
SAM3Open/export.py  (run once, ever)
  → weight/sam3_1008_open/text_encoder.engine     input_ids+attention_mask → text_features+text_mask
  → weight/sam3_1008_open/vision_decoder.engine   pixel_values+text_features+text_mask → detections

SAM3Open/ds9_rtsp.py  (runtime — live cameras)
  sam3open_text.py encodes each camera's prompt via text_encoder.engine
  (Python/TensorRT, not through nvinfer), writes text_features/text_mask
  snapshot files (one row per camera) to /dev/shm

  N× nvurisrcbin (RTSP)
    → nvstreammux → nvinfer (vision_decoder.engine, interval=5)
        │   nvdsparsebbox_sam3.cpp's NvDsInferInitializeInputLayers reads
        │   the snapshot files once per context load, injecting camera b's
        │   prompt encoding into batch row b -- so every camera is scored
        │   against its OWN prompt in the SAME forward pass
        → nvtracker → PromptLabeler (labels each camera's boxes from its own prompt string)
            → nvmultistreamtiler → nvdsosd → nvv4l2h264enc → mp4mux → output/
```

A prompt change only needs a snapshot rewrite + `nvinfer.set({"config-file-path":
<same path>})`, which reloads the context and re-triggers
`NvDsInferInitializeInputLayers` — verified empirically that reloading the
*same* config path re-reads the snapshot files. No TensorRT engine rebuild is
ever needed for a prompt change, unlike SAM3Fixed.

---

## File structure

```
SAM3-DeepStream/
├── export.py                    # unified export dispatcher: --prompt <classes> → SAM3Fixed/export.py
│                                # no --prompt → SAM3Open/export.py
├── bench_stages.py              # per-stage throughput: decode fps, text encoder, vision decoder, baked engine
├── weight/                      # generated engines from BOTH architectures
│   ├── sam3_{imgsz}_{mode}_{classes}/     # SAM3Fixed: onnx + engine + config_infer.txt + labels.txt
│   └── sam3_{imgsz}_open/                 # SAM3Open: text_encoder.engine + vision_decoder.engine
├── SAM3Fixed/                   # classes baked at export time, runs like YOLO in nvinfer -- self-contained
│   ├── export.py                # export SAM3 → ONNX → TRT engine (classes baked in)
│   ├── nvdsparsebbox_sam3.cpp   # custom nvinfer output parsers
│   ├── libnvdsinfer_sam3.so     # compiled
│   ├── ds9_rtsp.py              # multi-cam pipeline, per-camera prompts, hot-reload
│   ├── spec_cache.py            # union-prompt cache: build-or-reuse a weight/ engine for a class union
│   ├── config.txt               # sources + per-camera classes for live cameras
│   ├── config_tracker_NvSORT_i5.yml
│   └── output/                  # annotated mp4 outputs
├── SAM3Open/                    # open-vocabulary, prompt fed as a runtime text tensor -- self-contained
│   ├── export.py                # export text-encoder + vision-decoder engines (not baked)
│   ├── nvdsparsebbox_sam3.cpp   # non-image input injector + output parser
│   ├── libnvdsinfer_sam3.so     # compiled
│   ├── ds9_open_rtsp.py         # multi-cam pipeline, one free-text prompt per camera, hot-reload
│   ├── sam3open_text.py         # text encoder + non-image-input snapshot writer
│   ├── config.txt               # sources + per-camera free-text prompts
│   ├── config_tracker_NvSORT_i5.yml
│   └── output/
└── ESAM3_test/                   # EfficientSAM3 (EfficientViT backbone) variant, testing only
```

---

## Disclaimer

This is a personal project. It is not affiliated with, endorsed by, or connected to any company or organization. All work here represents individual research and experimentation.

---

## License

Derivative work of SAM3 by Meta. Distributed under the [SAM License](LICENSE).

`ESAM3_test/` is a derivative of [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) (Apache 2.0), by Simon Zeng.
