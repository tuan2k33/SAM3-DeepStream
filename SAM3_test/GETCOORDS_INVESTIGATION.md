# `_get_coords` cap — investigation log

Log of testing different cap values for the `_get_coords` fix in
`SAM3Fixed/export.py` (and the matching one in
`ESAM3_test/specialize_efficientsam3.py`). All tests use `test/bus.jpg`, the
`facebook/sam3` checkpoint (via `transformers`), and a **2-class
`['bus', 'person']` wrapper built together** — matching exactly what
`export.py --classes bus person` actually bakes into the ONNX/engine (see
"Important lesson" below for why this matters).

## The original bug

`Sam3DetrDecoder._get_coords` (in `transformers`, and the equivalent vendored
copy in `efficientsam3`) generates a coordinate grid
`coords_h = arange(0,H)/H` — never reaching `1.0`, max only `(H-1)/H`. This
grid feeds `_get_rpb_matrix` (relative position bias for the decoder's
cross-attention, used at EVERY box-refinement layer) → the model learns to
undershoot boxes toward the far corner (x2,y2), worse at smaller `imgsz`
(patch is larger relative to the image).

## Measurement method

- **Groundtruth proxy**: the `bus` box at `imgsz=1008`, using the **original
  buggy** formula (`before`) — since this is the resolution/condition
  documented as having "negligible" error.
- **Metric**: `mean|d|` = mean absolute pixel deviation (`dx1,dy1,dx2,dy2`,
  on the original `810×1080` image) from the groundtruth proxy, plus `IoU`.
- Tested at 4 sizes: `420, 700, 1008, 1400` (`Hp = imgsz/14` = `30, 50, 72, 100`).

## Combined results table (mean|d|, lower = better)

| imgsz | before (buggy) | cap=1.00 (raw) | cap=0.99 | cap=0.98 | **cap=0.987654321** | halfpixel | `/(H-0.9)` (raw) |
|---|---|---|---|---|---|---|---|
| 420  | 10.01 | 7.08 | 3.58 | 4.28 | **3.69** | 7.86 | 5.59 |
| 700  | 3.05  | 6.90 | 2.45 | 3.05 | **1.88** | 7.64 | 6.02 |
| 1008 | 0.00  | 6.15 | 1.76 | 2.71 | **0.73** | 6.59 | 5.55 |
| 1400 | 2.98  | 6.78 | 2.98 | 2.60 | **2.24** | 7.21 | 6.35 |

`cap=1.00` and `/(H-0.9)` columns are the **raw, unclamped** figures (see the
per-approach tables below for why — the clamped versions were quietly capped
at the image edge for 3 of 4 sizes, which understated their real error
slightly). `cap=0.99`/`cap=0.98`/`cap=0.987654321`/`halfpixel`/`before` never
overflowed `[0,1]` for this box, so clamping was a no-op for them and their
numbers are unaffected either way.

**`cap=0.987654321` wins 3/4 sizes, a close second at 420 (only slightly
behind `cap=0.99`).** `cap=1.00` (the fully "mathematically correct" fix, no
cap) is **worst everywhere** — including at 1008, the native resolution.

## Per-approach signed deltas (dx1, dy1, dx2, dy2)

Raw signed pixel deltas (original `810×1080` image, `d = mode − reference`)
for every approach tested, all 4 sizes. Sign convention: on `x1`/`y1`,
positive = edge moves inward (right/down); on `x2`/`y2`, positive = edge moves
outward (right/down, away from the box).

**`before` (buggy, unfixed)** — also the reference itself at 1008, hence all-zero there:

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +4.6 | -7.3 | -11.4 | -16.8 |
| 700  | +2.2 | -0.7 |  -3.3 |  -6.0 |
| 1008 |  0.0 |  0.0 |   0.0 |   0.0 |
| 1400 | +1.9 | -1.1 |  +4.2 |  +4.8 |

**`cap=1.00`** (full fix, no cap — worst overall). Shown **raw, unclamped**
(straight from `pred_boxes` in fp32, before `.clamp(0,1)` and before the
`.half()` cast) — the clamped/fp16 version made `dx2` look suspiciously
constant at `+8.8` across every size, which turned out to be an artifact: at
420/700/1400 the raw box genuinely overflows past normalized `x2>1.0` and
gets clamped flat to the image's right edge (`810px`); at 1008 it doesn't
overflow (`raw x2=0.99993<1`) but the final `.half()` cast rounds
`0.99993×1008=1007.93` to the nearest fp16-representable value, which
happens to also land on `1008.0` — coincidentally mimicking a clamp. The
**raw** numbers below are the real, unobstructed picture:

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +7.7 | +0.6 | +13.0 |  +7.0 |
| 700  | +4.5 | +3.9 | +10.7 |  +8.5 |
| 1008 | +2.0 | +3.2 |  +8.7 | +10.7 |
| 1400 | +3.7 | +0.9 |  +9.9 | +12.6 |

Aggregated (`mean|d|` of these raw deltas) traces a clean **U-shape, minimum
at 1008**:

```
420: 7.075   700: 6.900   1008: 6.150   1400: 6.775
```

i.e. error shrinks steadily from 420 up to 1008, then grows again past 1008
— consistent with 1008 being the one resolution `box_rpb_embed_x/y` and
`box_head` actually saw during training (see mechanism section below):
distribution-shift is smallest exactly at the trained `H`, and grows moving
away from it in *either* direction, not just for "small imgsz" as the
original bug's own description suggested.

**`cap=0.99`**:

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +5.8 | -1.5 | +6.9 | -0.1 |
| 700  | +2.8 | +1.8 | +4.2 | +0.9 |
| 1008 | +0.3 | +1.1 | +2.4 | +3.2 |
| 1400 | +1.9 | -1.1 | +4.2 | +4.8 |

**`cap=0.98`** (no-op at 700, reverses direction at 1008+):

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +5.1 | -4.0 | -0.8 | -7.2 |
| 700  | +2.2 | -0.7 | -3.3 | -6.0 |
| 1008 | -0.2 | -1.5 | -4.8 | -4.3 |
| 1400 | +1.2 | -3.8 | -3.3 | -2.2 |

**`cap=0.987654321`** (best overall):

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +5.6 | -2.1 | +5.0 | -2.0 |
| 700  | +2.6 | +1.3 | +3.1 | -0.6 |
| 1008 | +0.1 | +0.4 | +0.8 | +1.6 |
| 1400 | +1.6 | -1.6 | +2.5 | +3.2 |

**`halfpixel`** (patch-center convention — worst besides cap1.00):

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +17.7 | +11.1 | +2.1 | +0.5 |
| 700  | +10.3 | +10.1 | +5.4 | +4.8 |
| 1008 |  +5.6 |  +7.6 | +5.6 | +7.5 |
| 1400 |  +6.1 |  +4.3 | +8.3 | +10.2 |

**`/(H-0.9)`** (raw, unclamped — same clamp/fp16 artifact as `cap=1.00` affected the earlier version of this table, since `H/(H-0.9)` also pushes close to/past `1.0`):

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | +7.0 |  0.0 | +10.9 |  +4.5 |
| 700  | +4.1 | +3.5 |  +9.6 |  +6.9 |
| 1008 | +1.7 | +2.9 |  +7.9 |  +9.6 |
| 1400 | +3.5 | +0.7 |  +9.4 | +11.8 |

**`cap=1.00` + post-hoc shift `-0.5 patch`** (in imgsz-space, converted back — tried after the raw `-7px` experiment, did not generalize):

| imgsz | dx1 | dy1 | dx2 | dy2 |
|---|---|---|---|---|
| 420  | -5.8 | -17.4 | -4.7 | -11.0 |
| 700  | -3.6 |  -6.8 | +0.7 |  -2.1 |
| 1008 | -3.6 |  -4.3 | +3.2 |  +3.2 |
| 1400 | -0.3 |  -4.5 | +4.8 |  +7.1 |

### Other directions tried and rejected

- **`cap=0.98`**: an exact no-op at `imgsz=700` (`(H-1)/H` at H=50 already
  equals 0.98 exactly) and **reverses the fix's direction** at
  `imgsz≥1008` (cap sits below even the original bug). Not used.
- **`halfpixel`** (`(i+0.5)/H`, patch-center convention): worse than `before`
  in most cases — the model was trained with the edge-aligned convention
  (`i/H`); shifting both ends inward (including the near end, which was
  already correct) distorts things further instead of only fixing what's
  broken.
- **`/(H-0.9)`**: max value too close to `1.0` (~0.999), falls into the same
  bad zone as `cap=1.00`.
- **Post-hoc `-7px` shift** (uniformly shifting all 4 edges of `cap=1.00`'s
  output, in original-image pixel space): cut SSE by 67-89%, BUT when tried
  in a more "principled" form — subtracting exactly half a patch (`0.5/H`)
  in imgsz-space and converting back — it **did not beat `cap=0.987654321`**,
  and was even worse than plain `cap1.00` at 420. Conclusion: the "magic"
  `-7` value is most likely a coincidence specific to this one bus box in
  this one test image, not a generalizable physical constant.

## Why the "mathematically correct" fix (`cap=1.0`) is worse than `cap≈0.987-0.99`

Traced through the original `transformers` source
(`Sam3DetrDecoder._get_rpb_matrix`):

```python
deltas_y = coords_h.view(1,-1,1) - boxes_xyxy[...][:, :, 1:4:2]   # coords - box_edge
deltas_y_log = sign(deltas_y*8) * log2(|deltas_y*8|+1) / log2(8)  # log-scale, NONLINEAR
deltas_y = self.box_rpb_embed_y(deltas_y_log)                     # pretrained MLP, frozen weights
```

`box_rpb_embed_x/y` are two small MLPs (`Linear(2→256)→ReLU→Linear(256→8)`,
**no bounding activation on the output**) — weights loaded straight from the
`facebook/sam3` checkpoint (`detr_decoder.box_rpb_embed_{x,y}.layer{1,2}.
{weight,bias}`, 8 tensors total). Measured actual output range (hooked
directly during a real forward pass on bus.jpg):

| MLP | min | max | mean | std |
|---|---|---|---|---|
| `box_rpb_embed_x` | -46.13 | +72.72 | 5.63 | 23.04 |
| `box_rpb_embed_y` | -39.05 | +138.92 | 18.90 | 42.48 |
| `box_head` | -3.88 | +3.54 | -0.20 | 0.68 |

SAM3 was **only ever trained at one resolution: `imgsz=1008` (`H=72`)** —
every other `imgsz` only works at all thanks to the RoPE/position-embedding
retiling trick in `_VisionEncoder` (custom to this repo; the stock model
doesn't support it). So these two MLPs **only ever saw** inputs generated
from `coords_max=(72-1)/72=0.9861` during training. Pushing the cap all the
way to exactly `1.0` is not a subtle rounding nudge — `1.0 − 71/72 = 1/72`
**exactly**, i.e. `(H-1)/H` and `H/H` differ by precisely one grid step.
In other words, a full `1.0` cap is mathematically equivalent to shifting
the grid's last index one whole **patch** past where the MLPs were trained
to expect it, not a one-pixel-scale perturbation. Given the nonlinear
log-encoding in `_get_rpb_matrix` amplifies input deviations unevenly, a
full-patch-sized shift in the input distribution is large enough to push
these frozen MLPs well outside the region their weights were fit to → the
MLPs respond with the largest, least predictable drift → the box moves the
most. `cap≈0.987-0.99` stays close to `0.9861` (the value the model is
actually "used to") → more stable response, even though it's "less
correct" mathematically than `1.0`.

Deviation between `(H-1)/H` (the value naturally occurring at each size) and
the trained value:

```
H=30  (420):  0.9667  deviation -0.0194
H=50  (700):  0.9800  deviation -0.0061
H=72 (1008):  0.9861  deviation  0        <- the trained H
H=100(1400):  0.9900  deviation +0.0039   <- OPPOSITE sign, explains why "before" flips sign at 1400
```

## Important lesson: 1-class vs 2-class wrapper

`Sam3Wrapper` bakes N classes by looping the decoder once per class —
supposedly independent. Empirically, though: building the wrapper with
`['bus']` alone gives a DIFFERENT box than building `['bus','person']`
together (same image, same fix, same imgsz) — off by tens of pixels.
**Root cause not yet identified** (suspect some cross-class effect inside
the decoder, not traced down yet). Because of this, **every test in this
file uses the 2-class wrapper**, verified to match the real built TRT engine
exactly (`weight/sam3_1008_det_bus_person/sam3.engine`) — see
`sam3_bus_person_engine_infer.jpg`.

## Current code state

`SAM3Fixed/export.py` and `ESAM3_test/specialize_efficientsam3.py` now use
**`cap=0.987654321`** (applied, not yet committed) — finalized after the
71-point sweep (see below) confirmed it beats both `cap=0.99` and a full
`cap=1.00` on 3 of 4 spot-check sizes. `EfficientSAM3`'s value is carried
over from this investigation (run entirely on the original SAM3), not
independently re-verified on its own decoder.

## Image files in this folder

| File | Contents |
|---|---|
| `sam3_bus_person_engine_infer.jpg` | Real TRT engine inference result (`sam3_1008_det_bus_person`) |
| `sam3_bus_person_before_after_{420,700,1008}.jpg` | before(red)/after cap=0.99(green), bus+person, same image |
| `sam3_3way_cap_{420,700,1008,1400}.jpg` | before(red)/cap0.99(green)/cap0.98(blue) |
| `sam3_cap987654321_{420,700,1008,1400}.jpg` | before(red)/cap0.987654321(green) |

## Not done yet / possible next steps

- Test on more images/classes (currently only 1 image, 2 classes `bus`+
  `person`) to make sure `cap=0.987654321` isn't overfit to this one image.
- Trace the root cause of the 1-class vs 2-class wrapper discrepancy.
- To remove the distribution-shift issue entirely: fine-tune the 3 MLPs
  (`box_rpb_embed_x/y`, `box_head`) with `_get_coords` fixed and held
  constant + L1/GIoU loss on real labeled data — needs a training loop
  (not currently in this repo) plus labeled box data.
