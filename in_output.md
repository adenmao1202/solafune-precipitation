# Input/Output Pipeline: Problem & Fix

## Problem

Current pipeline:
  satellite (81~144px) -> resize 128x128 -> U-Net -> F.interpolate(bilinear) -> 41x41

The final F.interpolate is the root cause of degraded heavy rain prediction.

### Why it hurts

128 / 41 = 3.12 (non-integer ratio). Bilinear interpolation maps each 41x41 output
pixel to a weighted average of ~9.7 input pixels.

Heavy rain events occupy 1~3 pixels in the 41x41 GPM map.
In the 128x128 model output those pixels expand to ~9 pixels.
Interpolation averages them with surrounding dry pixels:

  model output (128x128): ... 0.1  0.1  8.5  0.1  0.1 ...
                                        |
                                 bilinear avg
                                        |
  final output (41x41):   ...       2.9       ...

Peak 8.5 becomes 2.9. Competition RMSE is computed on this 41x41 map,
so heavy rain error is systematically underestimated during training
and peak predictions are suppressed at inference.

This is also why val_RMSE_rain is stuck at 2.2~2.4 across all experiments:
the architecture itself is diluting the signal we are trying to learn.

---

## Fix: 96x96 input + center crop output

New pipeline:
  satellite -> resize 96x96 -> U-Net -> center crop -> 41x41

No interpolation. Cropping is lossless (just discards border pixels).

### Why 96x96, not 48x48

EfficientNet-B4 effective stride = 32 (5 stride-2 stages).
Bottleneck spatial size = input / 32.

| input  | bottleneck | notes |
|--------|-----------|-------|
| 48x48  | 2x2       | too small, decoder has little spatial info |
| 96x96  | 3x3       | acceptable, still much smaller than 128x128 |
| 128x128| 4x4       | current, but requires interpolation at output |

96x96 keeps 3x3 bottleneck (more spatial context than 48x48)
while enabling clean crop-based output.

### Center crop math

96 - 41 = 55. Crop: top=27, left=27, bottom=27+41=68, right=27+41=68.
(Asymmetric by 1px: 27 left/top, 28 right/bottom -- both valid, pick either.)

```python
def center_crop_to_gpm(t: torch.Tensor) -> torch.Tensor:
    # t: (B, C, H, W), returns (B, C, 41, 41)
    top  = (t.shape[-2] - GPM_SIZE[0]) // 2
    left = (t.shape[-1] - GPM_SIZE[1]) // 2
    return t[:, :, top:top + GPM_SIZE[0], left:left + GPM_SIZE[1]]
```

---

## Code Changes

### train.py

1. Add center_crop_to_gpm() after imports.

2. Replace all F.interpolate calls (3 occurrences):

```python
# Before
preds_41 = F.interpolate(preds, size=GPM_SIZE, mode="bilinear", align_corners=False)

# After
preds_41 = center_crop_to_gpm(preds)
```

Locations:
- Line 435: train loop
- Line 460: val loop (focal)
- Line 464: val loop (regression)

3. Change default input_size:

```python
# Before
parser.add_argument("--input_size", type=int, default=128, ...)

# After
parser.add_argument("--input_size", type=int, default=96, ...)
```

### predict.py

Same center_crop_to_gpm replacement for any F.interpolate(preds, ...) call.

---

## Training command

```bash
python src/train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --input_size 96 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 4 \
  --val_mode holdout \
  --holdout_locations florida,jakarta,cape_town,friuli_venezia_giulia \
  --run_name v14_96px_crop \
  2>&1 | tee ~/v14_96px_crop.log && \
curl -s -d "v14_96px_crop done!" ntfy.sh/solafune_luiz_train
```

---

## Checklist

- [x] Add center_crop_to_gpm() to train.py
- [x] Replace 3x F.interpolate in train.py
- [x] Replace F.interpolate in predict.py
- [ ] Smoke test locally: train_smoke.csv, input_size=48 and input_size=96
- [ ] Recompute stats.json on Vast.ai before running
- [ ] Push to Vast.ai, run v14_48px and v14_96px

---

## Two Experiment Commands

### Experiment A: 48x48 (bottleneck 2x2)

```bash
python src/train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --input_size 48 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 4 \
  --val_mode holdout \
  --holdout_locations florida,jakarta,cape_town,friuli_venezia_giulia \
  --run_name v14_48px_crop \
  2>&1 | tee ~/v14_48px_crop.log && \
curl -s -d "v14_48px_crop done!" ntfy.sh/solafune_luiz_train
```

### Experiment B: 96x96 (bottleneck 3x3)

```bash
python src/train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --input_size 96 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 4 \
  --val_mode holdout \
  --holdout_locations florida,jakarta,cape_town,friuli_venezia_giulia \
  --run_name v14_96px_crop \
  2>&1 | tee ~/v14_96px_crop.log && \
curl -s -d "v14_96px_crop done!" ntfy.sh/solafune_luiz_train
```

### Center crop math

| input  | bottleneck | crop top/left | notes |
|--------|-----------|---------------|-------|
| 48x48  | 2x2       | (48-41)//2 = 3 | tight border, less context |
| 96x96  | 3x3       | (96-41)//2 = 27 | recommended |
