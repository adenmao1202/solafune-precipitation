# Competition Insights (from paper analysis)
Sources: TUPANN, NPM, Global MetNet, GENESIS (UNet+ConvLSTM IMERG)

---

## Validated Current Approach
- log1p space training: confirmed correct (Global MetNet does the same)
- channel concat for 3 frames: confirmed correct (Global MetNet validates this, no need for LSTM)
- EfficientNet-B4 encoder: reasonable starting point

---

## Priority 1 — Quick wins (0.5 day each, implement first, submit immediately)

### 1. Weighted MSE loss
80.33% of GPM pixels are zero. Standard MSE lets the model get away with predicting near-zero everywhere.
```python
rain_weight = 1.0 + 5.0 * (target > 0).float()
loss = (rain_weight * (pred_log - target_log)**2).mean()
```
alpha=5 is starting point. Try 3, 8, 10. Source: general practice confirmed by all 4 papers.

### 2. Event-based sampling
Filter training CSV to only samples where GPM has at least one nonzero pixel.
TUPANN uses event-based sampling. Global MetNet filters patches with only missing values.
Effect: eliminates ~80% of zero-rain samples that contribute useless gradients.
```python
# pre-compute: flag rows where GPM tif has any pixel > 0
df_train = df_train[df_train['has_rain'] == True]
```

### 3. EMA (Exponential Moving Average) weights
Global MetNet uses Polyak averaging. Stabilizes training, improves generalization. Near-zero implementation cost.
```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
# update each step: ema_model.update_parameters(model)
# use ema_model for eval and inference
```

---

## Priority 2 — Highest single improvement (2 days)

### 4. Day + Hour Positional Encoding
NPM ablation: adding day-of-year embedding = +17% CSI (largest single improvement in the paper).
Reason: GPM rainfall has strong seasonal and diurnal cycles. Model currently has no awareness of time.
train_dataset.csv has `datetime` column (confirmed: "2023-01-01 00:00:00" format).

```python
import math

def time_embedding(datetime_series, embed_dim=32):
    day = datetime_series.dt.dayofyear.values  # 1-365
    hour = datetime_series.dt.hour.values      # 0-23
    pe = [
        math.sin(day / 182.5 * math.pi),
        math.cos(day / 182.5 * math.pi),
        math.sin(hour / 12.0 * math.pi),
        math.cos(hour / 12.0 * math.pi),
    ]
    return torch.tensor(pe)  # [B, 4] -> Linear -> [B, embed_dim]

# Inject into UNet bottleneck via FiLM
gamma = Linear(embed)  # [B, C]
beta  = Linear(embed)  # [B, C]
bottleneck = bottleneck * gamma[:, :, None, None] + beta[:, :, None, None]
```

### 5. Season-aware sampling
NPM: sample uniformly across 12 months per mini-batch, so rainy-season months don't dominate.
Prevents model from overfitting to monsoon-season cloud patterns.
```python
# Group train CSV by month, sample batch_size//12 per month per batch
```

---

## Priority 3 — Architecture change (choose one, 4-5 days)

### 6. ConvLSTM skip connections (recommended)
GENESIS (UNet+ConvLSTM) paper: replacing standard UNet skip connections with Recursive ConvLSTM
preserves temporal memory across skip paths. Improves sharpness of predicted precipitation fields.
Standard skip connection just concatenates spatial features; R-ConvLSTM carries temporal state.
Reference: Fig 4 in UNetConvLSTM_IMERG_2307.10843.pdf

### 6b. Larger backbone (fallback option)
If time is short, swap EfficientNet-B4 -> B5 or B6 (one-line change in smp).
Training time increases ~30%. Lower effort than ConvLSTM skip.

---

## Priority 4 — Final push

### 7. Ensemble
Train 3-5 models with different seeds. Average predictions at inference.
```python
pred_ensemble = torch.stack([m(x) for m in models]).mean(0)
```

### 8. Test-time augmentation (TTA)
Horizontal flip + vertical flip + original, average the 3 predictions.
No model change, only predict.py modification.

---

## Key finding: what NOT to do

- ConvLSTM for frame encoding (channel concat is fine per Global MetNet)
- Probabilistic output head (30-bin categorical: too complex, not worth 1 month)
- Full TUPANN replication (VED + MaxViT + optical flow: 3 months minimum)
- Full NPM replication (two-stage GAN: complex, and we target IMERG directly)

---

## Architecture ideas worth understanding (not implementing)

- Optical flow supervision: enforce motion field to match numerical optical flow (TUPANN/PIANO)
  -> Key insight: satellite imagery evolves by cloud motion, not pixel-wise mapping
- Advection-diffusion PINN loss: add PDE residual to loss as physical regularization (PIANO)
  -> Could be added to existing model as extra loss term with low risk
- Lead-time conditioning: condition model on forecast horizon via FiLM (Global MetNet)
  -> Not applicable for single-step prediction

---

## Data confirmed
- train_dataset.csv has `datetime` column: "2023-01-01 00:00:00" format
- Day-of-year and hour-of-day are directly extractable: datetime.dt.dayofyear, datetime.dt.hour
