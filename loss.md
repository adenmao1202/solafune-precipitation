# Loss Function Reference

---

## Current Loss: CombinedLoss (v8a baseline)

```python
loss = 0.7 * MSE(pred_log1p, target_log1p) + 0.3 * MAE(pred_log1p, target_log1p)
```

- All operations in log1p space
- No intensity weighting
- val_RMSE_rain stuck at 2.2-2.4 (heavy rain >5mm/hr has RMSE 7.97)

---

## LCB Loss (from DYffcast, https://github.com/Dseal95/DYffcast)

Source: rainnow/src/loss.py -- LPIPSMSELoss(mse_type="cb")

LCB = L(PIPS) + C(B loss). Three components:

### 1. CB Loss (Combined Balanced)

```
CB = (W * MSE + beta * W * MAE) / 2
```

W is a pixel-weight matrix computed from rainfall intensity (mm/hr):

| Rainfall (mm/hr) | Pixel weight W |
|-----------------|---------------|
| < 0.5           | 1             |
| 0.5 -- 2        | 2             |
| 2 -- 6          | 5             |
| 6 -- 10         | 10            |
| 10 -- 18        | 20            |
| 18 -- 30        | 30            |
| > 30            | 50            |

Heavy rain pixels contribute 50x more loss than dry pixels.
beta=0.1 (default, balances MSE vs MAE contribution).

### 2. LPIPS (Learned Perceptual Image Patch Similarity)

Uses pretrained AlexNet features to compare structural similarity.
Single-channel input is expanded to 3 channels (repeat).
Input range: [0,1] (normalize=True) or [-1,1] (normalize=False).

Advantage over MSE: penalizes blurry predictions, encourages sharp rain boundaries.
MSE/MAE encourage predicting the mean -> blurry outputs.

### 3. Final LCB formula

```
LCB = (1 - alpha) * gamma * LPIPS + alpha * CB
```

DYffcast used alpha=0.6, gamma=1e-3:
- 60% CB (pixel accuracy with intensity weighting)
- 40% * 1e-3 LPIPS (perceptual/structural similarity, scaled down to match CB magnitude)

---

## Adaptation Plan for Solafune

### Key differences vs DYffcast

| | DYffcast | Solafune |
|--|---------|---------|
| Output space | normalized image | log1p(mm/hr) |
| Output size | fixed resolution | 128x128 -> resize 41x41 |
| Architecture | diffusion model | UNet regression |

### Phase 1: CB Loss only (recommended first step)

Directly addresses val_RMSE_rain stuck at 2.2-2.4.
Drop-in replacement for CombinedLoss. No training loop changes needed.

Key adaptation: weights computed in mm/hr space, loss applied in log1p space.

```python
class AdaptedCBLoss(nn.Module):
    NODES_MM = [0.5, 2.0, 6.0, 10.0, 18.0, 30.0]  # mm/hr thresholds
    WEIGHTS  = [1,   2,   5,   10,   20,   30,  50]  # 1 more than nodes

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, pred_log1p, target_log1p):
        # weights from mm/hr space (physically meaningful)
        target_mm = torch.expm1(target_log1p.float().clamp(0, 8))
        W = torch.full_like(target_mm, float(self.WEIGHTS[-1]))
        for node, w in zip(reversed(self.NODES_MM), reversed(self.WEIGHTS[:-1])):
            W = torch.where(target_mm < node, torch.full_like(W, float(w)), W)

        # loss in log1p space (stable gradients)
        mse = (pred_log1p - target_log1p) ** 2
        mae = torch.abs(pred_log1p - target_log1p)
        return ((W * mse) + (self.beta * W * mae)).mean() / 2
```

Add --loss_type cb to train.py choices.

### Phase 2: Full LCB = LPIPS + CB (optional, after Phase 1 validated)

Problem: target is 41x41, model output is 128x128.
LPIPS on 41x41 is too small for patch-based perceptual loss.

Solution: compute LPIPS at 128x128 by upsampling the 41x41 target.
Normalize log1p to [0,1]: log1p(50mm/hr) ~= 3.9, so divide by 4.

```python
# In training loop:
preds_128 = model(inputs, cond)                          # (B, 1, 128, 128)
preds_41  = F.interpolate(preds_128, GPM_SIZE)           # (B, 1, 41, 41)

# CB loss on 41x41 (precise ground truth resolution)
cb = cb_loss(preds_41, targets_41)

# LPIPS on 128x128 (upsample target, normalize to [0,1])
target_128 = F.interpolate(targets_41, (128, 128))
pred_norm  = (preds_128.clamp(0, 4) / 4).expand(-1, 3, -1, -1)
tgt_norm   = (target_128.clamp(0, 4) / 4).expand(-1, 3, -1, -1)
lpips_val  = lpips_loss(pred_norm, tgt_norm)

loss = (1 - alpha) * gamma * lpips_val + alpha * cb
```

Requires: pip install torchmetrics (for LPIPS via torchmetrics.image)

### Decision: Phase 1 first

Reasons to start with CB only:
- Directly targets the known bottleneck (heavy rain underestimation)
- No training loop changes, easy to ablate
- LPIPS benefit on precipitation maps is unproven for our regression setup
- DYffcast is a diffusion model (diverse outputs), LPIPS is more natural there

---

## Status

- [ ] Implement AdaptedCBLoss in train.py
- [ ] Add --loss_type cb to choices
- [ ] Run v15_cb experiment (holdout val, same setup as v12)
- [ ] Compare val_RMSE_rain vs v12 (baseline: 2.2-2.4)
- [ ] If CB helps: evaluate LB
- [ ] If CB helps on val_RMSE_rain: consider adding LPIPS (Phase 2)
