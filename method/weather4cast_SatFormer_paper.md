# SaTformer: A Space-Time Transformer for Precipitation Nowcasting
**NeurIPS 2025 Weather4Cast "Cumulative Rainfall" Challenge -- 1st Place**
arXiv: 2511.11090 | Authors: Levi Harris, Tianlong Chen (UNC Chapel Hill)

---

## Task

Given 1 hour of low-resolution HRIT geostationary satellite radiances (T=4 frames, C=11 channels, H=W=32 pixels), predict cumulative rainfall over the following 4 hours (averaged over a 32x32 pixel area). Metric: CRPS (Cumulative Ranked Probability Score).

This is a **video-to-scalar regression** problem, not a pixel-wise prediction task.

---

## Core Innovations

### 1. Classification Instead of Regression

The most important design decision: treat precipitation as a **classification problem** with 64 discrete bins.

- Partition target space into n=64 non-overlapping bins of equal width delta
- Generate one-hot labels: i = round((y_reg - D_ymin) / delta)
- Model outputs softmax probability distribution over 64 bins
- At inference: convert to regression value by expected value (sum of bin_center * probability)

**Why:** Deep networks struggle on regression with imbalanced data. Reformulating as classification allows class-weighted loss and exploits proven classification architectures.

**Ablation (number of bins):**

| # Bins | CRPS  |
|--------|-------|
| 4      | 14.181 |
| 8      | 5.987  |
| 16     | 4.293  |
| 32     | 3.898  |
| 64     | 3.135  |
| 128    | 3.610  |
| 256    | 5.312  |

64 bins is the sweet spot -- more bins leads to sparse class representation and degenerate solutions.

### 2. Class-Weighted Cross-Entropy Loss

Standard CE biases toward majority class (no rain). Use log-scaled class weights:

```
w_i = -log(|D_i| / |D_total|)
L(y, y_hat) = -sum_i w_i * log(y_hat_i) * y_i
```

**Ablation result:**

| Loss Weighting | BW-Top-3 | BW-CRPS |
|----------------|----------|---------|
| Without        | 0.076    | 6.91    |
| With           | 0.272    | 2.64    |

Without weighting: model overfits to no-rain majority class.

### 3. Full Space-Time Self-Attention (ST^2)

Partition each frame into N=HW/P^2 non-overlapping patches (P=4, so N=64 patches per frame).
All T*N+1 tokens (including CLS) attend to each other across BOTH space and time simultaneously.

**Why full 3D attention:** Input is only 4 frames at 32x32 -- small enough that full attention is computationally feasible and empirically superior.

**Ablation (attention variants):**

| Attention Type     | BW-Top-3 | BW-CRPS |
|--------------------|----------|---------|
| Space then Time    | 0.250    | 4.39    |
| Time then Space    | 0.214    | 3.39    |
| Full S+T           | 0.272    | 2.64    |

### 4. CLS Token for Scalar Prediction

Prepend a randomly initialized CLS token to the token sequence. After L=12 transformer layers, splice out CLS token and pass through a 1-layer MLP head to predict class probabilities.

---

## Architecture Details

```python
SaTformer(
    dim=512,
    num_frames=4,       # T: temporal frames
    num_classes=64,     # output precipitation bins
    image_size=32,      # spatial input size
    patch_size=4,       # N=64 patches per frame
    channels=11,        # HRIT satellite bands
    depth=12,           # transformer encoder blocks
    heads=8,
    dim_head=64,
    attn_dropout=0.1,
    ff_dropout=0.1,
    rotary_emb=False,
    attn="ST^2"
)
# Total tokens per forward pass: 4*64 + 1 = 257
```

---

## Training Details (from train_categorical.json)

```
optimizer: Adam, lr=1e-5 (fixed, no scheduler)
batch_size: 128
epochs: 200, total_steps: 25000
hardware: 4x A6000 GPUs
loss: Class-weighted categorical cross-entropy
normalization: min-max to [0,1] using training set statistics
```

---

## Key Limitations Acknowledged

1. Full 3D attention is O(n^2) in sequence length -- only feasible because input is small (32x32, 4 frames)
2. Designed for scalar output, not pixel-wise video prediction
3. Requires modification for auto-regressive/spatial tasks

---

## Relevance to Solafune Task

| Aspect | Relevance |
|--------|-----------|
| Classification + 64 bins per pixel | High -- could address our 80% zero-value + right-skew problem |
| Class-weighted CE loss | High -- more principled than our weighted MSE attempts |
| Full Space-Time attention | Low -- our UNet architecture is very different |
| CLS token scalar output | N/A -- we need 41x41 spatial output |
| Fixed lr=1e-5 (no ReduceLROnPlateau) | Confirms our concern about adaptive LR |

**Bottom line:** The classification reformulation is the most transferable insight. Per-pixel bin prediction + expected value reconstruction could replace our MSE/MAE regression. High-risk but potentially high-reward.
