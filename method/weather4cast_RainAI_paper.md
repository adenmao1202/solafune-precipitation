# RainAI: Precipitation Nowcasting from Satellite Data
**NeurIPS 2023 Weather4Cast Competition**
arXiv: 2311.18398 | Authors: Rafael Pablos Sarabia, Joachim Nyborg, Morten Birk, Ira Assent
Affiliations: Aarhus University + Cordulus (weather intelligence company)

---

## Task

From 4 frames (1 hour) of 11-band EUMETSAT geostationary satellite radiances at low resolution (252x252 pixels = ~3000x3000 km), predict quantitative rainfall rates for the next 8 hours (32 timesteps at 15-min intervals). Output must be at high resolution (252x252 pixels = ~2km/pixel). This adds a **super-resolution** component on top of the forecasting task.

---

## Core Innovations

### 1. 2D UNet Beats 3D UNet

The baseline official model was a 3D UNet (spatial + temporal convolutions). RainAI's key finding:

**2D UNet outperforms 3D UNet** by treating temporal frames as additional channels:
- Input shape: (T x C x H x W) flattened to (T*C x H x W) = (44 x 252 x 252)
- Standard 2D convolutions process this input
- Skip connections combine high-level and fine-grained features as normal

Why 2D wins: 3D convolutions model time and space simultaneously but introduce unnecessary complexity. The simpler 2D approach with more training focus on spatial patterns generalizes better.

**Result:** Best 2D UNet (exp8) CSI=0.0507 vs official 3D UNet baseline CSI=0.0444. (+14% relative).

### 2. Importance Sampling (core finding)

The dataset is massively imbalanced: most samples have little or no rainfall.

Assign acceptance probability q_n to each sample proportional to precipitation intensity:
- High-rain samples: high acceptance probability (always included)
- No-rain samples: low but nonzero acceptance probability

**This is NOT the same as changing the loss weights.** Only the sampling distribution changes; the loss function treats all included samples equally.

**Ablation impact:**
- exp1 (ResNet 2D UNet, CE loss, NO sampling): CSI = 0.0306
- exp2 (same + importance sampling): CSI = 0.0491 (+60% relative)

This is the single biggest improvement in the paper.

### 3. Cross-Entropy Loss (Classification Framing)

Instead of MSE regression, frame as classification over discrete rainfall intensity buckets (Table 1 in paper lists the bin boundaries). Convert back to intensity for submission by: sum(bucket_center * predicted_probability).

Benefits:
- Enables class-weighting to handle imbalance
- Produces probabilistic outputs (full distribution, not just point estimate)
- Works better with highly skewed data than direct regression

Note: When class weights were added to CE (exp6 vs exp4), performance actually dropped slightly (0.0451 vs 0.0502), suggesting class weighting needs careful tuning.

### 4. Lead Time Conditioning

Model predicts for a single specified lead time (1 to 32 steps) rather than all 32 at once. The lead time scalar is injected as an additional input channel (uniform value across the spatial patch).

This avoids auto-regressive error accumulation. Similar to our FiLM day+hour conditioning but for prediction horizon rather than current time.

### 5. Super-Resolution Output

Two-stage output generation:
1. UNet output at input resolution (128x128) is center-cropped to 42x42 (the radar coverage area)
2. Cropped 42x42 is upsampled to 252x252 using learned super-resolution

Upsampling methods tested:
- Bilinear interpolation (fast, nearly as good)
- EDSR (learned, slightly better)
- NinaSR (best: CSI=0.0507)

**Insight for us:** We do the reverse -- train at 128x128, resize output to 41x41. Their experience suggests learned upsampling > bilinear but the gap is small.

### 6. Static Geographic Features

Include latitude, longitude, and topographic height per pixel as additional input channels. This provides the model with geographic context that may influence precipitation patterns.

**Relevance to us:** Our test locations are completely new -- adding lat/lon might help generalize to unseen geographies.

---

## Architecture Details

Two encoder variants tested:
- ResNet-18 encoder (residual blocks, default)
- Swin Transformer encoder (shifted window attention, global context)

Both paired with standard UNet decoder with skip connections.

Input: (T*C, H, W) = (4*11, 128, 128) = (44, 128, 128) for the cropped spatial context.

---

## Training Details

Not fully specified in paper, but key points:
- Optimizer: Adam (inferred from standard practice)
- Loss: Cross-entropy with optional class weights
- Dataset: 7 European regions for core, 3 additional for transfer track
- Importance sampling probability: proportional to rain intensity

---

## Results Summary

| Experiment | Description | Core CSI |
|------------|-------------|----------|
| 3D UNet baseline | Official baseline | 0.0444 |
| exp1 | ResNet 2D UNet + CE, no sampling | 0.0306 |
| exp2 | + importance sampling | 0.0491 |
| exp4 | + bilinear upsample, 128x128 input | 0.0502 |
| exp5 | + lead time conditioning | 0.0477 |
| exp6 | + class weights | 0.0451 |
| exp7 | + EDSR super-resolution | 0.0482 |
| exp8 | + NinaSR super-resolution | 0.0507 |

Top 5 in Weather4Cast 2023.

**Key failure case:** All models predict CSI=0 for precipitation above 5mm/hr. Extreme events are so rare in training data that the model never learns them.

---

## Relevance to Solafune Task

| Aspect | Applicability | Notes |
|--------|--------------|-------|
| 2D UNet beats 3D UNet | Confirms our current direction | We already use 2D UNet |
| Importance sampling (adjust sampling only) | Medium -- worth retrying | Our v6 failed because we ALSO changed the loss. Only sampling might be safer. |
| CE classification loss | High -- same idea as SatFormer | Could replace our MSE+MAE |
| Lead time conditioning | N/A | We don't do multi-step forecasting |
| Lat/lon as input features | Medium | Could help generalize to unseen locations |
| Learned upsampling | Low priority | Our 128->41 is already bilinear; small gain expected |
| Class weights in CE | Careful -- exp6 showed it can hurt | Need careful tuning |

**Key insight: Importance sampling alone (without loss reweighting) gave the biggest single boost (+60%). This is directly applicable to us and avoids the v6 mistake of over-correcting with both sampling and loss weighting simultaneously.**
