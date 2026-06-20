# TUPANN: Precipitation Nowcasting of Satellite Data Using Physically-Aligned Neural Networks
**arXiv: 2511.05471** | Authors: Andres Hernandez-Tello et al.
Task type: **FORECASTING** (satellite images at time t -> precipitation at t+10 to t+180 min)

---

## Task Definition

Given sequences of GOES-16 ABI satellite imagery, predict precipitation fields at lead times from 10 to 180 minutes. Ground truth is GOES-16 RRQPE (Rain Rate QPE, geostationary-derived) and IMERG (for generalization testing). Tested across 4 South American and North American cities: Rio de Janeiro, La Paz, Manaus, Miami. Also evaluated zero-shot on Toronto (unseen during training).

**Critical difference from our task:** TUPANN is FORECASTING from past satellite to future precipitation. We are RETRIEVAL from current satellite to current GPM-IMERG. TUPANN's optical flow supervision makes physical sense for temporal forecasting; it does not apply to our single-timestep retrieval setting.

---

## Architecture: VED + MaxViT

TUPANN uses a two-component architecture:

### Component 1: VED (Variational Encoder-Decoder)
- Purpose: models spatial motion dynamics via optical flow
- Variational autoencoder structure with latent space sampling
- **Optical flow supervision**: the encoder learns to estimate motion vectors between consecutive frames. Separate loss terms supervise this:
  - lambda_cos = 0.00165 (cosine similarity for motion direction)
  - lambda_motion = 0.0033 (motion magnitude)
  - lambda_int = 0.995 (intensity reconstruction)
  - lambda_KL = 1e-6 (KL divergence for latent regularization)
- The differentiable warp module applies learned motion vectors to warp current precipitation fields forward in time, producing "physics-aligned" predictions

### Component 2: MaxViT (lead-time conditioned)
- Vision Transformer backbone with local and global attention
- Receives lead-time as a conditioning signal (similar in spirit to our FiLM time conditioning, but for prediction horizon not current time)
- Applies correction on top of the VED warp output
- Depth=4, dim=64

### GAN variant (GAN-TUPANN)
- Adds a discriminator for perceptual sharpness
- Produces sharper spatial structures but sometimes less accurate intensity
- In IMERG test (Fig. 11), GAN-TUPANN and TUPANN are visually comparable; TUPANN is quantitatively more reliable

---

## Training Details

### Event-based training (key design choice)
TUPANN trains ONLY on rain events, defined as time windows where accumulated precipitation exceeds threshold tau=120,000 units in a symmetric +/-4-hour window. Rainy windows are merged if overlapping.

**Why event-based:** Metrics used are CSI and HSS, which compute precision/recall over binary detection at thresholds. These metrics explicitly ignore True Negatives (correct no-rain predictions). Training on all data wastes compute on the 80-90% no-rain background.

**Implication for us:** Our competition metric is RMSE, which penalizes errors on ALL pixels including the majority zero-rain background. Event-based sampling would actively hurt our RMSE since the model would never learn to suppress false positives on non-rainy pixels. Do NOT adopt this technique.

### Hyperparameters (Appendix Table 6)
VED:
- batch_size: 8
- learning_rate: 0.0001
- channels: 128
- embed_dim: 4
- reduc_factor: 4
- dropout: 0.2

MaxViT:
- batch_size: 8
- learning_rate: 0.0001
- MaxViT_depth: 4
- MaxViT_dim: 64

Optimizer: Adam for both components
Loss: L1 (intensity) + cosine similarity (motion direction) + KL divergence (latent) + motion magnitude

Hardware: RTX 3080, 16GB RAM. Inference: under 3 minutes per event.

---

## Results

### GOES-16 results across 4 cities (Table 7, HSS metrics)

| City | HSS-M | HSS_4 | HSS_8 | HSS_16 | HSS_32 | HSS_64 |
|------|-------|-------|-------|--------|--------|--------|
| Rio de Janeiro | **0.393** | **0.473** | **0.439** | **0.492** | **0.428** | **0.135** |
| Miami | **0.277** | 0.398 | 0.309 | 0.298 | 0.237 | **0.146** |
| Manaus | **0.435** | 0.479 | 0.464 | 0.469 | **0.430** | **0.333** |
| La Paz | **0.468** | 0.482 | **0.481** | **0.512** | **0.490** | **0.376** |

TUPANN is first or second across almost all settings. Most notably dominates at extreme thresholds (HSS_64), which reflects the optical-flow physics alignment better preserving intense cells.

Competitors: Earthformer, NowcastNet, PySTEPS (LK + DARTS), CasCast.

### Zero-shot generalization (Table 8, Toronto, GOES-16)

| Model | CSI-M (POOL1) | CSI-M (POOL4) |
|-------|--------------|--------------|
| NowcastNet | 0.206 | 0.216 |
| TUPANN | 0.219 | 0.222 |
| TUPANN-Multicity | **0.229** | **0.230** |

TUPANN-Multicity (trained on all 4 South/North American cities) outperforms single-city TUPANN on unseen Toronto. Cross-city performance degradation is at most ~20%.

**Insight for us:** We face a similar generalization challenge (training on labeled data, testing on different geographic regions). However, multi-city training would require labeled data from diverse regions -- in our case, that is already inherent since the competition data covers the full Meteosat/Himawari/GOES disk coverage.

---

## Relevance to Solafune

| Aspect | Applicability | Notes |
|--------|--------------|-------|
| Optical flow supervision | Not applicable | Only meaningful for temporal forecasting; we do single-timestep retrieval |
| VED + MaxViT dual architecture | Low | Adds complexity for a forecasting-specific problem we don't have |
| Event-based training | Counter-productive | Our RMSE penalizes ALL pixels; training only on rain events would increase false positives |
| Lead-time conditioning | Not applicable | We have no lead time; FiLM time conditioning already handles current-time encoding |
| Multi-city training improves generalization | Partially relevant | Our global training data already spans diverse geographies |
| Adam lr=1e-4, batch_size=8, dropout=0.2 | Reference only | Different task scale, but Adam+low-LR is consistent with other papers |
| L1 loss preferred over MSE | Note | TUPANN uses L1; consistent with our observation that MAE slightly helps |

**Bottom line:** TUPANN's core innovations (optical flow physics, event-based training) are specific to temporal forecasting and do not transfer to our retrieval task. The main takeaway is confirmation that L1/MAE loss is preferred for precipitation, and that geographic diversity in training data is beneficial.
