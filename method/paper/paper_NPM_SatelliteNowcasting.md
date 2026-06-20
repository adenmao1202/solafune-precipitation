# NPM: Neural Precipitation Model -- Satellite-Only Nowcasting
**arXiv: 2412.11480** | Authors: (Korean team)
Task type: **FORECASTING** (geostationary satellite -> radar-derived precipitation, 0-6 hour lead time)
Dataset: Sat2Rdr (based on GK2A Korean geostationary satellite), 41,637 hourly samples

---

## Task Definition

Given sequences of geostationary satellite imagery (no radar data required), predict 2km-resolution precipitation maps for the next 6 hours. Unlike most nowcasting systems that require existing radar coverage, NPM works globally from satellite-only input. Ground truth is Korean radar-derived precipitation at 2km/5min resolution, collocated with GK2A satellite data.

**Relevance to our task:** NPM is also satellite-to-precipitation without radar dependency. However, it is still FORECASTING (predict future from past), while our task is RETRIEVAL (predict current from current). Key transferable insights: Day+Hour encoding and season-aware sampling.

---

## Architecture: Two-Stage Pipeline

NPM decomposes the problem into two stages:

### Stage 1: Satellite Video Prediction
Goal: predict what the satellite imagery will look like at future timesteps.
- Input: current and past satellite frames
- Output: predicted future satellite frames (video extrapolation)
- Architecture: spatiotemporal neural network (likely ConvLSTM-based, specific architecture details limited in available content)
- Loss: reconstruction loss on satellite imagery

### Stage 2: Satellite-to-Radar Translation (StegoGAN)
Goal: translate (predicted or observed) satellite imagery to precipitation rates.
- Input: satellite imagery (real or stage-1 predicted)
- Output: precipitation rate maps
- Architecture: StegoGAN (a GAN-based image translation framework)
- Loss: adversarial + reconstruction

**The two-stage split is the key architectural innovation:** By separating satellite temporal evolution from satellite-to-precipitation mapping, each stage can be specialized and trained independently. Stage 2 alone (satellite-to-radar at current time) is essentially our task.

---

## Key Innovation 1: Day+Hour Positional Encoding

**This is the single most impactful component in NPM's ablation study.**

The model encodes the current observation time using:
- Day of year (1-365) encoded as sin/cos pair
- Hour of day (0-23) encoded as sin/cos pair
- All 4 values concatenated as a 4-dimensional time feature

This is EXACTLY what our FiLM time conditioning implements. NPM validates this approach rigorously:
- Ablation: removing Day+Hour encoding causes the largest performance drop of any component
- Day embedding alone accounts for most of the improvement (captures seasonal/diurnal precipitation cycles)
- Hour embedding adds smaller but significant improvement (captures diurnal cycle within day)

**Confirmation for our work:** Our v8a FiLM implementation is the right approach. NPM independently arrives at the same design and confirms it is the most important single component. We should NOT remove FiLM time conditioning.

---

## Key Innovation 2: Season-Aware Sampling

Training data spans multiple years with strong seasonal variation. Random sampling leads to over-representation of certain seasons.

NPM uses a stratified sampling strategy that ensures:
- Each season (spring/summer/fall/winter) is represented proportionally in each training batch
- Prevents the model from overfitting to the most common seasonal patterns
- Improves generalization across all months

**Relevance to us:** Our training data covers a specific time window. If it's not balanced across seasons, season-aware sampling could help. However, we don't know the exact temporal distribution of our training data.

---

## Key Innovation 3: Temporal Consistency Constraint

When predicting multiple lead times, NPM enforces that predictions are physically consistent across time steps. Precipitation at t+2h should not be radically different from t+1h predictions unless there's a strong physical reason.

Implemented as an auxiliary loss on temporal smoothness of predictions.

**Relevance to us:** N/A for our single-timestep retrieval task.

---

## Input Features

Only 3 satellite channels used (extremely minimal input):
- **IR 10.5 um (infrared window):** cloud-top temperature, correlates with deep convection
- **WV 6.3 um (water vapor):** upper tropospheric moisture
- **WV 7.3 um (water vapor):** mid-tropospheric moisture

Plus: DEM (Digital Elevation Model) as a static geographic channel.

**Insight:** NPM achieves strong results with only 3 channels + DEM. This validates the importance of IR/WV channels specifically (which are among our 16 bands per satellite). The WV 6.3um and WV 7.3um channels are our most physically informative bands for precipitation detection.

Hardware: 8x A6000 GPUs.

---

## Results

NPM outperforms radar-only extrapolation methods (like PySTEPS) specifically for **sudden-onset precipitation** -- rain events that begin with no prior radar signature. This is because:
- Radar-based extrapolation assumes precipitation cells move but new ones don't form suddenly
- Satellite can detect cloud formation (high, cold tops) before any precipitation falls to the surface
- NPM's Stage 1 predicts cloud evolution, enabling early warning of new precipitation systems

**Relevance to our task:** We also have this advantage -- our satellite inputs contain cloud formation information that GPM-IMERG does not capture until precipitation actually reaches the surface. This is a fundamental advantage of multi-satellite retrieval over persistence-based methods.

---

## Ablation Table (Day+Hour encoding is most important)

| Component | Performance |
|-----------|-------------|
| Full model | Best |
| - Day+Hour encoding | Largest drop (confirmed most important) |
| - Season-aware sampling | 2nd largest drop |
| - Temporal consistency constraint | Smaller drop |
| - DEM | Small drop |
| 3 channels only (no WV 7.3) | Noticeable drop |

---

## Relevance to Solafune

| Aspect | Applicability | Notes |
|--------|--------------|-------|
| Day+Hour sin/cos encoding | Already implemented! | Our FiLM v8a is exactly this; NPM validates it is the most important component |
| Season-aware (stratified) sampling | Medium | Could help; need to check if our data is temporally balanced |
| Stage 2 only (satellite -> precip) | High -- this IS our task | Stage 2 is a direct analog to our problem |
| 3-channel insight (IR + 2x WV) | Medium | Confirms these are the most important bands; could inform band selection experiments |
| GAN-based translation | Low | Adversarial training adds instability; lower priority than classification loss |
| Temporal consistency constraint | N/A | Single timestep retrieval; no temporal sequence to constrain |
| Satellite-only = no radar needed | Confirms our approach | Global applicability without radar; competition data is satellite-only |
| DEM as static feature | Low-Medium | Available globally; small benefit observed |

**Key actionable items:**
1. **FiLM time conditioning (already done):** NPM confirms Day+Hour is the single most important component. Keep it.
2. **Season-aware sampling:** If our training set is temporally skewed, worth implementing. Low risk.
3. **Band importance:** Consider experiments prioritizing IR 10.5um, WV 6.3um, WV 7.3um (their top 3 channels) when testing band ablations.
