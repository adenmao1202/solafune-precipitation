# GlobalMetNet: Global Precipitation Nowcasting with Multi-Satellite Data
**arXiv: 2510.13050** | Authors: Google Research team
Task type: **NOWCASTING/FORECASTING** (multi-satellite images -> global precipitation, 0-12 hour lead time)

---

## Task Definition

Given current and recent observations from 7 geostationary satellite sensors (18 spectral bands total), predict global precipitation at 0.05-degree spatial resolution for lead times from 0 to 12 hours. Training target: GPM CORRA (Combined Radar Retrieval Algorithm) -- a merged product closely related to GPM-IMERG.

**Relevance to our task:** GlobalMetNet is the most directly analogous paper to our competition. It also:
1. Uses multi-geostationary satellite imagery as input (7 sensors vs our 3)
2. Uses a GPM-family product as the training target (CORRA vs IMERG -- same satellite constellation, different algorithm variant)
3. Operates globally, covering regions without ground radar
4. Outputs quantitative precipitation rates (not just binary/category detection)

The difference: GlobalMetNet also predicts future precipitation (up to 12 hours), while we only predict current-time precipitation (lead time = 0). But for the 0-hour "nowcast" output, the problem is essentially identical.

---

## Architecture

Deep residual encoder-decoder.

### Input streams
- **Geostationary satellites (primary):** 7 sensors, 18 spectral bands total. Includes Himawari (Japan), GOES-E/W (Americas), Meteosat (Europe/Africa/Indian Ocean), Fengyun (China). Multiple time steps of past imagery are stacked.
- **NWP (secondary):** ECMWF HRES atmospheric model output (wind, humidity, temperature fields). Available globally but with several-hour delay.
- **Ground radar (tertiary):** Only available for limited regions (CONUS, Europe, Japan). Used as input where available; improves local accuracy.
- **Static features:** Latitude, longitude, terrain elevation per pixel.

### Target
GPM CORRA (Combined Radar Retrieval Algorithm) precipitation rates at 0.1-degree, half-hourly. Regridded to 0.05-degree for model output.

### Output representation
**Categorical: 30 bins** over precipitation rate range. Model outputs a probability distribution over 30 bins per pixel. At inference, convert to expected value (sum over bin_center * probability) for deterministic output, or sample for probabilistic.

### Lead time conditioning
A scalar lead time (0 to 720 minutes) is injected into the architecture (similar in concept to our FiLM day+hour conditioning, but for prediction horizon).

### Polyak Averaging (= EMA)
Model parameters are maintained as an exponential moving average of training checkpoints. At inference, the Polyak-averaged parameters are used rather than the latest checkpoint. This is identical to EMA stabilization used in diffusion models and mentioned in many competition solutions as beneficial.

---

## Training Details

- Hardware: 256 TPU chips, bfloat16 precision
- Scale: trained on multiple years of global satellite + CORRA data
- Normalization: per-channel statistics from training set

---

## Key Ablation: Input Source Importance

The most important finding for us: which satellite/data source matters most?

Ablation removing each input source (in order of importance):
1. **Geostationary satellite imagery** -- removing this causes the largest performance drop across ALL regions and ALL lead times. This is the single most important input.
2. **NWP (ECMWF HRES)** -- provides meaningful benefit at longer lead times (>3 hours) when satellite-only predictions degrade due to cloud/system movement. Less important at short lead times.
3. **Ground radar** -- improves performance only in regions where radar is available (CONUS, Europe). No benefit in data-sparse regions.

**Direct implication for us:** Our 3 geostationary satellites (Himawari + GOES + Meteosat) are the most important inputs. Our current 51-channel setup (3 frames x 16 bands + 3 masks) is the right focus. NWP would be beneficial IF we had it, but we don't.

## Key Finding: Global Forecast Equity

GlobalMetNet closes the "global equity gap" in precipitation forecasting:
- Prior NWP-based models (ECMWF, NOAA HRRR) perform much better over data-rich regions (North America, Europe) than over data-sparse regions (Africa, Southeast Asia, tropical ocean)
- GlobalMetNet achieves uniform performance globally because it uses only satellite data (globally uniform) as primary input
- This is especially important for GPM validation: CORRA quality is also lower over data-sparse regions

**Implication for our competition:** The test dataset likely covers geographic diversity including tropical/oceanic regions. Our model trained on globally diverse samples should not be biased toward well-observed regions. If training data is skewed toward CONUS/Europe, model might underperform on test regions. Check training data distribution.

## Result vs. Baselines

GlobalMetNet surpasses:
- ECMWF HRES (state-of-the-art NWP) across all 7 regions and all lead times 0-12h
- NOAA HRRR (high-resolution US regional model)
- Persistence baseline

Deployed in production on Google Search (visible on Google weather search results).

---

## 30-Bin Classification vs Our MSE

GlobalMetNet uses 30 categorical bins, similar to SatFormer's 64 bins. The rationale:
- Precipitation distribution is highly non-Gaussian (zero-inflated, right-skewed, heavy tail)
- Regression MSE underestimates extremes because the loss is dominated by the zero/light-rain majority
- Classification reformulation allows the model to learn the full distribution shape
- Expected value reconstruction recovers a continuous output

The specific bin boundaries are not specified in detail in the paper, but the same principle applies: equal-width or log-spaced bins from 0 to max_precipitation_rate.

---

## Relevance to Solafune

| Aspect | Applicability | Notes |
|--------|--------------|-------|
| Multi-satellite input (7 sensors, 18 bands) | High -- confirms our 3-satellite, 16-band setup | We have a subset of their inputs; our focus on IR/WV bands is correct |
| GPM CORRA as target | Very high -- closest paper to our task | CORRA and IMERG are both GPM multi-satellite merged products; methods directly transfer |
| 30 categorical bins output | High -- same idea as SatFormer 64 bins | Applicable to our per-pixel prediction problem |
| Polyak Averaging / EMA | High -- easy to implement | Stable training improvement, low risk |
| Lead-time conditioning | N/A for retrieval | We predict current-time only; our FiLM handles current-time already |
| Static features (lat/lon/elevation) | Medium | Could help with geographic generalization; low implementation cost |
| NWP as auxiliary input | Not available | We don't have NWP features in competition data |
| Global equity principle | Strategic | Ensure our training data covers diverse geographies, not just CONUS |

**Key actionable items:**
1. **EMA/Polyak averaging:** Add to our training loop. Low risk, potential stability improvement.
2. **Classification bins (30-64 bins):** High-priority experiment per SatFormer findings + GlobalMetNet confirmation.
3. **Verify training data geography:** If training set is skewed toward certain regions, model may not generalize well to test set's geographic diversity.
