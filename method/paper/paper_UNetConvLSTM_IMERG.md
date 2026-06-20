# GENESIS: Global Precipitation Nowcasting of Integrated Multi-satellitE Retrievals for GPM -- A U-Net Convolutional LSTM Architecture
**arXiv: 2307.10843** | Authors: Reyhaneh Rahimi, Praveen Ravirathinam, Ardeshir Ebtehaj, Ali Behrangi, Jackson Tan, Vipin Kumar
Affiliations: University of Minnesota, University of Arizona, NASA Goddard
Task type: **FORECASTING** (past IMERG + GFS -> future IMERG, 4-hour lead time)
Full paper name: "GENESIS" = Global prEcipitation Nowcasting using intEgrated multi-Satellite retrIevalS for GPM

---

## Task Definition

Given 6 hours of past IMERG precipitation sequences (12 timesteps at 30-min intervals) plus GFS NWP forecast variables (TPW, U/V wind), predict IMERG precipitation for the next 4 hours (8 timesteps at 30-min intervals). The model is trained and evaluated on global IMERG data from April 2020 to March 2023.

**Important distinction from our task:**
- GENESIS is FORECASTING: uses past IMERG as input to predict future IMERG
- Our task is RETRIEVAL: uses satellite band imagery to predict current IMERG
- GENESIS cannot be directly applied in our setting (we don't have past IMERG as input at test time)
- BUT: the loss function findings (MSE vs focal loss on IMERG data) transfer directly, since both use IMERG as the TARGET with the same zero-inflated, right-skewed distribution

---

## Data

### Input
- **IMERG Early Run (V06):** half-hourly, April 2020 - March 2023, 0.1-degree global grid
  - 12 timesteps (6 hours) used as input sequence
  - Auto-correlation analysis: IMERG correlation decays exponentially (alpha ~= 0.33 hr^-1, correlation length tau ~= 3 hours). After 6 hours, median correlation ~= 0.1, confirming 6-hour window captures essentially all memory.
- **GFS (NCEP v16):** hourly, 0.25-degree resolution
  - Variables: U/V horizontal wind velocity (2m), Total Precipitable Water (TPW, kg/m^2)
  - 6 past + 6 future GFS timesteps used (future GFS is available as forecast, paired with predicted IMERG)
  - Total C=4 predictors: [IMERG, TPW, U, V]

### Preprocessing
- Patches of 256x256 pixels randomly sampled globally
- 100,000 training patches (April 2020 - March 2022, 70/30 train/val)
- 15,000 test patches (April 2022 - March 2023)
- At inference: tile with 50% overlap, use inner 128x128 pixels to avoid boundary artifacts

### Feature Importance (MRMR analysis, Fig. 3)
Most informative features for T+30 min prediction (highest mutual information with output):
1. Recent IMERG precipitation (P_T, P_T-30, P_T-60): score 0.64 average
2. TPW (future values T+150 to T+300 min): score 0.48
3. U/V wind velocity: score 0.32

**Implication:** Past precipitation is the most informative predictor, followed by atmospheric moisture (TPW). For our task (no past IMERG available), the satellite imagery must substitute as the precipitation proxy. Bands sensitive to deep convection (cold cloud tops in IR, WV absorption) are our equivalent of this high-importance input.

---

## Architecture: GENESIS (U-Net + R-ConvLSTM)

Standard U-Net is insufficient for temporal sequence prediction because it has no explicit memory mechanism. GENESIS adds ConvLSTM cells to handle spatiotemporal dynamics.

### Encoder
- 5 encoder blocks, each block processes the input tensor sequence X in R^(HxWxC) for T=1,...,12 timesteps through a ConvLSTM cell
- Output at each block: spatial feature map Z^T_n in R^(H'xW'xC')
- Standard max-pooling, batch normalization, 15% dropout between blocks

### Recursive ConvLSTM (R-ConvLSTM) -- Key Innovation
At the bottleneck (deepest encoder level), a Recursive ConvLSTM operator takes the final latent embedding Z^12_n and auto-regressively generates future latent embeddings for 8 future timesteps:
- Z^12_n as initial input; H^0 and C^0 (hidden and cell states) initialized to zero
- Output: Y_n = [Y^1_n, ..., Y^8_n] for T=1,...,8 future steps
- The final context Z^12_n is ADDED to each step's input to prevent information loss during long unrolling

**Critical insight:** The R-ConvLSTM mechanism is applied not just at the bottleneck but ALSO at each skip connection level. This means the skip connections carry not static encoder features but RECURSIVELY PREDICTED future features at multiple scales. This prevents the skip connections from leaking "current time" information into the future prediction.

### Decoder
- At each decoder level: concatenate recursively-predicted skip features with upsampled features via Conv3DTranspose
- Apply Conv3D, batch normalization, 15% dropout
- Final layer differs between MSE and FL variants:
  - GENESIS_MSE: Conv1x1 -> normalization -> ReLU
  - GENESIS_FL: Conv1x1 -> normalization -> Softmax (over 10 precipitation classes)

---

## Loss Functions

### GENESIS_MSE (regression)
Standard mean squared error on predicted vs. actual IMERG precipitation rates. Adam optimizer, lr=1e-3, batch=8.

### GENESIS_FL (focal loss / classification)
Discretize IMERG into 10 classes on logarithmic scale: 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 32 mm/hr.
- Rates below 0.1 -> class 1
- Rates above 32 -> class 10

Focal loss:
```
FL(y, p) = -(1/N) * sum_{i,c} alpha_c * y_{i,c} * (1 - p_{i,c})^gamma * log(p_{i,c})
gamma = 2 (focusing parameter)
alpha_c proportional to inverse class frequency, sum(alpha_c) = 1
```

**Training details:**
- Adam optimizer, initial lr=1e-3
- LR decay: factor 0.1 every 10 epochs when validation loss does not improve
- Batch size: 8
- Weight initialization: Xavier method
- Hardware: AMD EPYC 7542 CPU + NVIDIA A-100 40GB GPU, TensorFlow-V2

---

## Results

### Main finding: regression vs classification crossover at r = 1.6 mm/hr

From Table 1 (Hurricane Ian case, single storm):
```
                T+30   T+60   T+90   T+120  T+150  T+180  T+210  T+240
CSI_1  MSE:    0.75   0.69   0.63   0.58   0.47   0.41   0.37   0.35
CSI_1  FL:     0.71   0.65   0.60   0.56   0.47   0.43   0.39   0.39
CSI_8  MSE:    0.55   0.43   0.36   0.30   0.28   0.27   0.25   0.18
CSI_8  FL:     0.59   0.48   0.41   0.37   0.35   0.35   0.32   0.31
```

From Fig. 7 multi-storm (15,000 test samples):
- CSI_1 (>=1 mm/hr threshold): GENESIS_MSE > GENESIS_FL at short lead times, converge later
- CSI_4 (>=4 mm/hr): FL begins to outperform MSE
- CSI_8 (>=8 mm/hr): FL clearly superior, especially at longer lead times

**The crossover threshold is approximately 1.6 mm/hr:**
- Below 1.6 mm/hr: MSE (regression) wins -- captures bulk of distribution better
- Above 8 mm/hr: FL (classification) wins -- focal penalty prevents heavy-rain class suppression

### FSS (Fractions Skill Score) at multiple spatial scales
GENESIS is skillful at 10km resolution for precipitation >= 1 mm/hr (GFS skillful only at 50km).
For rates >= 4 mm/hr, only GENESIS_FL remains FSS-skillful at scales > 50km within 2 hours.

### GFS auxiliary data helps at long lead times
Including GFS wind/TPW inputs improves performance more at T+120 to T+240 min than at T+30 min. At short lead times, past IMERG alone provides sufficient information for accurate prediction.

---

## Relevance to Solafune

| Aspect | Applicability | Notes |
|--------|--------------|-------|
| IMERG as training target | Direct -- same target as us | Architecture is forecasting-specific, but loss findings apply directly |
| MSE better for light rain (<1.6 mm/hr) | High relevance | Our RMSE is dominated by these low-intensity samples; MSE head might be necessary |
| Focal loss better for heavy rain (>8 mm/hr) | High relevance | Our 80% zero + right-skew: FL prevents the model ignoring high-intensity bins |
| Combined MSE+FL approach | Very high -- mirrors our 0.3*MSE + 0.7*MAE | Paper recommends FL but acknowledges hybrid might capture both regimes |
| 10 log-spaced bins for classification | Applicable | Simpler than SatFormer's 64 uniform bins; log-spacing better matches IMERG's distribution |
| Adam lr=1e-3, decay 0.1 per 10 epochs | Reference | Consistent with step-LR decay approach (vs our ReduceLROnPlateau) |
| R-ConvLSTM skip connections | Not directly applicable | This is a temporal forecasting architecture; our UNet processes single time step flattened |
| 6-hour IMERG auto-correlation analysis | Background knowledge | Confirms that 3 frames (90 min) we use captures most of the useful temporal context |
| GFS variables help | Not available | We don't have GFS variables in competition data |

**Key actionable items for us:**
1. **Focal loss experiment:** Add a focal loss variant (gamma=2, 10 log-spaced bins) as an alternative to our 0.3*MSE + 0.7*MAE. Expected gain at high-intensity precipitation.
2. **Consider hybrid loss:** Train with MSE head AND classification head jointly, or compare the two directly. The crossover at 1.6 mm/hr suggests a hybrid could outperform either alone.
3. **10 log-spaced bins as alternative to 64 uniform bins:** For our 0-50 mm/hr output range, log-spaced bins (0.1, 0.2, 0.4, ..., 32 mm/hr) might better match the IMERG distribution than SatFormer's 64 uniform bins.

**Critical note on task difference:** GENESIS uses past IMERG sequences as its primary input (MRMR score 0.64). We cannot do this because we're doing retrieval (no past IMERG available). Our satellite bands must compensate for this missing high-value feature. This confirms that maximizing information extraction from satellite bands (all 16 bands per sensor, FiLM time conditioning) is our correct focus.
