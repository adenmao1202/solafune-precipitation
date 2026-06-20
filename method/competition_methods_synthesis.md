# Competition Methods Synthesis
# 競賽方法論整合：對 Solafune 降水預測任務的參考價值分析

整理日期：2026-06-19（更新：2026-06-19，加入四篇 precipitation 論文）
資料來源：
- SpaceNet8 5th place README (method/spacenet8_5th_place_README.md)
- UNetTFI arXiv 2311.18341 (method/weather4cast_UNetTFI_README.md)
- RainAI arXiv 2311.18398 (method/weather4cast_RainAI_paper.md)
- SaTformer arXiv 2511.11090 (method/weather4cast_SatFormer_paper.md)
- TUPANN arXiv 2511.05471 (method/paper_TUPANN.md)
- GlobalMetNet arXiv 2510.13050 (method/paper_GlobalMetNet.md)
- NPM arXiv 2412.11480 (method/paper_NPM_SatelliteNowcasting.md)
- GENESIS arXiv 2307.10843 (method/paper_UNetConvLSTM_IMERG.md)

---

## 重要前提：任務差異說明

在讀各方法前，必須先理解我們的任務和下列競賽的根本差異：

| 項目 | 我們（Solafune） | Weather4Cast | TUPANN | GlobalMetNet | NPM | GENESIS |
|------|----------------|--------------|--------|--------------|-----|---------|
| 任務類型 | 衛星 -> 當前降水（Retrieval） | 過去衛星 -> 未來降水（Forecasting） | 過去衛星 -> 未來降水（Forecasting） | 多衛星 -> 全球降水 0-12h（Forecasting） | 衛星 -> 未來降水 0-6h（Forecasting） | 過去 IMERG+GFS -> 未來 IMERG 4h（Forecasting） |
| 輸入 | 3 衛星 x 3 frames x 16 bands + masks = 51ch | 1 衛星 x 4 frames x 11 bands | GOES-16 ABI 多波段影像序列 | 7 衛星 x 18 bands + NWP + 雷達（可選）| 3 channels (IR/WV) + DEM | 12 frames IMERG + GFS (TPW/U/V) |
| 輸出 | 當前降水量 mm/hr（回歸，41x41 每 pixel） | 未來 4-8 小時降水（scalar 或 multi-step） | 未來 10-180 分鐘降水場 | 全球降水 0.05 度，0-12h，30 bins | 未來 6h 降水，2km 解析度 | 未來 4h IMERG，8 個時間步 |
| 目標資料 | GPM-IMERG（衛星微波被動反演） | OPERA 地面雷達（歐洲） | GOES-16 RRQPE 或 IMERG | GPM CORRA（與 IMERG 同族產品）| 韓國地面雷達衍生降水（輸入是 GK2A 衛星）| IMERG Early Run V06 |
| 評估指標 | RMSE（全 pixel，含無雨區） | CRPS / CSI | CSI + HSS（多個閾值）| 比較 vs ECMWF HRES | 未明確（自訂 metric）| CSI 1/4/8 mm/hr + FSS |
| 資料不均衡 | 80% 零值像素 | 大多數樣本無雨 | 只在雨事件訓練（CSI 指標）| 全球分布不均 | 季節/時間不均 | IMERG 右偏，極端值稀少 |

**關鍵差異：我們是 retrieval（從當前輻射值推導當前降水），所有其他論文都是 forecasting（從過去預測未來）。
時序外插技巧不能直接移植。但資料不均衡、loss 設計、空間轉移泛化問題是完全共通的。
GlobalMetNet 和 GENESIS 使用 GPM 族群產品（CORRA/IMERG）作為訓練目標，是最直接相關的論文。**

---

---

## 0. 本文論文關聯性速查

在詳讀各方法前，以下是哪些論文對我們最重要的快速總結：

| 重要性 | 論文 | 最關鍵洞察 | 對應文件 |
|--------|------|-----------|---------|
| *** 最高 | GlobalMetNet | 同樣使用 GPM CORRA 作目標；30 bins 分類；EMA；衛星輸入最重要 | paper_GlobalMetNet.md |
| *** 最高 | GENESIS (UNetConvLSTM) | 同樣預測 IMERG；MSE < 1.6mm 好，FL > 8mm 好；10 log-spaced bins | paper_UNetConvLSTM_IMERG.md |
| ** 高 | NPM | 獨立驗證 Day+Hour encoding 是最重要單一組件（=我們的 FiLM）| paper_NPM_SatelliteNowcasting.md |
| * 中 | TUPANN | 物理對齊的光流監督；但 event-based 訓練對 RMSE 任務有害 | paper_TUPANN.md |
| ** 高 | RainAI | Importance sampling 唯一操作最大改善 +60%；2D UNet 優於 3D | weather4cast_RainAI_paper.md |
| ** 高 | SaTformer | 64 bins 分類 + class-weighted CE；固定 lr=1e-5（無 scheduler）| weather4cast_SatFormer_paper.md |
| * 中 | UNetTFI | TFI 時序監督；Transfer Learning Track 冠軍 | weather4cast_UNetTFI_README.md |
| * 中 | SpaceNet8 | Step LR decay；EMA；Ensemble；Backbone B5 | spacenet8_5th_place_README.md |

---

## 1. SpaceNet8 第 5 名解法（Motoki Kimura）

**來源：** method/spacenet8_5th_place_README.md
**任務：** 洪水偵測語意分割（建物 + 道路 x 淹水/未淹水）

### 目標
從災前/災後光學衛星影像，像素級分割出淹水建物與淹水道路。

### 實驗設計
- 5-fold cross-validation，按地理區塊劃分（非隨機），確保各 fold 地理多樣性
- 固定 random seed，讓不同 backbone 結果可直接比較
- 從輕量 backbone（B3）快速實驗 -> 最終採用 B5/B6

### 解法核心

**Backbone 升級：** EfficientNet-B3（快速實驗）-> B5/B6（最終提交）

**預訓練模型 Fine-tuning：** 從其他競賽的得主模型 fine-tune，是本解法最大的 LB 提升來源：
- 從 SpaceNet-5 winner（SE-ResNeXt-50）fine-tune 道路分割：+0.89 LB
- 從 xView2 winner（DenseNet-161）fine-tune 建物分割：+0.32 LB

**Step LR Decay（非 ReduceLROnPlateau）：** 固定在特定 epoch 降 LR（例如 epoch 80, 120 各乘 0.1）。不讓 val 表現影響 LR 排程。

**EMA（Exponential Moving Average）：** 每 epoch 更新，momentum=0.002。用途：val metric 從 epoch 到 epoch 震盪劇烈，EMA weights 更穩定，inference 用 EMA 模型。

**Ensemble：** 最終 35 個模型加權平均（EfficientNet-B5/B6 各 5 fold + pretrained fine-tune 各 5 fold）

**資料增強：** Mosaicing（將相鄰 4 個 tile 拼接成更大訓練樣本）；其他增強很少。

**TTA：** 只對 flooded building channel 做左右翻轉（TTA 增益在其他 channel 不顯著）。

### 結果
SpaceNet-8 第 5 名。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| Step LR decay | 高 | 比 ReduceLROnPlateau 更不容易卡死；設 epoch 20/35/50 各降一次 LR |
| EMA | 高 | 我們的 val RMSE 也有 epoch-to-epoch 震盪，EMA 可穩定 inference |
| Backbone B5/B6 | 中 | 他明確驗證 B5/B6 > B3/B4；需測試 GPU memory 是否允許 |
| Ensemble（3+ seeds） | 高 | 最穩定的性能提升，最終成績都靠這個 |
| 地理分組 k-fold | 間接 | 我們的 test 地點完全不重疊，k-fold 比 temporal split 更能評估真實泛化 |
| Flip TTA | 高 | 降水場在翻轉方向上物理合理，且零成本 |
| Domain-pretrained fine-tuning | 低 | 找不到公開的 GPM/衛星降水 pretrained model |

---

## 2. Weather4Cast 2023 UNetTFI（Transfer Learning Track 第 1 名）

**來源：** method/weather4cast_UNetTFI_README.md | arXiv: 2311.18341
**任務：** 從 1 小時衛星影像預測未來 4/8 小時降水，強調新地點（空間轉移）的泛化

### 解法核心

#### Temporal Frame Interpolation（TFI）
輸入有 4 個時間 frame，TFI 在相鄰 frame 之間插值出合成的中間 frame，強迫模型學習「雲的運動規律」：

```
原始：frame_t1 -> frame_t2 -> frame_t3 -> frame_t4
TFI：同時監督模型插出正確的 frame_t1.5, frame_t2.5, frame_t3.5
```

效果：使模型對空間轉移更強健——因為它學到的是通用的雲移動物理，而不是特定地區的氣候統計。

#### Multi-Level Dice Loss（ML-Dice）
把不同降水強度分成多個等級（輕雨/中雨/暴雨），分別計算 Dice loss 再加總。
加入序數關係：under-prediction 比 over-prediction 懲罰更重。
比 vanilla Dice 更好地處理降水強度的序數關係。

### 結果
Transfer Learning Track 第 1 名——在未見新地點上表現最好。這個結論直接和我們的問題相關。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| TFI 概念（時序運動監督） | 間接 | 我們有 3 frames，可以要求模型插出「中間時刻」的衛星影像作為額外監督；但我們是 retrieval 不是 forecasting，設計需仔細 |
| ML-Dice 序數 loss 概念 | 參考 | 我們的 RMSE 是回歸指標，但「輕/中/重雨分層監督」的思路可以和 classification bins 結合 |
| Transfer Track 冠軍洞察 | 重要 | TFI 確實提升了跨地理的泛化，這和我們 test 地點全不同的問題高度一致 |

---

## 3. Weather4Cast 2023 RainAI

**來源：** method/weather4cast_RainAI_paper.md | arXiv: 2311.18398
**任務：** 從 1 小時衛星影像（4 frames x 11 bands）預測未來 8 小時降水（32 timesteps，super-resolution）

### 解法核心

#### 2D UNet 優於 3D UNet（核心發現）
把時間 frames 當作 channels 處理（44ch = 4 frames x 11 bands），用 2D UNet：
- 更快、更好、更容易訓練
- 結果：2D UNet CSI=0.0507 vs 官方 3D UNet baseline CSI=0.0444（+14%）

#### Importance Sampling（最大單一改進：+60%）
對高降水量樣本給更高取樣機率，低降水樣本給低但非零機率：

| 實驗 | 說明 | CSI |
|------|------|-----|
| exp1 | ResNet 2D UNet + CE，無 sampling | 0.0306 |
| exp2 | + importance sampling | 0.0491 (+60%) |

**關鍵：只調取樣機率，不改 loss function。** 這和我們 v6 的失敗不同——v6 同時做了 stratified sampling + TieredWeightedLoss，是雙重過度修正。

#### Cross-Entropy Classification Loss
把降水量分成離散 bin，用 CE loss（同 SatFormer 的思路）。
轉換回連續值：sum(bin_center * predicted_probability)。

注意：加入 class weights 的效果不穩定（exp6 比 exp4 更差：0.0451 vs 0.0502），說明 class weighting 需要仔細調整，不是一加就好。

#### Lead Time Conditioning
模型每次只預測一個指定的未來時距，把時距值作為額外 input channel 注入（概念類似我們的 FiLM Day+Hour conditioning）。

#### 靜態地理特徵（Lat/Lon/高程）
把每個 pixel 的緯度、經度、地形高度加進 input channels。

#### 輸出超解析度
UNet 輸出 -> 中心裁剪 42x42 -> 學習型超解析度（NinaSR）到 252x252。
Bilinear 和 NinaSR 結果相近，bilinear 更快。

### 結果 - 完整 ablation table

| 實驗 | 說明 | Core CSI |
|------|------|----------|
| 官方 3D UNet baseline | - | 0.0444 |
| exp1 | 2D UNet + CE，無 sampling | 0.0306 |
| exp2 | + importance sampling | 0.0491 |
| exp4 | + bilinear upsample | 0.0502 |
| exp5 | + lead time conditioning | 0.0477 |
| exp6 | + class weights（效果反降）| 0.0451 |
| exp7 | + EDSR 超解析度 | 0.0482 |
| exp8 | + NinaSR 超解析度（最佳）| 0.0507 |

**失敗案例：** 所有模型對 5mm/hr 以上的極端降水 CSI=0，因訓練資料中此類樣本極稀少。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| 2D UNet 確認 | 已在用 | 我們的架構方向正確 |
| Importance sampling（只調取樣） | 值得重試 | 和 v6 的關鍵區別是：只動 sampling，不動 loss；scan_rain_labels 函數已存在於 train.py |
| CE classification loss | 高（參考 SatFormer） | 可能從根本解決右偏問題 |
| Class weights 效果不穩定 | 注意 | exp6 顯示 class weights 可能反而有害，不要直接加 |
| Lat/Lon 地理特徵 | 中 | 可幫助泛化到 test 新地點；加 2 個 channel，不需要改架構 |
| Lead time conditioning | N/A | 我們不是 multi-step |

---

## 4. Weather4Cast 2025 SaTformer（Cumulative Rainfall Track 第 1 名）

**來源：** method/weather4cast_SatFormer_paper.md | arXiv: 2511.11090
**任務：** 從 1 小時衛星影像預測未來 4 小時累積降水總量（scalar 輸出，非 pixel-wise）

### 解法核心

#### Classification + 64 Bins（最核心設計）
把連續降水值分成 64 個等寬 bin，轉為分類問題：
- 輸入：(T=4, C=11, H=32, W=32) 的衛星影像序列
- 輸出：64 個 bin 的 softmax 機率分布
- 推論時：預測值 = sum(bin_center * probability)（期望值）

**Ablation（bin 數量選擇）：**

| # Bins | CRPS  |
|--------|-------|
| 4      | 14.181 |
| 8      | 5.987  |
| 16     | 4.293  |
| 32     | 3.898  |
| 64     | 3.135  |
| 128    | 3.610  |
| 256    | 5.312  |

64 是最佳點：更多 bins 導致類別過稀疏，產生 degenerate solutions。

#### Class-Weighted CE Loss
```
w_i = -log(|D_i| / |D_total|)  # log-scaled inverse frequency
L = -sum_i w_i * log(y_hat_i) * y_i
```

**Ablation（有無 class weighting）：**

| 設定 | BW-Top-3 | BW-CRPS |
|------|----------|---------|
| 無 class weights | 0.076 | 6.91 |
| 有 class weights | 0.272 | 2.64 |

沒有 class weights 時模型直接 collapse 到只預測「無雨」。

#### Full Space-Time Attention（ST^2）
全部 T*N+1 tokens 彼此互相做 attention（不是分開的空間/時間軸 attention）。

**Ablation（attention 變體）：**

| Attention 設計 | BW-CRPS |
|---------------|---------|
| Space then Time | 4.39 |
| Time then Space | 3.39 |
| Full S+T | 2.64 |

Full 3D attention 可行是因為輸入小（32x32, 4 frames = 257 tokens）。

### 訓練細節（來自 train_categorical.json）

```
optimizer: Adam, lr=1e-5
scheduler: 無（固定 lr）
batch_size: 128
epochs: 200, total_steps: 25000
hardware: 4x A6000 GPUs
loss: Class-weighted categorical CE
norm: min-max [0,1]
```

### 結果
NeurIPS 2025 Weather4Cast Cumulative Rainfall Track 第 1 名，CRPS=3.135。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| Per-pixel classification（64 bins）| 高風險高回報 | 我們需要 41x41 spatial output，每個 pixel 都做 64-bin 分類；output shape (B, 64, 41, 41)，再 argmax 或 expected value 轉 regression |
| Class-weighted CE | 連帶採用 | 如果做 classification，必須加 class weighting，否則 collapse |
| Full Space-Time attention | N/A | 我們的 UNet 架構太不同，換架構成本太高 |
| Fixed lr=1e-5（無 scheduler）| 間接參考 | 確認主流不用 ReduceLROnPlateau；我們的 lr=1e-4 高 10x，可考慮降低 |

---

## 跨方法綜合洞察

### 1. Scheduler：沒有一個競賽頂解用 ReduceLROnPlateau

| 解法 | Scheduler |
|------|-----------|
| SpaceNet8 | Step decay（固定 milestone） |
| UNetTFI | 未明示，W4C 慣例為 cosine |
| RainAI | 未明示（論文未記載訓練細節） |
| SaTformer | 固定 lr=1e-5，無 scheduler |

**結論：ReduceLROnPlateau 是少數派。主流是讓 LR 按固定計劃降，不依賴 val 表現。**

---

### 2. 資料不均衡的三種對策（重要性排序）

| 對策 | 方法 | 效果 | 可用性 |
|------|------|------|--------|
| Importance Sampling（只調取樣） | RainAI | +60% CSI | 高，且已有 scan_rain_labels 函數 |
| Classification bins | RainAI + SaTformer | 最大幅改善 | 中（需改 loss + output head） |
| Class-weighted loss | SaTformer | 顯著但需搭配 classification | 中（不能單獨用在 MSE） |
| Weighted MSE/MAE | 我們的 v4/v6 | 失敗 | 不建議 |

**我們踩過的雷（v6）：同時做 stratified sampling + TieredWeightedLoss，雙重過度修正。
RainAI 的洞察：只動取樣，不動 loss，效果已足夠。**

---

### 3. 兩種解決右偏分布的哲學

| 哲學 | 做法 | 代表 | 對我們的適用性 |
|------|------|------|--------------|
| 保持 regression，調整資料分布 | Importance sampling，讓 batch 有更多有雨樣本 | RainAI | 低風險，可先試 |
| 改成 classification | 64 bins CE loss，最後轉回連續值 | SaTformer | 高風險高回報，需 output head 重設計 |

---

### 4. 空間轉移（Spatial Transfer）的對策

所有四個方法都面臨 train/test 地點不重疊的挑戰：

| 對策 | 來源 | 我們是否能用 |
|------|------|------------|
| TFI（學習通用雲移動物理） | UNetTFI | 間接，設計複雜 |
| 地理 k-fold（確保訓練多樣性）| SpaceNet8 | 可用，但計算成本高 |
| Lat/Lon 地理特徵 | RainAI | 可用，2 個 extra input channels |
| Ensemble（多樣化假設空間）| SpaceNet8 | 最穩定 |
| 更大 backbone（通用特徵）| SpaceNet8 | B5 是明確選項 |

---

### 5. 資料增強：普遍低投入

- SpaceNet8：Mosaicing（4 tiles 拼接）+ 左右翻轉 TTA（inference，非訓練增強）
- RainAI、SaTformer：隨機空間 crop
- 其他方法未明確提及 flip augmentation

**我們目前完全沒有任何資料增強。** 水平/垂直翻轉是最低成本的第一步（物理上合理，且其他降水競賽的慣例）。

---

## 下一步實驗建議（按風險/效益排序）

### 第一優先：低風險，有強理論支撐

1. **換 Scheduler（Cosine Annealing 或 Step Decay）**
   - 所有頂解的一致選擇
   - 解決 ReduceLROnPlateau 把模型卡死的問題
   - 實作：2 行修改

2. **Flip Augmentation（水平 + 垂直翻轉）**
   - 降水場翻轉後物理上仍合理
   - 零成本，15 行 transform 實作
   - 預期改善泛化能力（test 地點全不同）

### 第二優先：中風險，有 ablation 數據支撐

3. **Importance Sampling（只調取樣，不改 loss）**
   - RainAI 最大單一改進（+60% CSI）
   - 關鍵：scan_rain_labels + stratified_sample 已在 train.py 存在
   - 只需移除 v6 的 TieredWeightedLoss，保留 sampling
   - 和 v6 的區別：只動 sampling，不改 loss function

4. **EMA（訓練穩定性）**
   - SpaceNet8 明確驗證對震盪有效
   - 實作：~20 行（torch_ema 或手動）
   - 和其他改動可疊加

5. **Backbone B5 升級**
   - SpaceNet8 明確驗證 B5 > B4
   - 需確認 RTX 4090 24GB 在 batch_size=32 時能跑

### 第三優先：高風險，潛在大幅改善

6. **Per-pixel Classification Loss（64 bins）**
   - 同時被 RainAI 和 SaTformer 採用
   - Output head 改為 (B, 64, H, W)，loss 改為 weighted CE
   - 推論時：per-pixel expected value 轉回 mm/hr
   - 風險：整個 loss + output 設計需要重新驗證

7. **Lat/Lon 地理特徵**
   - RainAI 加入地理靜態特徵（緯度/經度/高程）
   - 對 test 地點完全不同的我們可能有幫助
   - 需要知道每個 TIF 的 rasterio CRS 和 bounding box（dataset.py 已用 rasterio）

---

## 5. GlobalMetNet（Google Research，全球降水即時預報）

**來源：** method/paper_GlobalMetNet.md | arXiv: 2510.13050
**任務：** 7 衛星 x 18 bands + NWP -> 全球降水，0-12h 超前時間，GPM CORRA 作為訓練目標

### 為什麼最直接相關

GlobalMetNet 和我們的任務高度相似：
- 輸入：多顆地球靜止衛星（Himawari、GOES-E/W、Meteosat、Fengyun）—— 我們有其中 3 顆
- 目標：GPM CORRA（Combined Radar Retrieval）—— 和 IMERG 同為 GPM 族群產品，使用相同衛星星座
- 評估範圍：全球，含無雷達覆蓋區域

### 核心設計

**輸出：30 個 categorical bins**
把降水率離散化為 30 個 bin，模型輸出每個 pixel 在 30 個 bin 上的 softmax 機率分布。
推論時轉為期望值（連續輸出）或保留機率分布（不確定性估計）。

**Polyak Averaging（= EMA）**
訓練時維護模型參數的指數移動平均，inference 時使用平均後的參數（而非當下 checkpoint）。
這和 SpaceNet8 的 EMA（momentum=0.002）完全相同概念，GlobalMetNet 從不同任務再次確認。

**靜態地理特徵：** 緯度、經度、地形高度。

**訓練規模：** 256 TPU chips，bfloat16，多年全球數據。

### 最重要 Ablation：哪種輸入最重要？

移除各輸入源後的性能下降排序：
1. **地球靜止衛星影像** -- 移除後性能下降最大（所有區域、所有超前時間）
2. **NWP（ECMWF HRES）** -- 長超前時間（>3h）有顯著幫助；短超前時間影響小
3. **地面雷達** -- 僅在有雷達的區域有益（CONUS/歐洲）；數據稀疏地區無改善

**對我們的直接意義：衛星影像是最重要的輸入，我們的 51ch 設計是正確重心。NWP 沒有就算了。**

### 全球公平性（Global Equity）

GlobalMetNet 在數據稀少的地區（非洲、東南亞、熱帶海洋）和數據豐富的地區（北美、歐洲）表現差距很小，因為它的主輸入（衛星）是全球均勻分布的。NWP 模型（ECMWF）則在數據稀少地區明顯更差。

**對競賽的意義：** 測試集很可能包含熱帶、海洋等多樣地理區域。如果訓練數據集中於某些地區，模型可能在測試集上泛化不佳。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| GPM CORRA 訓練目標 | 確認我們方向 | CORRA 和 IMERG 都是 GPM 族群；GlobalMetNet 的 loss 設計完全適用 |
| 30 bins 分類輸出 | 高 | 和 SatFormer 64 bins 方向相同；從 30/64 都有效可推斷分類優於回歸 |
| Polyak Averaging / EMA | 高 -- 易實作 | 兩個不同任務（SpaceNet8 + GlobalMetNet）都確認有效 |
| 衛星是最重要輸入 | 確認已知 | 我們的 51ch 衛星 input 設計是對的；不需要 NWP |
| 靜態地理特徵（lat/lon）| 中 | 可能幫助測試集多樣地理場景 |

---

## 6. NPM：神經降水模型（衛星唯一，無雷達依賴）

**來源：** method/paper_NPM_SatelliteNowcasting.md | arXiv: 2412.11480
**任務：** 地球靜止衛星（GK2A 韓星，3 個波段）-> 未來 6h 降水，Sat2Rdr 資料集

### 最重要發現：Day+Hour Encoding 是最關鍵組件

NPM 的 ablation 明確顯示，把當前時刻的「年中第幾天（Day）」和「一天中的小時（Hour）」用 sin/cos 編碼後注入模型，是所有組件中**單一最大的性能提升**。

這和我們的 FiLM 時間條件化（v8a，用 Day+Hour sin/cos 控制 encoder features 的 scale/shift）是完全相同的設計。

**NPM 的驗證結果：**
- 移除 Day+Hour encoding 導致最大性能下降
- Day 編碼（季節周期）貢獻最多
- Hour 編碼（日夜周期）添加額外但明顯的改善

**對我們的結論：v8a 的 FiLM time conditioning 必須保留。NPM 從完全獨立的任務和資料集獨立驗證了這一設計選擇。**

### 兩階段架構

Stage 1: 衛星視頻預測（過去衛星 -> 未來衛星影像）
Stage 2: StegoGAN（衛星影像 -> 降水）

**Stage 2 單獨就是我們的任務。**

### 只用 3 個波段：IR 10.5um + WV 6.3um + WV 7.3um

NPM 用非常少的輸入就能達到好結果：
- IR 10.5um（紅外窗口）：雲頂溫度，與深對流相關
- WV 6.3um（水汽通道）：上對流層水汽含量
- WV 7.3um（水汽通道）：中對流層水汽含量

**意義：** 我們 16 個波段中這三個最重要。如果考慮波段消融實驗，優先保留這三個。

### 季節平衡取樣（Season-Aware Sampling）

按季節平衡 batch 組成，確保每個季節在訓練中都有代表。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| Day+Hour sin/cos encoding | 已實作！ | 我們的 v8a FiLM 就是這個，NPM 確認是最重要組件 |
| Stage 2（衛星 -> 降水）| 等同我們的任務 | 直接確認衛星 -> 降水轉換的可行性 |
| 3 波段重要性（IR+WV+WV）| 中 | 波段消融實驗的參考：這 3 個最重要 |
| 季節平衡取樣 | 低優先 | 如果訓練資料時間分布不均才需要 |

---

## 7. TUPANN：物理對齊的衛星降水即時預測

**來源：** method/paper_TUPANN.md | arXiv: 2511.05471
**任務：** GOES-16 ABI 衛星影像序列 -> 10-180 分鐘後的降水場
**評估城市：** Rio de Janeiro、La Paz、Manaus、Miami，零樣本測試 Toronto

### 架構：VED + MaxViT

VED（變分編碼解碼器）學習光流運動，MaxViT Transformer 學習超前時間條件化的強度修正。
光流監督損失：cosine similarity + motion magnitude + KL divergence + intensity reconstruction。

GAN-TUPANN 加鑑別器，讓輸出更銳利，但定量精度略低。

### Event-Based 訓練：對 RMSE 任務有害

**TUPANN 只在降水事件期間訓練（threshold tau = 120,000）。**

原因：CSI 和 HSS 指標忽略「True Negative」（正確預測無雨）。只在有雨事件上訓練可以提升這些指標。

**但對我們的任務，這個策略會造成反效果：**
- 我們的指標是 RMSE，對所有 pixel 都計算誤差（包含 80% 無雨像素）
- 如果模型從不看無雨樣本，它不會學會在無雨地方預測接近 0
- 這會造成大量 false positive（把沒有雨的地方預測成有雨），顯著提高 RMSE
- **結論：不要採用 event-based 訓練。**

### 結果（Table 7, HSS）

TUPANN 在 4 個城市（Rio/Miami/Manaus/La Paz）所有閾值 HSS 中都是第 1 或第 2。
在極端降水（HSS_64）方面優勢最明顯，這正是光流對齊發揮作用的地方。

零樣本 Toronto 測試（Table 8）：TUPANN-Multicity（多城市聯合訓練）優於單城市 TUPANN，跨城市性能退化最多 20%。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| 光流監督（VED）| 不適用 | 時序預測才需要；我們是 retrieval，沒有時序外插 |
| Event-based 訓練 | 有害 | 我們的 RMSE 指標必須在全部 pixel 上訓練，包含無雨區 |
| MaxViT 超前時間條件化 | N/A | 我們沒有超前時間概念；FiLM 已處理當前時間 |
| L1 loss 偏好 | 佐證 | 和我們 0.7*MAE 的比重一致；L1 比 MSE 更適合降水分布 |
| 多地點聯合訓練有益 | 已在做 | 我們的訓練資料本就覆蓋全球多區域 |

---

## 8. GENESIS：IMERG 全球降水即時預報（U-Net + R-ConvLSTM）

**來源：** method/paper_UNetConvLSTM_IMERG.md | arXiv: 2307.10843
**任務：** 過去 6h IMERG 序列 + GFS (U/V/TPW) -> 未來 4h IMERG（8 個時間步）
**機構：** 明尼蘇達大學 + 亞利桑那大學 + NASA Goddard，TensorFlow-V2

### 為什麼重要：使用 IMERG 作為訓練目標

GENESIS 和我們唯一共同的核心要素是訓練目標：IMERG。
論文對 IMERG 的分布特性（零膨脹、右偏、重尾）做了詳細分析，
**這些分析的結論和 loss 設計直接適用於我們的任務。**

### 重要數據分析：IMERG 時間自相關

IMERG 降水場的時間自相關以 alpha ~= 0.33/hr 指數衰減，相關長度 tau ~= 3h。
6h 後相關係數降至 0.1 以下。

**對我們的意義：** 我們輸入的 3 frames（1.5h span）覆蓋了 IMERG 相關長度的重要部分。
更多 frames 在 tau=3h 之內仍有資訊量；超過 6h 後幾乎沒有額外資訊。

### 架構關鍵：R-ConvLSTM Skip Connections

標準 UNet skip connections 把當前時刻的 encoder features 直接傳給 decoder。
GENESIS 把 skip connections 換成 Recursive ConvLSTM（R-ConvLSTM）：
- 每個 encoder level 的 ConvLSTM 遞迴預測未來 8 個時間步的 latent features
- Decoder 在每個解析度 level 都接收遞迴預測的未來 features，而不是當前時刻 features

這防止 skip connections 洩露「當前時刻資訊」到未來預測中。

**對我們的任務：** 這個機制是 forecasting 特有的需求。我們是 retrieval（當前 -> 當前），不需要這種時序遞迴。我們的標準 UNet skip connections 是正確的。

### 最重要發現：MSE vs Focal Loss 交叉點在 1.6 mm/hr

這是對我們直接可用的最重要實驗結果：

**在 IMERG 資料集上的對比（CSI at different thresholds）：**

| 指標 | MSE 優勝條件 | FL 優勝條件 |
|------|------------|-----------|
| CSI_1 (>1 mm/hr) | T+30 至 T+180 min | 相差甚微 |
| CSI_4 (>4 mm/hr) | 幾乎持平 | FL 開始領先 |
| CSI_8 (>8 mm/hr) | MSE 明顯落後 | FL 清楚勝出（T+30: +7%，T+120: +23%，T+240: +72%）|

**結論（從 IMERG 資料集確認）：**
- MSE regression 在 <1.6 mm/hr 的輕雨優勢（因為 bulk of data 在此範圍，MSE 更好擬合）
- Focal loss classification 在 >8 mm/hr 的重雨優勢（FL 的 alpha 懲罰防止重雨類別被壓制）
- 跨越點大約在 1.6-4 mm/hr 之間

**直接應用：** 我們目前的 0.3*MSE + 0.7*MAE 是 MSE/MAE 混合，沒有 classification head。加入 focal loss（10 log-spaced bins，gamma=2）預計在高強度降水上顯著改善。

### Focal Loss 實作（10 類別，對數間距）

```
類別邊界（log scale）: [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 32] mm/hr
< 0.1 -> class 0
> 32  -> class 9（共 10 類別）

FL(y, p) = -(1/N) * sum_{i,c} alpha_c * y_{i,c} * (1 - p_{i,c})^gamma * log(p_{i,c})
gamma = 2
alpha_c = 1/freq_c（inverse frequency weighting，normalize to sum=1）
```

**訓練細節：**
- Adam, lr=1e-3
- LR decay factor 0.1 every 10 epochs when val loss no improvement（Step-like decay）
- Batch size=8
- Xavier initialization
- Hardware: NVIDIA A-100 40GB

### 輸入特徵重要性（MRMR 分析，Fig. 3）

最重要到最不重要：
1. **過去 IMERG 序列**（score 0.64）-- 過去 2h 最重要，6h 內漸減
2. **TPW（大氣可降水量）**（score 0.48）-- 未來 GFS TPW 對長超前時間有用
3. **U/V 風速**（score 0.32）-- 長超前時間增益更大

**對我們的意義：** GENESIS 能用過去 IMERG 作為主要輸入（MRMR 0.64）。我們在測試時沒有過去 IMERG，所以我們的衛星影像必須替代這個高價值輸入。這強調了衛星波段資訊的重要性，尤其是對深對流敏感的 IR/WV 波段。

### 對我們的啟示

| 方法 | 直接可用性 | 說明 |
|------|-----------|------|
| Focal loss (FL)，gamma=2，10 log-bins | 高 -- 直接可用 | 同樣針對 IMERG；MSE vs FL 的 1.6mm 交叉點直接告訴我們什麼時候 FL 更好 |
| 10 log-spaced bins（vs SatFormer 64 uniform bins）| 選擇之一 | Log spacing 更符合 IMERG 的對數尺度分布；64 uniform bins 是替代選項 |
| MSE 在輕雨更好 | 戰略意義 | 考慮 hybrid：分支預測 + 一個 MSE head + 一個 FL head |
| R-ConvLSTM skip connections | 不適用 | Forecasting 特定設計，我們不需要 |
| IMERG 自相關長度 tau=3h | 背景知識 | 3 frames (90min span) 覆蓋大部分有用記憶 |
| Adam lr=1e-3 + step decay | 參考 | 比 ReduceLROnPlateau 更穩定；和 SpaceNet8 Step Decay 一致 |

---

## 跨方法綜合洞察（更新版）

### 1. Scheduler：沒有一個競賽頂解用 ReduceLROnPlateau

| 解法 | Scheduler |
|------|-----------|
| SpaceNet8 | Step decay（固定 milestone：epoch 80/120 降 0.1x）|
| UNetTFI | 未明示，W4C 慣例為 cosine |
| RainAI | 未明示 |
| SaTformer | 固定 lr=1e-5，無 scheduler |
| GlobalMetNet | 未明示（Google 規模訓練，可能 cosine with warmup）|
| GENESIS | ReduceLROnPlateau（patience=10 epochs，factor=0.1）|
| NPM | 未明示 |
| TUPANN | Adam lr=1e-4，未明示 scheduler |

**結論：固定計劃的 Step/Cosine 是主流；GENESIS 雖用 val-loss 觸發的 plateau decay，但 patience=10 epochs 遠長於我們的預設，因此也不容易卡死。我們應至少換成 Step decay（固定 epoch milestone）。**

---

### 2. 資料不均衡的四種對策（重要性排序）

| 對策 | 方法 | 效果 | 可用性 |
|------|------|------|--------|
| Importance Sampling（只調取樣）| RainAI | +60% CSI（最大單一提升）| 高：函數已存在 scan_rain_labels |
| 10 log-spaced bins + Focal Loss | GENESIS | FL 在 >8mm 明顯優勝 | 高：改 output head + loss |
| 64 uniform bins + Class-weighted CE | SaTformer + RainAI | 整體 CRPS -55% | 中高：改 output head + loss |
| 30 bins + 期望值輸出 | GlobalMetNet | 生產級全球系統 | 中：可作為最終目標 |
| Weighted MSE/MAE | 我們的 v4/v6 | 失敗 | 不建議 |

**關鍵區別（v6 失敗的教訓）：**
- v6 同時做 stratified sampling + TieredWeightedLoss = 雙重過度修正
- RainAI 確認：只動 sampling，不動 loss，+60%
- SatFormer 確認：loss weighting（class weights）需要搭配 classification 框架，不是直接加在 MSE 上

---

### 3. 分類 vs 回歸：哪個更好？

兩個從不同角度研究這個問題的論文給出了一致的答案：

| 論文 | 結論 |
|------|------|
| GENESIS（IMERG 目標）| MSE 在 <1.6mm 好，FL（分類）在 >8mm 好 |
| SatFormer（OPERA 雷達目標）| class-weighted CE（CRPS=3.135）比 no-weight collapse（CRPS=6.91）好 55%；無 class weights 時模型退化為只預測無雨 |
| RainAI | CE 分類 + importance sampling > 回歸基線 +60% |
| GlobalMetNet | 30-bin 分類（生產級系統選擇）|

**共識：分類框架在高強度降水上明顯更好。問題是選幾個 bins 和用什麼 bin 邊界。**

兩種主要選項：
- **Log-spaced 10 bins（GENESIS 風格）：** 符合 IMERG 對數尺度分布；較少類別但每類別有更多樣本
- **Uniform 64 bins（SatFormer 風格）：** 更高解析度的機率分布；需要 class weighting 才不 collapse

---

### 4. 時間條件化的確認

多個論文獨立確認了 Day+Hour 時間編碼的重要性：

| 論文 | 時間條件化設計 | 重要性 |
|------|-------------|--------|
| NPM | Day+Hour sin/cos，直接拼接到輸入 | 最重要單一組件 |
| GlobalMetNet | Lead-time conditioning | 核心設計 |
| RainAI | Lead-time as uniform channel | 有效（exp5） |
| 我們（v8a）| FiLM Day+Hour -> scale/shift encoder features | 已實作，有效 |

**結論：我們的 FiLM v8a 設計是正確的，有獨立驗證。**

---

### 5. EMA 的確認

| 論文 | EMA 實作 | 確認效果 |
|------|---------|---------|
| SpaceNet8 | EMA，momentum=0.002，inference 用 EMA weights | val metric 震盪 -> 穩定 |
| GlobalMetNet | Polyak Averaging（= EMA），inference 用 Polyak weights | Google 生產級系統選擇 |

**結論：EMA 是低風險、有雙重確認的改進，應該加入我們的訓練流程。**

---

### 6. 空間轉移（Spatial Transfer）的對策

| 對策 | 來源 | 我們能用 |
|------|------|---------|
| TFI（學習通用雲移動物理）| UNetTFI | 設計複雜；間接適用 |
| 地理 k-fold | SpaceNet8 | 計算成本高 |
| Lat/Lon 地理特徵 | RainAI + GlobalMetNet | 可用，2 extra channels |
| Day+Hour encoding | NPM + 我們 v8a | 已實作（季節/日夜）|
| 多城市聯合訓練 | TUPANN | 我們已有全球數據 |
| Global equity 原則 | GlobalMetNet | 確認衛星輸入已足夠 |

---

## 下一步實驗建議（更新版，按風險/效益排序）

### 第一優先：低風險，有多方確認

1. **換 Scheduler（Step Decay）**
   - SpaceNet8 + GENESIS 都用 step decay
   - 改 ReduceLROnPlateau -> 固定 epoch 降 LR
   - 實作：2 行修改

2. **EMA（訓練穩定性）**
   - SpaceNet8（val 震盪）+ GlobalMetNet（Polyak）雙重確認
   - 實作：~20 行
   - 可和其他改動疊加

3. **Flip Augmentation**
   - 降水場翻轉後物理合理
   - 零成本，預期改善泛化

### 第二優先：中風險，有 ablation 數據支撐

4. **Importance Sampling（只調取樣，不改 loss）**
   - RainAI 最大單一改進（+60%）
   - scan_rain_labels + stratified_sample 已在 train.py 存在
   - 關鍵：只動 sampling，不改 loss（區別於 v6 失敗的雙重修正）

5. **Focal Loss（10 log-spaced bins，gamma=2）**
   - GENESIS 直接在 IMERG 上驗證：>8mm 明顯優勝
   - 實作：新 output head (B, 10, H, W) + FL loss
   - 先和 MSE 對比；然後考慮 hybrid

6. **Backbone B5 升級**
   - SpaceNet8 確認 B5 > B4
   - 需確認 24GB 在 batch=32 能跑

### 第三優先：高風險，潛在大幅改善

7. **Per-pixel Classification（64 uniform bins，SatFormer 風格）**
   - 需同時實作 class-weighted CE，否則 collapse
   - Output shape: (B, 64, 41, 41) -> argmax 或 expected value
   - 和 option 5（10 log-bins）對比後選一個方向

8. **Lat/Lon 地理特徵**
   - RainAI + GlobalMetNet 都加入
   - 需要每個訓練樣本的 bounding box（dataset.py 已用 rasterio，可提取）

---
   
## 附錄：各方法超參數對照（更新版）

| 超參數 | SpaceNet8 B5 | RainAI | SaTformer | GlobalMetNet | GENESIS | TUPANN | 我們（v8a）|
|--------|-------------|--------|-----------|--------------|---------|--------|-----------|
| Optimizer | Adam | Adam | Adam | 未明示 | Adam | Adam | AdamW |
| Base LR | 2e-4 | 未明示 | 1e-5 | 未明示 | 1e-3 | 1e-4 | 1e-4 |
| Scheduler | Step milestone | 未明示 | Fixed（無）| 未明示 | Step（10ep no improv）| 未明示 | ReduceLROnPlateau |
| Batch size | 8 | 未明示 | 128 | 未明示（TPU 256 chips）| 8 | 8 | 64 |
| Loss | Dice + BCE | CE（class bins）| Weighted CE（64 bins）| 30-bin CE | MSE 或 FL（10 log-bins）| L1 + motion | 0.3*MSE + 0.7*MAE |
| Target | 洪水 mask | OPERA 雷達 | OPERA 雷達（累積）| GPM CORRA | IMERG Early Run | GOES RRQPE / IMERG | GPM-IMERG |
| EMA | yes（0.002）| 未明示 | 未明示 | yes（Polyak）| 未明示 | 未明示 | 無 |
| Augmentation | Random crop + mosaic | 未明示 | Random crop | 未明示 | 未明示 | 未明示 | 無 |
| Input size | 448x448 | 128x128 | 32x32 | 0.05 deg 全球 | 256x256 | 全圖 | 128x128 |
