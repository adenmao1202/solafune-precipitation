# 概念學習路線圖
了解 SOTA 降水臨近預報模型的學習順序。
請按順序閱讀，每個 Level 都建立在前一個的基礎上。

---

## Level 1 - 基礎（所有內容的前提）

### UNet + Encoder-Decoder
**是什麼：** 
- 影像對影像的架構。Encoder 對輸入做下採樣（downsampling）來提取特徵，Decoder 做上採樣（upsampling）將輸出還原到原始解析度。
- Skip connection 將 Encoder 各層的高解析度空間細節直接傳遞給對應的 Decoder 層。

**為什麼重要：** 四篇論文全部建立在 UNet 或 Encoder-Decoder 結構上，你的 pipeline 也是。

### Depthwise Separable Convolution (DSC)
**是什麼：** 把標準 3x3 Conv 拆成兩步：
(1) Depthwise conv（每個 channel 獨立做卷積）+ 
(2) Pointwise 1x1 conv（混合 channel）。感受野相同，但參數量減少約 8 倍。

**為什麼重要：** EfficientNet（你目前的 backbone）大量使用 DSC。SmaAt-UNet 用 DSC 把模型縮小到原本的 1/4。
**搜尋：** "depthwise separable convolution MobileNet Chollet"

### Batch Normalization + Residual Connection
**是什麼：** Batch Normalization 對每層的 activation 做標準化，穩定訓練過程。Residual connection 把輸入直接加到輸出（x + F(x)），讓非常深的網路也能訓練。
**為什麼重要：** 所有 backbone 都有這兩個機制，EfficientNet 也不例外。
**搜尋：** "ResNet residual connections He 2016"

---

## Level 2 - 時序建模（理解如何處理多個時間幀）

### ConvLSTM
**是什麼：** 把 LSTM 中的矩陣乘法換成卷積運算。可以處理一連串的 2D feature map，並在時間步之間傳遞具有空間結構的 hidden state。
**為什麼重要：** GENESIS 論文（UNet+ConvLSTM）把 UNet 的 skip connection 換成 ConvLSTM，讓 skip path 帶有時序記憶。標準的 channel concat 把 3 幀直接疊在一起，ConvLSTM 則保留幀與幀之間的順序關係。
**與 LSTM 的差異：** 輸入和 hidden state 都是 2D feature map，而不是向量。
**搜尋：** "ConvLSTM precipitation nowcasting Shi 2015"

### Optical Flow（光流）
**是什麼：** 估算兩個連續幀之間每個像素的移動方向和速度（velocity field）的演算法。Lucas-Kanade（LK）用局部最小二乘法求解；DARTS 在 Fourier 空間求解（更適合氣象資料）。
**為什麼重要：** TUPANN 用 DARTS optical flow 算出的速度場作為 ground truth，直接監督模型的 motion field 輸出，強迫模型學到物理上合理的雲移動，而不只是統計上的像素對應。
**核心洞見：** 模型要學的不是「下一幀長什麼樣」，而是「每朵雲往哪個方向移動」。
**搜尋：** "optical flow Lucas-Kanade", "DARTS spectral optical flow radar"

### Video Prediction（影像序列預測）
**是什麼：** 給定過去 n 幀 [t-n, ..., t]，預測未來 k 幀 [t+1, ..., t+k]。屬於 spatiotemporal sequence modeling 的問題。
**為什麼重要：** NPM 的第一階段就是 video prediction model（用過去衛星影像預測未來衛星影像）。SimVP、PredRNN、SwinLSTM 是常見的 baseline。
**搜尋：** "SimVP video prediction", "spatiotemporal predictive learning"

---

## Level 3 - Attention 機制

### Channel Attention + Spatial Attention (CBAM)
**是什麼：** Convolutional Block Attention Module，包含兩個依序執行的注意力模組：(1) Channel attention — 對 feature map 做 global average/max pooling 後過 MLP，學習哪些 channel 最重要；(2) Spatial attention — 對 channel 方向做 pooling 後過 conv，學習哪些空間位置最重要。輸出 = 輸入 × attention map。
**為什麼重要：** SmaAt-UNet 在每個 Encoder block 後加 CBAM，幾乎不增加參數，但讓模型自動聚焦在與降水相關的波段和區域。
**搜尋：** "CBAM convolutional block attention Woo 2018"

### Self-Attention / Transformer 基礎
**是什麼：** Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V。每個位置可以注意到所有其他位置，捕捉卷積做不到的長距離依賴關係（long-range dependency）。
**為什麼重要：** 理解 Earthformer、MaxViT 的前提，也是理解為什麼 Transformer 在全球天氣預測上能贏過 CNN 的原因。
**搜尋：** "Attention is all you need Vaswani 2017"

### MaxViT
**是什麼：** Multi-axis Vision Transformer。結合 (1) Local window attention（只在小視窗內做 attention，高效，捕捉局部結構）+ (2) Global grid attention（在稀疏的網格點上做 attention，捕捉長距離關係）。避免了 full self-attention 的 O(n^2) 計算量。
**為什麼重要：** TUPANN 用 MaxViT 在 latent space 中跨 lead time 做時序推演，同時捕捉局部降水胞（local precipitation cells）和大尺度天氣系統。
**搜尋：** "MaxViT multi-axis vision transformer Tu 2022"

### Large Kernel Attention (LKA)
**是什麼：** 把大 kernel 卷積（例如 21x21）分解成三步：Depth-wise conv + Depth-wise dilated conv + Pointwise conv。用較低的計算成本達到大感受野。
**為什麼重要：** NPM 的 ST-Block 使用 LKA 處理空間維度，並將其延伸為 TKA（Temporal LKA）處理時間維度。Ablation study 顯示比標準卷積有穩定提升。
**搜尋：** "Large Kernel Attention VAN visual attention network"

---

## Level 4 - 物理先驗（Physics-Informed Methods）

### Advection-Diffusion Equation（平流擴散方程）
**是什麼：** 描述某個量（例如降水強度）在對流（advection）和擴散（diffusion）下如何隨時間演化的偏微分方程（PDE）：
```
du/dt = div(D * grad(u) - v * u) + R
```
其中 v = 速度場、D = 擴散係數、R = 源項。
**為什麼重要：** PIANO 把這個方程的殘差加進 loss function（PINN loss），強迫模型預測的速度場符合物理定律。
**核心洞見：** 雨胞隨風移動（advection）並向外擴散（diffusion），違反這個規律的預測在物理上是不合理的。
**搜尋：** "advection diffusion equation", "PINN physics-informed neural network Raissi 2019"

### PINN Loss（Physics-Informed Loss）
**是什麼：** 把 PDE 殘差當成額外的 loss term 加入訓練：
```
L_total = L_data + alpha * L_PDE
L_PDE = ||du/dt - div(D * grad(u) - v * u) - R||^2
```
不需要改變模型架構，純粹在 training 時加入物理約束。
**為什麼重要：** PIANO 顯示加入 PINN loss 可以降低預測的季節性變異（更穩健的泛化能力）。這是一種低風險的改進方式——最差情況下設 alpha=0 等同於沒加。
**搜尋：** "physics-informed neural networks", "PDE loss regularization"

### Variational Autoencoder (VAE)
**是什麼：** Encoder 把輸入映射到 latent space 中的分布（輸出 mu 和 sigma），Decoder 從採樣的 z 重建輸出。KL divergence loss 確保 latent space 結構規整（不會過於分散）。
**為什麼重要：** TUPANN 的 VED（Variational Encoder-Decoder）用 VAE 結構同時學習 motion field 和 intensity correction field。KL term 確保 latent space 平滑，方便 MaxViT 在其中做時序推演。
**搜尋：** "VAE variational autoencoder Kingma 2014"

---

## Level 5 - 生成模型（Generative Models）

### GAN + Pix2Pix
**是什麼：** Generator G 把輸入 X 映射到輸出 Y；Discriminator D 嘗試區分真實的 Y 和 G(X)。訓練交替進行：G 想騙過 D，D 想不被騙。
Pix2Pix 是 conditional GAN 的影像對影像翻譯版本，Generator 用 UNet，Discriminator 用 PatchGAN（對局部 patch 做分類）。
**為什麼重要：** NPM 第二階段和 PIANO 都用 Pix2Pix 把衛星影像轉換成雷達/降水圖。GAN loss 產生比純 MSE 更銳利（sharper）的輸出。
**GAN 的已知問題：** Mode collapse（生成多樣性不足）、訓練不穩定。
**搜尋：** "Pix2Pix image-to-image translation Isola 2017"

### Diffusion Model（擴散模型，背景知識）
**是什麼：** 學習逐步去除噪聲的過程（reverse diffusion）。推論時從純噪聲開始，迭代去噪產生輸出。訓練比 GAN 穩定，但推論速度慢（需要數百步）。
**為什麼重要：** CasCast 和 PreDiff（在 TUPANN related work 中提到）用 Diffusion model 做降水預測，是目前 SOTA 之一。本次競賽不建議實作，主因是推論成本太高。
**搜尋：** "latent diffusion model", "PreDiff precipitation nowcasting"

---

## Level 6 - 訓練技巧（Training Techniques）

### FiLM (Feature-wise Linear Modulation)
**是什麼：** 把條件訊號 c 注入 feature map x 的方式：
```
output = gamma(c) * x + beta(c)
```
其中 gamma 和 beta 都是 c 的線性映射。同時做乘法和加法調整，比直接把 c 加到 feature map 更有表達能力。
**為什麼重要：** 用來把 Day/Hour embedding 注入 UNet bottleneck（NPM 的做法）。Global MetNet 也用類似機制做 lead-time conditioning。
**搜尋：** "FiLM feature-wise linear modulation Perez 2018"

### Positional Encoding（位置編碼）
**是什麼：** 把純量位置 p 編碼成向量，用不同頻率的 sin/cos 組合：
```
PE(p) = [sin(p/T^0), cos(p/T^0), sin(p/T^(2/d)), cos(p/T^(2/d)), ...]
```
對週期性資料特別有效。Day-of-year 的週期是 365，Hour-of-day 的週期是 24。
**為什麼重要：** NPM 的 ablation study 顯示加入 day-of-year encoding 是單項最大改進（+17% CSI）。你的 `train_dataset.csv` 有 `datetime` 欄位，可以直接使用。
**搜尋：** "sinusoidal positional encoding", "Fourier features for time series"

### EMA (Exponential Moving Average) / Polyak Averaging
**是什麼：** 維護一份模型權重的指數移動平均作為 shadow copy：
```
ema_weights = decay * ema_weights + (1 - decay) * current_weights  (decay 通常設 0.999)
```
Eval 和 Inference 都用 ema_weights，不用 current_weights。
**為什麼重要：** Global MetNet 使用 Polyak averaging。可以平滑訓練過程中的權重震盪，通常可以再多降 0.5-1% RMSE，且幾乎沒有實作成本。
**搜尋：** "exponential moving average model weights", "Polyak averaging deep learning"


### Loss Functions for Precipitation（降水預測的 Loss 函數）

| Loss | 特性 | 適用情境 |
|------|------|---------|
| MSE | 懲罰誤差平方，對大值敏感。產生模糊（blurry）預測，因為模型傾向預測所有可能結果的平均 | 基本 baseline，log1p 空間下合理 |
| MAE | 懲罰絕對誤差，對 outlier 更穩健。比 MSE 稍微不那麼模糊 | 可以搭配 MSE 一起用 |
| Weighted MSE | 對有降水的 pixel 加大 weight，修正 80% 零值主導 loss 的問題 | **你應該優先實作的改進** |
| Focal Loss | 降低容易預測樣本的 weight，聚焦在難以預測的樣本。GENESIS 用在高強度降水的分類 head | 分類任務，RMSE 競賽不直接適用 |
| CRPS | 評估機率預測分布的 proper scoring rule | 當模型輸出分布而非單一數值時使用 |
| Log1p transform | 壓縮降水量的動態範圍。log1p(0)=0，保留零值的意義 | **你已經在用，所有論文都驗證這是正確的** |

---

## 論文閱讀順序（按難度由低到高）

**1. GENESIS — UNetConvLSTM_IMERG_2307.10843.pdf**
從這篇開始。架構和你最接近，都是 UNet-based。Ablation study 清楚展示不同 loss function 和 ConvLSTM skip connection 的效果。
需要：Level 1 + Level 2（ConvLSTM）

**2. NPM — NPM_SatelliteNowcasting_2412.11480.pdf**
重點看 Ablation Table（day embedding、season-aware sampling 的貢獻）和程式碼結構。有公開程式碼可以直接參考。
需要：Level 1-3

**3. TUPANN — Tupann.pdf**
重點看 motion field supervision 的設計和 VED 架構。了解為什麼 physically-aligned 比純 data-driven 更好。
需要：Level 1-4（VAE + Optical flow）

**4. Global MetNet — GlobalMetNet_2510.13050.pdf**
最後讀。重點看衛星 mosaic 的輸入設計（Himawari+GOES+Meteosat 的處理方式）和 Input Ablation 結果，這兩部分和你的多衛星輸入最直接相關。
需要：Level 1-6
