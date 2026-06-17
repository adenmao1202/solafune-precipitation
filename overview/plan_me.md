# 你的學習與研究計畫

這份文件規劃你在競賽期間應該讀什麼、問什麼、以什麼順序做。
原則：**等訓練跑的時候讀，不要為了讀完才開始動手。**

---

## 第一批（上傳等待期間，現在就可以開始）

### 必讀：concept.md Level 1-2
- UNet / Encoder-Decoder / Skip connection
- ConvLSTM（因為 Phase 4 可能要用到）
- 預計時間：1-2 小時

### 必讀：GENESIS 論文
- 路徑：`paper/precipitation/UNetConvLSTM_IMERG_2307.10843.pdf`
- 重點只看：Abstract / Fig 4（ConvLSTM skip connection 架構圖）/ Table 2（ablation）
- 目的：理解 ConvLSTM skip connection 長什麼樣，評估你要不要做 Phase 4
- 預計時間：40 分鐘

### 要能回答的問題
1. UNet 的 skip connection 是從哪裡連到哪裡？為什麼要有它？
2. ConvLSTM 和普通 LSTM 差在哪裡？
3. 為什麼 channel concat 3 幀和 ConvLSTM 處理 3 幀的效果會不一樣？

---

## 第二批（有 baseline LB 分數之後）

### 必讀：NPM 論文
- 路徑：`paper/precipitation/NPM_SatelliteNowcasting_2412.11480.pdf`
- 重點只看：Section 4 Method（Stage 1 架構）/ Table 4（ablation，特別是 Day Embedding 那行）/ 公開程式碼
- 目的：理解 Day/Hour embedding 怎麼實作，直接抄 ablation 結論
- 預計時間：1 小時

### 必讀：concept.md Level 6（FiLM + Positional Encoding）
- 這是 P3-1 Day embedding 的實作前置知識
- 預計時間：30 分鐘

### 要能回答的問題
1. Sin/cos positional encoding 為什麼比直接把數字輸入模型好？
2. FiLM 的 gamma * x + beta 和直接把 embedding 加到 x 有什麼差別？
3. Day embedding 讓模型「知道現在是幾月」——這對降水預測有什麼物理意義？
4. 你的資料裡有幾個地區？這些地區的雨季分布一樣嗎？（提示：亞洲 / 美洲 / 歐洲 / 非洲的季風不同）

---

## 第三批（P3 做完，準備 Phase 4）

### 選讀：TUPANN 論文
- 路徑：`paper/precipitation/Tupann.pdf`
- 重點只看：Section 4.1（VED）/ Section 4.1.2（Optical flow）/ Fig 5-6（motion field 視覺化）/ Table 1（結果）
- 目的：理解 motion field supervision 的概念，評估值不值得在最後一週試
- 預計時間：1.5 小時

### 選讀：concept.md Level 3-4
- Attention 機制（如果你想加 CBAM）
- Optical Flow（理解 TUPANN 的核心）
- 預計時間：1 小時

### 要能回答的問題
1. TUPANN 為什麼要把 motion field 和 intensity correction 分開學？
2. Optical flow supervision 和純 data-driven 學 motion 的差別是什麼？
3. 如果要在你的現有 UNet 上加 optical flow supervision，最簡單的切入點在哪裡？

---

## 第四批（最後一週，Ensemble 之前）

### 選讀：Global MetNet 論文
- 路徑：`paper/precipitation/GlobalMetNet_2510.13050.pdf`
- 重點只看：Section 4.1（Input features，特別是 satellite mosaic 的處理）/ Table 3-4（ablation，看哪種 input 最重要）/ Section 5（結果）
- 目的：確認你的多衛星輸入設計沒有明顯遺漏
- 預計時間：1.5 小時

### 要能回答的問題
1. Global MetNet 的 ablation 顯示哪種輸入移除後 CSI 下降最多？
2. 他們怎麼把不同衛星的影像合成（mosaic）？你的做法和他們有什麼不同？
3. 他們用了 NWP 資料（ERA5），而你不能用——這個缺口有沒有辦法用其他方式補？

---

## 競賽期間持續追蹤

### Discussion Page
定期看 Solafune 競賽的 Discussion，尤其是：
- 其他參賽者發現的資料異常
- Public LB 高分的方向提示
- 任何 data leakage 或資料格式問題的討論
把有用的資訊貼給我，我會幫你判斷要不要跟進。

### 每次實驗後記錄（照 coop.md 的格式）
```
實驗名稱: exp0X_xxx
改了什麼: ...
訓練 log:
  Epoch 001 | train_loss=... | val_RMSE=...
  Epoch 010 | train_loss=... | val_RMSE=...
  Epoch 030 | train_loss=... | val_RMSE=...
LB 分數: ...
你的觀察: ...
```

---

## 不需要讀的東西（節省時間）

- MaxViT 論文：TUPANN 用了，但你短期內不會實作
- Diffusion model 相關論文：推論太慢，競賽不適用
- GAN / Pix2Pix 細節：NPM 的兩階段架構太複雜，一個月內不值得全做
- ERA5 / NWP 相關資料：比賽規則禁止使用 external dataset
