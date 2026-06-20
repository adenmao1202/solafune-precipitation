# v10_focal 失敗分析：LB 1.95（史上最差）

日期：2026-06-20
實驗：v10_focal_b32
結果：LB RMSE = 1.95（vs 最佳 v8a = 0.6979）

---

## 核心問題：Expected Value 重建在重尾分布下爆炸

這是最主要的失敗原因，其他問題都在放大它。

Focal Loss 框架在推論時用「加權期望值」重建降雨量：

```
pred_mm = sum(softmax(logits)[i] * bin_center[i])
```

Bin centers（最後兩格）：..., 19.2, ~61.0 mm/hr

只要模型對高雨 bin（center = 19.2 或 61.0）給出 5-10% 的機率，
每個像素的預測值就被往上推 1-6 mm/hr，
即使是乾燥像素（真實值 = 0）也一樣。
只要全圖系統性高估，RMSE 就會從 0.7x 跳到 1.x 乃至 1.9x。

---

## 問題一：alpha[0] 約等於 0 = 讓 80% 的像素消失

```
alpha[0] = 0.0003（零雨 bin，佔 ~80% 像素）
alpha[9] = 0.7608（重雨 bin，極少數像素）
```

Focal Loss 公式：`loss = -alpha * focal_weight * log(p)`

對零雨像素，alpha[0] 極小，導致：
- 訓練時零雨像素的梯度訊號接近零
- 模型對零雨像素「學不到該預測 bin 0」
- 模型自由地在零雨像素預測任何 bin
- 結合 Expected Value 重建，零雨區域輸出大量非零預測值

這與直覺相反：我們希望 alpha 讓模型關注稀有的雨，卻讓模型完全忘記了多數的無雨。

**根本矛盾**：inverse-frequency alpha 是為了讓稀有類別不被忽略。
但零雨 bin（80% 頻率）反而得到最小 alpha，完全失去訓練訊號。
在 RMSE 任務中，零雨像素的預測誤差是 RMSE 的最大貢獻者，
alpha 設計方向完全錯誤。

---

## 問題二：Focal Loss 是偵測工具，不是回歸工具

Focal Loss 在 RetinaNet（物件偵測）中設計用途：
- 前景 vs 背景（2 類）
- 最終評估指標是 AP，而非 RMSE

本任務的目標：
- 預測連續降雨量（mm/hr）
- 評估指標是 RMSE

訓練時最小化 Focal Cross-Entropy，
推論時評估 RMSE（透過 E[X] 近似），
這兩個目標從來沒有直接對齊。
模型在訓練時學的是「哪個 bin 機率最高」，
不是「讓 E[X] 接近真實降雨量」。

---

## 問題三：訓練初期 val_RMSE 已是警告訊號

從記憶中的早期 epoch 記錄：

```
Epoch 1: val_RMSE = 4.54
Epoch 2: val_RMSE = 3.51
Epoch 3: val_RMSE = 3.85（震盪）
Epoch 5: val_RMSE = 2.85（最佳）
```

v8a epoch 12 的最佳 val_RMSE = 1.1441。

epoch 5 時 v10_focal 已經是 v8a 的 2.5 倍差，
這是「模型在往錯誤方向學習」的明確訊號，
應在 epoch 10 左右就中止實驗，不應等到完整訓練後才 submit。

---

## 問題四：Log-spaced Bins 的期望值計算失真

Bin centers：0.0, 0.15, 0.3, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, ~61.0

最後一個 bin 的 center 約 (25.6 + max_val) / 2 = (25.6 + 96.51) / 2 = 61.05 mm/hr

這個 bin 代表「極端暴雨」，center 高達 61 mm/hr。
若模型對它分配 3% 機率，全圖均值就被抬高 ~1.83 mm/hr。

有限數量的 log-spaced bins 無法準確表示重尾分布，
E[X] 對最後一個 bin 的 center 選擇極度敏感。

---

## 問題五：OneCycleLR + Focal Loss 不相容

v10 使用 OneCycleLR（max_lr = lr * 10）。
OneCycleLR 在前 30% epoch 快速升高 LR，
這在 Focal Loss 框架下會讓模型在分布已偏斜時「學習過快」，
在錯誤方向上 over-shoot。

v8a 使用 ReduceLROnPlateau，比較保守，遇到壞的方向時會降速。

---

## 結論：為什麼 Focal Loss 在這裡必然失敗

1. **設計目的不符**：Focal Loss 針對 AP/mAP，不針對 RMSE
2. **E[X] 重建脆弱**：log-spaced bins 的 E[X] 對高雨 bin 極敏感
3. **alpha 設計反效果**：讓模型忽略 80% 的無雨像素
4. **訓練/評估指標不對齊**：訓練 cross-entropy，評估 RMSE，模型從未被直接監督 RMSE

---

## 往後建議方向

**不要再嘗試純 Focal Classification 框架**。

如果要解決「零雨像素主導梯度」問題，考慮：

1. **Two-stage：先偵測有雨/無雨，再回歸降雨量**
   - Stage 1：Binary classifier（有雨 vs 無雨），這裡可以用 BCE + Focal weight
   - Stage 2：只對有雨像素做 regression（MSE/MAE）
   - 推論：mask * regression_output

2. **Huber Loss（delta=1.0）**：比 MSE/MAE 對 outlier 更穩健，不需要 bin 分類

3. **Asymmetric Loss**：對非零像素增加 weight，但用連續 weight，不做 discretization
   `loss = w(target) * MSE`，`w = 1 + k * target`

4. **保持 v8a 架構**，針對其他方向改進（TTA、ensemble、後處理）

---

## 教訓一句話

> Focal Loss 解決的是「前景/背景不平衡的偵測問題」，
> 不是「連續值回歸中的零值主導問題」。
> 把分類工具硬套回歸任務，E[X] 成為隱藏的爆炸點。
