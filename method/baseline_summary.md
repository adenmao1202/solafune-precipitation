# Solafune BL2 Baseline 分析（LB 0.708）

來源：競賽討論區分享，public LB 0.708

---

## 一、任務重新確認

- 輸入：過去 30 分鐘最多 3 張衛星影像，每張 16 個 spectral band
- 輸出：41x41 GPM-IMERG 降水圖（mm/hr），預測未來 30 分鐘
- **83% 的 target pixel 是精確的 0**，分布是極度零膨脹的右偏分布
- p99 = 5.4 mm/hr，max = 77.6 mm/hr（訓練集統計）
- RMSE 幾乎完全由稀有的重雨事件決定

---

## 二、Band 選擇的重要性

### Spearman 相關係數（1500 個訓練樣本）

| Band | rho with GPM | 物理意義 |
|------|-------------|--------|
| IR window (10.4um) | -0.293 | 雲頂溫度，最強訊號 |
| Mid-IR (3.9um) | -0.289 | 冰晶大小，深對流代理 |
| IR split (12.3um) | -0.287 | 分裂窗雲光學深度 |
| WV upper (6.2um) | -0.221 | 上對流層水氣 |
| Visible red (0.64um) | +0.129 | 白天雲反射，夜晚近零 |

**關鍵洞察：IR band 比 visible band 的相關係數高 2-3 倍。Visible band 在夜間幾乎無用。**

### Band 組合實驗結果（相同 U-Net 1.9M params，holdout val RMSE）

| Band 組合 | Bands | Val RMSE | vs. visible |
|----------|-------|---------|------------|
| Visible（官方 baseline） | B1, B2, B3 | 1.2857 | 基準 |
| IR classic | 10.4um, 11.2um, 12.3um | 0.8859 | -0.399 |
| WV moisture | 6.2um, 7.3um, 10.4um | 0.8825 | -0.403 |
| Ice proxy | 3.9um, 6.2um, 10.4um | 0.8809 | -0.405 |
| **Split-window（最佳）** | ir_window(10.4um), ir_split(12.3um), wv_upper(6.2um) | **0.8724** | **-0.413** |

**結論：Band 選擇的影響（1.286 vs 0.872，32% 相對提升）大於模型架構改動的影響。**

---

## 三、Split-Window（BTD）的正確用法

他們的 BTD 用法和我們（v9c）完全不同：

| | 我們的 v9c | BL2 的做法 |
|--|----------|----------|
| 用法 | 在 51ch 上再加 3ch BTD | 把 BTD 當成 3 個核心 band 之一 |
| 輸入 | 54ch（51 + BTD） | 9ch（3 bands x 3 frames，其中1個是BTD） |
| 結果 | val RMSE 從 1.14 退步到 1.28 | val RMSE 0.8724（最佳） |

**洞察：BTD 作為核心輸入特徵有效，但疊加在已有 51ch 上是冗餘且有害的。**

物理原因：BT10.4 - BT12.3 差值在冰晶存在時增加，是深對流的指標。但在 raw DN 空間（非亮溫）的 BTD 物理意義不成立，且我們的 stats.json 在 Meteosat band swap 前計算，進一步放大誤差。

---

## 四、模型架構

### BL2 的架構（1.9M params，LB 0.708）

```
Input (9, 41, 41)
  -> Encoder: Conv 9->32->64->128
  -> Bottleneck: Conv 128->256
  -> Decoder: Transposed conv + skip 256->128->64->32
  -> Head: Conv 32->1
Output (1, 41, 41)
```

- 從零訓練（無 pretrained）
- 輸入直接 resize 到 **41x41**（和 GPM target 相同解析度）
- 沒有 FiLM、沒有 EMA、沒有特殊技巧
- 只用 3 個 IR band x 3 frames = 9 channels

### 對比我們的架構

| 項目 | BL2 | 我們（v8a） |
|------|-----|----------|
| Encoder | Scratch Conv | EfficientNet-B4（ImageNet pretrained） |
| Params | 1.9M | ~19M |
| Input size | 41x41 | 128x128 |
| Input channels | 9（3 IR bands） | 51（16 bands x 3） |
| LB RMSE | 0.708 | 0.698 |

**洞察：更大的模型（我們）只比更小的模型（BL2）好 0.01 RMSE。模型大小不是瓶頸。**

---

## 五、Loss Function

### BL2 的做法

```python
loss = MSE(model(x), log1p(y))
pred_mm_hr = expm1(model(x)).clamp(0)
```

純 MSE 在 log1p 空間。

### 我們的做法

```python
loss = 0.7 * MSE(pred, log1p(y)) + 0.3 * MAE(pred, log1p(y))
```

**兩者都在 log1p 空間，差別只是我們多加了 MAE。**

### 為什麼 log1p 有效

Log1p 把梯度從極端值重新分配到所有降水強度：
- 純 MSE（raw mm/hr）：50 mm/hr 的誤差比 0.05 mm/hr 貢獻 2500x 更多梯度，導致訓練不穩定
- Log1p MSE：梯度均勻分佈在所有 bin，訓練訊號更穩定，但仍用 raw mm/hr 評估

---

## 六、Validation 策略的差異（重要）

### BL2 的做法：Location-grouped holdout

固定 4 個地點作為 val set（florida, france, jakarta, kinshasa），其餘 16 個地點訓練。

Val RMSE / LB RMSE 比值：0.87 / 0.708 = **1.23 倍**

### 我們的做法：Temporal split（按時間）

每個地點最後 20% 的時間點作為 val。

Val RMSE / LB RMSE 比值：1.14 / 0.698 = **1.63 倍**

**洞察：我們的 val/LB 差距比 BL2 大得多（1.63x vs 1.23x）。**

原因：訓練集和測試集是**地點不重疊**的。用 temporal split 時，train 和 val 都在同一個地點，讓模型記住地點的氣候特性，泛化性估計過於樂觀。Location-grouped holdout 才是誠實的估計。

---

## 七、誤差分析（BL2 split-window 模型）

| 降水強度（mm/hr） | pixel 佔比 | RMSE | Bias |
|----------------|----------|------|------|
| 0 - 0.1 | 88.1% | 0.230 | +0.076（過預測） |
| 0.1 - 0.5 | 4.7% | 0.630 | +0.159 |
| 0.5 - 1.0 | 2.3% | 0.814 | -0.096 |
| 1.0 - 2.0 | 2.1% | 1.162 | -0.571 |
| 2.0 - 5.0 | 2.0% | 2.322 | -1.754 |
| **>5.0** | **0.9%** | **7.968** | **-6.150（嚴重低估）** |

**關鍵洞察：**
- 輕雨（0-0.5 mm/hr）：模型**過預測**（到處預測薄薄的毛毛雨，不是乾淨的零/非零邊界）
- 重雨（>1 mm/hr）：模型**系統性低估**，回歸到均值
- 0.9% 的 >5 mm/hr pixel，RMSE = 7.97，是整體 RMSE（0.87）的 9 倍
- **這才是競賽勝負的關鍵，不是 88% 的零值 pixel**

---

## 八、Per-Satellite 性能

| 衛星 | Val RMSE | GPM 零值比例 |
|------|---------|------------|
| Himawari | 1.126 | 68.0% |
| GOES | 1.165 | 75.9% |
| Meteosat | 0.757 | 89.3% |

Meteosat 的 RMSE 低是因為 val 地點（france, kinshasa）降水少，零值比例高（89%），不代表模型真的更準。

---

## 九、時序訊號

- 97.5% 的訓練樣本有完整三個時間幀
- 雲頂降溫速率（BT_t0 - BT_t2，20 分鐘間隔）在有雨 pixel 明顯偏右——強化對流（變冷）預示重降水
- 三個時間幀**不是冗餘的**，它們編碼了對流的趨勢（加強/減弱）

---

## 十、對我們策略的影響

| 洞察 | 影響 |
|------|------|
| Location-grouped val 更誠實 | 換 val 方式後 val/LB 比值會縮小，判斷改動效果更可靠 |
| IR band 遠優於 visible | 我們 51ch 包含 IR，但也包含 visible（可能引入雜訊） |
| 輸入 41x41 vs 128x128 | 我們的 val RMSE 是在 upsampled GPM 上算的，本來就偏低 |
| BTD 作為核心 band 有效 | 不是疊加在 51ch，而是取代 visible 成為核心輸入 |
| 模型大小不是瓶頸 | 1.9M scratch 模型和我們 19M pretrained 差距只有 0.01 LB |

### 值得嘗試的方向

1. **換 val 方式成 location-grouped holdout**：讓 val 數字更接近 LB，歷史比較需要重新跑一次
2. **縮減 channel 到純 IR bands**：從 51ch 選出最強的 IR bands，去除 visible 雜訊
3. **純 MSE（拿掉 MAE）**：和 BL2 完全對齊，排除 MAE 的影響

---

## 十一、BL2 的三條 Key Takeaways

1. **Band 選擇比模型大小更重要。** Visible vs Split-window IR 的差距（1.286 vs 0.872）大於一般架構改動。了解哪些波段帶有降水資訊比加參數更有價值。

2. **Log1p loss 適合這個任務。** 讓重雨事件對梯度可見，不改變評估指標，仍用 raw mm/hr RMSE 選模型。

3. **用地點分組驗證，不要用時間。** 訓練測試集是地理分割，temporal val 可能產生和 LB 關係不大的數字。Location-grouped GroupKFold 是更誠實的泛化估計。


