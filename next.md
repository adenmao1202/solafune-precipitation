# 進度總覽（2026-06-21）

## 目前排名與成績

| Run | Val RMSE (holdout 41x41) | LB | 狀態 |
|-----|---------|-----|------|
| v8a | 1.1441 (temporal, 128x128) | **0.6979 (Rank 14)** | 目前最佳 LB |
| v12_holdout_41loss | 0.9036 | 未 submit | 新基準，最誠實的 val |
| v12_focal_14bins | 0.9344 | 未 submit | focal 效果不如 regression |
| v13_ir12ch_combined | 0.9216 | 未 submit | 12ch 比 51ch 差 |

前五名：0.6520 / 0.6617 / 0.6655 / 0.6721 / 0.6766，我們落後 #1 約 0.046。

---

## 完成的修正

### 1. Val RMSE 校正（最重要）
過去 val RMSE 在 128x128 bilinear upsampled GPM 上算，LB 是原始 41x41。
現在：model output interpolate 回 41x41 再算 loss 和 val RMSE，和 LB 同一個 metric。

### 2. 實驗基礎建設
- `experiments.yaml` + `run_all.py`：sequential queue，每個結束發 ntfy 通知
- `predict.py --run_name`：自動讀 args.json，一行生成 submission
- Per-epoch `history.csv`：train_loss / val_rmse / val_rmse_rain / lr
- Focal bin alpha cache：掃描 29,170 GPM 只做一次
- batch_size 8 -> 64（每 epoch 從 8.5 分鐘降到 1.5 分鐘）

### 3. 嘗試過、已排除的方向
| 方向 | 結果 | 結論 |
|------|------|------|
| Focal Loss + 14 bins | val 0.9344，val_RMSE_rain 完全不動（2.2x） | 梯度改善無效，排除 |
| 12ch IR band selection | val 0.9216，比 51ch 還差 | 51ch 已包含 IR，去掉其他 band 反而移除有用訊號 |
| OneCycleLR | 過去 v10 失敗 | 破壞 pretrained prior，永久排除 |
| BTD 疊加在 51ch | 過去 v9c 失敗 | DN 空間物理意義不成立，排除 |
| V-flip | 設計層面排除 | 破壞 FiLM 時間語意 |

---

## 核心問題（尚未解決）

**val_RMSE_rain 在所有實驗中都卡在 2.2~2.4，完全沒有改善。**
0.9% 的 >5mm/hr pixel 的 RMSE = 7.97（整體的 9 倍），這才是競賽勝負的關鍵。

BL2 error analysis：
- 0~0.5mm/hr（88%）：bias +0.076~+0.159，模型過度預測薄毛毛雨
- >5mm/hr（0.9%）：bias -6.15，系統性嚴重低估

**結論：loss function 和 band selection 都不是核心問題，輸出的 calibration 才是。**

前五名大概率用了：
- **Output thresholding**：預測值 < threshold（如 0.1mm/hr）設為 0，清除假陽性毛毛雨
- **Output scaling / bias correction**：對重雨區間放大預測值，修正系統性低估
- **更好的 input preprocessing**

---

## 下一步優先順序

### 立即（instance 上線後）
1. **submit v12_holdout_41loss** → 校準新 metric 下的 LB 比值
2. **scp checkpoint 到本地**（避免再次因 instance 問題損失進度）

### Phase 1：Output Post-processing
目前最高潛力、成本最低的改善方向：

**A. Zero thresholding**
```python
pred[pred < threshold] = 0.0  # threshold ~ 0.05~0.2 mm/hr
```
修正 light rain 區間的系統性正 bias。

**B. Heavy rain scaling**
對預測值 > X mm/hr 的區間乘以放大係數，修正重雨低估。
或用 isotonic regression 做 calibration（需要一組 holdout 資料）。

**C. 兩者組合**
先 threshold 清零，再對非零部分 apply scaling。

### Phase 2：架構改進（視 Phase 1 結果決定）
- 更長訓練（目前只跑 40 epochs，BL2 特徵：小模型跑更多 epoch）
- TTA（test-time augmentation）：H-flip 平均
- Ensemble：至少兩個不同 seed 的 v12_holdout_41loss

---

## 不再嘗試的方向

| 方向 | 原因 |
|------|------|
| Focal Loss | val_RMSE_rain 完全無改善，三次失敗（OneCycleLR / target misalign / alpha bug） |
| IR-only band selection | 51ch 本來就有 IR，移掉其他 band 不是 BL2 同樣的實驗情境 |
| OneCycleLR | 破壞 pretrained prior |
| BTD 疊加 51ch | DN 空間物理意義不成立 |
| stratified sampling + weighted loss 同時用 | 過度修正（v6 確認） |

---

## 參考數字

| 模型 | Val RMSE | LB RMSE | 比值 |
|------|----------|---------|------|
| BL2 visible | 1.2857 | ~0.913 | ~1.4x |
| BL2 split-window | 0.8724 | 0.708 | 1.23x |
| 我們 v8a（temporal, 128x128） | 1.1441 | 0.6979 | 1.63x |
| 我們 v12_holdout_41loss | 0.9036 | 未知 | 目標 ~1.23x |

若 v12_holdout_41loss 比值接近 1.23x，預期 LB = 0.9036 / 1.23 = 0.735（比 v8a 差）。
若比值縮小到 1.1x（holdout 更誠實），預期 LB = 0.9036 / 1.1 = 0.821（更差）。

**重點：submit 後才知道方向，先 submit。**

---

## EDA / Data Validation 驗證結果（src/eda.ipynb + src/data_validation.ipynb）

### 1. 資料集基本統計（已驗證）

| 項目 | 數值 |
|------|------|
| 訓練樣本總數 | 40,686 筆 |
| Himawari | 13,192 筆 |
| GOES | 10,272 筆 |
| Meteosat | 17,222 筆 |
| 時間範圍 | 2023-01-01 ~ 2026-01-31 |
| GPM 零值比例（100 sample EDA） | 80.33% |
| GPM 零值比例（全訓練集掃描） | 83.99% ← 100 sample 取樣偏差，以全集為準 |
| GPM mean | 0.335 mm/hr |
| GPM 99th pct | 7.4 mm/hr |
| GPM max（100 samples） | 39.96 mm/hr |
| GPM max（全訓練集） | **96.51 mm/hr** ← EDA 取樣太少沒看到真正極端值 |

### 2. 影像尺寸（已驗證，與 dataset.py 一致）

| 衛星 | 影像 HxW | GPM HxW | 比值 |
|------|---------|---------|------|
| Himawari | 81x81 | 41x41 | ~2x |
| GOES | 141x141 | 41x41 | ~3.4x |
| Meteosat | 144x144 | 41x41 | ~3.5x |

CRS=None（無地理投影），資料已預先空間對齊，不需要 reproject。
dataset.py 的 SAT_SIZE 常數正確。

### 3. Frame 數量分布（已驗證）

| Frames | 筆數 | 比例 |
|--------|------|------|
| 3 frames | 39,796 | 97.8% |
| 2 frames | 647 | 1.6% |
| 1 frame | 8 | 0.02% |
| 0 frames | 235 | 0.6% |

0 frames 的樣本用 zero padding + zero mask 處理，dataset.py 已正確實作。

### 4. 夜間 VIS 波段（已驗證）

夜間 VIS = **0（不是 NaN）**，z-score 後變成負值，不需要特殊 mask 處理。
Meteosat VIS 夜間零值比例 64~82%（最嚴重），Himawari 約 0~29%，GOES 約 0~1%。
mask channel（每 frame 一個）已記錄有效性，model 可自行學習夜間特性。

### 5. Band 物理波長對照（已驗證）

跨衛星的對應關係（0-based index）：

| 類型 | Himawari | GOES | Meteosat（swap 前） | Meteosat（swap 後） |
|------|---------|------|-------------------|-------------------|
| WV 6.2um | idx 7 (B08) | idx 7 (C08) | idx 9 (wv_63) | idx 9 (wv_63) |
| IR 10.4um | idx 12 (B13) | idx 12 (C13) | idx 13 (ir_105) | **idx 12** |
| IR 12.3um | idx 14 (B15) | idx 14 (C15) | idx 14 (ir_123) | idx 14 (ir_123) |

Meteosat band swap（arr[[12,13]] = arr[[13,12]]）後：
- idx 12 = ir_105（10.5um，雲頂溫度）
- idx 13 = ir_97（9.66um）

**IR_SPLIT_WINDOW_BANDS = {himawari:[7,12,14], goes:[7,12,14], meteosat:[9,12,14]}** 已驗證正確。

### 6. 重要發現：EDA 取樣不足導致的偏差

| 項目 | EDA（100 samples） | 全集掃描 |
|------|------------------|---------|
| 零值比例 | 80.33% | 83.99% |
| max precipitation | 39.96 mm/hr | 96.51 mm/hr |

100 sample EDA 沒看到真正的極端降水事件。全集掃描的數字才可靠。
這也解釋了為什麼固定 bin edge 上限設 26mm/hr 是錯的——真實 max 是 96mm/hr。
