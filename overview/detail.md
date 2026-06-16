# Solafune 降水預測競賽：環境配置與執行指南

版本：v3.1（2026-06-16 更新，加入 EDA + validation 結果）
適用環境：Mac Air M2（MPS）+ T7 外接硬碟（exFAT）+ 未來 Vast.ai RTX 4090

---

## 已完成事項

- [x] 資料下載並解壓至 `/Volumes/T7/new_code/solafune/data/`
- [x] `src/dataset.py` ---- 多衛星 Dataset，支援 MPS
- [x] `src/model.py` ---- UNet + EfficientNet-B4 encoder
- [x] `src/train.py` ---- 訓練迴圈，含 val RMSE、best model 儲存
- [x] `src/predict.py` ---- 推論 + 產生 submission.zip
- [x] `src/eda.ipynb` ---- EDA notebook（完整輸出已確認）
- [x] `src/data_validation.ipynb` ---- 尺寸/CRS/波段驗證（發現 GPM 尺寸不一致）

---

## 實際資料結構

```
/Volumes/T7/new_code/solafune/data/
|---- train_dataset.csv              <- 訓練 CSV（unique_id, satellite_target, ...）
|---- evaluation_target.csv          <- 測試 CSV（同欄位，gpm_imerg_filename 為待預測）
|---- himawari/                      <- train_*.tif + test_*.tif（~73,000 個）  81x81, 16ch
|---- goes/                          <- train_*.tif + test_*.tif（~30,000 個）  141x141, 16ch
|---- meteosat/                      <- train_*.tif + test_*.tif（~83,000 個）  144x144, 16ch
|---- gpm_imerg/                     <- train_*.tif（~40,686 個，訓練標籤）     41x41, 1ch
`---- test_files/                    <- 提交時放預測結果
```

### [!] 重大發現：GPM 與衛星影像尺寸不一致

| 衛星 | 影像 HxW | GPM HxW | 比例 |
|---|---|---|---|
| Himawari | 81x81 | 41x41 | ~2x |
| GOES | 141x141 或 282x282（多種尺寸）| 41x41 | ~3.4x / ~6.9x |
| Meteosat | 144x144 | 41x41 | ~3.5x |

**處理策略（[done] 已在 dataset.py + predict.py 實作）：**
- 方案 A（[done] 採用）：`dataset.py` 讀取 GPM 後 bilinear resize 到衛星解析度 -> 模型在衛星解析度訓練 -> `predict.py` 推論後 bilinear resize 回 41x41 提交
- 方案 B：所有衛星影像 resize 到 41x41（較簡單，但損失資訊，未採用）

### CSV 

| 欄位 | 說明 |
|---|---|
| `unique_id` | 唯一識別碼 |
| `name_location` | 地區名稱 |
| `satellite_target` | 衛星名稱（小寫：`himawari` / `goes` / `meteosat`） |
| `datetime` | 時間戳記 |
| `last_30_minutes_observation_filename` | Python list 字串 `"['a.tif','b.tif','c.tif']"` |
| `gpm_imerg_filename` | GPM-IMERG 目標 tif 檔名 |

### 重要注意事項

- 衛星名稱為**全小寫**（`himawari`，不是 `Himawari`）
- filename 欄位為 **Python list 字串**，用 `ast.literal_eval()` 解析，不是 `;` 分隔
- TIF 檔案路徑格式：`{data_dir}/{satellite_subdir}/{filename}`
- GPM 目標路徑格式：`{data_dir}/gpm_imerg/{gpm_imerg_filename}`

---

## 技術架構

```
src/
|---- dataset.py    <- PrecipDataset, parse_filenames(), get_device(), SATELLITE_SUBDIR
|---- model.py      <- build_model()（UNet + EfficientNet encoder）
|---- train.py      <- compute_stats() + 訓練迴圈
|---- predict.py    <- 推論 + 產生 submission.zip
|---- eda.py        <- 命令列 EDA 腳本
`---- eda.ipynb     <- Jupyter EDA notebook
```

**關鍵設計**：
- `IN_CHANNELS = 51` = 3 frames x 16 bands + 3 mask channels
- log1p / expm1：模型在 log 空間訓練，驗證時還原到 mm/hr 計算 RMSE
- `stats.json`：per-satellite, per-band mean/std（第一次跑 train.py 自動計算）
- MPS 支援：`get_device()` 自動選 cuda -> mps -> cpu
- **模型輸出尺寸**：UNet decoder 輸出與輸入同解析度（81/141/144）-> `predict.py` 推論後 bilinear resize 到 41x41 提交（[done] 已實作）
- **dataset.py GPM resize**：訓練時 GPM 41x41 -> bilinear resize 到衛星解析度，保留衛星全解析度資訊（[done] 已實作）

### EDA 關鍵數據（來自 eda.ipynb）

| 項目 | 數值 |
|---|---|
| 訓練集總筆數 | 40,686 |
| Himawari | 13,192 筆（32.4%） |
| GOES | 10,272 筆（25.2%） |
| Meteosat | 17,222 筆（42.3%） |
| GPM 零值佔比 | **80.33%** |
| GPM mean | 0.3348 mm/hr |
| GPM std | 1.4836 mm/hr |
| GPM max | 39.96 mm/hr |
| GPM 99th pct | 7.4 mm/hr |
| GPM 99.9th pct | 18.12 mm/hr |

**Frame 數量分布（過去 30 分鐘觀測）**：

| frames | 筆數 | 比例 |
|---|---|---|
| 3（完整） | 39,796 | 97.8% |
| 2 | 647 | 1.6% |
| 1 | 8 | 0.02% |
| 0（全遮蔽） | 235 | 0.6% |

-> 0 frame 的 235 筆需 mask padding 處理，不可直接跳過

### Band Mapping（來自 data_validation.ipynb）

| 衛星 | 雲頂溫度通道 | index（0-based） | 水氣通道 index |
|---|---|---|---|
| Himawari | B13（10.4um） | **12** | 7, 8, 9 |
| GOES | C13（10.3um） | **12** | 7, 8, 9 |
| Meteosat | ir_105（10.5um） | **13** [!] | 9, 10 |

[!] Meteosat 雲頂溫度 index 比另外兩顆多 1，特徵提取時需依衛星切換

**VIS 波段夜間行為**：
- 所有 NaN = 0%（無缺值）
- 夜間 VIS = 0（不是 NaN），z-score 後為負值，模型可正常學習
- Meteosat 樣本 VIS 零值達 64-82%（歐洲/非洲夜間樣本較多）

---

## 執行流程

### Step 1：安裝套件

```bash
pip install torch torchvision segmentation-models-pytorch rasterio pandas numpy tqdm
```

### Step 2：跑 EDA（可選）

```bash
# 在 Jupyter 開啟
cd /Volumes/T7/new_code/solafune/src
jupyter notebook eda.ipynb
# DATA_DIR 和 CSV_TRAIN 已設定好
```

### Step 3：訓練

```bash
cd /Volumes/T7/new_code/solafune/src
python train.py \
  --data_dir /Volumes/T7/new_code/solafune/data \
  --csv_train /Volumes/T7/new_code/solafune/data/train_dataset.csv \
  --encoder efficientnet-b4 \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4
```

第一次執行會自動計算 `stats.json`，之後跳過。

### Step 4：推論

```bash
python predict.py \
  --data_dir /Volumes/T7/new_code/solafune/data \
  --csv_test /Volumes/T7/new_code/solafune/data/evaluation_target.csv \
  --model_path best_model.pth \
  --out_dir ./submission
```

輸出：`submission.zip`（可直接上傳比賽平台）

---

## exFAT 磁碟注意事項（T7）

### `._` 伴生檔案問題
macOS 在 exFAT 磁碟上會自動產生 `._filename` 元資料檔案。**已永久關閉**：
```bash
# ~/.zshrc 已加入
export COPYFILE_DISABLE=1
```

清除已存在的 `._` 檔案：
```bash
find /Volumes/T7 -name "._*" -delete
```


---

*詳細執行計畫見 [plan.md](plan.md)*
