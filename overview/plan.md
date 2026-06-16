# 競賽執行計畫

*LB 第一：0.6848 RMSE（BlueLock）| 更新：2026-06-16*

---

## Week 1（進行中）：Pipeline 建立

- [x] 資料解壓與目錄結構確認
- [x] Dataset / Model / Train / Predict 程式碼
- [x] EDA 完成（降水分布、衛星樣本數、波段統計）
- [x] Data Validation 完成（HxW 確認、CRS、波段 mapping）
- [x] 修正 dataset.py/predict.py 的 GPM/衛星尺寸不一致問題
- [ ] 第一次在 Vast.ai 訓練，取得 baseline LB 分數

---

## Week 2：解決核心難點（目標：RMSE < LB 第三名）

- [ ] 確認 log1p transform 有效（理論支持：GPM 零值 80.33%，長尾分布）
- [ ] 嘗試 weighted loss（對非零像素加權；零值 80.33% 佔主導）
- [ ] 加入時間差分特徵（`frame_t - frame_{t-1}`）
- [ ] 嘗試 Huber Loss
- [ ] 處理 0-frame 的 235 筆樣本（mask padding 或排除）

---

## Week 3：特徵工程 + 架構升級（目標：超越 LB 第一 0.6848）

**衛星波段衍生特徵**（index 均為 0-based）

| 特徵 | 物理意義 | 衛星 | index |
|---|---|---|---|
| IR 亮溫（雲頂溫度） | 雲頂高度代理（最重要） | Himawari/GOES | 12 |
| IR 亮溫（雲頂溫度） | 雲頂高度代理 | Meteosat | **13**（[!] 不同）|
| 水氣通道（6-7um） | 大氣濕度 | Himawari/GOES | 7, 8, 9 |
| 水氣通道（wv_63, wv_73） | 大氣濕度 | Meteosat | 9, 10 |
| IR 亮溫差值（B13 - B15） | 雲厚度代理 | Himawari | 12 - 14 |
| 時間差分（frame_t - frame_{t-1}） | 對流發展速度 | 全部 | ---- |

**架構選項**

| 選項 | 說明 | 優先度 |
|---|---|---|
| A | UNet + EfficientNet-B4（已實作） | 已完成 |
| B | UNet + EfficientNet-B7 或 ResNet-50 | 先試 |
| C | ConvLSTM + UNet decoder | B 無顯著提升再試 |
| D | Swin Transformer encoder | 資源允許再試 |

---

## Week 4：Ensemble + 衝刺

- [ ] 訓練 3-5 個不同 seed / encoder 的模型
- [ ] Prediction averaging
- [ ] 後處理：低於閾值的像素設為 0
- [ ] 整理 source code 確認符合提交格式

---

## Vast.ai 部署（訓練用，Week 2 起）

### 租機規格

- GPU：RTX 4090 24GB
- Image：`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
- 磁碟：120GB+

### 部署腳本

```bash
#!/bin/bash
# 1. Clone 程式碼
git clone https://github.com/你的帳號/solafune-precipitation.git
cd solafune-precipitation

# 2. 安裝套件
pip install -r src/requirements.txt

# 3. 從 Google Drive 下載資料
mkdir -p data/
rclone copy gdrive:solafune-precipitation/data/ ./data/ \
  --progress --transfers 8 --drive-chunk-size 256M

# 4. 解壓
cd data
unzip train_dataset_*.zip
unzip evaluation_dataset_*.zip

# 5. 開始訓練
cd ../src
python train.py \
  --data_dir ../data \
  --csv_train ../data/train_dataset.csv \
  --encoder efficientnet-b4 \
  --epochs 50 \
  --batch_size 32
```

### Google Drive 同步（資料上傳，只做一次）

```bash
rclone copy /Volumes/T7/new_code/solafune/data/ gdrive:solafune-precipitation/data/ \
  --progress --transfers 4
```

---

## 風險與對策

| 風險 | 對策 |
|---|---|
| Private LB 翻車（只看 Public 35%） | 用本地 val set 為主要參考指標 |
| 跨地區泛化差 | val set 包含三顆衛星樣本；stratified split |
| 降水極端值主導 loss | log1p + weighted loss |
| exFAT 磁碟 I/O 不穩定 | 重要資料同步到 Google Drive |
