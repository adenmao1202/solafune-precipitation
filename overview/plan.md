# 競賽執行計畫

*LB 第一：0.6848 RMSE（BlueLock）| 更新：2026-06-16*
*Submit 上限：每天 5 次（GMT 0:00 重置）*

---

## 現況（截至 2026-06-16 晚）

- Pipeline 全部完成並 smoke test 通過（val RMSE ~0.68 on 3 epochs / 102 samples）
- GitHub repo 建立：https://github.com/adenmao1202/solafune-precipitation（private）
- rclone 設定完成，Google Drive remote 名稱為 "gdrive"
- train_dataset zip 上傳中（tmux session "upload"，--retries 20，ETA ~6hr）
- evaluation_dataset zip 尚未上傳（train 傳完後立刻跑）
- Vast.ai 帳號已建立並充值 $15，**尚未租機**

---

## 當前阻塞：等待 GDrive 上傳完成

```bash
# 確認上傳狀態
tmux attach -t upload

# 上傳完 train 後，立刻跑 evaluation
COPYFILE_DISABLE=1 rclone copy \
  "/Volumes/T7/new_code/solafune/data/evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip" \
  "gdrive:" \
  --progress --transfers 1 --drive-chunk-size 256M \
  --retries 20 --retries-sleep 30s
```

---

## Phase 1：取得 Baseline LB 分數

### 步驟

**1. 上傳完成後，租 Vast.ai**
- GPU：RTX 4090 24GB On-Demand
- Image：`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
- 磁碟：120GB+
- 確認 CUDA >= 11.8（規則要求）

**2. Vast.ai 部署**
```bash
# 安裝 rclone
curl https://rclone.org/install.sh | sudo bash
mkdir -p ~/.config/rclone
# 貼上 Mac 上的 rclone.conf 內容（cat ~/.config/rclone/rclone.conf）

# 下載資料
mkdir -p ~/solafune/data && cd ~/solafune/data
rclone copy "gdrive:train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip" . --progress
rclone copy "gdrive:evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip" . --progress
unzip train_dataset_*.zip && unzip evaluation_dataset_*.zip

# Clone 程式碼
cd ~/solafune
git clone https://github.com/adenmao1202/solafune-precipitation.git
cd solafune-precipitation
pip install segmentation-models-pytorch rasterio pandas numpy tqdm

# 跑訓練
python src/train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --encoder efficientnet-b4 \
  --epochs 30 --batch_size 32 --lr 1e-4 --num_workers 4
```

**3. 推論並下載提交檔**
```bash
python src/predict.py \
  --data_dir ~/solafune/data \
  --csv_test ~/solafune/data/evaluation_target.csv \
  --model_path best_model.pth \
  --out_dir ./submission

scp -P <port> root@<ip>:~/solafune/solafune-precipitation/submission.zip ~/Desktop/
```

**4. Submit → 記錄 LB 分數**

---

## Phase 2：快速提分（每項 0.5 天，有 baseline 後立刻做）

每次只改一件事，submit 前確認 local val RMSE 有改善。

### P2-1. Weighted MSE loss（改 3 行）
```python
rain_weight = 1.0 + 5.0 * (target > 0).float()
loss = (rain_weight * (pred_log - target_log)**2).mean()
```
- 解決 80.33% 零值主導 loss 的問題
- alpha=5 是起點，可試 3 / 8 / 10

### P2-2. Event-based sampling（過濾全零 GPM 樣本）
- 預掃 CSV，標記 GPM 有非零 pixel 的樣本
- 訓練時只用有雨的樣本（TUPANN / Global MetNet 都這樣做）
- 消除約 80% 廢棄梯度

### P2-3. EMA weights
```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
# 每 step 後：ema_model.update_parameters(model)
# eval 和 predict.py 用 ema_model
```

---

## Phase 3：最重要的單項改進（2 天）

### P3-1. Day + Hour Positional Encoding
- `train_dataset.csv` 有 `datetime` 欄位（已確認）
- NPM ablation：加 day-of-year encoding = 單項最大改進 +17% CSI
- 做法：sin/cos encoding → FiLM 注入 UNet bottleneck

```python
# dataset.py 中加入
day = pd.to_datetime(row['datetime']).day_of_year  # 1-365
hour = pd.to_datetime(row['datetime']).hour         # 0-23
time_feat = torch.tensor([
    math.sin(2*math.pi*day/365),
    math.cos(2*math.pi*day/365),
    math.sin(2*math.pi*hour/24),
    math.cos(2*math.pi*hour/24),
], dtype=torch.float32)  # shape: [4]

# model.py bottleneck 加 FiLM
# gamma = Linear(time_feat) -> [C]
# beta  = Linear(time_feat) -> [C]
# bottleneck = bottleneck * gamma[:,:,None,None] + beta[:,:,None,None]
```

### P3-2. Season-aware sampling
- 每個 mini-batch 從 12 個月均等抽取
- 防止模型過擬合雨季雲型

---

## Phase 4：架構改進（選一，4-5 天）

**優先：ConvLSTM skip connections**
- 把 UNet skip connection 換成 Recursive ConvLSTM
- GENESIS 論文（UNetConvLSTM_IMERG_2307.10843.pdf）有完整說明
- 讓 skip path 帶時序記憶，改善預測銳利度

**備選：換更大 backbone**
- EfficientNet-B4 -> B5/B6（改一行，訓練時間 +30%）

---

## Phase 5：最後衝刺

### Ensemble
```python
# 3-5 個不同 seed 的模型，平均預測
pred_ensemble = torch.stack([m(x) for m in models]).mean(0)
```

### TTA（Test-Time Augmentation）
- 水平翻轉 + 垂直翻轉 + 原始，取平均
- 只改 predict.py，不改模型

### 後處理
- 預測值低於閾值（例如 0.05 mm/hr）直接設為 0
- 因為 GPM 80% 是零，模型可能有系統性正偏差

---

## 得獎時的程式碼提交格式（規則要求）

需要拆成三個模組（現在就應照這結構寫）：
- `preprocess_train.py`：讀資料、計算 stats、輸出訓練就緒格式
- `preprocess_test.py`：讀測試資料、套用 stats
- training module：讀 preprocessed 資料、訓練、存 best_model.pth
- prediction module：讀測試資料 + model，輸出 submission.zip
- 建議提供 Dockerfile，CUDA >= 11.8

---

## 衛星波段衍生特徵（Phase 3-4 可以試）

| 特徵 | 物理意義 | 衛星 | channel index |
|------|---------|------|--------------|
| IR 亮溫（雲頂溫度） | 雲頂高度代理，最重要 | Himawari/GOES | 12 |
| IR 亮溫（雲頂溫度） | 同上 | Meteosat | **13（不同！）** |
| 水氣通道 | 大氣濕度 | Himawari/GOES | 7, 8, 9 |
| 水氣通道 | 大氣濕度 | Meteosat | 9, 10 |
| 時間差分 frame_t - frame_{t-1} | 對流發展速度 | 全部 | 衍生特徵 |
| IR 亮溫差值 B13-B15 | 雲厚度代理 | Himawari | 12-14 |

---

## 風險與對策

| 風險 | 對策 |
|------|------|
| Private LB 翻車（只看 Public 35%） | 以 local val set 為主要指標，stratified split 含三顆衛星 |
| 跨地區泛化差 | val set 包含三顆衛星樣本 |
| 降水極端值低估 | Weighted MSE + log1p |
| 網路中斷導致上傳失敗 | tmux + --retries 20，單次斷線不影響 |
| Vast.ai 費用超支 | 每次訓練完立刻停機，$15 約可跑 35-40hr |
