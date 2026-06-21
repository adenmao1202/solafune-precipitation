# Vast.ai 部署完整步驟

Google Drive 資料夾：https://drive.google.com/drive/u/0/folders/1r9P6UfWD4iZCuWweDje1Lb3H9xvxBOiF

內容：
- solafune_parts/   <- train zip 切成 9 份 (part_aa ~ part_ai, 共 17.75 GB)
- evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip  (13 GB)
- sample_submission_95c3b1e094034f5fbba421f...zip

---

## Step 1：租機器

1. 進 https://vast.ai -> 左側 Templates -> 搜尋 `pytorch`
2. 選：`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
3. Filter 設定：
   - GPU Type: RTX 4090
   - # GPUs: 1
   - Disk Space: 120 GB
   - Instance Type: On-Demand
4. 選最便宜那台 ($0.35-0.45/hr)，按 Rent
5. 等狀態變 Running，點 Connect -> 取得 SSH 指令

---

## Step 2：SSH 連線後，安裝 rclone

```bash
apt-get update -qq && apt-get install -y unzip
curl https://rclone.org/install.sh | sudo bash
mkdir -p ~/.config/rclone
```

**在本機執行**，把 rclone 設定複製過去：
```zsh
# 本機執行（換掉 port 和 ip）
scp -P <port> ~/.config/rclone/rclone.conf root@<ip>:~/.config/rclone/rclone.conf
```

確認可以讀到 GDrive solafune 資料夾：
```bash
rclone lsd gdrive: --drive-root-folder-id 1r9P6UfWD4iZCuWweDje1Lb3H9xvxBOiF
# 應看到 solafune_parts/ 資料夾
```

GDrive 資料夾 ID（所有 rclone 指令都要加此參數）：
```
--drive-root-folder-id 1r9P6UfWD4iZCuWweDje1Lb3H9xvxBOiF
```

---

## Step 3：下載並還原 train 資料

```bash
mkdir -p ~/solafune/data/train_parts

# 下載 9 個 parts（約 10-20 分鐘）
rclone copy "gdrive:solafune_parts/" ~/solafune/data/train_parts/ \
  --drive-root-folder-id 1r9P6UfWD4iZCuWweDje1Lb3H9xvxBOiF \
  --progress

# 組回 zip 並解壓（約 15-20 分鐘）
cat ~/solafune/data/train_parts/part_* > ~/solafune/data/train.zip
cd ~/solafune/data
unzip train.zip -d . && rm train.zip
rm -rf train_parts/
```

---

## Step 4：下載 evaluation 資料

```bash
rclone copy "gdrive:evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip" ~/solafune/data/ \
  --drive-root-folder-id 1r9P6UfWD4iZCuWweDje1Lb3H9xvxBOiF \
  --progress
cd ~/solafune/data
unzip evaluation_dataset_*.zip -d . && rm evaluation_dataset_*.zip
```

---

## Step 5：Clone 程式碼 + 安裝套件

```bash
cd ~/solafune
git clone https://github.com/adenmao1202/solafune-precipitation.git
cd solafune-precipitation
pip install segmentation-models-pytorch rasterio pandas tqdm
```

---

## Step 6：確認資料路徑正確

```bash
ls ~/solafune/data/
# 預期看到：train_dataset.csv, evaluation_target.csv, himawari/, goes/, meteosat/, gpm_imerg/, test_files/
```

---

## Step 7：跑訓練

先開 tmux，避免 SSH 斷線中斷訓練：
```bash
tmux new -s train
```

```bash
cd ~/solafune/solafune-precipitation
git pull origin main && \
python src/train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --batch_size 64 --lr 1e-4 --num_workers 16 \
  --stats_max_samples 3000 \
  --run_name <run_name> && \
curl -s -d "<run_name> done!" ntfy.sh/solafune_luiz_train
```

- best_model.pth 儲存在 `runs/<run_name>/best_model.pth`
- tmux 離開不中斷：`Ctrl+B` 再按 `D`；重新連線：`tmux attach -t train`

---

## Step 8：推論 + 下載提交檔

```bash
python src/predict.py \
  --data_dir ~/solafune/data \
  --csv_test ~/solafune/data/evaluation_target.csv \
  --model_path runs/<run_name>/best_model.pth \
  --out_dir ./submission
```

**在本機執行**，下載 submission.zip：
```zsh
scp -P <port> root@<ip>:~/solafune/solafune-precipitation/submission.zip ~/Desktop/
```

---

## Step 9：Submit 到 Solafune

上傳 submission.zip -> 記錄 LB 分數。

---

## 注意事項

- 訓練完立刻 Stop 機器（不要 Destroy），磁碟費極低
- $15 credits 約可跑 35-40 小時
- val_RMSE 是 128x128 upsampled GPM，不等於 LB RMSE（LB 是原始 41x41）
