# rclone 設定指南

資料傳輸路徑：Mac T7 -> Google Drive -> Vast.ai

---

## Step 1：Mac 上安裝並設定 rclone（只做一次）

```bash
brew install rclone
rclone config
```

互動設定流程：
```
n                    <- New remote
名稱: gdrive
類型: drive          <- 輸入 drive 選 Google Drive
client_id:           <- 直接 Enter（留空）
client_secret:       <- 直接 Enter（留空）
scope: 1             <- 全部存取
root_folder_id:      <- 直接 Enter
service_account:     <- 直接 Enter
Edit advanced? n
Use auto config? y   <- 會自動開瀏覽器，用 luizmao1202@gmail.com 登入授權
```

授權完畢後測試：
```bash
rclone ls gdrive: | head -10
```

看到 Google Drive 檔案列表就成功。設定存在：`~/.config/rclone/rclone.conf`

---

## Step 2：確認 zip 在 Google Drive 上的位置

```bash
rclone ls "gdrive:" | grep zip
```

確認以下兩個 zip 存在：
- `train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip`（~19GB）
- `evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip`（~13GB）

---

## Step 3：Vast.ai 上設定 rclone

在 Vast.ai instance 上執行：

```bash
# 安裝 rclone
curl https://rclone.org/install.sh | sudo bash

# 複製 Mac 上的設定（把 ~/.config/rclone/rclone.conf 內容貼進來）
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
# 貼上 Mac 上 rclone.conf 的完整內容，Ctrl+X 存檔
```

Mac 上查看 rclone.conf 內容：
```bash
cat ~/.config/rclone/rclone.conf
```

---

## Step 4：Vast.ai 上下載並解壓資料

```bash
mkdir -p ~/solafune/data
cd ~/solafune/data

# 下載兩個 zip（從 Google Drive 根目錄，依實際路徑調整）
rclone copy "gdrive:train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip" . --progress
rclone copy "gdrive:evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip" . --progress

# 解壓
unzip train_dataset_*.zip
unzip evaluation_dataset_*.zip

# 確認目錄結構
ls -la
```

預期看到：`himawari/`, `goes/`, `meteosat/`, `gpm_imerg/`, `test_files/`, `train_dataset.csv`, `evaluation_target.csv`

---

## Step 5：上傳 code 到 Vast.ai

```bash
# 在 Mac 上打包 src/
cd /Volumes/T7/new_code/solafune
tar -czf src.tar.gz src/

# scp 上傳到 Vast.ai（IP 和 port 從 Vast.ai 控制台取得）
scp -P <port> src.tar.gz root@<ip>:~/solafune/

# 在 Vast.ai 上解壓
cd ~/solafune
tar -xzf src.tar.gz
```

---

## Step 6：Vast.ai 上安裝套件並訓練

```bash
pip install segmentation-models-pytorch rasterio pandas numpy tqdm

cd ~/solafune/src
python train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --encoder efficientnet-b4 \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 4
```

---

## Step 7：推論並下載提交檔案

```bash
python predict.py \
  --data_dir ~/solafune/data \
  --csv_test ~/solafune/data/evaluation_target.csv \
  --model_path best_model.pth \
  --out_dir ./submission

# 下載到 Mac
scp -P <port> root@<ip>:~/solafune/src/submission.zip ~/Desktop/
```

上傳 `submission.zip` 到 Solafune 比賽平台。

---

## 注意事項

- Google Drive zip 路徑依你實際上傳的資料夾而定，用 `rclone ls gdrive:` 確認
- Vast.ai SSH 連線資訊在租機後的控制台可以看到
- 訓練完記得在 Vast.ai 上按停止避免持續計費
