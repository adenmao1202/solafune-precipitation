# 程式碼說明

本文件說明 src/ 下三個核心檔案的每個函數用途與設計邏輯。

---

## dataset.py

負責「把原始 TIF 檔案讀進來，轉換成模型可以吃的 tensor」。

### 全域常數

```python
MAX_FRAMES  = 3      # 每個樣本最多 3 張衛星影像（過去 30 分鐘，每 10 分鐘一張）
N_BANDS     = 16     # 每張影像有 16 個波段
IN_CHANNELS = 51     # 3 frames x 16 bands + 3 mask channels
```

`IN_CHANNELS = 51` 的由來：
- 3 frames x 16 bands = 48 個衛星資料 channel
- 3 個 mask channel（每個 frame 一個，標記該 frame 是否有效，1=有資料，0=遺失/損壞）

```python
SAT_SIZE = {
    "himawari": (81, 81),
    "goes":     (141, 141),
    "meteosat": (144, 144),
}
GPM_SIZE = (41, 41)  # GPM-IMERG 輸出固定是 41x41
```

三顆衛星的原生解析度不同，這是 dataset.py 要處理的核心挑戰之一。

---

### `get_device()`

```python
def get_device() -> torch.device:
```

偵測可用的硬體，依優先順序回傳：CUDA（NVIDIA GPU）> MPS（Mac M 系列）> CPU。

之所以不直接寫 `device = "cuda"`，是因為本機是 Mac M2（MPS），Vast.ai 才有 CUDA，讓這個函數自動判斷即可，同一份 code 兩邊都能跑。

---

### `parse_filenames(raw)`

```python
def parse_filenames(raw: str) -> list[str]:
    return ast.literal_eval(raw)
```

CSV 中的 `last_30_minutes_observation_filename` 欄位長這樣：

```
"['himawari_20200101_0000.tif', 'himawari_20200101_0010.tif', 'himawari_20200101_0020.tif']"
```

這是 Python list 被序列化成字串存進 CSV，用 `ast.literal_eval()` 安全地把它還原成真正的 list。

---

### `read_tif(path)`

```python
def read_tif(path: Path) -> np.ndarray | None:
```

用 rasterio 開啟 TIF 檔案，回傳 shape `(bands, H, W)` 的 float32 陣列。

若檔案損壞或不存在則回傳 `None`（不讓程式 crash）。呼叫端收到 `None` 時會自動補零 + 把對應的 mask 設為 0，告知模型「這個 frame 沒有資料」。

---

### `normalize_per_band(arr, stats, satellite)`

```python
def normalize_per_band(arr, stats, satellite) -> np.ndarray:
```

對每個波段做 z-score 標準化：

```
output = (arr - mean) / (std + 1e-6)
```

- `mean` 和 `std` 來自 `stats.json`（由 `compute_stats()` 預先計算）
- 分母加 `1e-6` 防止除以零
- `[:, None, None]` 把 shape `(16,)` 的 mean/std 廣播到 `(16, H, W)`

為什麼要標準化：衛星影像是 0-255 的整數，且不同波段（例如可見光 vs 紅外線）數值範圍差很多。標準化後各波段的值域接近，梯度下降更穩定、收斂更快。

---

### `resize_to(tensor, size)`

```python
def resize_to(tensor, size) -> torch.Tensor:
```

把 `(C, H, W)` 的 tensor 用雙線性插值縮放到指定尺寸。

使用時機：
1. 讓不同衛星（81x81、141x141、144x144）統一到同一個 `input_size`（128x128），這樣 DataLoader 才能把不同衛星的樣本放進同一個 batch。
2. 把 GPM 的 41x41 縮放到跟衛星影像一樣大，讓 loss 計算時兩者尺寸一致。

---

### `class PrecipDataset`

PyTorch 的標準 Dataset 類別，負責把一筆 CSV row 轉成模型輸入。

#### `__init__()`

儲存所有設定：CSV 路徑、資料目錄、stats、是否為訓練模式、input_size。

`input_size` 若設定（例如 `(128, 128)`），所有影像都會被 resize 到這個尺寸；若為 `None`，則保持各衛星的原生解析度（這樣每個 batch 只能放同一顆衛星的資料）。

#### `__len__()`

回傳資料集的樣本數（CSV 的 row 數）。

#### `_sat_path(satellite, fname)`

組合出 TIF 檔案的完整路徑：

```
data_dir / himawari / himawari_20200101_0000.tif
```

#### `__getitem__(idx)`

最核心的函數，每次 DataLoader 要取一筆資料時就呼叫一次。

**流程：**

```
1. 從 CSV 讀取第 idx 筆 row
2. 解析出衛星名稱、3 個 frame 的檔名
3. 對每個 frame（最多 3 個）：
   a. 用 read_tif() 讀取 TIF
   b. 若讀取成功：
      - 修正波段數（補零或截斷到 16 bands）
      - 修正空間尺寸（某些 TIF 不是標準大小，強制 resize）
      - normalize_per_band() 做 z-score 標準化
      - mask = 1（有效）
   c. 若讀取失敗（None）：
      - 補全零陣列
      - mask = 0（無效，讓模型知道這個 frame 沒資料）
4. 把 3 個 frame（各 16 bands）和 3 個 mask 在 channel 維度串接：
   shape: (16+16+16+1+1+1, H, W) = (51, H, W)
5. 若設定了 input_size，把整個 input_tensor resize 到指定尺寸
6. 若是訓練模式，讀取 GPM TIF：
   - log1p 轉換（把右偏的降水分布壓縮成接近常態）
   - resize 到跟 input 一樣大
7. 回傳 (input_tensor, target_tensor, unique_id)
```

**為什麼 target 用 log1p：**

GPM 降水量分布極度偏態：80.33% 是零，剩下的有少數極端大雨值（可能到幾百 mm/hr）。直接用原始值訓練，loss 會被大雨樣本主導，模型傾向忽略小雨。`log1p(x) = log(1 + x)` 把這個長尾分布壓縮，讓模型均勻學習所有降水量級。驗證和推論時再用 `expm1` 還原回原始單位。

---

## model.py

### `build_model(encoder_name, encoder_weights)`

```python
def build_model(encoder_name="efficientnet-b4", encoder_weights=None) -> nn.Module:
```

用 `segmentation_models_pytorch`（smp）套件建立 UNet 模型。

**參數說明：**

- `encoder_name`：encoder 骨幹網路，預設 EfficientNet-B4。可換成 `efficientnet-b7`、`resnet50` 等（Week 3 實驗項目）。
- `encoder_weights`：預訓練權重。`None` 表示從零訓練；`"imagenet"` 表示使用 ImageNet 預訓練權重（遷移學習，通常收斂更快）。
- `in_channels=51`：對應 IN_CHANNELS，即 3 frames x 16 bands + 3 masks。
- `classes=1`：輸出 1 個 channel（降水量預測圖）。
- `activation=None`：輸出原始數值，不過任何激活函數。因為這是回歸問題，MSE loss 不需要輸出被限制在特定範圍。

**資料流：**

```
(batch, 51, 128, 128)  <- 輸入
        |
  EfficientNet-B4 Encoder
  逐步下採樣，提取特徵：
  128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        |
  UNet Decoder
  逐步上採樣 + skip connection（從 encoder 各層複製特徵過來）：
  8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        |
(batch, 1, 128, 128)   <- 輸出（log1p 降水量預測圖）
```

**Skip connection 的作用：**
UNet 的關鍵設計。Encoder 在壓縮過程中會損失細節位置資訊，skip connection 把 encoder 各層的 feature map 直接複製給 decoder 對應層，讓 decoder 在放大時能參考原始細節，產生精確的空間定位。這對預測像素級降水分布非常重要。

---

## train.py

### `compute_stats(csv_path, data_dir, out_path, max_samples)`

```python
def compute_stats(...) -> dict:
```

**用途：** 預先計算每顆衛星每個波段的 mean 和 std，儲存為 `stats.json`。

**流程：**

```
1. 讀取 CSV，可選擇只取 max_samples 筆（每顆衛星各取 1/3，保持平衡）
2. 對每一筆樣本的每個 frame：
   - 用 rasterio 讀取 TIF
   - 對每個 band，取出所有像素值（用 [::10] 每隔 10 個取一個，避免記憶體爆炸）
3. 統計完後對每個 band 計算 mean 和 std
4. 存成 JSON：
   {
     "himawari": {"mean": [b1_mean, b2_mean, ...], "std": [b1_std, b2_std, ...]},
     "goes":     {...},
     "meteosat": {...}
   }
```

**只需要執行一次**。之後每次訓練直接讀 `stats.json` 即可。本機用 `--stats_max_samples 150`（快速採樣），Vast.ai 上建議用全量（`--stats_max_samples 0`）以確保統計準確。

---

### `class CombinedLoss`

```python
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7):
```

定義訓練用的 loss 函數，結合 MSE 和 MAE：

```
loss = 0.7 * MSE(pred, target) + 0.3 * MAE(pred, target)
```

**為什麼混合：**

| Loss | 優點 | 缺點 |
|------|------|------|
| MSE（均方誤差） | 對大誤差懲罰重，促使模型預測大雨 | 對極端值非常敏感，不穩定 |
| MAE（平均絕對誤差） | 對極端值穩健，不會暴衝 | 對大誤差懲罰較輕，可能低估大雨 |

7:3 混合取兩者平衡，在降水這種偏態分布上比單獨使用任一種效果更好。

#### `forward(pred, target)`

計算並回傳加權 loss 值。每個 batch 前向傳播時自動被呼叫。

---

### `train(args)`

主訓練函數，控制整個訓練流程。

#### 階段 1：載入 stats

```python
stats_path = Path(args.data_dir) / "stats.json"
if not stats_path.exists():
    stats = compute_stats(...)
else:
    stats = json.load(...)
```

若 `stats.json` 不存在就先算，存在就直接讀。

#### 階段 2：建立 Dataset 和 DataLoader

```python
n_val   = int(len(full_ds) * 0.1)   # 10% 作為 val set
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(..., generator=torch.Generator().manual_seed(42))
```

- 90% 訓練、10% 驗證，seed 固定為 42（確保每次執行切法相同，結果可重現）
- `pin_memory=True` 只在 CUDA 下啟用（MPS 不支援，否則報錯）
- `shuffle=True` 只在訓練集，驗證集不需要打亂

#### 階段 3：建立模型

```python
model = build_model(encoder_name=args.encoder).to(device)
```

把模型移到 GPU 上（`.to(device)`），之後所有計算在 GPU 執行。

#### 階段 4：Optimizer 和 Scheduler

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=lr*0.01)
```

**AdamW：** Adam 加上 weight decay（L2 正規化），讓模型權重不要長太大，防止 overfitting。

**CosineAnnealingLR：** Learning rate 的變化曲線如下：

```
lr
^
|*                *
| *              * *
|  **          **   *
|    **      **      **
|      ******          **...
+--------------------------------> epoch
0                   T_max
```

訓練初期 lr 較高，快速收斂；後期 lr 降低，細緻調整；最後降到 `lr * 0.01`（1% 的初始值）。比固定 lr 或 step decay 更平滑。

#### 階段 5：訓練迴圈

**Train phase（每個 batch）：**

```python
optimizer.zero_grad()       # 清除上一個 batch 留下的梯度
preds = model(inputs)       # 前向傳播
loss  = criterion(preds, targets)  # 計算 loss
loss.backward()             # 反向傳播，計算每個參數的梯度
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
optimizer.step()            # 用梯度更新參數
```

**梯度裁剪（clip_grad_norm_）的作用：**
把所有參數的梯度向量的 L2 norm 限制在 1.0 以內。降水預測中偶爾出現的極端值可能導致梯度暴衝（gradient explosion），使模型訓練不穩定。裁剪後確保每一步更新幅度不超過合理範圍。

**Val phase（每個 batch）：**

```python
with torch.no_grad():                   # 不計算梯度，節省記憶體
    preds = model(inputs)               # 前向傳播（在 log1p 空間）
    preds_real   = torch.expm1(preds.clamp(min=0))   # 還原到 mm/hr
    targets_real = torch.expm1(targets)              # 還原到 mm/hr
    sq_errors = (preds_real - targets_real) ** 2     # 平方誤差
```

`clamp(min=0)`：把負數預測值裁剪為 0（降水量不可能是負數），再做 `expm1`。

最後把所有 batch 的平方誤差串接起來，計算整體 RMSE：

```python
val_rmse = sqrt(mean(all_squared_errors))
```

#### 階段 6：儲存最佳模型

```python
if val_rmse < best_val_rmse:
    best_val_rmse = val_rmse
    torch.save(model.state_dict(), "best_model.pth")
```

只存「驗證 RMSE 最低」的那個 epoch 的模型，而不是最後一個 epoch。這樣即使訓練後期 overfitting，我們用的還是泛化能力最好的版本。

`model.state_dict()` 只儲存參數數值，不儲存模型結構（結構在 `build_model()` 裡定義）。

---

## 整體資料流總結

```
train_dataset.csv
    |
    | (每一筆 row)
    v
PrecipDataset.__getitem__()
    |-- 讀取 3 個 frame 的 TIF (rasterio)
    |-- 處理損壞/遺失 frame (補零 + mask=0)
    |-- z-score 標準化 (normalize_per_band)
    |-- resize 到 128x128 (resize_to)
    |-- 串接成 (51, 128, 128) tensor
    |-- 讀取 GPM TIF，log1p 轉換
    v
DataLoader (batch_size=32)
    v
build_model() -- UNet + EfficientNet-B4
    |-- Encoder: (batch, 51, 128, 128) -> feature maps
    |-- Decoder: feature maps -> (batch, 1, 128, 128)
    v
CombinedLoss: 0.7*MSE + 0.3*MAE (在 log1p 空間計算)
    v
AdamW + CosineAnnealingLR: 更新模型參數
    v
Val: expm1 還原 -> 計算真實 RMSE (mm/hr)
    v
best_model.pth (val RMSE 最低的 epoch)
    v
predict.py -> submission.zip -> Solafune 排行榜
```
