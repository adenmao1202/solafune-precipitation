## Here are info for edge: 

1. GPM 80% 0 
- from official website : 
The GPM-IMERG is a multiband global precipitation dataset that contains both calibrated and uncalibrated precipitation values.
We only used the calibrated values under the name band of 'precipitation'. One band only.
The images have been processed to the region of interest from their own perspective locations with their own unique datetimes.
For more information, please visit this site.
This is the target variable.

---

## Edge 方向 A：解決 GPM 80% 零值問題

### 問題根源與分布分析

GPM 降水量分布（來自 EDA）：
```
80%   -> 0 mm/hr（無雨）
19%   -> 0.1~7.4 mm/hr（小到中雨）
0.9%  -> 7.4~18 mm/hr（大雨）
0.1%  -> >18 mm/hr（暴雨，最大 39.96 mm/hr）
```

非零像素的條件均值 ≈ 0.3348 / 0.2 ≈ 1.67 mm/hr。

**真正的問題不只是「零太多」，而是「RMSE 被 0.1% 的暴雨主導」：**
```
預測 0，真實 20 mm/hr -> 誤差 = 400
預測 0，真實 0.1 mm/hr -> 誤差 = 0.01
```
一個暴雨 pixel 預測失敗 = 4 萬個小雨 pixel 預測失敗。
這是為什麼 LB 分數卡關的根本原因——不是零值預測不好，是暴雨預測太差。

GENESIS 論文發現：**MSE 在輕雨表現好，在 >8 mm/hr 暴雨表現很差。**

---

### Solution A-1：分層取樣（Stratified Sampling）

[重要修正] 不是「完全移除零值樣本」，而是「調整比例」。
完全移除零值的風險：模型從沒見過晴天衛星影像，推論時對乾燥區域會輸出假降水。

**建議比例：有雨:無雨 = 50:50**

```
原始比例：80% 零 + 20% 有雨  -> 模型學到「預測零就贏」
完全過濾：0% 零 + 100% 有雨  -> 不知道晴天長什麼樣（有系統性假降水偏差）
建議比例：50% 零 + 50% 有雨  -> 強迫學降水特徵，仍保留晴天知識
```

**效果（來自 TUPANN 和 Global MetNet 論文）：**
- 有效 gradient 比例從 20% 提升到 50%，等效訓練效率提升 2.5 倍
- 模型對晴天和有雨情況都有足夠的學習樣本

**實作步驟：**

Step 1：在 train.py 或獨立腳本中預掃 CSV，生成一個「有雨樣本的 CSV」：

```python
import rasterio
import pandas as pd
from pathlib import Path

def has_precipitation(gpm_path: Path) -> bool:
    """回傳 True 表示這個 GPM 檔案有非零降水"""
    try:
        with rasterio.open(gpm_path) as src:
            arr = src.read(1)
            return arr.max() > 0
    except Exception:
        return False

def filter_rainy_samples(csv_path, data_dir, out_csv):
    df = pd.read_csv(csv_path)
    data_dir = Path(data_dir)
    mask = []
    for _, row in df.iterrows():
        gpm_path = data_dir / "gpm_imerg" / row["gpm_imerg_filename"]
        mask.append(has_precipitation(gpm_path))
    df_rainy = df[mask]
    df_rainy.to_csv(out_csv, index=False)
    print(f"全樣本：{len(df)}，有雨樣本：{len(df_rainy)} ({len(df_rainy)/len(df)*100:.1f}%)")
```

Step 2：生成一次，之後訓練時用新 CSV：
```bash
# 跑一次，約 5-10 分鐘
python scripts/filter_rainy.py \
  --csv ~/solafune/data/train_dataset.csv \
  --data_dir ~/solafune/data \
  --out ~/solafune/data/train_rainy.csv

# 訓練時改用新 CSV
python src/train.py --csv_train ~/solafune/data/train_rainy.csv ...
```

**注意事項：**
- 不需要更改 dataset.py 或 model.py，只換 CSV 輸入
- val set 仍然用全樣本（包含零值），這樣 val RMSE 才能反映真實 LB 情況
- 預估有雨樣本約佔 20%，即約 8,000 筆，訓練更快

---

### Solution A-2：Weighted MSE（在 pixel 層級加權）

在樣本層級分層取樣之外，還可以在 pixel 層級對高降水區域加更強的懲罰。
現在的 WeightedMSE 只區分「零/非零」，更細緻的版本：

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=3.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # target 在 log1p 空間
        # log1p(7.4) ≈ 2.1，用這個門檻區分大雨
        weight = 1.0 + self.alpha * (target > 0).float()
        # 進階：暴雨區域加更強 weight
        # weight = weight + 2.0 * (target > 2.1).float()  # >7.4 mm/hr
        return (weight * (pred - target) ** 2).mean()
```

**alpha 選擇建議：**
- v4 用了 alpha=5 -> val 變差（太激進，或 imagenet 不穩定導致）
- 搭配分層取樣（A-1）使用時，建議 alpha=2.0~3.0
- 單獨使用時，建議 alpha=3.0

### Solution A-3：針對暴雨的 Loss 設計（來自 GENESIS 論文）

GENESIS 論文發現 MSE 對暴雨（>8 mm/hr）預測很差。
進階選項：把降水預測拆成兩個子任務：
1. **分類頭**：是否有雨（Binary cross-entropy）
2. **迴歸頭**：有雨情況下強度是多少（Huber loss，對極端值更穩健）

這是「兩段式架構」，比單純改 loss 效果更大，但實作成本也更高（放 Phase 4 之後考慮）。

---

## Edge 方向 B：注入氣象物理知識（Domain Knowledge）

這是純 ML 競賽者不一定會做的部分，是真正的競爭優勢來源。

---

### Solution B-1：Day + Hour Positional Encoding（最高優先）

**為什麼重要：**
降水強度高度依賴季節和時間：
- 亞洲季風：6-9 月降水量是 1-3 月的 10 倍以上
- 午後對流：熱帶地區 14:00-17:00 是強對流高峰
- 歐洲/非洲（Meteosat）、美洲（GOES）、亞洲（Himawari）各有不同的季節節律

目前模型完全不知道「現在是幾月幾點」。NPM 論文（NPM_SatelliteNowcasting_2412.11480.pdf）
的 ablation table 顯示：加入 day-of-year encoding 是單一最大的改進項目（+17% CSI）。

**為什麼用 sin/cos 而不是直接輸入數字：**
- 直接輸入 day=1 ~ 365：模型看到 1 和 365 以為距離很遠，但它們其實是相鄰的（12/31 到 1/1）
- sin/cos encoding 把時間變成一個「圓形」，首尾相連：
  ```
  day_sin = sin(2 * pi * day / 365)
  day_cos = cos(2 * pi * day / 365)
  ```
  day=1 和 day=365 在 (sin, cos) 空間中距離很近 -> 模型正確理解季節連續性

**實作：在 dataset.py 提取時間特徵**

```python
import math
import pandas as pd

# 在 __getitem__ 中，取出 datetime 欄位後：
dt = pd.to_datetime(row['datetime'])
day = dt.day_of_year   # 1~365
hour = dt.hour          # 0~23

time_feat = torch.tensor([
    math.sin(2 * math.pi * day / 365),
    math.cos(2 * math.pi * day / 365),
    math.sin(2 * math.pi * hour / 24),
    math.cos(2 * math.pi * hour / 24),
], dtype=torch.float32)  # shape: [4]

# __getitem__ 回傳改成：
return input_tensor, target_tensor, unique_id, time_feat
```

**實作：在 model.py 用 FiLM 把時間注入 UNet bottleneck**

FiLM（Feature-wise Linear Modulation）：用時間 embedding 生成「縮放因子 gamma」和「偏移量 beta」，
直接調整 bottleneck feature map 的每個 channel。
直覺：「現在是季風季，把和強降水相關的 feature channel 放大」。

```python
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, time_dim=4, feature_channels=None):
        super().__init__()
        # feature_channels = UNet bottleneck 的 channel 數（EfficientNet-B4 約 448）
        self.gamma_fc = nn.Linear(time_dim, feature_channels)
        self.beta_fc  = nn.Linear(time_dim, feature_channels)

    def forward(self, features, time_feat):
        # features: (batch, C, H, W)
        # time_feat: (batch, 4)
        gamma = self.gamma_fc(time_feat)  # (batch, C)
        beta  = self.beta_fc(time_feat)   # (batch, C)
        # 廣播到空間維度
        gamma = gamma[:, :, None, None]   # (batch, C, 1, 1)
        beta  = beta[:, :, None, None]
        return features * (1 + gamma) + beta  # (1 + gamma) 讓初始值接近 identity
```

然後在 UNet forward 中，在 encoder 最後一層（bottleneck）後插入 FiLM：
```python
# model.py 中的 forward
x = self.encoder(x)          # bottleneck features
x = self.film(x, time_feat)  # 時間調制
x = self.decoder(x)          # 上採樣還原
```

---

### Solution B-2：BTD 亮溫差衍生特徵（Brightness Temperature Difference）

**物理原理：**
B13（10.4um）受大氣水氣吸收影響小，B15（12.4um）受影響大。
兩者差值 BTD = B13 - B15：
- BTD 大（正值）= 雲頂真實溫度比水氣遮蔽後更冷 = 雲厚、對流旺盛 = 大雨
- BTD 接近 0 = 薄雲或晴空
- 這是氣象學 40 年歷史的降水代理指標（Split Window Technique）

**各衛星對應 index：**
| 衛星 | B13 index | B15 index | 差值計算 |
|------|-----------|-----------|---------|
| Himawari | 12 | 14 | arr[12] - arr[14] |
| GOES | 12 | 14 | arr[12] - arr[14] |
| Meteosat | 13 | 15 | arr[13] - arr[15] |

**實作（加在 dataset.py __getitem__ 中）：**

```python
# 在 normalize_per_band 之前，對每個 frame 計算 BTD
satellite = row['satellite_target']
btd_idx_a = 12 if satellite in ('himawari', 'goes') else 13
btd_idx_b = 14 if satellite in ('himawari', 'goes') else 15

btd_channels = []
for i, arr in enumerate(frame_arrays):  # frame_arrays: 3 個 (16, H, W) array
    if masks[i] == 1:
        btd = arr[btd_idx_a] - arr[btd_idx_b]  # (H, W)
        btd_channels.append(btd[None])          # (1, H, W)
    else:
        btd_channels.append(np.zeros((1, H, W), dtype=np.float32))

# 把 BTD channels 串接到原始輸入後面
# 原本：(51, H, W)，加了 BTD 後：(54, H, W)
```

注意：IN_CHANNELS 要從 51 改成 54（3 frames x 1 BTD channel）。

---

### Solution B-3：IR 亮溫時間差分（Convective Development Rate）

**物理原理：**
雲頂溫度在 10 分鐘內下降多少 K = 對流發展速度。
快速下降（例如 -5K/10min）= 積雨雲正在長高 = 接下來強降水概率高。

```python
# frame_2 = t=0, frame_1 = t-10min, frame_0 = t-20min
ir_idx = 12 if satellite in ('himawari', 'goes') else 13

delta_recent = frame_2[ir_idx] - frame_1[ir_idx]  # 最近 10 分鐘的 IR 變化
delta_prev   = frame_1[ir_idx] - frame_0[ir_idx]  # 前 10 分鐘的 IR 變化
# 負值 = 溫度下降 = 雲頂升高 = 對流發展
```

加入後 IN_CHANNELS 從 54 變 56（再加 2 channels）。

---

## 關於 ImageNet 預訓練的結論（論文調查結果）

四篇同領域論文（GENESIS、NPM、TUPANN、Global MetNet）**全部從零訓練，沒有一篇用 ImageNet**。

原因：
- 衛星多光譜影像（紅外線能量、亮溫、水氣吸收率）和自然照片（RGB）的低階特徵差異太大
- 我們的 input 是 51 channels，ImageNet 只有 3 channels
- SMP 套件的處理：前 3 channel 用 ImageNet 權重，後 48 channel 隨機初始化
- 造成 gradient 更新嚴重不平衡，反而讓訓練不穩定

**v5 的目的：確認 imagenet=None（從零訓練）是否比 imagenet="imagenet" 更穩定。**
預期結果：val RMSE 應該比 v4 更低，回到接近 v3 的水準。

---

## 實驗優先順序

| 優先 | 實驗 | 主要改動 | 預估效果 |
|------|------|---------|---------|
| v5 | imagenet only | 移除 WeightedMSE，只保留 imagenet | 確認 v4 問題來源 |
| v6 | event-based sampling | 只用有雨樣本訓練（train_rainy.csv） | 預估最大單項提升 |
| v7 | day+hour encoding + FiLM | dataset.py + model.py 修改 | 第二大提升 |
| v8 | BTD + IR 差分 | dataset.py 加衍生特徵，IN_CHANNELS=56 | 中等提升 |
| v9 | 以上全部 ensemble | 3-5 個不同 seed | 穩定最終分數 |

---

## 參考論文

- **NPM（2412.11480）**：Day embedding ablation，直接支持 B-1 的做法
- **TUPANN（Tupann.pdf）**：Event-based sampling，支持 A-1
- **Global MetNet（2510.13050）**：多衛星輸入處理方式，支持衛星 alignment 修正
- **GENESIS（2307.10843）**：ConvLSTM skip connections，Phase 4 架構改進
