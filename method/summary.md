# Solafune 降水預測競賽：實驗策略全流程

整理日期：2026-06-20（v10 失敗後更新）

---

## 一、現況確認

### 目前最佳成績

| 版本 | val RMSE | LB RMSE | 排名 | 備註 |
|------|----------|---------|------|------|
| v8a | ~1.1441 | 0.6979 | #9 | 目前最佳，FiLM Day+Hour，EfficientNet-B4 |
| #1 BlueLock | -- | ~0.6848 | #1 | 差距 0.0131 |

### v8a 架構（最佳成績基準）

- Encoder: EfficientNet-B4（ImageNet pretrained）
- Decoder: UNet decoder（skip connections）
- Time conditioning: FiLM（Day+Hour sin/cos -> scale/shift encoder features）
- Input: 51 channels = 3 satellites x 3 frames x 16 bands + 3 coverage masks，resize 到 128x128
- Output: 1 channel regression，bilinear downsample 到 41x41，ReLU
- Loss: 0.7 * MSE + 0.3 * MAE
- Optimizer: AdamW，lr=1e-4
- Scheduler: ReduceLROnPlateau（factor=0.5，patience=3，min_lr=lr*0.01）
- Augmentation: 無
- EMA: 無

### 目前 Code 狀態（HEAD: ad03faa，2026-06-20）

v10 系列在 v8a 基礎上改動，目前 code 的實際狀態：

| 項目 | 目前 code | v11 計劃目標 | 是否已修 |
|------|-----------|------------|--------|
| Loss default | combined（ad03faa 剛改） | combined | 已修 |
| Scheduler | **OneCycleLR(max_lr=1e-3)**（80dd257 改入） | ReduceLROnPlateau(patience=5) | **未修** |
| Flip aug | **H-flip + V-flip 都在**（dataset.py） | H-flip only，移除 V-flip | **未修** |
| EMA | **decay=0.995**（80dd257 改入） | decay=0.999 | **未修** |
| FocalLossIMERG class | 仍在 train.py（可用 --loss_type focal 啟動） | 保留，待乾淨測試 | -- |

> **結論：** v11 的 code 目前只修了 loss default，scheduler / V-flip / EMA decay 三個問題還未在 code 裡修正，需要在開下一個 instance 前先 fix + push。

### 已知失敗實驗

| 版本 | 改動 | val RMSE | 結論 |
|------|------|---------|------|
| v9c | 加入 BTD 特徵（51->54ch） | ~1.28 | BTD 有害，禁止再用 |
| v9d | MAE 比重從 0.3 調高到 0.7（0.3*MSE+0.7*MAE），同時移除 BTD | 1.1540 | 比 v8a 更差（+0.01）；regression 任何加權方式上限都差不多，框架本身是瓶頸 |
| v6 | stratified sampling + TieredWeightedLoss | 失敗 | 雙重過度矯正：sampling 和 loss 同時放大有雨梯度，互相疊加導致不穩定 |
| **v10_focal** [DONE - 失敗] | Focal Loss（10 log-bins, gamma=2） + **OneCycleLR**（原本計劃是 CosineWarmRestarts，commit 80dd257 換成 OneCycleLR） + EMA(0.995) + H/V-flip + batch=32（RTX 3060） | 最差一次（>v8a 大幅倒退） | 四個原因見下方「v10 失敗根因分析」 |
| **v10_regression** [DONE - 失敗] | 把 Focal 換回 combined loss（commit ad03faa），**OneCycleLR / H/V-flip / EMA 0.995 都未改** | 無法到 1.3 以下 | **主因：OneCycleLR max_lr=1e-3 對 pretrained EfficientNet-B4 太激進，破壞 ImageNet prior；Focal 只是次要因素** |

---

## 一之一、v10 失敗根因分析（2026-06-20）

v10 是最差的一次，且把 Focal 拿掉後換回 regression，val RMSE 仍無法到 1.3 以下（v8a 是 1.1441）。以下按影響排序：

### 主因 1：OneCycleLR 破壞 pretrained ImageNet prior（無法到 1.3 的真正原因）

OneCycleLR 設定：`max_lr = lr * 10 = 1e-3`，前 30% epoch 從 4e-6 **猛拉到 1e-3**（放大 250 倍），接著退火到 1e-7。

這個 LR 對 pretrained EfficientNet-B4 極高——相當於把 encoder 從 ImageNet 初始點踢走，強迫 fine-tuning 重新在一個更差的位置收斂。v8a 的 ReduceLROnPlateau 保守地從 1e-4 出發，讓模型安全地找到更好的收斂盆地。

**確認**：即使把 Focal Loss 換回 combined regression，model 仍卡在 1.3+，說明倒退的根源是 scheduler，不是 loss 函數。

### 主因 2：Focal Loss 的結構性精度下限

val 時預測方式是 `(probs * bin_center).sum(dim=1)`（期望值）。問題在最後一個 bin（25.6 ~ max_val，寬 ~70 mm/hr），任何落在此 bin 的 pixel 預測值都固定是 ~61 mm/hr，無法捕捉 bin 內的連續分佈。這讓 Focal 框架即使分類準確，RMSE 也存在不可消除的 floor。

### 次因 3：V-flip 破壞 FiLM 時間語意

V-flip 把衛星影像上下翻轉，但 FiLM 的 `sin(2*pi*day/365)` 沒有跟著翻。北半球夏天（day~180）的對流場被 V-flip 後在視覺上像南半球冬天，但時間條件仍是「夏天」，造成 FiLM 語意矛盾，梯度方向衝突。H-flip（左右）沒有這個問題。

### 次因 4：EMA 在高 LR 期追不穩定 weights

EMA decay=0.995，但 OneCycleLR 前期 LR 爆升時每個 step 的 weights 都在大幅震盪。EMA shadow 追的是「不穩定高 LR 期」的平均，validate 時的 EMA weights 比真實 model 更差。

---

## 二、根本診斷：為什麼卡在 0.69x

### 問題一：Loss 函數被零值像素壓倒

GPM-IMERG 的分布是「零膨脹的右偏分布」：
- 約 80% 的 pixel 是零雨（0 mm/hr）
- 非零部分大多是輕雨（0.1-1 mm/hr）
- 重雨（>8 mm/hr）佔不到 1%，但它們的值大，對 RMSE 的貢獻不成比例地高

在任何 regression loss 框架下（無論是 0.7*MSE+0.3*MAE 的 v8a，還是 0.3*MSE+0.7*MAE 的 v9d），假設一個 batch 有 16384 個 pixel：
- 13100 個零值 pixel：梯度方向是「預測值降到 0 附近」
- 3284 個有雨 pixel：梯度方向是「預測值升到真實值附近」

13100 個 vs 3284 個，梯度被零值淹沒。模型的最優解是「全部預測接近零」：這樣能最小化大量零值 pixel 的 RMSE，但代價是系統性低估重雨事件。

v9d 把 MAE 比重從 0.3 調高到 0.7，val RMSE 反而從 1.1441 升到 1.1540（比 v8a 更差），正好說明：regression 框架調整 loss 權重已無法解決根本問題，不管怎麼調比例都逃不出同一個上限。

### 問題二：Scheduler 過早降 LR，卡在局部解

我們的 ReduceLROnPlateau 設定 patience=3、factor=0.5，代表「3 epoch 沒改善就把 LR 砍半」。這太激進：3 epoch 的震盪很正常，過早降 LR 會讓模型卡在某個局部解附近，之後的訓練只是在局部解周圍做更細的搜索，無法跳脫。

文獻中的做法：SpaceNet8 用固定 milestone step decay，SatFormer 用完全固定 lr=1e-5 不降，GENESIS 是 ReduceLROnPlateau(patience=10, factor=0.1)（patience 比我們長 3 倍、factor 比我們溫和），這些設計都給模型更長的探索空間。我們的 patience=3 是文獻中最激進的設定，很可能是訓練提早收斂的原因之一。

### 問題三：沒有任何資料增強 [v10 已加入，但 V-flip 有反效果]

v10 在 dataset.py 加入了 H-flip + V-flip（各 p=0.5）。H-flip 沒問題；V-flip 破壞了 FiLM 時間條件的地理語意（見「一之一 次因3」）。

v11 應只保留 H-flip。H-flip 讓每個樣本出現 2 個方向，等效擴大 2 倍有效訓練集，且物理合理（降水場在東西方向無固有偏好）。

### 問題四：沒有 EMA [v10 已加入，但在 OneCycleLR 下效果差]

v10 加入了手動 EMA（decay=0.995）。實作本身正確，但 OneCycleLR 高 LR 期 weights 震盪劇烈，EMA shadow 追的是不穩定狀態，val 時的 EMA weights 反而比真實 model 更差。

v11 的 EMA 應調整 decay=0.999（更保守），並在 scheduler 換回保守設定後才能正常發揮作用。

---

## 三、實驗計劃

### 總體方向（2026-06-20 更新後）

原計劃是「All-in-one 打包」，但 v10 失敗揭示了打包多個改動的風險：scheduler 問題掩蓋了 Focal Loss 真正的效果。

**新方向：分步驗證**
1. **v11 step 1**：先修正 scheduler + flip + EMA decay，用 combined loss，確認 baseline 回到 ~1.14
2. **v11 step 2**：在穩定 baseline 上單獨加 Focal Loss，才能乾淨評估其效果
3. **實驗二（Ensemble）**：最佳設定確認後，3-5 seeds 集成

**不使用 importance sampling**（原因不變：FL alpha_c 和 sampling 雙重矯正，重演 v6 問題）

---

### 實驗一：All-in-one（原計劃）[DONE - 失敗，見一之一]

**原性質：** 把所有不衝突的改進打包成一次訓練（OneCycleLR + H/V-flip + EMA + Focal Loss）。

**失敗原因：** 多個問題同時引入，OneCycleLR 的破壞效果掩蓋了其他改動的效果，無法判斷各自貢獻。

**為什麼不加 importance sampling（結論不變）：**
RainAI 的 ablation 顯示，在 CE 分類 loss 之上再加 class weights 會讓性能下降（exp6: 0.0451 < exp4: 0.0502）。Focal Loss 的 alpha_c（inverse frequency weighting）和 importance sampling 本質上都在修正同一個問題（class imbalance），同時使用兩者是雙重矯正，可能重演 v6 的問題。

#### 改動一：Scheduler [嘗試過 OneCycleLR - 失敗；CosineWarmRestarts - 未乾淨測試]

**v10 git 歷史：**
- `1a3718e`：原始 v10 計劃加入 CosineAnnealingWarmRestarts
- `80dd257`：中途改為 OneCycleLR（max_lr=1e-3）——這才是 v10 失敗的主因

**結論：** OneCycleLR 確認有害，絕不再用。CosineWarmRestarts 從未在乾淨環境下測試過，效果未知。

**v11 step 1 採用：** 回到 ReduceLROnPlateau（patience=5, factor=0.5, min_lr=1e-7），與 v8a 相同基調但 patience 放寬（v8a 是 3）。這是最保守的選擇，先確認能回到 ~1.14。

**v11 step 2 或後續可嘗試：** CosineAnnealingWarmRestarts，作為 scheduler 的單一變數實驗：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-7
)
```

T_0=20 表示第一個週期 epoch 0-20 完成一次完整的 1e-4 -> 1e-7 掃描，之後每次 restart 週期加倍。patience 同步調到 30。

**目前 code 狀態：train.py 仍是 OneCycleLR，v11 需要改掉。**

#### 改動二：Flip Augmentation [H+V flip 已加入 - V-flip 需移除]

**目前 code 狀態（dataset.py HEAD）：** H-flip + V-flip 都在，各 p=0.5。

**v11 需要修改：** 移除 V-flip（dims=[-2]），只保留 H-flip：

```python
# 目前 code（需修改）
if random.random() < 0.5:
    input_tensor  = torch.flip(input_tensor,  dims=[-1])   # H-flip: 保留
    target_tensor = torch.flip(target_tensor, dims=[-1])
if random.random() < 0.5:
    input_tensor  = torch.flip(input_tensor,  dims=[-2])   # V-flip: 移除
    target_tensor = torch.flip(target_tensor, dims=[-2])
```

input 和 target 必須同步翻轉，只在 train set 使用。

**理由：** H-flip 物理合理（降水場東西方向無偏好），零成本。V-flip 破壞 FiLM 時間條件的地理語意（見一之一 次因3）。

#### 改動三：EMA（Exponential Moving Average）[已加入 decay=0.995 - 需調整為 0.999]

**目前 code 狀態（train.py HEAD）：** 手動 EMA 已實作（apply/restore 模式），decay 在 v10 期間從 0.999 改為 0.995（commit 80dd257）。

**v11 需要調整：** decay 改回 0.999，更保守，EMA shadow 追蹤更長期的平均。

```python
# 目前 code：decay=0.995（預設 --ema_decay 0.995）
# v11 需改為：
ema = EMA(model, decay=0.999)
```

**EMA 實作（目前 code 的手動版，實作正確）：**
```python
# train.py 的 EMA 類別已有 apply / restore / state_dict 方法
# Validation: ema.apply(model) -> eval -> ema.restore(model)
# Checkpoint: 存 ema.state_dict()，inference 時 load 後 apply
```

**理由：** val RMSE 有 epoch-to-epoch 震盪，EMA 平滑化 weights 得到更穩定的 checkpoint。decay=0.995 對高 LR 期追蹤太快（等效 200 steps），decay=0.999 追蹤更長期。SpaceNet8 + GlobalMetNet 雙重確認有效。

#### 改動四：Focal Loss（核心突破）[已嘗試 - 結論尚未確定]

> **v10 教訓：** v10 中 Focal Loss 和 OneCycleLR 同時上，無法判斷 Focal 本身的效果。Focal 的結構性問題（最後 bin 期望值 floor）確實存在，但真正能否突破 v8a 仍未被乾淨驗證。v11 先用 regression 框架 + 修正 scheduler，確認 baseline 回到 ~1.14 後，再單獨測試 Focal 是否有效。

**核心思想：** 把 per-pixel regression 問題改成 per-pixel classification 問題。原來直接預測 mm/hr 連續值，改成預測「落在哪個降水強度區間的機率」，推論時用期望值轉回連續值。**不使用 importance sampling**（FL 的 alpha_c 和 sampling 都在修正 class imbalance，同時使用是雙重矯正，會重演 v6 問題）。

**Bin 邊界（10 個等級，對數間距）：**

先從訓練集 label 計算第 99.9 百分位數（記為 max_val，通常在 30-50 mm/hr）。

```
edges = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, max_val]
bin_center = [0.0, 0.15, 0.3, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, (25.6 + max_val) / 2]
```

GENESIS 直接在 IMERG 資料上驗證了 10 log-bins；比 SatFormer 的 64 uniform bins 更穩定（per-pixel 時 64 bins 每 bin 樣本密度過稀疏）。

**預先計算 alpha（inverse frequency weighting）：**

```python
freq = [0] * 10
for label_path in train_label_paths:
    label = load_label(label_path)  # (41, 41) numpy array
    for pixel_val in label.flatten():
        bin_idx = np.searchsorted(edges[1:], pixel_val)
        bin_idx = min(bin_idx, 9)
        freq[bin_idx] += 1

total = sum(freq)
freq_norm = [f / total for f in freq]
alpha = [1.0 / (f + 1e-6) for f in freq_norm]
alpha_sum = sum(alpha)
alpha = [a / alpha_sum for a in alpha]  # normalize to sum = 1
```

**Model Output Head 改動：**

```python
# 原來
self.head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
out = F.relu(self.head(x))
out = F.interpolate(out, size=(41, 41), mode='bilinear', align_corners=False)
return out  # (B, 1, 41, 41)

# 改成
self.head = nn.Conv2d(decoder_channels, 10, kernel_size=1)
logits = self.head(x)  # 不加 ReLU
logits = F.interpolate(logits, size=(41, 41), mode='bilinear', align_corners=False)
return logits  # (B, 10, 41, 41)
```

推論時：
```python
probs = F.softmax(logits, dim=1)  # (B, 10, 41, 41)
bin_center_t = torch.tensor(bin_center, dtype=torch.float32, device=logits.device).view(1, 10, 1, 1)
pred_mm = (probs * bin_center_t).sum(dim=1, keepdim=True).clamp(min=0)  # (B, 1, 41, 41)
```

**Focal Loss 函數：**

```python
def focal_loss_imerg(logits, targets_mm, bin_edges, alpha, gamma=2.0):
    B, _, H, W = logits.shape
    device = logits.device

    targets_flat = targets_mm.view(-1)
    edges_tensor = torch.tensor(bin_edges[1:], device=device)
    targets_bin = torch.bucketize(targets_flat, edges_tensor).clamp(0, 9).view(B, H, W)

    targets_onehot = F.one_hot(targets_bin, num_classes=10).permute(0, 3, 1, 2).float()

    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-8)
    focal_weight = (1.0 - probs) ** gamma

    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device).view(1, 10, 1, 1)
    loss = -alpha_t * focal_weight * targets_onehot * log_probs
    return loss.sum(dim=1).mean()
```

**Training Loop：**

> **AMP 已在 train.py 實作**（`torch.amp.autocast("cuda") + GradScaler`），Focal Loss 框架下已生效，不需要另外補。

```python
# 目前 train.py 的實際寫法（已正確）：
with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
    preds = model(inputs, time_feat)
    loss  = criterion(preds, targets)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()
ema.update(model)
```

**為什麼 Focal Loss 能突破 0.69x：**

GENESIS 在 IMERG 資料的結果：MSE 在輕雨（<1.6 mm/hr）略好，Focal Loss 在重雨（>8 mm/hr）明顯更好（CSI 差距 7-23%）。對 RMSE 而言，重雨 pixel 雖稀少但值大（8-30 mm/hr），其誤差貢獻不成比例地高——Focal Loss 的 gamma=2 focusing term 壓低零值 pixel 的 loss 貢獻，alpha_c 確保重雨梯度不被淹沒。

#### 實驗一的評估標準

對比 v8a（基準：val RMSE 1.1441）：
- val RMSE 下降 >0.01：確認有效，提交 LB 確認
- val RMSE 持平或下降 <0.005：先檢查 EMA 在 validation 時是否正確使用 EMA weights；再確認 bin 邊界 max_val 合理（np.percentile(all_labels, 99.9)），alpha 計算是否正確
- val RMSE 上升：若是 Focal Loss 版本，嘗試 alpha = sqrt(1/freq) 而非線性 inverse，或把 gamma 從 2 降到 1；若只是改了 scheduler/flip/EMA，檢查 flip 的 input/label random state 是否一致

---

### 實驗二：Ensemble（最終得分保底）

**時機：** 實驗一的最佳設定確認後，用 3-5 個不同 random seed 訓練相同模型。

**Seeds：**
```python
seeds = [42, 1337, 2024, 777, 314]
```

**Ensemble 策略（Focal Loss 版）：**

在 softmax 之前平均 logits（比在 mm/hr 空間平均更穩定，softmax 是非線性的）：

```python
avg_logits = sum(logits_list) / len(logits_list)  # (B, 10, 41, 41)
probs = F.softmax(avg_logits, dim=1)
pred_mm = (probs * bin_center_t).sum(dim=1, keepdim=True)
```

**理由：** SpaceNet8 的最終提交是 35 個模型的集成。3-5 seeds 的 ensemble 通常比單模型好 0.5-2% RMSE。在 #1 和 #9 只差 0.0131 的情況下，這個差距可能就是排名的決定性因素。

---

### v11 Step 1：恢復 baseline（當前最優先）

**需修改的 code（尚未 push）：**

1. `train.py`：scheduler 換回 ReduceLROnPlateau
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)
# 並在 val 後呼叫：scheduler.step(val_rmse)
```

2. `train.py`：EMA decay 預設改回 0.999
```python
parser.add_argument("--ema_decay", type=float, default=0.999)
```

3. `dataset.py`：移除 V-flip（保留 H-flip）
```python
# 移除這兩行：
# if random.random() < 0.5:
#     input_tensor  = torch.flip(input_tensor,  dims=[-2])
#     target_tensor = torch.flip(target_tensor, dims=[-2])
```

**執行指令（修完 push 後）：**
```bash
cd ~/solafune/code && git pull origin main && \
python src/train.py \
  --data_dir ~/solafune/data \
  --csv_train ~/solafune/data/train_dataset.csv \
  --epochs 100 --batch_size 32 --num_workers 4 \
  --loss_type combined --run_name v11_baseline && \
curl -s -d "v11_baseline done!" ntfy.sh/solafune_luiz_train
```

**目標：** val RMSE 回到 ~1.14（v8a 水準）。若達到，說明 v10 問題確認，可進入 Step 2。

### v11 Step 2：在乾淨 baseline 上測試 Focal Loss

Step 1 確認後，僅改 loss_type：
```bash
--loss_type focal --run_name v11_focal_clean
```
其餘設定（ReduceLROnPlateau + H-flip only + EMA 0.999）不變，才能乾淨評估 Focal 是否有效。

**Ensemble 策略（回歸版，若 FL 無效）：**
```python
pred_mm = sum(pred_list) / len(pred_list)
```

---

## 四、實驗時間線（2026-06-20 更新）

```
v8a 現況（LB 0.6979, val 1.1441）
 |
 +--> [DONE - 失敗] 實驗一：OneCycleLR + H/V-flip + EMA + Focal Loss（v10）
      val RMSE 最差；拿掉 FL 後仍 >1.3
      根因：OneCycleLR 太激進 + V-flip 語意矛盾 + Focal bin 精度 floor
           |
           v
 +--> [v11 - 下一步] 先恢復 baseline：
      ReduceLROnPlateau(patience=5) + H-flip only + EMA(0.999) + combined loss
      目標：確認 val RMSE 回到 ~1.14（v8a 水準）
           |
           +--> val 回到 ~1.14 -> 確認 OneCycleLR 是主因
           |       |
           |       +--> 單獨測試 Focal Loss（在穩定 baseline 上加）
           |       |       +--> FL 有效 -> 提交 LB
           |       |       +--> FL 無效 -> Ensemble（3-5 seeds，baseline 框架）
           |
           +--> val 仍 >1.3 -> 還有其他未知問題，需要進一步診斷
```

---

## 五、每次實驗的重要注意事項

### Val vs LB 的解讀

我們的 val RMSE（~1.14）和 LB RMSE（0.6979）有接近 2 倍的差距。這不完全是地理分布差異，可能也和 LB 的計算（不同分辨率、不同後處理）有關。

因此：
- Val RMSE 的相對變化（下降 or 上升）是可信的判斷基準
- Val RMSE 的絕對值與 LB 沒有直接可比性
- 每次 val RMSE 改善超過 0.01 後，都值得提交一次 LB 確認，不要等所有實驗跑完

### 必須保留的設計

- **FiLM Day+Hour conditioning 必須保留。** NPM 論文從完全獨立的資料集和任務確認：Day+Hour 是最重要的單一組件。所有後續實驗都必須保持 FiLM。
- **3 frames x 16 bands 的全 51ch 輸入必須保留。** GlobalMetNet 確認衛星輸入是最重要的輸入類型，刪減 bands 只會讓性能下降。

### 絕對不做的事

- BTD 特徵：已驗證有害（val RMSE 從 1.14 升到 1.28）
- 同時改 sampling 和 loss weighting（v6 的失敗）
- ReduceLROnPlateau(patience=3)：文獻最激進設定，太早降 LR——**v11 用 patience=5 的 ReduceLROnPlateau 作為過渡，確認 baseline 後再換 CosineAnnealing**
- **OneCycleLR：v10 確認，max_lr=1e-3 對 pretrained EfficientNet-B4 過於激進，破壞 ImageNet prior**
- **V-flip（垂直翻轉）：破壞 FiLM 時間條件的地理語意，只用 H-flip**
- Event-based 訓練（只在有雨事件上訓練）：RMSE 指標對全部 pixel 計算，event-based 訓練會造成大量 false positive，顯著提高 RMSE
- Local 用完整訓練集：只用 train_smoke.csv（102 rows）做 sanity check

---

## 六、決策摘要

| 決策 | 選擇 | 最重要的理由 |
|------|------|-----------|
| Scheduler（v11 起） | **ReduceLROnPlateau(patience=5)** | v10 確認 OneCycleLR 破壞 pretrained prior；先恢復 baseline 再考慮 CosineAnnealing |
| Scheduler（實驗一原設計） | ~~CosineAnnealingWarmRestarts~~ | **已改為先用 ReduceLROnPlateau 確認 baseline** |
| Flip augmentation | **H-flip only** | v10 確認 V-flip 破壞 FiLM 時間語意；H-flip 仍合理且零成本 |
| EMA | 加入 decay=0.999 | SpaceNet8 + GlobalMetNet 雙重確認；OneCycleLR 下效果差，保守 scheduler 下應正常 |
| Importance sampling | 不使用 | FL 的 alpha_c 已處理 class imbalance；再加 sampling 是雙重矯正（v6 前車之鑑） |
| Loss 框架（v11 第一步） | **combined regression（0.7*MSE+0.3*MAE）** | 先恢復 baseline，再單獨評估 Focal 效果 |
| Loss 框架（後續） | Focal Loss（10 log-bins）| 回歸框架邊際報酬耗盡（v9d 確認）；GENESIS 在 IMERG 上直接驗證；v10 中因 scheduler 問題效果未被乾淨驗證 |
| OneCycleLR | **絕對不用** | v10 確認，max_lr=1e-3 對 EfficientNet-B4 pretrained 過激 |
| Bins 類型 | Log-spaced（10 bins）| IMERG 對數尺度分布；比 64 uniform bins 更穩定 |
| Backbone | 維持 B4 | 先修 scheduler/loss 框架，模型大小是次要問題 |
| Ensemble | 最後做，3-5 seeds | 最穩定的最終得分提升 |
| BTD | 絕對不加 | v9c 已驗證有害（val 從 1.14 升到 1.28） |
| Event-based 訓練 | 絕對不做 | RMSE 指標對全部 pixel 計算，event-based 訓練造成大量 false positive |
| FiLM conditioning | 保留 | NPM 確認是最重要單一組件 |

---

## 七、參考文獻

| 論文 | 關鍵貢獻 | 詳細分析文件 |
|------|---------|------------|
| GENESIS（arXiv 2307.10843）| IMERG 上 MSE vs Focal Loss；10 log-bins；Adam lr=1e-3 step decay | method/paper_UNetConvLSTM_IMERG.md |
| GlobalMetNet（arXiv 2510.13050）| GPM CORRA 目標；30 bins；Polyak/EMA；衛星輸入最重要 | method/paper_GlobalMetNet.md |
| NPM（arXiv 2412.11480）| Day+Hour 是最重要單一組件（= 我們的 FiLM）| method/paper_NPM_SatelliteNowcasting.md |
| RainAI（arXiv 2311.18398）| Importance sampling +60%，只動 sampling 不動 loss | method/weather4cast_RainAI_paper.md |
| SaTformer（arXiv 2511.11090）| 64 bins + class-weighted CE；固定 lr=1e-5 | method/weather4cast_SatFormer_paper.md |
| SpaceNet8 5th（Motoki Kimura）| Step LR；EMA；Backbone B5；Ensemble | method/spacenet8_5th_place_README.md |
| TUPANN（arXiv 2511.05471）| Event-based 訓練對 RMSE 有害（反面教材）| method/paper_TUPANN.md |
| 跨方法整合 | 所有方法的統一對比和建議 | method/competition_methods_synthesis.md |
