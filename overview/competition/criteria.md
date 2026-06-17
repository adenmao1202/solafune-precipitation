# 評估方法（Evaluation Criteria）

## 主要評估指標：RMSE

**Root Mean Squared Error（均方根誤差）**

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- $y_i$：GPM-IMERG 校正降水量真實值
- $\hat{y}_i$：模型預測降水量
- 分數越低越好
- 目前 Public LB 最高分約 **0.685**

---

## 實驗性指標：Efficiency Score（效率分數）

競賽結束後，Solafune 將對 **前 10 名**的提交模型額外計算效率分數（不影響最終排名，僅作參考）：

$$
\text{Efficiency Score} = \frac{1}{(\text{RMSE} + \varepsilon)^2 \times \text{Inference Time}}
$$

- $\varepsilon$：穩定常數，避免近無限值
- 同時考量精度與推論速度
- 此指標為實驗性質，未來可能正式納入評分

---

## Leaderboard 說明

| 階段 | 資料比例 | 說明 |
|---|---|---|
| Public LB（競賽期間） | 約 35% 評估資料 | 即時顯示排名 |
| Private LB（競賽結束後） | 100% 評估資料 | **最終正式排名** |

---

## 目前 Public LB 排名（參考）

| 名次 | 隊伍 | 最佳分數（RMSE） |
|---|---|---|
| 1 | BlueLock | 0.6848 |
| 2 | Kobugi | 0.7219 |
| 3 | techrend08 | 0.7451 |
| 4 | jalesneves | 0.7462 |
| 5 | ExaltedJoseph | 0.8551 |
| 6 | zhiyan | 0.8964 |

---

## 是否需要大型 GPU？

**結論：建議使用 GPU，但不一定需要頂級大型 GPU。**

分析如下：

| 面向 | 說明 |
|---|---|
| **任務性質** | 像素級空間迴歸（每張 GeoTIFF 預測降水分布），類似氣象降尺度或影像對影像任務 |
| **輸入規模** | 每個樣本最多 3 張衛星影像 × 16 波段，影像解析度視地區而定（通常數十至數百像素） |
| **模型選擇空間** | 從輕量 CNN（UNet）到 Transformer（SwinIR、ViT）皆可，後者較吃記憶體 |
| **官方規定** | 要求使用 **CUDA 11.8+** 環境（若使用 NVIDIA GPU），暗示 GPU 為預期使用環境 |
| **推論效率** | 效率分數中含推論時間，鼓勵輕量高效模型 |
| **無外部資料限制** | 禁止使用外部資料，無法直接用大型預訓練氣象模型（如 Pangu-Weather）的外部訓練集 |

**實務建議**：
- 單張 RTX 3090 / A100（24GB VRAM）即可應付大多數有競爭力的方案
- 若使用 UNet 系架構，消費級 GPU（8~16GB）也足夠
- 若要嘗試 ViT / Transformer 結合多衛星多時序輸入，建議 16GB+ VRAM
- 重點在於**跨地區泛化能力**與**特徵工程**，而非單純堆疊模型規模
