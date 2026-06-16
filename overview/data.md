# 資料集說明（Data）

## 資料來源概覽

| 衛星 / 資料集 | 機構 | 授權 |
|---|---|---|
| Himawari-8/9 | JMA（日本氣象廳） | Government of Japan Standard Terms of Use v2.0 |
| GOES（GOES-East/West） | NOAA | Creative Commons CC0 1.0 Universal |
| Meteosat | EUMETSAT | EUMETSAT Data Policy |
| GPM-IMERG | NASA / JAXA | Free and Open Archive |

---

## 各資料集格式說明

### 1. Himawari-8/9（輸入特徵）
- 儀器：Multi-Spectral Instrument（MSI），Level-1B
- 波段數：**16 個波段**
  - `B01`, `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B09`, `B10`, `B11`, `B12`, `B13`, `B14`, `B15`, `B16`
- 原始資料為全圓盤影像（Full Disk），已裁切至各地區感興趣區域（ROI）
- 每筆資料有獨立的時間戳記

### 2. GOES（輸入特徵）
- 儀器：Multi-Spectral Instrument（MSI），Level-1B
- 波段數：**16 個波段**
  - `C01`, `C02`, `C03`, `C04`, `C05`, `C06`, `C07`, `C08`, `C09`, `C10`, `C11`, `C12`, `C13`, `C14`, `C15`, `C16`
- 原始資料為全圓盤影像，已裁切至各地區 ROI
- 每筆資料有獨立的時間戳記

### 3. Meteosat（輸入特徵）
- 儀器：Multi-Spectral Instrument（MSI），Level-1C
- 波段數：**16 個波段**
  - `vis_04`, `vis_05`, `vis_06`, `vis_08`, `vis_09`（可見光）
  - `nir_13`, `nir_16`, `nir_22`（近紅外）
  - `ir_38`, `wv_63`, `wv_73`, `ir_87`, `ir_97`, `ir_105`, `ir_123`, `ir_133`（紅外 / 水氣）
- 原始資料為全圓盤影像，已裁切至各地區 ROI
- 每筆資料有獨立的時間戳記

### 4. GPM-IMERG（預測目標）
- 全球多衛星降水資料集，包含校正與未校正降水值
- **本比賽只使用校正值**，波段名稱為 `precipitation`（單波段）
- 已裁切至各地區 ROI
- **此為迴歸目標變數（Target Variable）**

---

## CSV 訓練資料格式

訓練資料以 CSV 格式提供，欄位如下：

| 欄位名稱 | 說明 |
|---|---|
| `data_id` | 每筆資料的唯一識別碼 |
| `name_location` | 地區名稱 |
| `satellite_target` | 使用的衛星名稱（Himawari / GOES / Meteosat） |
| `datetime` | 時間戳記 |
| `last_30_minutes_observation_filename` | 過去 30 分鐘衛星影像檔名（最多 3 張） |
| `gpm_imerg_filename` | GPM-IMERG 降水目標檔案名稱（迴歸目標） |

---

## 提交格式

```
your_submission.zip
├── evaluation_target.csv
└── test_files/
        ├── {location_X}_GPM_IMERG_{datetime_X}.tif
        ├── {location_X}_GPM_IMERG_{datetime_X}.tif
        └── ...
```


---

## 資料特性與挑戰

- 多衛星來源（3 顆不同靜止衛星），各自有不同波段命名與投影方式
- 跨地區泛化：模型需在不同地理區域（亞洲 / 美洲 / 歐洲）均有良好表現
- 影像為時序資料，輸入為過去 30 分鐘最多 3 張影像
- 輸出為像素級降水量預測（空間迴歸）
