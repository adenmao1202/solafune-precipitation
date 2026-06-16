"""
Day 1 EDA 腳本：快速了解資料分布。
執行後會印出關鍵統計並存圖。

  python eda.py --data_dir /path/to/data --csv_train train.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def run_eda(args):
    data_dir = Path(args.data_dir)
    df = pd.read_csv(args.csv_train)

    print(f"Total samples : {len(df)}")
    print(f"Satellites    : {df['satellite_target'].value_counts().to_dict()}")
    print(f"Locations     : {df['name_location'].nunique()} unique")
    print(f"Date range    : {df['datetime'].min()} ~ {df['datetime'].max()}")

    # --- 降水值分布 ---
    precip_vals = []
    for _, row in df.sample(min(200, len(df)), random_state=0).iterrows():
        fname = row["gpm_imerg_filename"]
        with rasterio.open(data_dir / fname) as src:
            arr = src.read(1).ravel()
        precip_vals.append(arr)

    precip = np.concatenate(precip_vals)
    print(f"\nPrecipitation stats (mm/hr):")
    print(f"  zero fraction : {(precip == 0).mean():.2%}")
    print(f"  mean          : {precip.mean():.4f}")
    print(f"  std           : {precip.std():.4f}")
    print(f"  max           : {precip.max():.4f}")
    print(f"  99th pct      : {np.percentile(precip, 99):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(precip[precip > 0], bins=100, log=True)
    axes[0].set_title("Precipitation > 0 (raw)")
    axes[0].set_xlabel("mm/hr")
    axes[1].hist(np.log1p(precip[precip > 0]), bins=100, log=True)
    axes[1].set_title("Precipitation > 0 (log1p)")
    plt.tight_layout()
    plt.savefig("eda_precip_dist.png", dpi=120)
    print("\nSaved: eda_precip_dist.png")

    # --- 衛星影像範例可視化 ---
    row = df.iloc[0]
    fname = str(row["last_30_minutes_observation_filename"]).split(";")[0].strip()
    with rasterio.open(data_dir / fname) as src:
        img = src.read()   # (16, H, W)
    print(f"\nSample image shape: {img.shape}")
    print(f"Band value range: [{img.min():.2f}, {img.max():.2f}]")

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for b in range(16):
        ax = axes[b // 8][b % 8]
        ax.imshow(img[b], cmap="gray")
        ax.set_title(f"B{b+1}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Satellite: {row['satellite_target']} | All 16 bands")
    plt.tight_layout()
    plt.savefig("eda_bands.png", dpi=120)
    print("Saved: eda_bands.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--csv_train", required=True)
    args = parser.parse_args()
    run_eda(args)
