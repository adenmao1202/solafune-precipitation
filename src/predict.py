"""
推論腳本：產生比賽提交格式。

使用方式：
  python predict.py --data_dir /path/to/data --csv_test test.csv \
                    --model_path best_model.pth --out_dir ./submission
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PrecipDataset, get_device, GPM_SIZE
from model import build_model


def predict(args):
    device = get_device()

    with open(Path(args.data_dir) / "stats.json") as f:
        stats = json.load(f)

    input_size = (args.input_size, args.input_size) if args.input_size else None
    test_ds = PrecipDataset(
        csv_path=Path(args.csv_test),
        data_dir=Path(args.data_dir),
        stats=stats,
        is_train=False,
        input_size=input_size,
    )
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model = build_model()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    out_dir = Path(args.out_dir) / "test_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(args.csv_test)

    # Read GPM profile from any placeholder in test_files (all are 41×41)
    placeholder_dir = Path(args.data_dir) / "test_files"
    placeholder_path = next(placeholder_dir.glob("*.tif"), None)
    if placeholder_path is None:
        raise FileNotFoundError(f"No placeholder TIF found in {placeholder_dir}")
    with rasterio.open(placeholder_path) as ref:
        gpm_profile = ref.profile.copy()
    gpm_profile.update(count=1, dtype="float32")

    result_rows = []

    with torch.no_grad():
        for inputs, _, unique_ids in loader:
            inputs = inputs.to(device)
            # preds: (B, 1, sat_H, sat_W) — satellite native resolution
            preds = model(inputs)
            # Resize to GPM_SIZE (41×41) for submission
            preds = F.interpolate(
                preds, size=GPM_SIZE, mode="bilinear", align_corners=False
            )
            preds = torch.expm1(preds.clamp(min=0)).cpu().numpy()

            for i, unique_id in enumerate(unique_ids):
                row       = df_test[df_test["unique_id"] == unique_id].iloc[0]
                out_fname = row["gpm_imerg_filename"]
                arr       = preds[i, 0]  # (41, 41)

                with rasterio.open(out_dir / out_fname, "w", **gpm_profile) as dst:
                    dst.write(arr.astype(np.float32), 1)

                result_rows.append({"unique_id": unique_id, "gpm_imerg_filename": out_fname})

    # evaluation_target.csv
    pd.DataFrame(result_rows).to_csv(
        Path(args.out_dir) / "evaluation_target.csv", index=False
    )

    # 打包成 zip
    zip_path = Path(args.out_dir).parent / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in Path(args.out_dir).rglob("*"):
            zf.write(p, p.relative_to(Path(args.out_dir).parent))
    print(f"Submission saved: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--csv_test",   required=True)
    parser.add_argument("--model_path", default="best_model.pth")
    parser.add_argument("--out_dir",    default="./submission")
    parser.add_argument("--input_size", type=int, default=128)
    args = parser.parse_args()
    predict(args)
