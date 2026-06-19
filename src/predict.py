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
from train import NUM_BINS, BIN_EDGES_FIXED, BIN_CENTERS_FIXED


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

    use_focal = (args.loss_type == "focal")
    num_classes = NUM_BINS if use_focal else 1
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    if use_focal:
        # max_val 用 50.0 作為推論時的保守上界（訓練時 99.9th percentile 通常 30-50）
        bin_edges   = BIN_EDGES_FIXED + [max(args.max_val, 26.0)]
        bin_centers = BIN_CENTERS_FIXED + [(25.6 + bin_edges[-1]) / 2]
        bin_center_t = torch.tensor(bin_centers, dtype=torch.float32, device=device).view(1, NUM_BINS, 1, 1)

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

    from tqdm import tqdm
    with torch.no_grad():
        for inputs, _, unique_ids, time_feat in tqdm(loader, desc="Predicting"):
            inputs    = inputs.to(device)
            time_feat = time_feat.to(device)
            preds = model(inputs, time_feat)
            # Resize to GPM_SIZE (41×41) for submission
            preds = F.interpolate(
                preds, size=GPM_SIZE, mode="bilinear", align_corners=False
            )
            if use_focal:
                probs = F.softmax(preds.float(), dim=1)
                preds = (probs * bin_center_t).sum(dim=1, keepdim=True).clamp(min=0).cpu().numpy()
            else:
                preds = torch.expm1(preds.clamp(min=0)).cpu().numpy()

            for i, unique_id in enumerate(unique_ids):
                row       = df_test[df_test["unique_id"] == unique_id].iloc[0]
                out_fname = row["gpm_imerg_filename"]
                arr       = preds[i, 0]  # (41, 41)

                with rasterio.open(out_dir / out_fname, "w", **gpm_profile) as dst:
                    dst.write(arr.astype(np.float32), 1)

                result_rows.append({"unique_id": unique_id, "gpm_imerg_filename": out_fname})

    # evaluation_target.csv — 保留原始所有欄位，只替換 gpm_imerg_filename
    eval_df = pd.read_csv(args.csv_test)
    result_df = pd.DataFrame(result_rows)
    submission_csv = Path(args.out_dir) / "evaluation_target.csv"
    eval_df[["unique_id", "gpm_imerg_filename"]].merge(
        result_df[["unique_id"]], on="unique_id"
    ).to_csv(submission_csv, index=False)

    # 打包成 zip：evaluation_target.csv 和 test_files/ 在根目錄
    zip_path = Path(args.out_dir).parent / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_csv, "evaluation_target.csv")
        for p in out_dir.glob("*.tif"):
            zf.write(p, Path("test_files") / p.name)
    print(f"Submission saved: {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--csv_test",   required=True)
    parser.add_argument("--model_path", default="best_model.pth")
    parser.add_argument("--out_dir",    default="./submission")
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--loss_type", default="focal",
                        help="focal: Focal Loss checkpoint (10ch); combined: regression checkpoint (1ch).")
    parser.add_argument("--max_val", type=float, default=50.0,
                        help="Max bin edge (mm/hr). Should match training max_val printed during training.")
    args = parser.parse_args()
    predict(args)
