"""
推論腳本：產生比賽提交格式。

最簡單的用法（從 run_name 自動讀取所有訓練設定）：
  python predict.py --run_name v12_focal_14bins \
                    --data_dir ~/solafune/data \
                    --csv_test ~/solafune/data/test_dataset.csv

手動指定（不需要 run_name）：
  python predict.py --data_dir /path/to/data --csv_test test.csv \
                    --model_path runs/v12_focal_14bins/best_model.pth \
                    --loss_type focal --band_selection all
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

from dataset import PrecipDataset, get_device, GPM_SIZE, IN_CHANNELS_12, IN_CHANNELS_18, COND_DIM
from model import build_model


def center_crop_to_gpm(t: torch.Tensor) -> torch.Tensor:
    """Center-crop U-Net output to GPM_SIZE without interpolation."""
    top  = (t.shape[-2] - GPM_SIZE[0]) // 2
    left = (t.shape[-1] - GPM_SIZE[1]) // 2
    return t[:, :, top:top + GPM_SIZE[0], left:left + GPM_SIZE[1]]


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
        band_mode=args.band_mode,
    )
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    use_focal = (args.loss_type == "focal")
    in_channels = IN_CHANNELS_18 if args.band_mode == "18slot" else IN_CHANNELS_12

    if use_focal:
        focal_cfg_path = Path(args.model_path).parent / "focal_config.json"
        if not focal_cfg_path.exists():
            raise FileNotFoundError(f"focal_config.json not found at {focal_cfg_path}. "
                                    "Run training first or use --loss_type=combined.")
        with open(focal_cfg_path) as f:
            focal_cfg = json.load(f)
        bin_centers = focal_cfg["bin_centers"]
        n_bins = len(bin_centers)
        bin_center_t = torch.tensor(bin_centers, dtype=torch.float32, device=device).view(1, n_bins, 1, 1)
        num_classes = n_bins
    else:
        num_classes = 1
        bin_center_t = None

    model = build_model(num_classes=num_classes, in_channels=in_channels, cond_dim=COND_DIM)

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

    from tqdm import tqdm
    with torch.no_grad():
        for inputs, _, unique_ids, time_feat in tqdm(loader, desc="Predicting"):
            inputs    = inputs.to(device)
            time_feat = time_feat.to(device)
            preds = model(inputs, time_feat)
            # Center-crop to GPM_SIZE (41×41) — no interpolation
            preds = center_crop_to_gpm(preds)
            if use_focal:
                assert bin_center_t is not None
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
    # Convenience: auto-load all settings from a previous run
    parser.add_argument("--run_name", default=None,
                        help="Load loss_type/band_selection/input_size from runs/<run_name>/args.json. "
                             "Also sets model_path and out_dir automatically.")
    parser.add_argument("--data_dir",  required=True,
                        help="Data directory (must contain stats.json and test_files/).")
    parser.add_argument("--csv_test",  required=True,
                        help="Path to test_dataset.csv.")
    # Manual overrides (ignored when --run_name is given, unless explicitly set)
    parser.add_argument("--model_path",     default=None)
    parser.add_argument("--out_dir",        default=None)
    parser.add_argument("--input_size",     type=int, default=None)
    parser.add_argument("--band_mode", default=None, choices=["12slot", "18slot"])
    parser.add_argument("--loss_type",      default=None)
    args = parser.parse_args()

    # If run_name given, load training args and fill in defaults
    if args.run_name:
        run_dir  = Path("runs") / args.run_name
        args_path = run_dir / "args.json"
        if not args_path.exists():
            parser.error(f"args.json not found: {args_path}")
        with open(args_path) as f:
            train_args = json.load(f)
        if args.model_path is None:
            args.model_path = str(run_dir / "best_model.pth")
        if args.out_dir is None:
            args.out_dir = str(run_dir / "submission")
        if args.input_size is None:
            args.input_size = train_args.get("input_size", 128)
        if args.band_mode is None:
            args.band_mode = train_args.get("band_mode", "12slot")
        if args.loss_type is None:
            args.loss_type = train_args.get("loss_type", "combined")
        print(f"Loaded from {args_path}:")
        print(f"  loss_type={args.loss_type}  band_mode={args.band_mode}  "
              f"input_size={args.input_size}")
        print(f"  model_path={args.model_path}")
    else:
        # Apply plain defaults when no run_name
        if args.model_path is None:
            args.model_path = "best_model.pth"
        if args.out_dir is None:
            args.out_dir = "./submission"
        if args.input_size is None:
            args.input_size = 128
        if args.band_mode is None:
            args.band_mode = "12slot"
        if args.loss_type is None:
            args.loss_type = "combined"

    predict(args)
