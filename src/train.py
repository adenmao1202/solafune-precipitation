"""
Week 1 Baseline 訓練腳本。

使用方式：
  python train.py --data_dir /path/to/data --csv_train train.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PrecipDataset, get_device, parse_filenames, SATELLITE_SUBDIR
from model import build_model


# ---------------------------------------------------------------------------
# Step 0: 先跑這個函數，統計訓練集各衛星各波段的 mean/std，存成 stats.json
# ---------------------------------------------------------------------------
def compute_stats(csv_path: Path, data_dir: Path, out_path: Path, max_samples: int = 0):
    import pandas as pd
    import rasterio

    df = pd.read_csv(csv_path)
    if max_samples > 0:
        df = pd.concat([
            g.sample(min(len(g), max_samples // 3), random_state=42)
            for _, g in df.groupby("satellite_target")
        ]).reset_index(drop=True)
    accum = {}  # satellite -> {band_idx -> [values]}

    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing stats"):
        sat = row["satellite_target"]
        filenames = parse_filenames(str(row["last_30_minutes_observation_filename"]))
        for fname in filenames:
            with rasterio.open(data_dir / SATELLITE_SUBDIR[str(sat)] / fname) as src:
                arr = src.read().astype(np.float32)  # (16, H, W)
            if sat not in accum:
                accum[sat] = [[] for _ in range(arr.shape[0])]
            for b in range(arr.shape[0]):
                # 用 subsample 避免記憶體爆炸
                accum[sat][b].append(arr[b].ravel()[::10])

    stats = {}
    for sat, bands in accum.items():
        all_means, all_stds = [], []
        for b_vals in bands:
            v = np.concatenate(b_vals)
            all_means.append(float(v.mean()))
            all_stds.append(float(v.std()))
        stats[sat] = {"mean": all_means, "std": all_stds}

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {out_path}")
    return stats


# ---------------------------------------------------------------------------
# 損失函數：MSE + MAE 組合（對降水的偏態分布更穩健）
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) \
             + (1 - self.mse_weight) * self.mae(pred, target)


# ---------------------------------------------------------------------------
# 主訓練迴圈
# ---------------------------------------------------------------------------
def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # 1. 載入 stats（若不存在則先計算）
    stats_path = Path(args.data_dir) / "stats.json"
    if not stats_path.exists():
        print("stats.json not found, computing...")
        stats = compute_stats(Path(args.csv_train), Path(args.data_dir), stats_path,
                              max_samples=args.stats_max_samples)
    else:
        with open(stats_path) as f:
            stats = json.load(f)

    # 2. Dataset
    from tqdm import tqdm
    print("Loading dataset...")
    input_size = (args.input_size, args.input_size) if args.input_size else None
    full_ds = PrecipDataset(
        csv_path=Path(args.csv_train),
        data_dir=Path(args.data_dir),
        stats=stats,
        is_train=True,
        input_size=input_size,
    )
    n_val   = int(len(full_ds) * 0.1)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    print(f"Dataset ready: {n_train} train / {n_val} val samples")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # 3. Model
    print("Building model...")
    model = build_model(encoder_name=args.encoder).to(device)
    print("Model ready.")

    # 4. Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = CombinedLoss(mse_weight=0.7)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for inputs, targets, _ in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(inputs)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # --- Validate ---
        model.eval()
        sq_errors = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
            for inputs, targets, _ in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = model(inputs)
                # expm1 還原到原始降水空間再計算 RMSE
                # clamp max=8: expm1(8)~2981 mm/hr, 防止未收斂模型數值溢位
                preds_real   = torch.expm1(preds.float().clamp(0, 8))
                targets_real = torch.expm1(targets.float())
                sq = (preds_real - targets_real) ** 2
                sq_errors.append(sq[torch.isfinite(sq)].cpu().numpy().ravel())

        val_rmse = float(np.sqrt(np.concatenate(sq_errors).mean()))
        avg_train = train_loss / len(train_loader)
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (RMSE={best_val_rmse:.4f})")

    print(f"\nTraining done. Best val RMSE: {best_val_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--csv_train",  required=True)
    parser.add_argument("--encoder",    default="efficientnet-b4")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers",       type=int, default=0)
    parser.add_argument("--input_size",        type=int, default=128,
                        help="Resize all satellite inputs to (N×N). Required to batch mixed satellites.")
    parser.add_argument("--stats_max_samples", type=int, default=0,
                        help="Max rows for stats computation (0=all). Use ~300 for smoke test.")
    args = parser.parse_args()
    train(args)
