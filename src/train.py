"""
Week 1 Baseline 訓練腳本。

使用方式：
  python train.py --data_dir /path/to/data --csv_train train.csv
"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import PrecipDataset, get_device, parse_filenames, SATELLITE_SUBDIR
from model import build_model


def temporal_split(csv_path: Path, val_ratio: float = 0.2):
    """每個地點按時間排序，取最後 val_ratio 為 val，避免 data leakage。"""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["name_location", "datetime"]).reset_index(drop=True)

    train_idx, val_idx = [], []
    for loc in df["name_location"].unique():
        loc_idx = df[df["name_location"] == loc].index.tolist()
        split = int(len(loc_idx) * (1 - val_ratio))
        train_idx.extend(loc_idx[:split])
        val_idx.extend(loc_idx[split:])
    return train_idx, val_idx


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
# 損失函數：CombinedLoss（v3 baseline，所有 pixel 同等權重）
# v4/v6 的加權 loss 實驗均比 v3 差：零值過度修正讓 val RMSE 上升
# 結論：log1p 已足夠壓縮右偏分布，加權 loss 不是目前 LB 瓶頸
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        mse = ((pred - target) ** 2).mean()
        mae = (pred - target).abs().mean()
        return 0.7 * mse + 0.3 * mae


# ---------------------------------------------------------------------------
# 分層取樣：掃描 GPM 檔案標記有雨/無雨，讓訓練集有雨:無雨 = 50:50
# val set 不動，保留全樣本以得到無偏的 RMSE 估計
# ---------------------------------------------------------------------------
def scan_rain_labels(csv_path: Path, data_dir: Path, cache_path: Path) -> dict:
    """回傳 {row_index: True/False}，True 代表該樣本 GPM 有非零降水。結果快取到 cache_path。"""
    import rasterio
    from tqdm import tqdm

    if cache_path.exists():
        df_cache = pd.read_csv(cache_path)
        return dict(zip(df_cache["idx"].tolist(), df_cache["has_rain"].astype(bool).tolist()))

    df = pd.read_csv(csv_path)
    results = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning GPM labels"):
        gpm_path = Path(data_dir) / "gpm_imerg" / row["gpm_imerg_filename"]
        try:
            with rasterio.open(gpm_path) as src:
                results[idx] = bool(src.read(1).max() > 0)
        except Exception:
            results[idx] = False

    pd.DataFrame({"idx": list(results.keys()), "has_rain": list(results.values())}).to_csv(cache_path, index=False)
    print(f"Rain labels cached -> {cache_path}")
    return results


def stratified_sample(train_idx: list, rain_labels: dict, seed: int = 42) -> list:
    """有雨樣本全保留，無雨樣本隨機抽取到與有雨數量相同（50:50）。"""
    rainy = [i for i in train_idx if rain_labels.get(i, False)]
    dry   = [i for i in train_idx if not rain_labels.get(i, False)]
    rng = random.Random(seed)
    dry_sampled = rng.sample(dry, min(len(rainy), len(dry)))
    combined = rainy + dry_sampled
    rng.shuffle(combined)
    print(f"Stratified train: {len(rainy)} rainy + {len(dry_sampled)} dry "
          f"= {len(combined)} total (original dry: {len(dry)})")
    return combined


# ---------------------------------------------------------------------------
# 主訓練迴圈
# ---------------------------------------------------------------------------
def save_experiment(args, best_val_rmse: float, epochs_run: int):
    log_path = Path("experiments.csv")
    fieldnames = ["run_name", "datetime", "epochs_run", "best_val_rmse",
                  "lr", "batch_size", "encoder", "loss_type", "lb_score", "notes"]
    row = {
        "run_name":       args.run_name,
        "datetime":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "epochs_run":     epochs_run,
        "best_val_rmse":  f"{best_val_rmse:.4f}",
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "encoder":        args.encoder,
        "loss_type":      args.loss_type,
        "lb_score":       "",
        "notes":          "",
    }
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Experiment logged -> {log_path}")


def train(args):
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run: {args.run_name}  (output -> {run_dir})")

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
    train_idx, val_idx = temporal_split(Path(args.csv_train), val_ratio=0.2)
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    print(f"Temporal split: {len(train_ds)} train / {len(val_ds)} val")

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=args.lr * 0.01
    )
    criterion = CombinedLoss()  # args.loss_type logged; extend here if adding new losses

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_rmse = float("inf")
    patience_counter = 0
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for inputs, targets, _, time_feat in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            time_feat = time_feat.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(inputs, time_feat)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validate ---
        model.eval()
        sq_errors = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
            for inputs, targets, _, time_feat in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                time_feat = time_feat.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = model(inputs, time_feat)
                # expm1 還原到原始降水空間再計算 RMSE
                # clamp max=8: expm1(8)~2981 mm/hr, 防止未收斂模型數值溢位
                preds_real   = torch.expm1(preds.float().clamp(0, 8))
                targets_real = torch.expm1(targets.float().clamp(0, 8))
                sq = (preds_real - targets_real) ** 2
                sq_errors.append(sq[torch.isfinite(sq)].cpu().numpy().ravel())

        val_rmse = float(np.sqrt(np.concatenate(sq_errors).mean()))
        avg_train = train_loss / len(train_loader)
        scheduler.step(val_rmse)
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            print(f"  -> Saved best model (RMSE={best_val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.early_stop_patience})")
            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nTraining done. Best val RMSE: {best_val_rmse:.4f}")
    save_experiment(args, best_val_rmse, epochs_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--csv_train",  required=True)
    parser.add_argument("--encoder",    default="efficientnet-b4")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers",       type=int, default=0)
    parser.add_argument("--input_size",        type=int, default=128,
                        help="Resize all satellite inputs to (N×N). Required to batch mixed satellites.")
    parser.add_argument("--stats_max_samples", type=int, default=0,
                        help="Max rows for stats computation (0=all). Use ~300 for smoke test.")
    parser.add_argument("--early_stop_patience", type=int, default=7,
                        help="Stop training if val RMSE does not improve for this many epochs.")
    parser.add_argument("--loss_type", default="combined",
                        help="Loss function identifier for logging (e.g. combined, weighted_mse, tiered).")
    parser.add_argument("--run_name", default=datetime.now().strftime("%Y%m%d_%H%M"),
                        help="Experiment name. Output saved to runs/{run_name}/")
    args = parser.parse_args()
    train(args)
