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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import (PrecipDataset, get_device, parse_filenames, SATELLITE_SUBDIR,
                     GPM_SIZE, IN_CHANNELS, IR_CHANNELS)
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


def location_holdout_split(csv_path: Path, holdout_locations: list[str]):
    """固定地點 holdout：holdout_locations 的所有樣本作為 val，其餘訓練。
    比 temporal split 更誠實：train/test 地點不重疊，val 應該也不重疊。"""
    df = pd.read_csv(csv_path)
    val_mask = df["name_location"].isin(holdout_locations)
    train_idx = df[~val_mask].index.tolist()
    val_idx   = df[val_mask].index.tolist()
    train_locs = sorted(df[~val_mask]["name_location"].unique())
    val_locs   = sorted(df[val_mask]["name_location"].unique())
    print(f"Location holdout split: {len(train_locs)} train locs / {len(val_locs)} val locs")
    print(f"  Train: {train_locs}")
    print(f"  Val:   {val_locs}")
    print(f"  Samples: {len(train_idx)} train / {len(val_idx)} val")
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
# Focal Loss for IMERG precipitation bins (GENESIS 10 log-spaced bins)
# v8a regression (0.7*MSE+0.3*MAE) confirmed ceiling; classification framework
# needed to break through 80% zero-pixel gradient dominance.
# ---------------------------------------------------------------------------
NUM_BINS = 14  # dynamic log-spaced bins generated at runtime from training max_val


def make_log_bins(max_val: float, num_bins: int = NUM_BINS):
    """Generate num_bins log-spaced bins from 0.1 to max_val.
    Returns (edges, centers): len(edges)=num_bins+1, len(centers)=num_bins.
    centers[0]=0 (dry bin), centers[i]=geometric mean of edges[i]..edges[i+1]."""
    edges = [0.0] + list(np.logspace(np.log10(0.1), np.log10(max(max_val, 0.2)), num_bins))
    centers = [0.0]
    for i in range(1, num_bins):
        centers.append(float(np.sqrt(edges[i] * edges[i + 1])))
    return edges, centers


class FocalLossIMERG(nn.Module):
    def __init__(self, bin_edges: list, alpha: list, gamma: float = 2.0):
        super().__init__()
        self.edges_list = bin_edges[1:]  # exclude leading 0
        self.alpha_list = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets_log1p: torch.Tensor) -> torch.Tensor:
        """logits: (B, n_bins, H, W); targets_log1p: (B, 1, H, W) in log1p(mm/hr)"""
        device = logits.device
        n_bins = len(self.alpha_list)
        edges = torch.tensor(self.edges_list, dtype=torch.float32, device=device)
        alpha_t = torch.tensor(self.alpha_list, dtype=torch.float32, device=device).view(1, n_bins, 1, 1)

        targets_mm = torch.expm1(targets_log1p.float().clamp(0, 8)).squeeze(1)  # (B, H, W)
        B, H, W = targets_mm.shape

        targets_bin = torch.bucketize(targets_mm.reshape(-1), edges).clamp(0, n_bins - 1).view(B, H, W)
        targets_onehot = F.one_hot(targets_bin, num_classes=n_bins).permute(0, 3, 1, 2).float()

        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        focal_weight = (1.0 - probs) ** self.gamma

        loss = -alpha_t * focal_weight * targets_onehot * log_probs
        return loss.sum(dim=1).mean()


# Regression fallback — kept for --loss_type=combined experiments
class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        mse = ((pred - target) ** 2).mean()
        mae = (pred - target).abs().mean()
        return 0.7 * mse + 0.3 * mae  # v8a setting


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) — manual implementation, no extra deps.
# Validation and checkpoint both use EMA weights.
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        self._backup = {k: v.clone().detach() for k, v in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            param.data.copy_(self._backup[name])

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict):
        self.shadow = {k: v.clone() for k, v in state.items()}


# ---------------------------------------------------------------------------
# Compute inverse-frequency alpha for Focal Loss bins.
# Scans all training GPM files once; results are printed for verification.
# ---------------------------------------------------------------------------
def compute_bin_alpha(train_idx: list, full_ds, cache_path: Path | None = None):
    """Two-pass scan: pass1 finds max_val, generates log-spaced bins; pass2 counts freq.
    Returns (alpha, bin_edges, bin_centers). Result cached to cache_path if provided."""
    import rasterio
    from tqdm import tqdm

    # Load from cache if available (same data → same result)
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("num_bins") == NUM_BINS:
            print(f"Loaded bin alpha from cache: {cache_path}")
            print(f"max_val={cached['bin_edges'][-1]:.2f} mm/hr")
            print(f"Bin alpha: {[f'{a:.4f}' for a in cached['alpha']]}")
            return cached["alpha"], cached["bin_edges"], cached["bin_centers"]

    # Pass 1: find max_val
    max_val = 0.0
    for idx in tqdm(train_idx, desc="Pass1: max_val", leave=False):
        row = full_ds.df.iloc[idx]
        gpm_path = full_ds.data_dir / "gpm_imerg" / row["gpm_imerg_filename"]
        try:
            with rasterio.open(gpm_path) as src:
                arr = src.read(1).astype(np.float32)
            max_val = max(max_val, float(arr.max()))
        except Exception:
            pass

    bin_edges, bin_centers = make_log_bins(max_val)
    edges_arr = np.array(bin_edges[1:], dtype=np.float32)

    # Pass 2: count frequency per bin
    freq = np.zeros(NUM_BINS, dtype=np.float64)
    for idx in tqdm(train_idx, desc="Pass2: bin freq", leave=False):
        row = full_ds.df.iloc[idx]
        gpm_path = full_ds.data_dir / "gpm_imerg" / row["gpm_imerg_filename"]
        try:
            with rasterio.open(gpm_path) as src:
                arr = src.read(1).astype(np.float32)
            bin_idx = np.searchsorted(edges_arr, arr.ravel()).clip(0, NUM_BINS - 1)
            np.add.at(freq, bin_idx, 1)
        except Exception:
            pass

    freq_norm = freq / (freq.sum() + 1e-8)

    # Power scaling (exponent=0.5) prevents near-empty bins from dominating.
    # 1/freq gives bin13 (~0% freq) an alpha 200,000x larger than light-rain bins.
    # sqrt(1/freq) compresses that ratio to ~450x, then clipping to 50x keeps it sane.
    alpha_raw = 1.0 / np.sqrt(freq_norm + 1e-6)
    # Zero out bins with essentially no samples (< 0.001% of pixels)
    alpha_raw[freq_norm < 1e-4] = 0.0
    # Clip to 50x the lightest non-dry rain bin to prevent any single bin dominating
    rain_bins = alpha_raw[1:]  # exclude dry bin
    if rain_bins[rain_bins > 0].size > 0:
        cap = float(rain_bins[rain_bins > 0].min()) * 50
        alpha_raw = np.minimum(alpha_raw, cap)
    alpha = (alpha_raw / (alpha_raw.sum() + 1e-8)).tolist()

    print(f"max_val={max_val:.2f} mm/hr | edges(first 5): {[f'{e:.3f}' for e in bin_edges[:6]]}")
    print(f"Bin freq%: {[f'{f*100:.2f}' for f in freq_norm]}")
    print(f"Bin alpha: {[f'{a:.4f}' for a in alpha]}")

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump({"num_bins": NUM_BINS, "alpha": alpha,
                       "bin_edges": bin_edges, "bin_centers": bin_centers}, f)
        print(f"Bin alpha cached -> {cache_path}")

    return alpha, bin_edges, bin_centers


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
                  "lr", "batch_size", "encoder", "loss_type", "band_selection",
                  "lb_score", "notes"]
    row = {
        "run_name":       args.run_name,
        "datetime":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "epochs_run":     epochs_run,
        "best_val_rmse":  f"{best_val_rmse:.4f}",
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "encoder":        args.encoder,
        "loss_type":      args.loss_type,
        "band_selection": getattr(args, "band_selection", "all"),
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
    # Expand ~ in paths immediately so all downstream code sees absolute paths
    args.data_dir  = str(Path(args.data_dir).expanduser().resolve())
    args.csv_train = str(Path(args.csv_train).expanduser().resolve())

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
    band_selection = getattr(args, "band_selection", "all")
    if band_selection == "all":
        band_selection = None
    input_size = (args.input_size, args.input_size) if args.input_size else None
    full_ds = PrecipDataset(
        csv_path=Path(args.csv_train),
        data_dir=Path(args.data_dir),
        stats=stats,
        is_train=True,
        input_size=input_size,
        band_selection=band_selection,
    )
    if args.val_mode == "holdout":
        holdout_locs = [s.strip() for s in args.holdout_locations.split(",")]
        train_idx, val_idx = location_holdout_split(Path(args.csv_train), holdout_locs)
    else:
        train_idx, val_idx = temporal_split(Path(args.csv_train), val_ratio=0.2)
        print(f"Temporal split: {len(train_idx)} train / {len(val_idx)} val")
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # 3. Model + Loss
    use_focal = (args.loss_type == "focal")
    num_classes = NUM_BINS if use_focal else 1
    in_channels = IR_CHANNELS if band_selection == "ir_split_window" else IN_CHANNELS
    print(f"Building model (in_channels={in_channels}, num_classes={num_classes})...")
    model = build_model(encoder_name=args.encoder, num_classes=num_classes,
                        in_channels=in_channels).to(device)
    print("Model ready.")

    if use_focal:
        print("Computing Focal Loss bin frequencies (14 dynamic log-spaced bins)...")
        bin_cache = Path(args.data_dir) / f"focal_alpha_cache_{args.val_mode}.json"
        alpha, bin_edges, bin_centers = compute_bin_alpha(train_idx, full_ds, cache_path=bin_cache)
        criterion = FocalLossIMERG(bin_edges=bin_edges, alpha=alpha, gamma=args.gamma)
        bin_center_t = torch.tensor(bin_centers, dtype=torch.float32, device=device).view(1, NUM_BINS, 1, 1)
        with open(run_dir / "focal_config.json", "w") as f:
            json.dump({"max_val": float(bin_edges[-1]), "bin_edges": bin_edges,
                       "bin_centers": bin_centers, "alpha": alpha, "gamma": args.gamma}, f, indent=2)
    else:
        criterion    = CombinedLoss()
        bin_center_t = None

    # 4. Optimizer + Scheduler (ReduceLROnPlateau)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=1e-7,
    )

    # 5. EMA
    ema = EMA(model, decay=args.ema_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_rmse = float("inf")
    patience_counter = 0
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)
        for inputs, targets, _, time_feat in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            time_feat = time_feat.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(inputs, time_feat)
                preds_41 = F.interpolate(preds, size=GPM_SIZE, mode="bilinear", align_corners=False)
                loss  = criterion(preds_41, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validate (using EMA weights) ---
        ema.apply(model)
        model.eval()
        sq_errors = []
        sq_errors_rain = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, dynamic_ncols=True)
            for inputs, targets, _, time_feat in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                time_feat = time_feat.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = model(inputs, time_feat)
                targets_real = torch.expm1(targets.float().clamp(0, 8))
                if use_focal:
                    preds_41 = F.interpolate(preds.float(), size=GPM_SIZE, mode="bilinear", align_corners=False)
                    probs    = F.softmax(preds_41, dim=1)
                    pred_mm  = (probs * bin_center_t).sum(dim=1, keepdim=True)
                else:
                    preds_41 = F.interpolate(preds.float(), size=GPM_SIZE, mode="bilinear", align_corners=False)
                    pred_mm = torch.expm1(preds_41.clamp(0, 8))
                sq = (pred_mm - targets_real) ** 2
                sq_errors.append(sq[torch.isfinite(sq)].cpu().numpy().ravel())
                rain_mask = (targets_real > 0) & torch.isfinite(sq)
                if rain_mask.any():
                    sq_errors_rain.append(sq[rain_mask].cpu().numpy().ravel())
        ema.restore(model)

        val_rmse = float(np.sqrt(np.concatenate(sq_errors).mean()))
        val_rmse_rain = float(np.sqrt(np.concatenate(sq_errors_rain).mean())) if sq_errors_rain else float("nan")
        avg_train = train_loss / len(train_loader)
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | val_RMSE={val_rmse:.4f} | val_RMSE_rain={val_rmse_rain:.4f} | lr={current_lr:.2e}")

        # Per-epoch history CSV
        history_path = run_dir / "history.csv"
        history_fields = ["epoch", "train_loss", "val_rmse", "val_rmse_rain", "lr"]
        write_header = not history_path.exists()
        with open(history_path, "a", newline="") as hf:
            hw = csv.DictWriter(hf, fieldnames=history_fields)
            if write_header:
                hw.writeheader()
            hw.writerow({"epoch": epoch, "train_loss": f"{avg_train:.6f}",
                         "val_rmse": f"{val_rmse:.6f}", "val_rmse_rain": f"{val_rmse_rain:.6f}",
                         "lr": f"{current_lr:.2e}"})

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # Save EMA weights directly so best_model.pth is ready for inference
            ema.apply(model)
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_rmse": best_val_rmse,
            }, run_dir / "checkpoint.pth")
            ema.restore(model)
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
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="Stop training if val RMSE does not improve for this many epochs.")
    parser.add_argument("--band_selection", default="all",
                        choices=["all", "ir_split_window"],
                        help="all: 51ch (16 bands x 3 frames + masks); ir_split_window: 12ch IR only.")
    parser.add_argument("--loss_type", default="combined",
                        help="focal: FocalLossIMERG (14 dynamic log-bins); combined: 0.7*MSE+0.3*MAE regression.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal Loss gamma (focusing parameter). Default 2.0 per GENESIS.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay factor. Default 0.999.")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience. Default 5.")
    parser.add_argument("--val_mode", default="temporal",
                        help="temporal: last 20pct per location; holdout: fixed locations as val.")
    parser.add_argument("--holdout_locations", default="florida,france,jakarta,kinshasa",
                        help="Comma-separated location names for holdout val (used when --val_mode=holdout).")
    parser.add_argument("--run_name", default=datetime.now().strftime("%Y%m%d_%H%M"),
                        help="Experiment name. Output saved to runs/{run_name}/")
    args = parser.parse_args()
    train(args)
