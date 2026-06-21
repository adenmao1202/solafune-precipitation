import ast
import math
import random

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-satellite official band order (source: competition data spec)
# All indices are 0-based.
# ---------------------------------------------------------------------------
SATELLITE_SUBDIR = {
    "himawari": "himawari",
    "goes":     "goes",
    "meteosat": "meteosat",
}

# Native spatial size per satellite (validated: data_validation.ipynb)
SAT_SIZE = {
    "himawari": (81, 81),
    "goes":     (141, 141),
    "meteosat": (144, 144),
}

# GPM-IMERG native output size (always 41x41 regardless of satellite)
GPM_SIZE = (41, 41)

MAX_FRAMES = 3
N_BANDS    = 16  # raw bands per satellite TIF

# ---------------------------------------------------------------------------
# Canonical band mapping (see /Volumes/T7/new_code/solafune/mapping.md)
# Raw 0-based indices into the satellite TIF. No Meteosat swap needed.
# None = satellite has no band at this wavelength -> filled with zeros.
# ---------------------------------------------------------------------------
CANONICAL_BANDS_12 = {
    # Slot: [0.64, 0.8, 1.6, 2.25, 3.9, 6.2, 7.3, 8.6, 9.7, 10.4, 12.3, 13.3]
    "himawari": [2,  3,  4,  5,  6,  7,  9,  10, 11, 12, 14, 15],
    "goes":     [1,  2,  4,  5,  6,  7,  9,  10, 11, 12, 14, 15],
    "meteosat": [2,  3,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15],
}

CANONICAL_BANDS_18 = {
    # Slots 0-11: same as 12-slot
    # Slots 12-17: satellite-specific; None -> zero-pad
    # Slot 12: 0.47um blue   (H=0, G=0, M=None)
    # Slot 13: 0.5um green   (H=1, G=None, M=1)
    # Slot 14: 0.9um NIR2    (H=None, G=None, M=4)
    # Slot 15: 1.38um cirrus (H=None, G=3, M=5)
    # Slot 16: 6.9um mid-WV  (H=8, G=8, M=None)
    # Slot 17: 11.2um IR2    (H=13, G=13, M=None)
    "himawari": [2,  3,  4,  5,  6,  7,  9,  10, 11, 12, 14, 15, 0,    1,    None, None, 8,    13  ],
    "goes":     [1,  2,  4,  5,  6,  7,  9,  10, 11, 12, 14, 15, 0,    None, None, 3,    8,    13  ],
    "meteosat": [2,  3,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, None, 1,    4,    5,    None, None],
}

N_SLOTS_12 = 12
N_SLOTS_18 = 18

# Input channel counts: n_slots * 3 frames + 3 frame-valid masks
IN_CHANNELS_12 = N_SLOTS_12 * MAX_FRAMES + MAX_FRAMES  # 39
IN_CHANNELS_18 = N_SLOTS_18 * MAX_FRAMES + MAX_FRAMES  # 57

# FiLM conditioning dim: 4 time features + 3 satellite one-hot
COND_DIM = 7

# Satellite one-hot encoding: himawari=[1,0,0], goes=[0,1,0], meteosat=[0,0,1]
SAT_ONEHOT = {
    "himawari": [1.0, 0.0, 0.0],
    "goes":     [0.0, 1.0, 0.0],
    "meteosat": [0.0, 0.0, 1.0],
}

# Legacy alias kept for any external scripts that import IN_CHANNELS
IN_CHANNELS = IN_CHANNELS_12


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_filenames(raw: str) -> list[str]:
    return ast.literal_eval(raw)


def read_tif(path: Path) -> np.ndarray | None:
    """Returns None if file is corrupt or unreadable."""
    try:
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)  # (bands, H, W)
    except Exception:
        return None


def normalize_per_band(arr: np.ndarray, stats: dict, satellite: str) -> np.ndarray:
    mean = np.array(stats[satellite]["mean"], dtype=np.float32)[:, None, None]
    std  = np.array(stats[satellite]["std"],  dtype=np.float32)[:, None, None]
    return (arr - mean) / (std + 1e-6)


def resize_to(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(C, H, W) -> (C, size[0], size[1]) using bilinear interpolation."""
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)


def select_canonical_bands(arr: np.ndarray, satellite: str, band_mode: str) -> np.ndarray:
    """
    Select canonical slots from a normalized (N_BANDS, H, W) array.
    Returns (n_slots, H, W). None slots become zero rows.
    """
    if band_mode == "12slot":
        slot_indices = CANONICAL_BANDS_12[satellite]
        n_slots = N_SLOTS_12
    else:
        slot_indices = CANONICAL_BANDS_18[satellite]
        n_slots = N_SLOTS_18

    H, W = arr.shape[1], arr.shape[2]
    out = np.zeros((n_slots, H, W), dtype=np.float32)
    for s, raw_idx in enumerate(slot_indices):
        if raw_idx is not None:
            out[s] = arr[raw_idx]
    return out


class PrecipDataset(Dataset):
    """
    input_tensor  : (IN_CHANNELS_12 or IN_CHANNELS_18, H, W)
    target_tensor : (1, H, W)  -- GPM log1p(mm/hr) resized to GPM_SIZE (41x41)
    time_feat     : (COND_DIM,) = [sin_day, cos_day, sin_hour, cos_hour, sat_0, sat_1, sat_2]

    band_mode: "12slot" (39ch) or "18slot" (57ch)
    input_size: if set (e.g. (128,128)), all satellites are resized to this fixed size
                so DataLoader can batch samples from different satellites.
    """

    def __init__(self, csv_path: Path, data_dir: Path, stats: dict,
                 is_train: bool = True, transform=None,
                 input_size: tuple[int, int] | None = None,
                 band_mode: str = "12slot"):
        if band_mode not in ("12slot", "18slot"):
            raise ValueError(f"band_mode must be '12slot' or '18slot', got '{band_mode}'")
        self.df         = pd.read_csv(csv_path)
        self.data_dir   = Path(data_dir)
        self.stats      = stats
        self.is_train   = is_train
        self.transform  = transform
        self.input_size = input_size
        self.band_mode  = band_mode

    def __len__(self):
        return len(self.df)

    def _sat_path(self, satellite: str, fname: str) -> Path:
        return self.data_dir / SATELLITE_SUBDIR[satellite] / fname

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        satellite = str(row["satellite_target"])
        filenames = parse_filenames(str(row["last_30_minutes_observation_filename"]))

        sat_h, sat_w = SAT_SIZE[satellite]
        out_h, out_w = self.input_size if self.input_size else (sat_h, sat_w)

        n_slots = N_SLOTS_12 if self.band_mode == "12slot" else N_SLOTS_18

        frames, masks = [], []
        for i in range(MAX_FRAMES):
            if i < len(filenames):
                arr = read_tif(self._sat_path(satellite, filenames[i]))
                if arr is not None:
                    # Pad/trim to N_BANDS
                    if arr.shape[0] < N_BANDS:
                        pad = np.zeros((N_BANDS - arr.shape[0], arr.shape[1], arr.shape[2]),
                                       dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                    elif arr.shape[0] > N_BANDS:
                        arr = arr[:N_BANDS]
                    # Fix spatial size
                    if arr.shape[1] != sat_h or arr.shape[2] != sat_w:
                        t = torch.from_numpy(arr)
                        t = F.interpolate(t.unsqueeze(0), size=(sat_h, sat_w),
                                          mode='bilinear', align_corners=False).squeeze(0)
                        arr = t.numpy()
                    # Normalize all 16 bands using per-satellite stats
                    arr = normalize_per_band(arr, self.stats, satellite)
                    # Select canonical slots (no Meteosat swap needed)
                    slot_arr = select_canonical_bands(arr, satellite, self.band_mode)
                    frames.append(slot_arr)
                    masks.append(np.ones((1, sat_h, sat_w), dtype=np.float32))
                else:
                    frames.append(np.zeros((n_slots, sat_h, sat_w), dtype=np.float32))
                    masks.append(np.zeros((1, sat_h, sat_w), dtype=np.float32))
            else:
                frames.append(np.zeros((n_slots, sat_h, sat_w), dtype=np.float32))
                masks.append(np.zeros((1, sat_h, sat_w), dtype=np.float32))

        input_tensor = torch.from_numpy(np.concatenate(frames + masks, axis=0))
        # shape: (n_slots*3 + 3, sat_h, sat_w)

        if self.input_size and (sat_h, sat_w) != (out_h, out_w):
            input_tensor = resize_to(input_tensor, (out_h, out_w))

        if self.is_train:
            gpm_arr = read_tif(self.data_dir / "gpm_imerg" / row["gpm_imerg_filename"])
            if gpm_arr is not None:
                target_tensor = torch.from_numpy(np.log1p(gpm_arr))
                if target_tensor.shape[-2:] != GPM_SIZE:
                    target_tensor = resize_to(target_tensor, GPM_SIZE)
            else:
                target_tensor = torch.zeros(1, *GPM_SIZE)
        else:
            target_tensor = torch.zeros(1, *GPM_SIZE)

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if self.is_train:
            if random.random() < 0.5:
                input_tensor  = torch.flip(input_tensor,  dims=[-1])
                target_tensor = torch.flip(target_tensor, dims=[-1])

        dt   = pd.to_datetime(row["datetime"])
        day  = dt.day_of_year
        hour = dt.hour
        sat_oh = SAT_ONEHOT[satellite]
        time_feat = torch.tensor([
            math.sin(2 * math.pi * day / 365),
            math.cos(2 * math.pi * day / 365),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            sat_oh[0],
            sat_oh[1],
            sat_oh[2],
        ], dtype=torch.float32)

        return input_tensor, target_tensor, row["unique_id"], time_feat
