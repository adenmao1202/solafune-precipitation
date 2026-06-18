import ast
import math

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


SATELLITE_BANDS = {
    "himawari": ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12","B13","B14","B15","B16"],
    "goes":     ["C01","C02","C03","C04","C05","C06","C07","C08","C09","C10","C11","C12","C13","C14","C15","C16"],
    "meteosat": ["vis_04","vis_05","vis_06","vis_08","vis_09","nir_13","nir_16","nir_22","ir_38","wv_63","wv_73","ir_87","ir_97","ir_105","ir_123","ir_133"],
}

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

# GPM-IMERG native output size (always 41×41 regardless of satellite)
GPM_SIZE = (41, 41)

MAX_FRAMES  = 3
N_BANDS     = 16
IN_CHANNELS = MAX_FRAMES * N_BANDS + MAX_FRAMES  # 51


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
    """(C, H, W) → (C, size[0], size[1]) using bilinear interpolation."""
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)


class PrecipDataset(Dataset):
    """
    input_tensor  : (IN_CHANNELS, H, W)
    target_tensor : (1, H, W)  — GPM log1p(mm/hr) resized to match H×W

    input_size: if set (e.g. (128,128)), all satellites are resized to this fixed size so
                DataLoader can batch samples from different satellites together.
                If None, native satellite resolution is used — batches must be same-satellite.
    """

    def __init__(self, csv_path: Path, data_dir: Path, stats: dict,
                 is_train: bool = True, transform=None,
                 input_size: tuple[int, int] | None = None):
        self.df         = pd.read_csv(csv_path)
        self.data_dir   = Path(data_dir)
        self.stats      = stats
        self.is_train   = is_train
        self.transform  = transform
        self.input_size = input_size  # (H, W) or None

    def __len__(self):
        return len(self.df)

    def _sat_path(self, satellite: str, fname: str) -> Path:
        return self.data_dir / SATELLITE_SUBDIR[satellite] / fname

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        satellite = str(row["satellite_target"])
        filenames = parse_filenames(str(row["last_30_minutes_observation_filename"]))

        sat_h, sat_w = SAT_SIZE[satellite]
        out_h, out_w  = self.input_size if self.input_size else (sat_h, sat_w)

        frames, masks = [], []
        for i in range(MAX_FRAMES):
            if i < len(filenames):
                arr = read_tif(self._sat_path(satellite, filenames[i]))
                if arr is not None:
                    # Fix band count
                    if arr.shape[0] < N_BANDS:
                        pad = np.zeros((N_BANDS - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                    elif arr.shape[0] > N_BANDS:
                        arr = arr[:N_BANDS]
                    # Fix spatial size (some TIFs differ from SAT_SIZE)
                    if arr.shape[1] != sat_h or arr.shape[2] != sat_w:
                        t = torch.from_numpy(arr)
                        t = F.interpolate(t.unsqueeze(0), size=(sat_h, sat_w),
                                          mode='bilinear', align_corners=False).squeeze(0)
                        arr = t.numpy()
                    if satellite == "meteosat":
                        arr[[12, 13]] = arr[[13, 12]]
                    arr = normalize_per_band(arr, self.stats, satellite)
                    frames.append(arr)
                    masks.append(np.ones((1, sat_h, sat_w), dtype=np.float32))
                else:
                    # Corrupt file: treat as missing frame
                    frames.append(np.zeros((N_BANDS, sat_h, sat_w), dtype=np.float32))
                    masks.append(np.zeros((1, sat_h, sat_w), dtype=np.float32))
            else:
                frames.append(np.zeros((N_BANDS, sat_h, sat_w), dtype=np.float32))
                masks.append(np.zeros((1, sat_h, sat_w), dtype=np.float32))

        input_tensor = torch.from_numpy(np.concatenate(frames + masks, axis=0))
        # shape: (IN_CHANNELS, sat_h, sat_w)

        # Resize to target output size if specified (needed to batch mixed satellites)
        if self.input_size and (sat_h, sat_w) != (out_h, out_w):
            input_tensor = resize_to(input_tensor, (out_h, out_w))

        if self.is_train:
            gpm_arr = read_tif(self.data_dir / "gpm_imerg" / row["gpm_imerg_filename"])
            if gpm_arr is not None:
                target_tensor = torch.from_numpy(np.log1p(gpm_arr))
                if target_tensor.shape[-2:] != (out_h, out_w):
                    target_tensor = resize_to(target_tensor, (out_h, out_w))
            else:
                target_tensor = torch.zeros(1, out_h, out_w)
        else:
            target_tensor = torch.zeros(1, out_h, out_w)

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        dt   = pd.to_datetime(row["datetime"])
        day  = dt.day_of_year
        hour = dt.hour
        time_feat = torch.tensor([
            math.sin(2 * math.pi * day / 365),
            math.cos(2 * math.pi * day / 365),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
        ], dtype=torch.float32)

        return input_tensor, target_tensor, row["unique_id"], time_feat
