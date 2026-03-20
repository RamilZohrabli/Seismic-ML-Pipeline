from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_DT_MS = 2.0
TARGET_TIME_MS = 1500.0
TARGET_SAMPLES = 751  # 0..1500 ms inclusive at 2 ms sampling


def load_csv_rows(csv_path):
    csv_path = Path(csv_path)
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["shot_index"] = int(row["shot_index"])
            row["shot_id"] = int(row["shot_id"])
            row["n_traces"] = int(row["n_traces"])
            row["n_samples"] = int(row["n_samples"])
            row["valid_labels"] = int(row["valid_labels"])
            row["label_ratio"] = float(row["label_ratio"])
            row["file"] = str(row["file"])
            rows.append(row)

    return rows


def standardize_time_axis(traces, labels_ms, samp_rate_us):
    """
    Convert every gather to a common time axis:
    - target dt = 2 ms
    - target window = 0..1500 ms
    - target samples = 751

    traces: shape (n_traces, n_samples_native)
    labels_ms: shape (n_traces,)
    """
    current_dt_ms = float(samp_rate_us) / 1000.0

    old_t = np.arange(traces.shape[1], dtype=np.float32) * current_dt_ms
    new_t = np.arange(TARGET_SAMPLES, dtype=np.float32) * TARGET_DT_MS

    traces_std = np.empty((traces.shape[0], TARGET_SAMPLES), dtype=np.float32)

    # lazy per-trace interpolation
    for i in range(traces.shape[0]):
        traces_std[i] = np.interp(
            new_t,
            old_t,
            traces[i],
            left=0.0,
            right=0.0,
        )

    labels_idx = np.full(labels_ms.shape, -1, dtype=np.int64)
    valid = (labels_ms > 0) & (labels_ms <= TARGET_TIME_MS)

    if np.any(valid):
        idx = np.rint(labels_ms[valid] / TARGET_DT_MS).astype(np.int64)
        idx = np.clip(idx, 0, TARGET_SAMPLES - 1)
        labels_idx[valid] = idx

    return traces_std, labels_idx


def robust_normalize(traces, clip_percentile=99.0):
    """
    Robust amplitude normalization per gather.
    """
    clip = np.percentile(np.abs(traces), clip_percentile)

    if not np.isfinite(clip) or clip <= 0:
        clip = 1.0

    traces = np.clip(traces, -clip, clip) / clip
    return traces.astype(np.float32), float(clip)


def build_pick_mask(labels_idx, n_samples, n_traces, half_width=1):
    """
    Build a segmentation mask around the first-break line.
    Output shape: (n_samples, n_traces)
    """
    mask = np.zeros((n_samples, n_traces), dtype=np.float32)

    for x, y in enumerate(labels_idx):
        if y < 0:
            continue

        y0 = max(0, y - half_width)
        y1 = min(n_samples, y + half_width + 1)
        mask[y0:y1, x] = 1.0

    return mask


class FirstBreakShotDataset(Dataset):
    """
    Shot-level dataset.
    Each item is one full gather:
      image -> (1, 751, n_traces)
      mask  -> (1, 751, n_traces)

    Note:
    - Width (n_traces) varies by shot
    - So for now use batch_size=1
    """

    def __init__(
        self,
        csv_path,
        clip_percentile=99.0,
        mask_half_width=1,
    ):
        self.csv_path = Path(csv_path)
        self.rows = load_csv_rows(self.csv_path)
        self.clip_percentile = clip_percentile
        self.mask_half_width = mask_half_width

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        file_path = Path(row["file"])

        arr = np.load(file_path)

        traces = arr["traces"].astype(np.float32)          # (n_traces, n_samples_native)
        labels_ms = arr["labels_ms"].astype(np.float32)    # (n_traces,)
        samp_rate_us = float(arr["samp_rate_us"][0])
        shot_id = int(arr["shot_id"][0])

        traces_std, labels_idx = standardize_time_axis(
            traces=traces,
            labels_ms=labels_ms,
            samp_rate_us=samp_rate_us,
        )

        traces_std, clip_value = robust_normalize(
            traces_std,
            clip_percentile=self.clip_percentile,
        )

        n_traces = traces_std.shape[0]
        mask = build_pick_mask(
            labels_idx=labels_idx,
            n_samples=TARGET_SAMPLES,
            n_traces=n_traces,
            half_width=self.mask_half_width,
        )

        # convert to image layout: (C, H, W)
        image = traces_std.T[None, :, :]   # (1, 751, n_traces)
        mask = mask[None, :, :]            # (1, 751, n_traces)

        sample = {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "labels_sample": torch.from_numpy(labels_idx),
            "asset": row["asset"],
            "split": row["split"],
            "shot_id": shot_id,
            "clip_value": clip_value,
            "n_traces": n_traces,
        }
        return sample


def single_item_collate(batch):
    """
    Use this with batch_size=1 because widths vary.
    """
    return batch[0]