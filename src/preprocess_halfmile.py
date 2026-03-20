from __future__ import annotations

from pathlib import Path
import json
import math

import h5py
import numpy as np


GROUP_PATH = "TRACE_DATA/DEFAULT"


def read_constant(ds) -> float:
    """Read one constant-like value from an HDF5 dataset."""
    arr = np.asarray(ds[0]).reshape(-1)
    return float(arr[0])


def decode_scale(raw_scale: float) -> float:
    """
    SEG-Y style scale decoding:
    0  -> 1
    >0 -> multiply by scale
    <0 -> divide by abs(scale)
    """
    raw_scale = float(raw_scale)
    if raw_scale == 0:
        return 1.0
    if raw_scale > 0:
        return raw_scale
    return 1.0 / abs(raw_scale)


def first_break_ms_to_sample(spare1_ms: np.ndarray, samp_rate_us: float, n_samples: int) -> np.ndarray:
    """
    Convert SPARE1 from milliseconds to sample index.
    Unlabeled values (<= 0) become -1.
    """
    dt_ms = samp_rate_us / 1000.0
    out = np.full(spare1_ms.shape, -1, dtype=np.int32)

    valid = spare1_ms > 0
    idx = np.rint(spare1_ms[valid] / dt_ms).astype(np.int32)
    idx = np.clip(idx, 0, n_samples - 1)
    out[valid] = idx
    return out


def iter_shot_ranges(shot_ids_1d: np.ndarray):
    """
    Yield contiguous shot ranges from a 1D SHOTID array.
    Assumes same SHOTID values appear in contiguous blocks.
    """
    start = 0
    n = len(shot_ids_1d)

    for i in range(1, n):
        if shot_ids_1d[i] != shot_ids_1d[i - 1]:
            yield int(shot_ids_1d[start]), start, i
            start = i

    yield int(shot_ids_1d[start]), start, n


def preprocess_asset(
    input_path: str | Path,
    output_dir: str | Path,
    min_valid_labels: int = 50,
    save_float16: bool = False,
):
    """
    Preprocess one HDF5 asset into shot-based .npz files.

    For each SHOTID:
    - extract traces
    - sort by REC_X
    - convert SPARE1(ms) -> sample index
    - save one compressed .npz file

    Parameters
    ----------
    input_path : path to one HDF5 asset
    output_dir : directory where processed shots will be saved
    min_valid_labels : skip shots with too few valid labels
    save_float16 : if True, saves traces as float16 to reduce disk usage
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_name = input_path.stem.replace(" ", "_")
    asset_out_dir = output_dir / asset_name
    asset_out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    with h5py.File(input_path, "r") as f:
        g = f[GROUP_PATH]

        data_ds = g["data_array"]
        shot_ids = g["SHOTID"][:, 0]   # okay to load metadata column
        spare1 = g["SPARE1"][:, 0]
        rec_x_all = g["REC_X"][:, 0]
        rec_y_all = g["REC_Y"][:, 0]

        coord_scale = decode_scale(read_constant(g["COORD_SCALE"])) if "COORD_SCALE" in g else 1.0
        ht_scale = decode_scale(read_constant(g["HT_SCALE"])) if "HT_SCALE" in g else 1.0
        samp_rate_us = read_constant(g["SAMP_RATE"])
        samp_num = int(read_constant(g["SAMP_NUM"]))

        print(f"\nProcessing asset: {asset_name}")
        print(f"Total traces: {len(shot_ids):,}")
        print(f"Samples per trace: {samp_num}")
        print(f"Sample rate: {samp_rate_us} us")
        print(f"Coord scale factor: {coord_scale}")
        print(f"Height scale factor: {ht_scale}")

        shot_ranges = list(iter_shot_ranges(shot_ids))
        print(f"Detected shots: {len(shot_ranges)}")

        for shot_idx, (shot_id, start, end) in enumerate(shot_ranges, start=1):
            traces = np.asarray(data_ds[start:end], dtype=np.float32)
            labels_ms = spare1[start:end].astype(np.float32)
            rec_x = rec_x_all[start:end].astype(np.float64) * coord_scale
            rec_y = rec_y_all[start:end].astype(np.float64) * coord_scale

            # sort traces spatially
            order = np.argsort(rec_x, kind="stable")
            traces = traces[order]
            labels_ms = labels_ms[order]
            rec_x = rec_x[order]
            rec_y = rec_y[order]

            labels_sample = first_break_ms_to_sample(
                labels_ms,
                samp_rate_us=samp_rate_us,
                n_samples=samp_num,
            )

            valid_count = int(np.sum(labels_sample >= 0))
            if valid_count < min_valid_labels:
                continue

            # Optional disk size reduction
            if save_float16:
                traces_to_save = traces.astype(np.float16)
            else:
                traces_to_save = traces.astype(np.float32)

            file_name = f"shot_{shot_idx:04d}_id_{shot_id}.npz"
            save_path = asset_out_dir / file_name

            np.savez_compressed(
                save_path,
                traces=traces_to_save,          # shape: (n_traces, n_samples)
                labels_ms=labels_ms,            # shape: (n_traces,)
                labels_sample=labels_sample,    # shape: (n_traces,)
                rec_x=rec_x,
                rec_y=rec_y,
                shot_id=np.array([shot_id], dtype=np.int64),
                samp_rate_us=np.array([samp_rate_us], dtype=np.float32),
                samp_num=np.array([samp_num], dtype=np.int32),
            )

            manifest.append(
                {
                    "asset": asset_name,
                    "shot_index": shot_idx,
                    "shot_id": int(shot_id),
                    "file": file_name,
                    "n_traces": int(traces.shape[0]),
                    "n_samples": int(traces.shape[1]),
                    "valid_labels": valid_count,
                    "label_ratio": float(valid_count / len(labels_sample)),
                }
            )

            if shot_idx % 50 == 0:
                print(f"Processed {shot_idx}/{len(shot_ranges)} shots...")

    manifest_path = asset_out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {asset_name}")
    print(f"Saved shots to: {asset_out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Saved {len(manifest)} usable shot gathers.")


if __name__ == "__main__":
    # Change this path to whichever dataset you want to preprocess first
    input_file = r"data\raw\Halfmile3D_add_geom_sorted.hdf5"
    output_folder = r"data\processed"

    preprocess_asset(
        input_path=input_file,
        output_dir=output_folder,
        min_valid_labels=50,
        save_float16=False,   # keep False first; turn True later to save disk
    )