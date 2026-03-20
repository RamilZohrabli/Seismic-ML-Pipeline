from pathlib import Path
import json
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#asset split
TRAIN_ASSETS = {
    "Halfmile3D_add_geom_sorted",
    "Brunswick_orig_1500ms_V2_(1)",
}

VAL_ASSETS = {
    "Lalor_raw_z_1500ms_norp_geom_v3",
}

TEST_ASSETS = {
    "preprocessed_Sudbury3D",
}


def assign_split(asset_name: str) -> str:
    if asset_name in TRAIN_ASSETS:
        return "train"
    if asset_name in VAL_ASSETS:
        return "val"
    if asset_name in TEST_ASSETS:
        return "test"
    raise ValueError(f"Unknown asset: {asset_name}")


def common_window_samples(asset_name: str, samp_rate_us: float, samp_num: int) -> int:
    """
    Common time window = 1500 ms
    Convert that to sample count depending on sample rate.
    Crop later during Dataset loading.
    """
    target_ms = 1500.0
    dt_ms = samp_rate_us / 1000.0
    n_target = int(round(target_ms / dt_ms))

    return min(n_target, samp_num)


def main():
    rows = []

    asset_dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir()]
    asset_dirs = sorted(asset_dirs)

    print("Found processed asset directories:")
    for d in asset_dirs:
        print(" -", d.name)

    for asset_dir in asset_dirs:
        manifest_path = asset_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"Skipping {asset_dir.name} (no manifest.json)")
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        split = assign_split(asset_dir.name)

        for item in manifest:
            file_path = asset_dir / item["file"]

            # infer sample rate / sample count from saved npz metadata later if needed,
            # but here we use known asset-based assumptions only through manifest + filename
            # We will store asset name and file path; actual loading logic will read npz.
            rows.append({
                "asset": asset_dir.name,
                "split": split,
                "shot_index": item["shot_index"],
                "shot_id": item["shot_id"],
                "file": str(file_path),
                "n_traces": item["n_traces"],
                "n_samples": item["n_samples"],
                "valid_labels": item["valid_labels"],
                "label_ratio": item["label_ratio"],
            })

    # Write master CSV
    master_csv = OUTPUT_DIR / "all_shots.csv"
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "asset",
                "split",
                "shot_index",
                "shot_id",
                "file",
                "n_traces",
                "n_samples",
                "valid_labels",
                "label_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Write per-split CSVs
    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split_name]
        split_csv = OUTPUT_DIR / f"{split_name}_shots.csv"

        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "asset",
                    "split",
                    "shot_index",
                    "shot_id",
                    "file",
                    "n_traces",
                    "n_samples",
                    "valid_labels",
                    "label_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(split_rows)

        print(f"{split_name}: {len(split_rows)} shots")

    print("\nSaved split files to:", OUTPUT_DIR)
    print("Master file:", master_csv)


if __name__ == "__main__":
    main()