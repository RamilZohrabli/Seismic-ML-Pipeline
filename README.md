# Seismic First-Break Project

This repository contains data processing and training utilities for seismic first-break picking workflows.

## Project Overview

The project is organized around three main stages:

1. Preprocessing raw seismic datasets into shot-level `.npz` files.
2. Building train/validation/test splits.
3. Training segmentation-style models (Tiny U-Net) for first-break related tasks.

## Repository Structure

- `data/raw/`: source datasets (`.hdf5`/`.hdf`).
- `data/processed/`: preprocessed shot-level outputs and manifests.
- `data/splits/`: CSV files defining train/val/test splits.
- `src/`: preprocessing scripts, dataset loaders, model code, and training utilities.
- `notebooks/`: exploratory analysis notebooks.

## Main Scripts

- `src/preprocess_brunswick.py`
- `src/preprocess_halfmile.py`
- `src/preprocess_lalor.py`
- `src/preprocess_sudbury.py`
- `src/build_splits.py`
- `src/first_break_dataset.py`
- `src/first_break_window_dataset.py`
- `src/tiny_unet.py`
- `src/train_utils.py`

## Quick Start

## 1) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

If you already have a requirements file, install it:

```powershell
pip install -r requirements.txt
```

If not, install core packages as needed (example):

```powershell
pip install numpy pandas h5py torch torchvision matplotlib jupyter
```

## 3) Preprocess a dataset

Run the corresponding preprocessing script for your dataset:

```powershell
python src/preprocess_brunswick.py
```

Repeat similarly for other datasets:

- `python src/preprocess_halfmile.py`
- `python src/preprocess_lalor.py`
- `python src/preprocess_sudbury.py`

## 4) Build data splits

```powershell
python src/build_splits.py
```

## 5) Explore and train

- Use `notebooks/exploration.ipynb` for data inspection.
- Use the utilities in `src/train_utils.py` with model definitions from `src/tiny_unet.py` to train experiments.

## Data Notes

- Processed shots are stored as `.npz` files under dataset-specific folders in `data/processed/`.
- Each processed dataset folder includes a `manifest.json` describing generated contents.
- Split files are saved under `data/splits/` as CSV.

## Tips

- Keep raw datasets unchanged in `data/raw/`.
- Version split CSV files when comparing experiments.
- Log key hyperparameters for reproducibility (batch size, learning rate, window size, augmentations).

