"""
Preprocess CICIDS2017 raw CSV files into feature matrices.

CICIDS2017 data quality issues handled here:
  - Leading/trailing whitespace in column headers (strip all immediately after read)
  - Embedded duplicate header rows mid-file (rows where Label == 'Label')
  - inf / -inf values in Flow Bytes/s and Flow Packets/s columns
  - Non-numeric stray values in feature columns (coerced to NaN, then imputed)
  - Monday CSV contains benign-only traffic — expected, logged for transparency
  - Memory constraint: CSVs loaded sequentially; arrays stored as float32

Leakage prevention:
  - train_test_split() is called BEFORE any fitting operation
  - SimpleImputer(strategy='median') fit on train split only, applied to both
  - StandardScaler fit on imputed train only, applied to both
  - No test-split data influences any fitted transformer

NOTE: The original preprocess.py applied pd.get_dummies() on the concatenated
train+test DataFrame, which leaks test-set category distributions into the
encoder. This script avoids that pattern entirely (CICIDS2017 has no
categorical features after dropping identifier columns).

Outputs → data/processed/cicids2017/:
    X_train.npy, y_train.npy, X_test.npy, y_test.npy, feature_names.json

Usage:
    python src/features/preprocess_cicids2017.py --config configs/experiment.yaml
    make preprocess_cicids2017
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Identifier columns to drop (not informative features; would cause overfitting
# or data leakage from network addresses/ports unique to the capture environment)
_IDENTIFIER_COLS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Timestamp",
}


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load one CICIDS2017 CSV with robust handling of known data quality issues.
    Returns a cleaned DataFrame (identifiers dropped, duplicate headers removed,
    inf replaced with NaN).
    """
    df = pd.read_csv(path, low_memory=False)

    # 1. Strip whitespace from all column names (CIC headers have leading spaces)
    df.columns = df.columns.str.strip()

    # 2. Remove embedded duplicate header rows (artifact of CIC's tool output)
    label_col = "Label"
    if label_col in df.columns:
        mask = df[label_col] == label_col
        n_dupes = mask.sum()
        if n_dupes:
            df = df[~mask].copy()

    # 3. Drop identifier columns (present only in some files; ignore missing)
    drop = [c for c in df.columns if c in _IDENTIFIER_COLS]
    if drop:
        df.drop(columns=drop, inplace=True)

    # 4. Replace inf values with NaN (common in Flow Bytes/s, Flow Packets/s)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def load_and_clean(raw_dir: Path, label_col: str) -> pd.DataFrame:
    """
    Discover and load all CICIDS2017 CSV files in raw_dir sequentially to
    manage memory. Returns a single concatenated DataFrame.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            f"Run 'make data' for download instructions."
        )

    frames = []
    for path in tqdm(csv_files, desc="  Loading CSVs"):
        df = _load_csv(path)
        label_counts = df[label_col].value_counts() if label_col in df.columns else {}
        tqdm.write(f"    {path.name}: {len(df):,} rows  {dict(label_counts)}")
        frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def binarize_labels(series: pd.Series, benign_label: str) -> np.ndarray:
    """0 for benign traffic, 1 for all attack categories."""
    return (series.str.strip() != benign_label).astype(np.int8).values


def to_numeric_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Drop the label column and coerce all remaining columns to float32.
    Non-numeric values are coerced to NaN (handled later by SimpleImputer).
    """
    feature_df = df.drop(columns=[label_col])
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    return feature_df.astype(np.float32)


def fit_and_transform(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Leakage-safe imputation and scaling pipeline.
    All fitting is done on X_train only.

    Steps:
      1. SimpleImputer(strategy='median'): fit on train, transform both.
         Median is preferred over mean for skewed network traffic distributions.
      2. StandardScaler: fit on imputed train, transform both.

    Returns:
      X_train_arr, X_test_arr (float32 numpy arrays), feature_names (list)
    """
    feature_names = list(X_train_df.columns)
    X_train_vals = X_train_df.values
    X_test_vals = X_test_df.values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_vals)
    X_test_imp = imputer.transform(X_test_vals)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_imp).astype(np.float32)

    return X_train_scaled, X_test_scaled, feature_names


def preprocess(cfg: dict) -> None:
    ds_cfg = cfg["datasets"]["cicids2017"]
    raw_dir = Path(cfg["paths"]["raw_data"]) / ds_cfg["raw_subdir"]
    out_dir = Path(cfg["paths"]["processed_data"]) / "cicids2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_col = ds_cfg["label_col"]
    benign_label = ds_cfg["benign_label"]
    random_state = cfg["experiment"]["random_state"]

    print("Loading CICIDS2017 CSV files ...")
    df = load_and_clean(raw_dir, label_col)
    print(f"\n  Total rows after cleaning: {len(df):,}")
    print(f"  Label distribution:")
    for label, count in df[label_col].value_counts().items():
        tag = " (benign)" if label.strip() == benign_label else " (attack)"
        print(f"    {label:<45s} {count:>8,}{tag}")

    # Binarize labels
    y_all = binarize_labels(df[label_col], benign_label)

    # Extract numeric features (label column dropped)
    X_df = to_numeric_features(df, label_col)
    del df  # free memory before split

    # Stratified 80/20 split — BEFORE any fitting
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y_all,
        test_size=ds_cfg["test_size"],
        random_state=random_state,
        stratify=y_all,
    )
    del X_df  # free memory

    print(f"\n  Split: train={len(y_train):,}  test={len(y_test):,}")

    # Impute + scale (train-fit only)
    print("  Imputing and scaling (fit on train only) ...")
    X_train, X_test, feature_names = fit_and_transform(X_train_df, X_test_df)

    # Save
    print(f"  Saving to {out_dir} ...")
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"\n  Train: {X_train.shape}  attacks: {y_train.sum():,}/{len(y_train):,}")
    print(f"  Test : {X_test.shape}  attacks: {y_test.sum():,}/{len(y_test):,}")
    print("CICIDS2017 preprocessing complete.")


def main(config_path: str) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    preprocess(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 dataset")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    main(args.config)
