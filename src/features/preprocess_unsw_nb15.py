"""
Preprocess UNSW-NB15 raw CSV files into feature matrices.

Uses the official train/test split:
    UNSW_NB15_training-set.csv  (175,341 rows)
    UNSW_NB15_testing-set.csv   (82,332 rows)

Features:
    - 49 features total; 3 categorical: proto, service, state
    - Drop: id (row identifier), attack_cat (multi-class label — would leak
      attack type into features if retained)
    - Binary label column: 'label' (already 0/1 in the official files)

Leakage prevention:
    - One-hot encoding: pd.get_dummies() applied to train only, then test is
      reindexed to match train columns (fill_value=0 for unseen categories,
      extra test-only columns dropped). No test categories influence the
      encoding schema.
    - SimpleImputer(strategy='median'): fit on train, applied to both.
    - StandardScaler: fit on train, applied to both.

NOTE: The original preprocess.py applied pd.get_dummies() on the combined
train+test DataFrame — a well-known leakage pattern. This script fixes that
by fitting the encoder on the training split only.

Known gotchas:
    - ct_ftp_cmd: NaN for non-FTP flows; semantic meaning is "0 FTP commands",
      so NaN is explicitly overridden to 0 after median imputation.
    - service: '-' is a valid category meaning "generic/none", not a missing
      value; treat as a regular string category.
    - label: already binary 0/1; asserted at load time to prevent silent errors.

Outputs → data/processed/unsw_nb15/:
    X_train.npy, y_train.npy, X_test.npy, y_test.npy, feature_names.json

Usage:
    python src/features/preprocess_unsw_nb15.py --config configs/experiment.yaml
    make preprocess_unsw_nb15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Columns that carry FTP-specific count semantics: NaN means 0, not missing measurement.
_FTP_ZERO_COLS = {"ct_ftp_cmd", "is_ftp_login"}


def load_split(path: Path, label_col: str) -> pd.DataFrame:
    """Load one UNSW-NB15 CSV and validate the label column."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            f"Run 'make data' for download instructions."
        )
    df = pd.read_csv(path, low_memory=False)

    # Verify binary label
    label_vals = set(df[label_col].dropna().unique())
    if not label_vals.issubset({0, 1}):
        raise ValueError(
            f"Expected binary 0/1 in column '{label_col}', "
            f"found: {label_vals}. Check the source file."
        )
    return df


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe one-hot encoding using pd.get_dummies + column reindex.

    1. Encode train categorical columns.
    2. Encode test categorical columns independently.
    3. Reindex test to match train columns:
         - Missing columns filled with 0 (category present in train, absent in test)
         - Extra columns dropped (category present in test, absent in train)
    4. Drop the original categorical columns from both DataFrames.
    5. Concatenate numeric + encoded columns.
    """
    cat_cols_present = [c for c in cat_cols if c in train_df.columns]
    if not cat_cols_present:
        return train_df, test_df

    train_enc = pd.get_dummies(train_df[cat_cols_present], prefix=cat_cols_present)
    test_enc = pd.get_dummies(test_df[cat_cols_present], prefix=cat_cols_present)

    # Align test to train's column set
    test_only_cols = set(test_enc.columns) - set(train_enc.columns)
    if test_only_cols:
        print(
            f"  INFO: {len(test_only_cols)} OHE column(s) present in test but not train "
            f"(dropped — unseen categories): {sorted(test_only_cols)}"
        )
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)

    # Drop original categorical columns; concatenate encoded
    train_out = pd.concat(
        [train_df.drop(columns=cat_cols_present), train_enc], axis=1
    )
    test_out = pd.concat(
        [test_df.drop(columns=cat_cols_present), test_enc], axis=1
    )
    return train_out, test_out


def preprocess(cfg: dict) -> None:
    ds_cfg = cfg["datasets"]["unsw_nb15"]
    raw_dir = Path(cfg["paths"]["raw_data"]) / ds_cfg["raw_subdir"]
    out_dir = Path(cfg["paths"]["processed_data"]) / "unsw_nb15"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_col = ds_cfg["label_col"]
    drop_cols = ds_cfg.get("drop_cols", [])
    cat_cols = ds_cfg.get("categorical_cols", [])

    print("Loading UNSW-NB15 CSVs ...")
    train_df = load_split(raw_dir / ds_cfg["train_file"], label_col)
    test_df = load_split(raw_dir / ds_cfg["test_file"], label_col)
    print(f"  Train: {len(train_df):,} rows   Test: {len(test_df):,} rows")

    # Extract labels (already binary)
    y_train = train_df[label_col].values.astype(np.int8)
    y_test = test_df[label_col].values.astype(np.int8)
    print(f"  Train attacks: {y_train.sum():,}/{len(y_train):,}")
    print(f"  Test  attacks: {y_test.sum():,}/{len(y_test):,}")

    # Drop label + identifier + multi-class label columns
    to_drop = [c for c in (drop_cols + [label_col]) if c in train_df.columns]
    train_df.drop(columns=to_drop, inplace=True)
    to_drop_test = [c for c in (drop_cols + [label_col]) if c in test_df.columns]
    test_df.drop(columns=to_drop_test, inplace=True)

    # One-hot encode categorical columns (leakage-safe)
    print("  Encoding categorical features (fit on train only) ...")
    train_df, test_df = encode_categoricals(train_df, test_df, cat_cols)

    # Override FTP-specific NaN columns: semantically 0, not missing
    for col in _FTP_ZERO_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0)

    feature_names = list(train_df.columns)

    # Convert to float32
    X_train_vals = train_df.values.astype(np.float32)
    X_test_vals = test_df.values.astype(np.float32)

    # Impute remaining NaN (fit on train only)
    print("  Imputing remaining NaN (strategy=median, fit on train only) ...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_vals)
    X_test_imp = imputer.transform(X_test_vals)

    # Scale (fit on train only)
    print("  Standardizing (fit on train only) ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test = scaler.transform(X_test_imp).astype(np.float32)

    # Save
    print(f"  Saving to {out_dir} ...")
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"\n  Train: {X_train.shape}  attacks: {y_train.sum():,}/{len(y_train):,}")
    print(f"  Test : {X_test.shape}  attacks: {y_test.sum():,}/{len(y_test):,}")
    print("UNSW-NB15 preprocessing complete.")


def main(config_path: str) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    preprocess(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UNSW-NB15 dataset")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    main(args.config)
