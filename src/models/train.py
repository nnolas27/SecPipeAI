"""
Train intrusion detection baseline models on preprocessed feature matrices.

Models (CPU-only):
    dummy               : DummyClassifier(strategy="most_frequent") — chance baseline
    logistic_regression : LogisticRegression(solver="lbfgs", n_jobs=-1)
    random_forest       : RandomForestClassifier(n_jobs=-1)
    xgboost             : XGBClassifier(tree_method="hist", device="cpu")

Artifacts saved per dataset:
    outputs/models/<dataset>/<model_name>.joblib
    outputs/models/<dataset>/<model_name>_meta.json

Usage:
    python src/models/train.py --config configs/experiment.yaml
    python src/models/train.py --config configs/experiment.yaml --dataset cicids2017
    make train
    make train DATASET=unsw_nb15
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(model_cfg: dict):
    name = model_cfg["name"]
    params = model_cfg.get("params", {})

    if name == "dummy":
        return DummyClassifier(
            strategy=params.get("strategy", "most_frequent"),
        )
    elif name == "logistic_regression":
        return LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            C=params.get("C", 1.0),
            random_state=params.get("random_state", 42),
            solver="lbfgs",
            n_jobs=-1,
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    elif name == "xgboost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=params.get("random_state", 42),
            tree_method="hist",
            device="cpu",
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train_dataset(cfg: dict, dataset: str) -> None:
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset
    model_dir = Path(cfg["paths"]["models"]) / dataset
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train_path = proc_dir / "X_train.npy"
    if not X_train_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {X_train_path}\n"
            f"Run 'make preprocess_{dataset}' first."
        )

    print(f"  Loading processed training data from {proc_dir} ...")
    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    print(f"  X_train: {X_train.shape}  attacks: {y_train.sum():,}/{len(y_train):,}")

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        print(f"\n  Training: {name} ...")
        if name == "logistic_regression":
            print("    (LogisticRegression may take several minutes on large datasets)")

        model = build_model(model_cfg)

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        out_path = model_dir / f"{name}.joblib"
        joblib.dump(model, out_path)

        meta = {
            "model": name,
            "dataset": dataset,
            "params": model_cfg.get("params", {}),
            "train_shape": list(X_train.shape),
            "train_time_s": round(elapsed, 2),
        }
        (model_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Saved {out_path}  ({elapsed:.1f}s)")

    print(f"\nTraining on {dataset} complete.")


def main(config_path: str, dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())
    for ds in datasets:
        print(f"\n=== Training on dataset: {ds} ===")
        train_dataset(cfg, ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IDS baseline models")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to train on (cicids2017 or unsw_nb15). Omit to train on all.",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
