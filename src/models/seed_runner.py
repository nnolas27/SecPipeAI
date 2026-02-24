"""
Multi-seed runner: trains and evaluates all models across multiple seeds.

For each (dataset, seed), the model random_state is set to the seed value.
Preprocessing is deterministic and NOT re-run per seed — this avoids
confounding model variance with data-split variance.

CICIDS2017 cap: if datasets.cicids2017.cap_rows is set in config, the
training set is stratified-subsampled to that size before each seed's
training run. The test set is never capped so evaluation is on the full
held-out split.

Outputs per (dataset, seed):
    outputs/models/<dataset>/seed_<seed>/<model>.joblib
    outputs/models/<dataset>/seed_<seed>/<model>_meta.json
    outputs/metrics/<dataset>/runs/seed_<seed>/<model>_metrics.json
    outputs/metrics/<dataset>/runs/seed_<seed>/<model>_predictions.npy
    outputs/metrics/<dataset>/runs/seed_<seed>/<model>_probas.npy

Metrics saved (extended set for aggregation / advanced stats):
    accuracy, precision, recall, f1 (binary)
    macro_f1, weighted_f1
    pr_auc  (average_precision_score)
    roc_auc
    fpr     (false positive rate = FP / (FP + TN))

Usage:
    python src/models/seed_runner.py --config configs/experiment.yaml
    python src/models/seed_runner.py --config configs/experiment.yaml --n-seeds 10
    python src/models/seed_runner.py --config configs/experiment.yaml --seeds 1 2 3 4 5
    python src/models/seed_runner.py --config configs/experiment.yaml --dataset cicids2017
    make seeds
    make seeds SEEDS=10
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


def build_model(model_cfg: dict, seed: int):
    """Build a model with the given seed as random_state (where applicable)."""
    name = model_cfg["name"]
    params = model_cfg.get("params", {})

    if name == "dummy":
        return DummyClassifier(strategy=params.get("strategy", "most_frequent"))

    elif name == "logistic_regression":
        return LogisticRegression(
            max_iter=params.get("max_iter", 2000),
            C=params.get("C", 1.0),
            random_state=seed,
            solver="lbfgs",
            n_jobs=-1,
        )

    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=min(params.get("n_estimators", 300), 300),
            max_depth=params.get("max_depth", 20),  # None allowed
            random_state=seed,
            n_jobs=-1,
        )

    elif name == "xgboost":
        return XGBClassifier(
            n_estimators=min(params.get("n_estimators", 500), 500),
            max_depth=min(params.get("max_depth", 8), 8),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=min(params.get("subsample", 0.9), 0.9),
            colsample_bytree=min(params.get("colsample_bytree", 0.9), 0.9),
            random_state=seed,
            tree_method="hist",
            device="cpu",
            eval_metric="logloss",
        )

    else:
        raise ValueError(f"Unknown model: {name}")


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute extended metrics for aggregation and advanced stats."""
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_test, y_prob))
    except ValueError:
        pr_auc = None

    return {
        "accuracy":    round(float(accuracy_score(y_test, y_pred)), 6),
        "precision":   round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall":      round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1":          round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "macro_f1":    round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 6),
        "pr_auc":      round(pr_auc, 6) if pr_auc is not None else None,
        "roc_auc":     round(roc_auc, 6) if roc_auc is not None else None,
        "fpr":         round(fpr_val, 6),
        "n_test":      int(len(y_test)),
        "n_positive":  int(y_test.sum()),
    }


def run_seed(cfg: dict, dataset: str, seed: int) -> None:
    """Train and evaluate all models for one (dataset, seed) pair."""
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset
    seed_model_dir = Path(cfg["paths"]["models"]) / dataset / f"seed_{seed}"
    seed_metrics_dir = Path(cfg["paths"]["metrics"]) / dataset / "runs" / f"seed_{seed}"
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics_dir.mkdir(parents=True, exist_ok=True)

    if not (proc_dir / "X_train.npy").exists():
        raise FileNotFoundError(
            f"Processed data not found at {proc_dir}.\n"
            f"Run 'make preprocess_{dataset}' first."
        )

    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    # Optional training-set cap (stratified) — test set is never capped
    ds_cfg = cfg["datasets"][dataset]
    cap = ds_cfg.get("cap_rows")
    if cap and len(y_train) > cap:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
        cap_idx, _ = next(sss.split(X_train, y_train))
        X_train = X_train[cap_idx]
        y_train = y_train[cap_idx]
        print(f"  [cap] Training set reduced to {len(y_train):,} rows (stratified, seed={seed})")

    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}  seed={seed}")

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        print(f"  [{seed}] Training {name} ...", end="", flush=True)

        model = build_model(model_cfg, seed)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s")

        # Save model and meta
        joblib.dump(model, seed_model_dir / f"{name}.joblib")
        meta = {
            "model": name,
            "dataset": dataset,
            "seed": seed,
            "train_shape": list(X_train.shape),
            "train_time_s": round(elapsed, 2),
        }
        (seed_model_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        metrics["dataset"] = dataset
        metrics["seed"] = seed

        # Save metrics, predictions, and probabilities
        (seed_metrics_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        np.save(seed_metrics_dir / f"{name}_predictions.npy", y_pred)
        np.save(seed_metrics_dir / f"{name}_probas.npy", y_prob)

        print(
            f"    macro_f1={metrics['macro_f1']:.4f}  "
            f"pr_auc={metrics['pr_auc'] if metrics['pr_auc'] is not None else 'N/A'}  "
            f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}"
        )


def resolve_seeds(args_seeds, args_n_seeds, cfg: dict) -> list[int]:
    """Resolve the seed list from CLI args or config."""
    if args_seeds:
        return args_seeds
    if args_n_seeds is not None:
        return list(range(1, args_n_seeds + 1))
    return cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])


def main(config_path: str, seeds: list[int], dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())
    print(f"Seeds: {seeds}")

    for ds in datasets:
        print(f"\n=== Seed runs: {ds} ===")
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            run_seed(cfg, ds, seed)

    print("\nAll seed runs complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed model training and evaluation")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        help="Number of seeds (generates [1..n]). E.g. --n-seeds 5 → seeds [1,2,3,4,5]",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seed list (overrides --n-seeds and config). E.g. --seeds 1 2 3 4 5",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset to run (cicids2017 or unsw_nb15). Omit to run all.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        _cfg = yaml.safe_load(fh)

    seeds = resolve_seeds(args.seeds, args.n_seeds, _cfg)
    main(args.config, seeds, args.dataset)
