"""
Evaluate trained models on held-out test sets.

All metrics come from executed code — no fabricated numbers.

Per model outputs (under outputs/metrics/<dataset>/):
    <model>_metrics.json  — accuracy, precision, recall, F1, ROC-AUC
    <model>_confusion.csv — confusion matrix

Per dataset outputs:
    summary_<dataset>.csv                          — metrics table for all models
    outputs/figures/<dataset>/roc_comparison_<dataset>.png  — all 4 ROC curves overlaid

Usage:
    python src/evaluation/evaluate.py --config configs/experiment.yaml
    python src/evaluation/evaluate.py --config configs/experiment.yaml --dataset cicids2017
    make eval
    make eval DATASET=unsw_nb15
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Line style per model for the combined ROC plot
_ROC_STYLES: dict[str, tuple[str, str, float]] = {
    "dummy":               ("dimgray",    ":",  1.2),
    "logistic_regression": ("steelblue",  "-.", 1.5),
    "random_forest":       ("forestgreen", "-", 2.0),
    "xgboost":             ("firebrick",   "-", 2.0),
}


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    metrics_dir: Path,
) -> dict:
    """
    Compute and persist per-model metrics and confusion matrix.
    Returns the metrics dict (including fpr/tpr for the combined ROC plot).

    DummyClassifier note: predict_proba returns a constant-class vector,
    yielding ROC-AUC ≈ 0.5. This is the correct behaviour for a chance
    baseline and is intentionally preserved, not suppressed.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        roc_auc = round(roc_auc_score(y_test, y_prob), 6)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
    except ValueError as exc:
        # Only one class present in y_test (should not happen with these datasets)
        print(f"  WARNING: roc_auc_score failed for {name}: {exc}")
        roc_auc = None
        fpr = tpr = np.array([0.0, 1.0])

    metrics = {
        "model": name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 6),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 6),
        "roc_auc":   roc_auc,
        "n_test":    int(len(y_test)),
        "n_positive": int(y_test.sum()),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_normal", "actual_attack"],
        columns=["pred_normal", "pred_attack"],
    )
    cm_df.to_csv(metrics_dir / f"{name}_confusion.csv")

    # Attach curve arrays for the combined plot (not written to JSON)
    metrics["_fpr"] = fpr
    metrics["_tpr"] = tpr
    return metrics


def plot_combined_roc(
    model_results: list[dict],
    dataset: str,
    figures_dir: Path,
) -> None:
    """Overlay all model ROC curves on a single axes and save as PNG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    for r in model_results:
        color, ls, lw = _ROC_STYLES.get(r["model"], ("black", "-", 1.0))
        auc_label = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "n/a"
        ax.plot(
            r["_fpr"], r["_tpr"],
            label=f"{r['model']} (AUC={auc_label})",
            color=color, linestyle=ls, linewidth=lw,
        )

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="random")
    ax.set_title(f"ROC Curves — {dataset}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = figures_dir / f"roc_comparison_{dataset}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC plot saved: {out_path}")


def evaluate_dataset(cfg: dict, dataset: str) -> None:
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset
    model_dir = Path(cfg["paths"]["models"]) / dataset
    metrics_dir = Path(cfg["paths"]["metrics"]) / dataset
    figures_dir = Path(cfg["paths"]["figures"]) / dataset
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X_test_path = proc_dir / "X_test.npy"
    if not X_test_path.exists():
        raise FileNotFoundError(
            f"Processed test data not found: {X_test_path}\n"
            f"Run 'make preprocess_{dataset}' first."
        )

    print(f"  Loading test data from {proc_dir} ...")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")
    print(f"  X_test: {X_test.shape}  attacks: {y_test.sum():,}/{len(y_test):,}")

    all_metrics: list[dict] = []
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        model_path = model_dir / f"{name}.joblib"
        if not model_path.exists():
            print(f"  SKIP {name}: model file not found at {model_path}")
            continue

        print(f"\n  Evaluating: {name} ...")
        model = joblib.load(model_path)
        metrics = evaluate_model(model, X_test, y_test, name, metrics_dir)

        # Strip internal curve arrays before writing to JSON
        json_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        (metrics_dir / f"{name}_metrics.json").write_text(json.dumps(json_metrics, indent=2))
        all_metrics.append(metrics)

        for k, v in json_metrics.items():
            if k not in ("model", "n_test", "n_positive"):
                print(f"    {k:12s}: {v}")

    if not all_metrics:
        print(f"  No model results found for {dataset}.")
        return

    # Combined ROC plot
    plot_combined_roc(all_metrics, dataset, figures_dir)

    # Summary CSV (public metrics only)
    summary_rows = [{k: v for k, v in m.items() if not k.startswith("_")}
                    for m in all_metrics]
    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_path = metrics_dir / f"summary_{dataset}.csv"
    summary_df.to_csv(summary_path)
    print(f"  Summary CSV: {summary_path}")

    print(f"\nEvaluation on {dataset} complete.")


def main(config_path: str, dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())
    for ds in datasets:
        print(f"\n=== Evaluating dataset: {ds} ===")
        evaluate_dataset(cfg, ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IDS baseline models")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to evaluate (cicids2017 or unsw_nb15). Omit to evaluate all.",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
