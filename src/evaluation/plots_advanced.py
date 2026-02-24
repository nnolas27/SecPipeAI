"""
Publication-quality advanced plots.

Generates three figure types per dataset:
    1. PR curve overlay — all models, reference seed (seed=1 by default)
       outputs/figures/<dataset>/pr_comparison_<dataset>.png
    2. Confusion matrix heatmap — best model, reference seed
       outputs/figures/<dataset>/confusion_best_<dataset>.png
    3. Macro-F1 bar chart with error bars (mean ± std across all seeds)
       outputs/figures/<dataset>/macro_f1_bar_<dataset>.png

Best model is read from stats_advanced.json if it exists; otherwise falls
back to the non-dummy model with the highest mean macro_f1 in seed results.

All figures are also mirrored to outputs/figures/ for the paper_artifacts pipeline.

Usage:
    python src/evaluation/plots_advanced.py --config configs/experiment.yaml
    python src/evaluation/plots_advanced.py --config configs/experiment.yaml --dataset cicids2017
    make plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless / CPU-only renderer
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_curve

# ── Shared style constants ────────────────────────────────────────────────────
_MODEL_STYLES: dict[str, tuple[str, str, float]] = {
    "dummy":               ("dimgray",     ":",  1.2),
    "logistic_regression": ("steelblue",  "-.", 1.5),
    "random_forest":       ("forestgreen", "-", 2.0),
    "xgboost":             ("firebrick",   "-", 2.0),
}
_MODEL_DISPLAY: dict[str, str] = {
    "dummy":               "Dummy",
    "logistic_regression": "LR",
    "random_forest":       "RF",
    "xgboost":             "XGBoost",
}


# ── 1. PR curve overlay ───────────────────────────────────────────────────────

def plot_pr_curves(cfg: dict, dataset: str, figures_dir: Path) -> None:
    seeds = cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])
    ref_seed = seeds[0]
    model_names = [m["name"] for m in cfg["models"]]
    metrics_root = Path(cfg["paths"]["metrics"]) / dataset / "runs"
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset

    y_test_path = proc_dir / "y_test.npy"
    if not y_test_path.exists():
        print(f"  SKIP PR curves: y_test not found at {y_test_path}")
        return

    y_test = np.load(y_test_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False

    for name in model_names:
        prob_path = metrics_root / f"seed_{ref_seed}" / f"{name}_probas.npy"
        metric_path = metrics_root / f"seed_{ref_seed}" / f"{name}_metrics.json"
        if not prob_path.exists():
            continue

        y_prob = np.load(prob_path)
        pr_auc = None
        if metric_path.exists():
            data = json.loads(metric_path.read_text())
            pr_auc = data.get("pr_auc")

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        color, ls, lw = _MODEL_STYLES.get(name, ("black", "-", 1.0))
        label = _MODEL_DISPLAY.get(name, name)
        if pr_auc is not None:
            label += f" (AP={pr_auc:.4f})"
        ax.plot(recall, precision, label=label, color=color, linestyle=ls, linewidth=lw)
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"  SKIP PR curves: no probabilities found for {dataset} (run 'make seeds' first)")
        return

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {dataset} (seed={ref_seed})")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out_path = figures_dir / f"pr_comparison_{dataset}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PR curves: {out_path}")


# ── 2. Confusion matrix ───────────────────────────────────────────────────────

def _find_best_model(cfg: dict, dataset: str) -> str | None:
    """Determine best model: prefer stats_advanced.json, else highest mean macro_f1."""
    agg_dir = Path(cfg["paths"]["metrics"]) / dataset / "aggregate"
    stats_path = agg_dir / "stats_advanced.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        best = stats.get("best_model")
        if best:
            return best

    # Fallback: compute from seed metric files
    seeds = cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])
    model_names = [m["name"] for m in cfg["models"] if m["name"] != "dummy"]
    metrics_root = Path(cfg["paths"]["metrics"]) / dataset / "runs"
    best_model, best_mean = None, -1.0
    for name in model_names:
        vals = []
        for seed in seeds:
            p = metrics_root / f"seed_{seed}" / f"{name}_metrics.json"
            if p.exists():
                v = json.loads(p.read_text()).get("macro_f1")
                if v is not None:
                    vals.append(float(v))
        if vals and np.mean(vals) > best_mean:
            best_mean = float(np.mean(vals))
            best_model = name
    return best_model


def plot_confusion_matrix(cfg: dict, dataset: str, figures_dir: Path) -> None:
    seeds = cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])
    ref_seed = seeds[0]
    metrics_root = Path(cfg["paths"]["metrics"]) / dataset / "runs"
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset

    y_test_path = proc_dir / "y_test.npy"
    if not y_test_path.exists():
        print(f"  SKIP confusion matrix: y_test not found at {y_test_path}")
        return

    best_model = _find_best_model(cfg, dataset)
    if best_model is None:
        print(f"  SKIP confusion matrix: cannot determine best model for {dataset}")
        return

    pred_path = metrics_root / f"seed_{ref_seed}" / f"{best_model}_predictions.npy"
    if not pred_path.exists():
        print(f"  SKIP confusion matrix: predictions not found for {best_model} seed={ref_seed}")
        return

    y_test = np.load(y_test_path)
    y_pred = np.load(pred_path)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Normal", "Attack"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    display_name = _MODEL_DISPLAY.get(best_model, best_model)
    ax.set_title(f"Confusion Matrix — {display_name}\n{dataset} (seed={ref_seed})")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = figures_dir / f"confusion_best_{dataset}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix: {out_path}")


# ── 3. Macro-F1 bar chart with error bars ────────────────────────────────────

def plot_macro_f1_bar(cfg: dict, dataset: str, figures_dir: Path) -> None:
    seeds = cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])
    model_names = [m["name"] for m in cfg["models"]]
    metrics_root = Path(cfg["paths"]["metrics"]) / dataset / "runs"

    means: list[float] = []
    stds: list[float] = []
    labels: list[str] = []
    colors: list[str] = []

    for name in model_names:
        vals = []
        for seed in seeds:
            path = metrics_root / f"seed_{seed}" / f"{name}_metrics.json"
            if path.exists():
                v = json.loads(path.read_text()).get("macro_f1")
                if v is not None:
                    vals.append(float(v))
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        labels.append(_MODEL_DISPLAY.get(name, name))
        colors.append(_MODEL_STYLES.get(name, ("steelblue", "-", 1.0))[0])

    if not means:
        print(f"  SKIP bar chart: no macro_f1 data for {dataset}")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.5, alpha=0.85,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Macro-F1 Mean ± Std ({len(seeds)} seeds) — {dataset}")

    # Headroom for text labels
    y_top = max(m + s for m, s in zip(means, stds)) + 0.07
    ax.set_ylim(0, min(1.05, y_top))

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    out_path = figures_dir / f"macro_f1_bar_{dataset}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Macro-F1 bar chart: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str, dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())

    for ds in datasets:
        print(f"\n=== Advanced plots: {ds} ===")
        figures_dir = Path(cfg["paths"]["figures"]) / ds
        figures_dir.mkdir(parents=True, exist_ok=True)

        plot_pr_curves(cfg, ds, figures_dir)
        plot_confusion_matrix(cfg, ds, figures_dir)
        plot_macro_f1_bar(cfg, ds, figures_dir)

    print("\nAdvanced plots complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PR curves, confusion matrix, and macro-F1 bar chart"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset (cicids2017 or unsw_nb15). Omit for all.",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
