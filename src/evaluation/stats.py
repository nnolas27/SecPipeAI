"""
McNemar's significance test for pairwise model comparison.

For each dataset, all pairs of models are compared using McNemar's test on
their binary prediction vectors against the held-out test labels.

Theory:
    McNemar's test uses the off-diagonal cells of the 2x2 agreement table:
        b = cases where model A is correct, model B is wrong
        c = cases where model A is wrong, model B is correct
    Statistic (with continuity correction):
        chi2 = (|b - c| - 1)^2 / (b + c)    (when b + c > 0)
        p = 1 - chi2.cdf(chi2, df=1)

    NOTE: scipy.stats.chi2_contingency tests independence on the full 2x2
    table — it is NOT McNemar's test. We compute the McNemar statistic
    directly using scipy.stats.chi2.sf to avoid this common mistake.

Outputs per dataset (under outputs/metrics/<dataset>/):
    mcnemar_<dataset>.csv  — model_A, model_B, b, c, chi2, p_value, significant
    mcnemar_<dataset>.tex  — LaTeX table of the same results

Usage:
    python src/evaluation/stats.py --config configs/experiment.yaml
    python src/evaluation/stats.py --config configs/experiment.yaml --dataset cicids2017
    make stats
    make stats DATASET=unsw_nb15
"""

import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2


def mcnemar_test(
    y_test: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """
    Compute McNemar's test with continuity correction.

    Returns dict with keys: b, c, chi2_stat, p_value, significant (p < 0.05).
    When b + c == 0 (models agree on all samples), p_value is set to 1.0.
    """
    correct_a = y_pred_a == y_test
    correct_b = y_pred_b == y_test

    b = int((correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a & correct_b).sum())   # A wrong, B right

    if (b + c) == 0:
        return {
            "b": b, "c": c,
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    chi2_stat = float((abs(b - c) - 1) ** 2 / (b + c))
    p_val = float(chi2.sf(chi2_stat, df=1))

    return {
        "b": b,
        "c": c,
        "chi2_stat": round(chi2_stat, 6),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05,
    }


def to_latex(df: pd.DataFrame, caption: str) -> str:
    """
    Generate a LaTeX table suitable for inclusion in a paper.
    Significant rows are bolded via a post-processing step.
    """
    df_display = df.copy()
    # Format floats for readability
    df_display["chi2_stat"] = df_display["chi2_stat"].map("{:.4f}".format)
    df_display["p_value"] = df_display["p_value"].map("{:.4f}".format)

    latex = df_display.to_latex(
        index=False,
        escape=True,
        float_format="%.4f",
        caption=caption,
        label=f"tab:{caption.lower().replace(' ', '_')[:30]}",
    )
    return latex


def run_stats_dataset(cfg: dict, dataset: str) -> None:
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset
    model_dir = Path(cfg["paths"]["models"]) / dataset
    metrics_dir = Path(cfg["paths"]["metrics"]) / dataset
    metrics_dir.mkdir(parents=True, exist_ok=True)

    y_test_path = proc_dir / "y_test.npy"
    if not y_test_path.exists():
        raise FileNotFoundError(
            f"Processed test labels not found: {y_test_path}\n"
            f"Run 'make preprocess_{dataset}' first."
        )

    y_test = np.load(y_test_path)
    X_test = np.load(proc_dir / "X_test.npy")

    # Load models and collect predictions
    model_names = [m["name"] for m in cfg["models"]]
    predictions: dict[str, np.ndarray] = {}
    for name in model_names:
        model_path = model_dir / f"{name}.joblib"
        if not model_path.exists():
            print(f"  SKIP {name}: model not found at {model_path}")
            continue
        model = joblib.load(model_path)
        predictions[name] = model.predict(X_test)
        print(f"  Loaded predictions for {name}")

    available = list(predictions.keys())
    if len(available) < 2:
        print(f"  Need at least 2 models for McNemar tests. Found: {available}")
        return

    rows = []
    for name_a, name_b in itertools.combinations(available, 2):
        result = mcnemar_test(y_test, predictions[name_a], predictions[name_b])
        rows.append({
            "model_A": name_a,
            "model_B": name_b,
            **result,
        })
        sig = "*" if result["significant"] else ""
        print(
            f"  {name_a} vs {name_b}: "
            f"b={result['b']} c={result['c']} "
            f"chi2={result['chi2_stat']:.4f} p={result['p_value']:.4f}{sig}"
        )

    df = pd.DataFrame(rows)

    csv_path = metrics_dir / f"mcnemar_{dataset}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  McNemar CSV saved: {csv_path}")

    caption = f"McNemar pairwise significance tests — {dataset} (* p < 0.05)"
    tex_path = metrics_dir / f"mcnemar_{dataset}.tex"
    tex_path.write_text(to_latex(df, caption))
    print(f"  McNemar LaTeX saved: {tex_path}")


def main(config_path: str, dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())
    for ds in datasets:
        print(f"\n=== McNemar tests: {ds} ===")
        run_stats_dataset(cfg, ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pairwise McNemar significance tests between models"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to test (cicids2017 or unsw_nb15). Omit for all.",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
