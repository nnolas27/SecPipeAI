"""
Advanced statistical tests for paper-quality rigor.

Tests performed:
    1. Bootstrap 95% CI — macro_f1 and pr_auc, using seed=1 predictions
       (configurable bootstrap_reps in experiment.yaml; default 1000)
    2. Wilcoxon signed-rank test — best model vs LogisticRegression,
       paired macro_f1 values across seeds (two-sided)
    3. Cliff's delta effect size — best model vs LR macro_f1 across seeds
       Interpretation: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large
    4. McNemar test — best model vs LR using seed=1 prediction vectors
       (same implementation as stats.py; continuity-corrected chi-square)

Best model: determined as highest mean macro_f1 across seeds, excluding dummy.

Writes:
    outputs/metrics/<dataset>/aggregate/stats_advanced.json
    outputs/metrics/<dataset>/aggregate/stats_advanced.csv
    outputs/tables/stats_advanced_<dataset>.csv
    outputs/tables/stats_advanced_<dataset>.tex

Usage:
    python src/evaluation/stats_advanced.py --config configs/experiment.yaml
    python src/evaluation/stats_advanced.py --config configs/experiment.yaml --dataset cicids2017
    make stats_advanced
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2, wilcoxon


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def _fast_macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    """
    Vectorised macro-F1 for binary (0/1) labels — avoids sklearn overhead.
    Returns 0.0 when a class has zero support in the bootstrap sample.
    """
    tp1 = int(((yp == 1) & (yt == 1)).sum())
    fp1 = int(((yp == 1) & (yt == 0)).sum())
    fn1 = int(((yp == 0) & (yt == 1)).sum())
    denom1 = 2 * tp1 + fp1 + fn1
    f1_1 = (2 * tp1 / denom1) if denom1 > 0 else 0.0

    tp0 = int(((yp == 0) & (yt == 0)).sum())
    fp0 = int(((yp == 0) & (yt == 1)).sum())
    fn0 = int(((yp == 1) & (yt == 0)).sum())
    denom0 = 2 * tp0 + fp0 + fn0
    f1_0 = (2 * tp0 / denom0) if denom0 > 0 else 0.0

    return (f1_0 + f1_1) / 2.0


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
    max_samples: int = 50000,
) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI for a scalar metric over the test set.

    metric: 'macro_f1' or 'pr_auc'

    For speed on large test sets, bootstrap is performed on a stratified
    subsample of up to max_samples rows (default 50,000). The point
    estimate itself (in the metrics JSON) always uses the full test set.

    Returns (lower_2.5%, upper_97.5%) or (nan, nan) if insufficient valid
    bootstrap samples (< 10).
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Stratified subsample for speed — does NOT change the point estimate
    if n > max_samples:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos = max(1, int(max_samples * len(pos_idx) / n))
        n_neg = max(1, max_samples - n_pos)
        sub_idx = np.concatenate([
            rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
        ])
        y_true = y_true[sub_idx]
        y_pred = y_pred[sub_idx]
        y_prob = y_prob[sub_idx]
        n = len(y_true)

    stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "macro_f1":
            stats.append(_fast_macro_f1(yt, y_pred[idx]))
        elif metric == "pr_auc":
            stats.append(float(average_precision_score(yt, y_prob[idx])))

    if len(stats) < 10:
        return float("nan"), float("nan")

    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ── Cliff's delta ─────────────────────────────────────────────────────────────

def cliffs_delta(x: list[float], y: list[float]) -> float:
    """
    Cliff's delta effect size between two distributions x and y.

    δ = (# pairs where xi > yj  −  # pairs where xi < yj) / (n * m)
    Positive δ means x tends to dominate y.
    """
    nx, ny = len(x), len(y)
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (nx * ny)


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.330:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


# ── McNemar ───────────────────────────────────────────────────────────────────

def mcnemar_test(
    y_test: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test with continuity correction (matches stats.py implementation).

    b = cases where A is correct and B is wrong
    c = cases where A is wrong and B is correct
    chi2 = (|b − c| − 1)² / (b + c)
    """
    correct_a = y_pred_a == y_test
    correct_b = y_pred_b == y_test
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())

    if (b + c) == 0:
        return {"b": b, "c": c, "chi2_stat": 0.0, "p_value": 1.0, "significant": False}

    chi2_stat = float((abs(b - c) - 1) ** 2 / (b + c))
    p_val = float(chi2.sf(chi2_stat, df=1))
    return {
        "b": b,
        "c": c,
        "chi2_stat": round(chi2_stat, 6),
        "p_value": round(p_val, 6),
        "significant": bool(p_val < 0.05),
    }


# ── Dataset runner ────────────────────────────────────────────────────────────

def run_stats(cfg: dict, dataset: str) -> None:
    seeds = cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])
    n_bootstrap = cfg.get("experiment", {}).get("bootstrap_reps", 1000)
    bootstrap_max_samples = cfg.get("experiment", {}).get("bootstrap_max_samples", 50000)
    model_names = [m["name"] for m in cfg["models"]]
    metrics_root = Path(cfg["paths"]["metrics"]) / dataset / "runs"
    agg_dir = Path(cfg["paths"]["metrics"]) / dataset / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-seed macro_f1 for all models ─────────────────────────────────
    seed_macro_f1: dict[str, list[float]] = {name: [] for name in model_names}
    for seed in seeds:
        for name in model_names:
            path = metrics_root / f"seed_{seed}" / f"{name}_metrics.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            v = data.get("macro_f1")
            if v is not None:
                seed_macro_f1[name].append(float(v))

    # ── Determine best model (highest mean macro_f1, excluding dummy) ─────────
    non_dummy = {
        k: v for k, v in seed_macro_f1.items()
        if k != "dummy" and len(v) > 0
    }
    if not non_dummy:
        print(f"  No non-dummy results for {dataset}. Run 'make seeds' first.")
        return

    best_model = max(non_dummy, key=lambda k: np.mean(non_dummy[k]))
    best_mean = np.mean(non_dummy[best_model])
    print(f"  Best model: {best_model}  (mean macro_f1={best_mean:.4f} over {len(non_dummy[best_model])} seeds)")

    results: dict = {"dataset": dataset, "best_model": best_model}

    # ── Bootstrap CI (reference seed = seeds[0]) ──────────────────────────────
    ref_seed = seeds[0]
    proc_dir = Path(cfg["paths"]["processed_data"]) / dataset
    y_test = np.load(proc_dir / "y_test.npy")

    for model_key in [best_model, "logistic_regression"]:
        if model_key not in model_names:
            continue
        pred_path = metrics_root / f"seed_{ref_seed}" / f"{model_key}_predictions.npy"
        prob_path = metrics_root / f"seed_{ref_seed}" / f"{model_key}_probas.npy"
        if not pred_path.exists() or not prob_path.exists():
            print(f"  WARNING: predictions not found for {model_key} seed={ref_seed} — skipping bootstrap")
            continue

        y_pred = np.load(pred_path)
        y_prob = np.load(prob_path)

        for metric in ["macro_f1", "pr_auc"]:
            lo, hi = bootstrap_ci(
                y_test, y_pred, y_prob, metric,
                n_bootstrap, seed=ref_seed, max_samples=bootstrap_max_samples,
            )
            key = f"{model_key}_bootstrap_{metric}"
            results[key] = {
                "ci_lower": lo, "ci_upper": hi,
                "seed_used": ref_seed,
                "n_bootstrap": n_bootstrap,
                "bootstrap_max_samples": bootstrap_max_samples,
            }
            ci_str = f"[{lo:.4f}, {hi:.4f}]" if not np.isnan(lo) else "[n/a]"
            print(f"  Bootstrap CI ({model_key}, {metric}, seed={ref_seed}): {ci_str}")

    # ── Wilcoxon signed-rank test (best vs LR, paired across seeds) ───────────
    best_vals = seed_macro_f1.get(best_model, [])
    lr_vals = seed_macro_f1.get("logistic_regression", [])

    if len(best_vals) >= 2 and len(lr_vals) >= 2 and len(best_vals) == len(lr_vals):
        try:
            w_stat, w_p = wilcoxon(best_vals, lr_vals, alternative="two-sided")
            results["wilcoxon"] = {
                "model_A": best_model,
                "model_B": "logistic_regression",
                "statistic": round(float(w_stat), 6),
                "p_value": round(float(w_p), 6),
                "significant": bool(float(w_p) < 0.05),
                "n_seeds": len(best_vals),
            }
            sig = " *" if float(w_p) < 0.05 else ""
            print(f"  Wilcoxon ({best_model} vs LR): stat={w_stat:.4f}  p={w_p:.4f}{sig}")
        except ValueError as exc:
            # wilcoxon raises if all differences are zero (perfect agreement)
            results["wilcoxon"] = {"error": str(exc), "n_seeds": len(best_vals)}
            print(f"  Wilcoxon: {exc}")
    else:
        msg = f"insufficient seeds (best={len(best_vals)}, LR={len(lr_vals)})"
        results["wilcoxon"] = {"error": msg}
        print(f"  WARNING: Wilcoxon skipped — {msg}")

    # ── Cliff's delta (best vs LR) ────────────────────────────────────────────
    if best_vals and lr_vals:
        delta = cliffs_delta(best_vals, lr_vals)
        mag = cliffs_magnitude(delta)
        results["cliffs_delta"] = {
            "model_A": best_model,
            "model_B": "logistic_regression",
            "delta": round(float(delta), 6),
            "magnitude": mag,
        }
        print(f"  Cliff's delta ({best_model} vs LR): δ={delta:.4f}  ({mag})")

    # ── McNemar (best vs LR, seed=ref_seed) ───────────────────────────────────
    best_pred = metrics_root / f"seed_{ref_seed}" / f"{best_model}_predictions.npy"
    lr_pred = metrics_root / f"seed_{ref_seed}" / "logistic_regression_predictions.npy"
    if best_pred.exists() and lr_pred.exists():
        mcn = mcnemar_test(y_test, np.load(best_pred), np.load(lr_pred))
        results["mcnemar"] = {
            "model_A": best_model,
            "model_B": "logistic_regression",
            "seed_used": ref_seed,
            **mcn,
        }
        sig = " *" if mcn["significant"] else ""
        print(
            f"  McNemar ({best_model} vs LR, seed={ref_seed}): "
            f"b={mcn['b']}  c={mcn['c']}  chi2={mcn['chi2_stat']:.4f}  p={mcn['p_value']:.4f}{sig}"
        )
    else:
        print(f"  WARNING: McNemar skipped — prediction files not found (run 'make seeds')")

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = agg_dir / "stats_advanced.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Stats JSON: {json_path}")

    _write_csv_tex(results, dataset, agg_dir, tables_dir)


def _write_csv_tex(results: dict, dataset: str, agg_dir: Path, tables_dir: Path) -> None:
    """Flatten results dict into a tabular CSV + LaTeX for paper inclusion."""
    rows = []

    # Bootstrap CI rows
    for key, val in results.items():
        if "_bootstrap_" not in key or not isinstance(val, dict):
            continue
        parts = key.split("_bootstrap_")
        model_name = parts[0]
        metric_name = parts[1]
        lo, hi = val.get("ci_lower", float("nan")), val.get("ci_upper", float("nan"))
        ci_str = f"[{lo:.4f}, {hi:.4f}]" if not (np.isnan(lo) or np.isnan(hi)) else "[n/a]"
        rows.append({
            "test": "bootstrap_95ci",
            "model_A": model_name,
            "model_B": "",
            "metric": metric_name,
            "statistic": "",
            "ci_95": ci_str,
            "p_value": "",
            "significant": "",
            "extra": f"seed={val.get('seed_used','')} n_bootstrap={val.get('n_bootstrap','')}",
        })

    # Wilcoxon
    w = results.get("wilcoxon", {})
    if "p_value" in w:
        rows.append({
            "test": "wilcoxon",
            "model_A": w.get("model_A", ""),
            "model_B": w.get("model_B", ""),
            "metric": "macro_f1",
            "statistic": f"{w['statistic']:.4f}",
            "ci_95": "",
            "p_value": f"{w['p_value']:.4f}",
            "significant": w.get("significant", ""),
            "extra": f"n_seeds={w.get('n_seeds','')}",
        })

    # Cliff's delta
    cd = results.get("cliffs_delta", {})
    if "delta" in cd:
        rows.append({
            "test": "cliffs_delta",
            "model_A": cd.get("model_A", ""),
            "model_B": cd.get("model_B", ""),
            "metric": "macro_f1",
            "statistic": f"{cd['delta']:.4f}",
            "ci_95": "",
            "p_value": "",
            "significant": "",
            "extra": cd.get("magnitude", ""),
        })

    # McNemar
    m = results.get("mcnemar", {})
    if "chi2_stat" in m:
        rows.append({
            "test": "mcnemar",
            "model_A": m.get("model_A", ""),
            "model_B": m.get("model_B", ""),
            "metric": "predictions",
            "statistic": f"{m['chi2_stat']:.4f}",
            "ci_95": "",
            "p_value": f"{m['p_value']:.4f}",
            "significant": m.get("significant", ""),
            "extra": f"seed={m.get('seed_used','')}",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)

    csv_path = agg_dir / "stats_advanced.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Stats CSV: {csv_path}")

    tables_csv = tables_dir / f"stats_advanced_{dataset}.csv"
    df.to_csv(tables_csv, index=False)

    tex = df.to_latex(
        index=False,
        escape=True,
        caption=f"Advanced statistical tests — {dataset} (* p $<$ 0.05)",
        label=f"tab:stats_advanced_{dataset}",
    )
    tables_tex = tables_dir / f"stats_advanced_{dataset}.tex"
    tables_tex.write_text(tex)
    print(f"  Tables: {tables_csv}  {tables_tex}")


def main(config_path: str, dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())
    for ds in datasets:
        print(f"\n=== Advanced stats: {ds} ===")
        run_stats(cfg, ds)

    print("\nAdvanced stats complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap CI, Wilcoxon, Cliff's delta, and McNemar tests"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset (cicids2017 or unsw_nb15). Omit for all.",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
