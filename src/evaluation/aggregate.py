"""
Aggregate per-seed metrics into mean ± std summary tables.

Reads:
    outputs/metrics/<dataset>/runs/seed_<seed>/<model>_metrics.json

Writes:
    outputs/metrics/<dataset>/aggregate/summary_mean_std.csv
    outputs/metrics/<dataset>/aggregate/summary_mean_std.tex
    outputs/tables/aggregate_<dataset>.csv
    outputs/tables/aggregate_<dataset>.tex

Metrics aggregated: macro_f1, weighted_f1, pr_auc, roc_auc, fpr, f1, accuracy

Usage:
    python src/evaluation/aggregate.py --config configs/experiment.yaml
    python src/evaluation/aggregate.py --config configs/experiment.yaml --dataset cicids2017
    make aggregate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_METRICS = ["macro_f1", "weighted_f1", "pr_auc", "roc_auc", "fpr", "f1", "accuracy"]


def aggregate_dataset(cfg: dict, dataset: str) -> pd.DataFrame:
    """Read all per-seed JSON files and compute mean±std per model per metric."""
    seeds = cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])
    model_names = [m["name"] for m in cfg["models"]]
    metrics_root = Path(cfg["paths"]["metrics"]) / dataset / "runs"

    rows = []
    for name in model_names:
        seed_vals: dict[str, list[float]] = {m: [] for m in _METRICS}
        found_seeds: list[int] = []

        for seed in seeds:
            path = metrics_root / f"seed_{seed}" / f"{name}_metrics.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            found_seeds.append(seed)
            for m in _METRICS:
                v = data.get(m)
                if v is not None:
                    seed_vals[m].append(float(v))

        if not found_seeds:
            print(f"  WARNING: no seed results found for {name} on {dataset}")
            continue

        row: dict = {"model": name, "n_seeds": len(found_seeds)}
        for m in _METRICS:
            vals = seed_vals[m]
            if vals:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            else:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("model")


def df_to_latex(df: pd.DataFrame, dataset: str) -> str:
    """Build a compact LaTeX table showing mean±std for key metrics."""
    key_cols = []
    for m in ["macro_f1", "weighted_f1", "pr_auc", "roc_auc"]:
        for suffix in ["mean", "std"]:
            col = f"{m}_{suffix}"
            if col in df.columns:
                key_cols.append(col)

    display = df[key_cols + (["n_seeds"] if "n_seeds" in df.columns else [])].copy()
    latex = display.to_latex(
        float_format="%.4f",
        na_rep="—",
        escape=True,
        caption=f"Aggregate metrics (mean \\pm std over seeds) — {dataset}",
        label=f"tab:aggregate_{dataset}",
    )
    return latex


def print_summary(df: pd.DataFrame) -> None:
    for m in _METRICS:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col not in df.columns:
            continue
        print(f"  {m}:")
        for model, row in df.iterrows():
            mean_v = row.get(mean_col)
            std_v = row.get(std_col, 0.0)
            if mean_v is not None:
                print(f"    {model:<30s}: {mean_v:.4f} ± {std_v:.4f}")


def main(config_path: str, dataset: str | None = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    datasets = [dataset] if dataset else list(cfg["datasets"].keys())
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        print(f"\n=== Aggregating: {ds} ===")
        df = aggregate_dataset(cfg, ds)

        if df.empty:
            print(f"  No results found for {ds}. Run 'make seeds' first.")
            continue

        print_summary(df)

        # Per-dataset aggregate directory
        agg_dir = Path(cfg["paths"]["metrics"]) / ds / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)

        csv_path = agg_dir / "summary_mean_std.csv"
        df.to_csv(csv_path, float_format="%.6f")
        print(f"\n  CSV : {csv_path}")

        tex_path = agg_dir / "summary_mean_std.tex"
        tex_path.write_text(df_to_latex(df, ds))
        print(f"  TEX : {tex_path}")

        # Mirror to outputs/tables/ for paper-ready access
        tables_csv = tables_dir / f"aggregate_{ds}.csv"
        df.to_csv(tables_csv, float_format="%.6f")

        tables_tex = tables_dir / f"aggregate_{ds}.tex"
        tables_tex.write_text(df_to_latex(df, ds))
        print(f"  Also: {tables_csv}  {tables_tex}")

    print("\nAggregation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate per-seed metrics into mean±std tables")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset to aggregate (cicids2017 or unsw_nb15). Omit for all.",
    )
    args = parser.parse_args()
    main(args.config, args.dataset)
