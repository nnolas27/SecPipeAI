# SecPipeAI

Reproducible ML-based network intrusion detection pipeline using the **CICIDS2017** and **UNSW-NB15** benchmark datasets. Four baseline classifiers are trained and evaluated with full statistical testing — all metrics come from executed code.

## Features

- CPU-only, 8 GB RAM compatible
- Leakage-free preprocessing (train-only fitting for imputers, scalers, and encoders)
- Four baseline models: Dummy, Logistic Regression, Random Forest, XGBoost
- Per-model metrics, confusion matrices, and ROC curves
- Pairwise McNemar significance tests
- Consolidated CSV + LaTeX export for papers
- Fully reproducible via Makefile and pinned dependencies

## Datasets

| Dataset | Source | Size |
|---|---|---|
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | UNB CIC | ~2.8M rows, 8 CSV files |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | UNSW | Official train/test split |

Raw data must be placed under `data/raw/` manually (see `make data` for instructions).

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/nnolas27/SecPipeAI.git
cd SecPipeAI
make setup
source .venv/bin/activate

# 2. Download datasets (follow printed instructions)
make data

# 3. Run full pipeline
make all
```

## Pipeline

```
make setup              # create venv, install pinned deps
make data               # print download instructions + verify checksums
make preprocess_cicids2017
make preprocess_unsw_nb15
make train              # train 4 models × 2 datasets
make eval               # metrics, confusion matrices, ROC plots
make stats              # pairwise McNemar tests → CSV + LaTeX
make export             # consolidated summary CSV + LaTeX table
make clean              # remove outputs/ and data/processed/ (raw data preserved)
```

Scope any step to a single dataset:

```bash
make train DATASET=cicids2017
make eval  DATASET=unsw_nb15
```

## Models

| Model | Notes |
|---|---|
| `dummy` | Most-frequent baseline |
| `logistic_regression` | `lbfgs` solver, `max_iter=1000` |
| `random_forest` | 100 estimators |
| `xgboost` | `tree_method=hist`, CPU-only |

## Outputs

```
outputs/
├── models/<dataset>/       # *.joblib + *_meta.json
├── metrics/<dataset>/      # *_metrics.json, *_confusion.csv,
│                           # mcnemar_<ds>.{csv,tex}, summary.tex
└── figures/<dataset>/      # roc_comparison_<dataset>.png
```

## Requirements

Python 3.10+ with pinned dependencies in `requirements.txt`:

```
numpy==1.26.4   pandas==2.2.2   scikit-learn==1.5.1
xgboost==2.1.1  matplotlib==3.9.2  scipy==1.13.1
```

## Reproducibility

- `configs/experiment.yaml` — all hyperparameters and dataset paths
- `configs/checksums.yaml` — SHA-256 checksums for raw data files
- `requirements.txt` — fully pinned Python dependencies
- `random_state=42` used throughout
