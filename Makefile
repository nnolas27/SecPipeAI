PYTHON := python3
PIP    := pip3
CONFIG := configs/experiment.yaml

# Optional: scope train/eval/stats/seeds to one dataset.
# Example: make train DATASET=cicids2017
DATASET_ARG := $(if $(DATASET),--dataset $(DATASET),)

# Number of seeds for multi-seed runs (generates seeds [1..N]).
# Override: make seeds SEEDS=10
SEEDS ?= 5

.PHONY: all setup data preprocess_cicids2017 preprocess_unsw_nb15 \
        train eval stats export clean help \
        seeds aggregate stats_advanced plots paper_artifacts

## Default: full pipeline — preprocess, single-run train/eval/stats, seed runs, paper artifacts
all: preprocess_cicids2017 preprocess_unsw_nb15 train eval stats seeds paper_artifacts

## Set up Python virtual environment and install dependencies
setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

## Print download instructions; verify SHA-256 checksums for any files already present
data:
	$(PYTHON) src/data/download.py --config $(CONFIG)

## Preprocess CICIDS2017 CSV files → data/processed/cicids2017/
preprocess_cicids2017:
	$(PYTHON) src/features/preprocess_cicids2017.py --config $(CONFIG)

## Preprocess UNSW-NB15 CSV files → data/processed/unsw_nb15/
preprocess_unsw_nb15:
	$(PYTHON) src/features/preprocess_unsw_nb15.py --config $(CONFIG)

## Train all 4 models (Dummy, LR, RF, XGB); optional: make train DATASET=cicids2017
train:
	$(PYTHON) src/models/train.py --config $(CONFIG) $(DATASET_ARG)

## Evaluate saved models and write metrics/ROC plots; optional: make eval DATASET=unsw_nb15
eval:
	$(PYTHON) src/evaluation/evaluate.py --config $(CONFIG) $(DATASET_ARG)

## Pairwise McNemar significance tests; optional: make stats DATASET=cicids2017
stats:
	$(PYTHON) src/evaluation/stats.py --config $(CONFIG) $(DATASET_ARG)

## Export consolidated CSV + LaTeX tables (outputs/metrics/summary.tex)
export:
	$(PYTHON) src/evaluation/export.py --config $(CONFIG)

## Run all models across N seeds (default N=5). Override: make seeds SEEDS=10
## Seeds list will be [1, 2, ..., N]. Optional: make seeds SEEDS=5 DATASET=cicids2017
seeds:
	$(PYTHON) src/models/seed_runner.py --config $(CONFIG) --n-seeds $(SEEDS) $(DATASET_ARG)

## Build mean±std aggregate tables from per-seed results
## Optional: make aggregate DATASET=unsw_nb15
aggregate:
	$(PYTHON) src/evaluation/aggregate.py --config $(CONFIG) $(DATASET_ARG)

## Bootstrap 95% CI + Wilcoxon + Cliff's delta + McNemar advanced stats
## Optional: make stats_advanced DATASET=cicids2017
stats_advanced:
	$(PYTHON) src/evaluation/stats_advanced.py --config $(CONFIG) $(DATASET_ARG)

## Generate PR curves, confusion matrix, and macro-F1 bar chart (per dataset)
## Optional: make plots DATASET=unsw_nb15
plots:
	$(PYTHON) src/evaluation/plots_advanced.py --config $(CONFIG) $(DATASET_ARG)

## Run aggregate + stats_advanced + plots + export + generate docs/results_summary.md
paper_artifacts:
	$(PYTHON) src/evaluation/paper_artifacts.py --config $(CONFIG)

## Remove all generated outputs and processed data; raw data in data/raw/ is preserved
clean:
	rm -rf outputs/models/cicids2017/ outputs/models/unsw_nb15/
	rm -rf outputs/metrics/cicids2017/ outputs/metrics/unsw_nb15/
	rm -f  outputs/metrics/summary.tex
	rm -rf outputs/figures/cicids2017/ outputs/figures/unsw_nb15/
	rm -rf outputs/tables/
	rm -rf data/processed/cicids2017/ data/processed/unsw_nb15/
	rm -f  docs/results_summary.md docs/reproducibility.md
	@echo "Cleaned outputs and processed data. Raw data in data/raw/ preserved."

help:
	@grep -E '^##' Makefile | sed 's/## /  /'
