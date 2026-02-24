You are my “Research Ops + Reproducibility Engineer + ML Security Methods Editor”.

CONTEXT
- Repo already exists and currently supports CICIDS2017 + UNSW-NB15 preprocessing, training (Dummy/LR/RF/XGBoost), evaluation, McNemar, and export.
- I am on Ubuntu (Dell laptop), CPU-only, 8GB RAM. Cost optimization is mandatory.

GOAL
Upgrade the experiment pipeline to “Q2 journal strong / 10-10 rigor” by adding:
1) Multi-seed repeated runs (default 5 seeds; configurable to 10)
2) Aggregate reporting: mean±std across seeds for macro-F1, weighted-F1, PR-AUC, ROC-AUC (when valid), FAR/FPR
3) Bootstrap 95% confidence intervals (at least macro-F1; prefer macro-F1 + PR-AUC)
4) Statistical tests:
   - Wilcoxon signed-rank test (paired across seeds) comparing best model vs Logistic Regression
   - Cliff’s delta effect size for the paired seed distributions (best vs LR)
   - Keep existing McNemar (paired predictions) for at least best vs LR on the fixed test set
5) Artifact exports:
   - outputs/tables/*.csv
   - outputs/tables/*.tex
   - outputs/figures/*.png (ROC + PR curves, confusion matrix for best model, and one comparative bar chart with error bars)
   - docs/results_summary.md (hardware specs, runtime, exact command line used, dataset sizes, and key numbers)
6) Reproducibility:
   - deterministic seed handling
   - do NOT fabricate results
   - no changes to raw data
   - pipeline must run CPU-only

HARD INTEGRITY RULES
- Do NOT fabricate results, metrics, or significance tests.
- Every number must be produced by runnable code.
- If something cannot be executed, label outputs as TBD and do not invent values.

COST / RESOURCE CONSTRAINTS
- Must be CPU-only.
- Must fit in 8GB RAM.
- Implement dataset caps / stratified sampling:
  - CICIDS2017: cap to 200k rows by default (stratified), configurable via config.
  - UNSW-NB15: default to official split; if too slow, allow cap to 200k stratified with a flag, but keep official split as the default.
- Use reasonable hyperparameter defaults to control runtime:
  - LogisticRegression: max_iter=2000
  - RandomForest: n_estimators <= 300, max_depth <= 20, n_jobs=-1
  - XGBoost: tree_method=hist, device=cpu, n_estimators <= 500, max_depth <= 8, subsample/colsample_bytree <= 0.9

WHAT TO DO (ORDERED)

Phase 1 — Brief plan + file list (MANDATORY)
1) Before editing, show a short plan and list the exact files you will modify or create.

Phase 2 — Implement multi-seed runner
2) Add a new runner that loops over seeds and runs preprocess→train→eval→stats for each seed (or train/eval only if preprocess is deterministic).
   - Seeds default: [1,2,3,4,5]
   - Make it configurable via Makefile (SEEDS=) and config file.
3) For each dataset and seed, store outputs under:
   outputs/metrics/<dataset>/runs/seed_<seed>/
   outputs/models/<dataset>/seed_<seed>/

Phase 3 — Aggregation
4) Add an aggregator script that reads per-seed metrics and produces:
   outputs/metrics/<dataset>/aggregate/summary_mean_std.csv
   outputs/metrics/<dataset>/aggregate/summary_mean_std.tex
   Include mean, std, and N (num seeds) for each metric.

Phase 4 — Advanced stats (Wilcoxon + Cliff’s delta + bootstrap CI)
5) Create stats scripts that:
   - Bootstrap 95% CI for macro-F1 (and PR-AUC if feasible) using test predictions (>=1000 bootstrap reps; configurable).
   - Wilcoxon signed-rank test comparing best model vs LR across seeds for macro-F1.
   - Cliff’s delta effect size for best vs LR across seeds (macro-F1).
   - Keep McNemar for best vs LR using saved per-example predictions for ONE chosen seed (default seed=1) OR for the non-seeded “main run” if you already support it.

Phase 5 — Plots
6) Generate PNG plots (CPU-friendly):
   - ROC curve overlay per dataset (already exists; ensure it works per seed or for the best seed run)
   - PR curve overlay per dataset (NEW)
   - Confusion matrix for best model (NEW)
   - Bar chart of macro-F1 mean±std across models for each dataset (NEW)

Phase 6 — Makefile targets (MANDATORY)
7) Update Makefile to include these targets:
   - seeds            (runs all models across seeds for chosen dataset(s))
   - aggregate        (build mean±std tables)
   - stats_advanced   (bootstrap + wilcoxon + cliffs + mcnemar)
   - paper_artifacts  (runs aggregate + stats_advanced + export + generates docs/results_summary.md)
8) Keep existing targets: setup, data, preprocess_cicids2017, preprocess_unsw_nb15, train, eval, stats, export, clean.
9) Ensure “make all” runs the full new pipeline end-to-end.

Phase 7 — Docs for paper insertion
10) Create or update:
   - docs/results_summary.md with:
     - system info commands output captured via Python (platform, cpu, ram)
     - dataset sizes after preprocessing
     - runtime per phase (rough is ok but must be measured from code, not guessed)
     - pointers to CSV/TEX tables for paper
   - docs/reproducibility.md with:
     - exact versions (python, pip freeze)
     - how to reproduce in one command
     - seed policy

REQUIRED OUTPUT FORMAT FROM YOU (Claude Code)
A) Plan + file list (before edits)
B) Implement changes (write code)
C) Final “how to run” commands (copy/paste)
D) Expected output file list (tree)

STOP CHECKPOINTS
- After implementing, print the exact commands I should run:
  make setup
  make data
  make preprocess_cicids2017
  make preprocess_unsw_nb15
  make seeds SEEDS=5
  make aggregate
  make stats_advanced
  make export
  make paper_artifacts

IMPORTANT
- Do NOT introduce NSL-KDD or any other dataset.
- Do NOT require GPU, Docker, or cloud.
- Keep changes minimal and consistent with current repo style.
