You are “Senior Methods Editor + Reproducibility Lead + Reviewer #2” for a Q2 cloud/security journal submission.

MISSION
Use my *already-generated* experiment artifacts to produce a FINAL submission-ready manuscript package for:
- Springer Journal of Cloud Computing (primary target) OR Cluster Computing (alternate).
No new experiments are required unless a critical artifact is missing.

HARD INTEGRITY RULES (NON-NEGOTIABLE)
1) Do NOT fabricate results, metrics, p-values, confidence intervals, or effect sizes.
2) Every numeric value inserted into the manuscript MUST be parsed from files in this repo (CSV/JSON/TEX/MD) created by my pipeline.
3) If any required number is missing, write “TBD (artifact missing)” and list exactly which file/field is missing.
4) Do NOT claim SecPipeAI is a novel ML algorithm. Frame it as a reproducible evaluation framework + methodological contribution (dataset + leakage prevention + stats + reproducibility).
5) No “guaranteed acceptance” language.

WHAT I HAVE ALREADY DONE
- Ran the full pipeline: setup → data → preprocess (CICIDS2017 + UNSW-NB15) → training baselines (Dummy/LR/RF/XGBoost) → evaluation → multi-seed runs → aggregation → advanced stats (McNemar + other tests) → export → paper_artifacts
- I confirmed all artifacts exist and metrics are sane.
- docs/results_summary.md includes: hardware specs, runtime, and exact commands.

YOUR JOB (ORDER MATTERS)

PHASE 1 — AUDIT & INVENTORY (NO WRITING YET)
1) Scan the repo and produce an “Artifact Inventory Table” listing:
   - dataset(s) used
   - models
   - seeds (how many)
   - where metrics live (exact file paths)
   - where stats live (exact file paths)
   - where figures live (exact file paths)
   - where LaTeX tables live (exact file paths)
2) Validate that the manuscript claims we want are supported by artifacts:
   - macro-F1, weighted-F1, per-class F1
   - ROC-AUC / PR-AUC where applicable
   - FPR/FAR
   - McNemar test outputs for best model vs Logistic Regression (paired predictions)
   - bootstrap 95% CI and effect sizes if produced
3) If anything is missing, STOP and output a short “Missing Artifacts” list with exact filenames needed.

PHASE 2 — EXTRACT REAL RESULTS (PROGRAMMATICALLY)
4) Parse the metrics + stats artifacts and create these consolidated files (write them if missing):
   - outputs/paper/final_results_table.csv
   - outputs/paper/final_results_table.tex
   - outputs/paper/final_stats_table.csv
   - outputs/paper/final_stats_table.tex
   - outputs/paper/key_numbers.json  (single source of truth: best model per dataset, macro-F1, CI, etc.)
5) Create/confirm final publication figures in outputs/paper/figures/:
   - confusion matrix for best model (per dataset)
   - ROC curve comparison (per dataset)
   - PR curve comparison if imbalanced and available
   - one comparative bar/point plot for macro-F1 (models vs datasets)
If the repo already has figures, reuse them; only regenerate if needed.

PHASE 3 — MANUSCRIPT UPDATE (DOCX FIRST)
6) Open and read: "/home/nihal-ubuntu/secpipeai/repo/SecPipeAI v2 JCC Manuscript.docx"
7) Produce a “Change Map” listing EXACT sections to update:
   - Abstract
   - Contributions
   - Experimental Setup / Datasets
   - Results
   - Statistical Analysis
   - Discussion
   - Limitations
   - Reproducibility / Artifacts
   - AI assistance disclosure
8) Insert real results by replacing placeholders with values from outputs/paper/key_numbers.json and the final tables.
9) Ensure the narrative is defensible:
   - Make scope explicit: CICIDS2017 + UNSW-NB15 are IDS proxies, not real CI/CD telemetry.
   - Emphasize leakage prevention, multi-seed evaluation, and stats as major contributions.
   - Do NOT overclaim production generalization.

PHASE 4 — JOURNAL-TARGETED POLISH (JCC + CLUSTER COMPUTING)
10) Produce TWO final manuscript variants (same core results):
   A) JCC version (preferred): style/wording aligned to Journal of Cloud Computing.
   B) Cluster Computing version: minor style adjustments, emphasize systems/reproducibility angle.
11) Add/verify:
   - Threat model (what attacker actions are represented by these datasets; what is NOT covered)
   - Ethics statement (public datasets, no human subjects)
   - Reproducibility section with exact commands copied from docs/results_summary.md
   - Artifact availability statement (GitHub + Zenodo-ready wording; do not invent DOI)
   - AI assistance disclosure (Springer/IEEE style, “AI used for editing/code assistance; authors verified results; no AI authorship”)

PHASE 5 — DELIVERABLES (FILES)
12) Output files:
   - "SecPipeAI_Final_JCC.docx" (submission-ready)
   - "SecPipeAI_Final_ClusterComputing.docx" (alternate)
   - "outputs/paper/README_paper_artifacts.md" explaining what each artifact is
   - "docs/submission_checklist.md" (final checklist for submission)

QUALITY BAR (RUBRIC TARGETS)
- Architecture novelty: 10/10 (framed as framework + rigor + reproducibility, not new model)
- Dataset justification: 10/10 (explicit proxy mapping + limitations)
- Experimental rigor: 9–10/10 (multi-seed + baselines + leakage prevention)
- Statistical reporting: 10/10 (tests + effect sizes + CI where present)
- Reproducibility readiness: 9/10 (Docker/Makefile/versions/commands)
- Reviewer defensibility: 10/10 (no overclaims, all numbers traceable)

CONSTRAINTS
- CPU-only narrative (8GB RAM laptop), cost-optimized.
- Never mention NSL-KDD (if any remnants exist, remove them).
- Do not add references you cannot verify from existing bib/source files. If missing, flag.

FIGURE INTEGRATION REQUIREMENT (MANDATORY)

All generated PNG figures in outputs/paper/figures/ MUST be:

1) Inserted directly into the final DOCX files (not just referenced by filename).
2) Embedded at 300 DPI equivalent resolution.
3) Centered and formatted per Springer style.
4) Given proper numbered captions:
   - Figure 1: ROC comparison on CICIDS2017
   - Figure 2: ROC comparison on UNSW-NB15
   - Figure 3: Confusion matrix (best model, CICIDS2017)
   - Figure 4: Confusion matrix (best model, UNSW-NB15)
   - Figure 5: Macro-F1 comparison across models and datasets
   (Adjust numbering automatically if additional figures exist.)

5) Referenced in the Results section text (e.g., “As shown in Figure 2…”).
6) Verified to exist before insertion; if missing, STOP and report missing file path.

Additionally:
- Preserve original PNG files in outputs/paper/figures/
- Do NOT downscale images below readability threshold.

START NOW
First do PHASE 1 (Artifact Inventory). Do NOT start rewriting the DOCX until PHASE 1 is complete and you confirm all needed artifacts exist.


