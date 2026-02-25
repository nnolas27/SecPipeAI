# Submission Checklist — SecPipeAI

## Target Journals
- [ ] **Primary**: Journal of Cloud Computing (Springer Nature)
  - Submission portal: https://www.springer.com/journal/13677
  - Manuscript: `SecPipeAI_Final_JCC.docx`
- [ ] **Alternate**: Cluster Computing (Springer Nature)
  - Manuscript: `SecPipeAI_Final_ClusterComputing.docx`

## Pre-Submission Checklist

### Manuscript
- [x] Abstract contains actual numeric results (no placeholders)
- [x] All "[PLANNED — NOT YET EXECUTED]" replaced with actual results
- [x] Section 7 (Results) fully populated
- [x] Statistical tests reported (McNemar, bootstrap CI, Cliff's delta, Wilcoxon)
- [x] Figures embedded (Figures 1–5)
- [x] Figure captions numbered and descriptive
- [x] All figures referenced in text ("As shown in Figure N…")
- [x] Limitations section complete (L1–L9)
- [x] AI assistance disclosure present (Declarations)
- [x] Ethics statement: public datasets, no human subjects
- [ ] Author affiliation up to date
- [ ] ORCID added to author block
- [ ] Word count within journal limits (check JCC guidelines)

### Integrity
- [x] No fabricated results — all numbers from outputs/paper/key_numbers.json
- [x] No NSL-KDD references
- [x] No BETH results claimed (not executed)
- [x] No LSTM-AE results claimed (not executed)
- [x] No overclaims about production generalization
- [x] SecPipeAI framed as evaluation framework, not novel algorithm
- [x] Proxy dataset limitations explicitly stated

### Statistics
- [x] McNemar's test: χ² and p-value reported
- [x] Bootstrap 95% CI: method (n=1000, stratified subsample=50000) stated
- [x] Cliff's δ with magnitude interpretation
- [x] Wilcoxon p = 0.0625 with explanation (n=5 seeds insufficient for significance)
- [x] All stats from outputs/metrics/*/aggregate/stats_advanced.json

### Reproducibility
- [x] Exact commands in docs/results_summary.md
- [x] Seeds documented (1, 2, 3, 4, 5)
- [x] Hardware specs in docs/results_summary.md
- [x] SHA-256 checksums in configs/checksums.yaml
- [x] Makefile pipeline executable
- [ ] GitHub repository made public
- [ ] Zenodo DOI obtained for artifact archive (replace TBD in manuscript)
- [ ] Docker image built and tested

### Figures (all in outputs/paper/figures/)
- [x] Figure 1: roc_comparison_cicids2017.png
- [x] Figure 2: roc_comparison_unsw_nb15.png
- [x] Figure 3: confusion_best_cicids2017.png
- [x] Figure 4: confusion_best_unsw_nb15.png
- [x] Figure 5: macro_f1_bar_cicids2017.png

### References
- [x] All DOIs verified against cited papers
- [ ] Check for any unverifiable references — flag and remove

### Final Steps
- [ ] Re-read full manuscript for typos and consistency
- [ ] Confirm table/figure numbering is sequential
- [ ] Generate PDF from DOCX and verify figure quality
- [ ] Prepare cover letter (Appendix D in manuscript)
- [ ] Prepare supplementary materials zip (code + data download scripts)

## Known Limitations to Disclose (reviewers will ask)
1. No CI/CD-specific dataset (proxy evaluation only)
2. LSTM-AE component not evaluated in this paper
3. No ablation study (LSTM-AE deferred)
4. Wilcoxon p = 0.0625 with 5 seeds (power limitation)
5. CICIDS2017 may have latent CICFlowMeter artifacts
