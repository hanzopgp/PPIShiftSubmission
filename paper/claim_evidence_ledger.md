# Claim–evidence ledger

The ledger separates protocol evidence, diagnostic evidence, and confirmatory
results. Synthetic rows are `synthetic_unobserved_pair` and are never described
as verified biological non-interactions.

| claim or sentence | state | required evidence | current artifact |
| --- | --- | --- | --- |
| The positive backbone and exclusion registries are versioned and hashable. | SUPPORTED | Gold/data manifests and source hashes | `data/manifests/e001_v2_panel_manifest.yaml` |
| E001v1 degree construction has a global tie-breaking artifact. | SUPPORTED diagnostic | Independent cached-vs-reference review across all stages | `audit/e001_v1_degree_review.json`; `audit/negative_environments/E001_V1_DEGREE_REVIEW.md` |
| The protein-disjoint test makes train-visible degree intrinsically degenerate. | SUPPORTED diagnostic | Stage degree-zero distributions and tie counts | `audit/e001_v1_degree_review.json` |
| E001v1 remains immutable historical/provenance evidence. | SUPPORTED process claim | Parent manifest and unchanged v1 hashes | `data/manifests/e001_panel_manifest.yaml` |
| E001v2 removes the global degree tie winner without changing the legal universe. | SUPPORTED construction claim | v2 sampler tests, manifest, and distribution audit | `audit/negative_environments/E001_V2_DEGREE_DECISION.md`; `audit/negative_environments/E001_V2_PANEL_FREEZE.md` |
| V2 degree partner concentration is materially lower than v1. | SUPPORTED diagnostic | Before/after candidate reuse, HHI, and effective-count audit | `audit/e001_v2_panel_distribution_audit.json` plus v1 review |
| The three-model primary factorial is executable without test adaptation. | SUPPORTED protocol claim | Frozen config, runner contracts, request manifests, and run manifests | `configs/experiments/e001_v2_confirmatory.yaml`; `scripts/analyze_e001_v2_results.py` |
| Real A100 scheduler/GPU allocation is available. | SUPPORTED runtime claim | Slurm state, in-job `nvidia-smi`, CUDA/PyTorch probe | `audit/E001_V2_GPU_AND_RUNNER_SMOKE.md` |
| D-SCRIPT and corrected TUnA complete the integrated runner smoke. | SUPPORTED isolated-runtime claim | Token-cache-filtered smoke provenance and jobs 103058/103059 | `audit/E001_V2_GPU_AND_RUNNER_SMOKE.md` |
| Frozen-ESM completed 48 valid E001v2 cells. | SUPPORTED partial-result claim | Twelve full-data training units, frozen-test manifests, and result registry | `audit/e001_v2_result_analysis.json`; `results/run_registry.csv` |
| The audited D-SCRIPT/TUnA runtime can score the complete frozen E001v2 panels. | BLOCKED | Full-panel length/cache compatibility and valid external run manifests | `audit/E001_V2_EXTERNAL_LENGTH_BLOCKER.md` |
| Frozen-ESM is a complete three-model confirmatory conclusion. | NOT SUPPORTED | D-SCRIPT/TUnA full-panel cells are absent; ranking analysis is incomplete | `audit/e001_v2_result_analysis.json` |
| E001v3 is the CASE B common-model-compatible successor with a 512-residue cohort. | SUPPORTED construction claim | Native length contract, materialization provenance, retention audit, and frozen v3 manifest | `audit/E001_V3_MODEL_LENGTH_CONTRACT.md`; `data/manifests/e001_v3_common_cohort_manifest.yaml`; `audit/E001_V3_RESULTS_SUMMARY.md` |
| E001v3 has a complete common-cohort factorial registry. | SUPPORTED confirmatory execution claim | 144 valid rows, 108 primary rows, 36 degree-supplementary rows, three seeds per model × train × test group, and postprocess gate | `audit/e001_v3_result_analysis.json`; `audit/e001_v3_run_registry.csv`; `audit/e001_v3_postprocess_gate.json` |
| Mean model performance varies across the declared E001v3 train/test environments. | SUPPORTED descriptive result | Machine-readable AUPRC/AUROC matrices with frozen test scoring and validation-owned selection | `audit/e001_v3_result_analysis.json`; `audit/E001_V3_RESULTS_SUMMARY.md` |
| Model order reverses across some E001v3 negative-environment comparisons. | SUPPORTED descriptive result | 66 environment-pair comparisons, 39 nonzero reversal comparisons, and 54 total pairwise reversal events | `audit/e001_v3_result_analysis.json`; `paper/figures/e001_v3/figure_C_ranking_stability.svg` |
| Architecture × environment response surfaces differ in the observed E001v3 matrix. | SUPPORTED descriptive result | Predeclared train × test interaction residuals, reported with explicit non-causal boundary | `audit/e001_v3_result_analysis.json`; `audit/E001_V3_RESULTS_SUMMARY.md` |
| Seed-level variability is small or large in a biologically interpretable sense. | NOT SUPPORTED | The registry has training seeds, not biological replicates or IID protein resamples | `audit/e001_v3_result_analysis.json` |
| E001v2-to-v3 Frozen-ESM deltas isolate negative-environment repair. | NOT SUPPORTED | `same_positive_cohort` is false because v3 excludes over-threshold positive pairs | `audit/e001_v2_v3_frozen_esm_comparison.json` |
| E001v2 performance changes substantially across N_test. | HYPOTHESIS / v2 incomplete | Complete primary result registry and predeclared sensitivity analysis | `results/run_registry.csv` and `audit/e001_v2_result_analysis.json` |
| E001v2 model ranking reverses across negative environments. | HYPOTHESIS / v2 incomplete | Complete three-model matrix, Kendall tau, pairwise reversals, paired deltas | `audit/e001_v2_result_analysis.json` |
| A particular architecture is more robust in general. | NOT SUPPORTED | A complete synthetic matrix can describe response surfaces but cannot establish general robustness without the stated external/biological validation | `audit/e001_v3_result_analysis.json`; `audit/E001_V3_RESULTS_SUMMARY.md` |
| Synthetic E001 negatives are biological non-interactions. | NOT SUPPORTED | Would require a distinct biological validation design | No such claim is permitted by the protocol |
| E001v3 model-pair AUPRC reversals are stable under paired evaluation resampling. | SUPPORTED uncertainty result | 54 point-estimate events, 2,000 paired positive-unit and 2,000 anchor-cluster bootstrap replicates, with 41 uncertainty-supported and 13 descriptive-only events | `audit/e001_v3_uncertainty_analysis.json`; `audit/E001_V3_UNCERTAINTY_AUDIT.md` |
| E001v3 train/test effects and architecture-by-environment interactions have explicit effect-size summaries. | SUPPORTED descriptive result | Grand means, train/test effect ranges, interaction residual RMS, and maximum residual from the machine-readable analysis | `audit/e001_v3_uncertainty_analysis.json`; `audit/E001_V3_UNCERTAINTY_AUDIT.md` |
| E001v3 ranking conclusions are independent of training-seed choice. | NOT SUPPORTED as a biological or universal claim | 27 empirical seed combinations per model order quantify training-seed sensitivity only | `audit/e001_v3_uncertainty_analysis.json`; `audit/E001_V3_UNCERTAINTY_AUDIT.md` |
| Vidal provides an independent measured-assay classification supplement. | SUPPORTED scoped result | Frozen 4,031-row compatible cohort, 846 positives, 3,185 assay non-detections, 36 scoring-only runs, reconstruction audit pass | `audit/ood/VIDAL_E001_V3_RESULTS.md`; `audit/ood/e001_v3_vidal_results.json` |
| Gold model ordering transfers unchanged to Vidal. | NOT SUPPORTED | Top-model agreement is 3/4 for AUPRC and 2/4 for AUROC; Vidal ranking remains a descriptive transfer check | `audit/ood/e001_v3_vidal_results.json`; `audit/ood/VIDAL_E001_V3_RESULTS.md` |
| E001v3 internal homology novelty sensitivity materially changes the result. | NOT SUPPORTED | All strata are estimable; maximum aggregate delta is 0.000363 at i40 and 0.001538 at i30/cov80 | `audit/ood/e001_v3_internal_homology_sensitivity.json`; `audit/ood/E001_V3_INTERNAL_HOMOLOGY_SENSITIVITY.md` |
| The project is the first study of negative-sampling bias, leakage-aware splitting, or topology-driven negatives in PPI. | NOT SUPPORTED | These are established by prior work; the current gap is only a bounded, narrow combination claim | `audit/PRIOR_ART_NOVELTY_CLOSURE_2026-09-15.md`; `LITERATURE.md` |
| E001v3 offers a narrow crossed-environment/ranking-stability/assay/novelty combination not identified in the bounded search. | SUPPORTED bounded novelty wording | Primary-source search boundary and closest-threat comparison; no first-ever or universal novelty claim | `audit/PRIOR_ART_NOVELTY_CLOSURE_2026-09-15.md` |
| The submission-ready title, abstract, and paper scaffold describe the evidence without overclaiming. | SUPPORTED process claim | Title candidates, abstract, main-paper scaffold, and linked evidence artifacts | `paper/title_candidates.md`; `paper/abstract_draft.md`; `paper/main.md` |
| E003 corrected Bernett and TPPNI panels are method-faithful and frozen before corrected scoring. | SUPPORTED protocol claim | Protocol freeze, upstream source map, sampler diagnostics, candidate/pool hashes, solver/SCM diagnostics, panel-contract audit, and PASS signal audit | `audit/E003_PROTOCOL_FREEZE.md`; `audit/E003_BERNETT2026_EXACT_AUDIT.md`; `audit/E003_TPPNI2025_EXACT_AUDIT.md`; `audit/E003_PANEL_CONTRACT_AUDIT.md`; `results/analysis/e003_signal_audit/e003_signal_audit_summary.json` |
| E003 model ordering and absolute performance depend on the declared candidate environment. | SUPPORTED scoped descriptive result | Complete 432-cell Core-4 registry, three-seed means, complete train×test analysis, and uncertainty categories with denominators | `results/analysis/e003_analysis.json`; `audit/E003_FINAL_SCIENTIFIC_SUMMARY.md` |
| E003 measured-assay transfer changes with Gold training-negative environment. | SUPPORTED scoped transfer result | Frozen Vidal/IntAct scoring-only bundles with prevalence-normalized AP, AUROC, validation-owned thresholds, and no external selection | `results/analysis/external_metric_summary.csv`; `audit/E003_FINAL_SCIENTIFIC_SUMMARY.md` |
| E003 IntAct performance is interpreted against its full-cohort prevalence and post-hoc sensitivity. | SUPPORTED diagnostic result | Full 10,347-row natural-prevalence metrics, explicit prevalence baseline, and deterministic 236-negative sensitivity diagnostic | `results/analysis/external_metric_summary.csv`; `results/analysis/intact_posthoc_prevalence_sensitivity.csv` |
| E003 official DataSAIL sensitivity supports or weakens the crossed-environment conclusion. | SUPPORTED supplementary sensitivity result | Official split identity, compatibility retention, complete 36-cell supplementary matrix, and result comparison | `audit/E003_DATASAIL_SENSITIVITY.md`; `results/analysis/e003_datasail_results.json` |
| E003 publication figures are reproducible from canonical result files without manually entered values. | SUPPORTED reproducibility gate | Adjacent CSV/JSON source data, input/script hashes, rendered PDF/SVG/PNG files, signal-audit PASS, and recomputation checks | `audit/E003_FIGURE_DATA_AUDIT.md`; `results/analysis/e003_figure_hashes.json`; `results/analysis/e003_figure_integrity_audit.json`; `paper/figures/e003_final/` |

| In the E003 Gold factorial, changing only the test-negative construction changes AUPRC by as much as 0.264 for Frozen-ESM2 and 0.249 for corrected TUnA. | SUPPORTED scoped descriptive result | Mean AUPRC over seeds 11/22/33 for fixed model/training environment across six Gold test environments | `paper/figures/e003_final/figure_data__figure_02_negative_transfer.csv`; `audit/E003_FINAL_SCIENTIFIC_SUMMARY.md` |
| On the ESM-matched Gold test surface, within-class partner shuffling increases mean AUROC by 0.0863 for Frozen-ESM2 and 0.0866 for corrected TUnA. | SUPPORTED scoped diagnostic result | Original-vs-shuffled frozen-model scores with labels and endpoint marginals preserved by the declared broken-pair protocol | `paper/figures/e003_final/figure_data__figure_03_pair_dependence.csv`; `audit/E003_SIGNAL_AUDIT_PROTOCOL.md` |
| Low-capacity shortcut diagnostics differ by negative construction: Bernett easy-properties AUROC is 0.514, while TPPNI single-protein-sequence and pooled-ESM diagnostics reach 0.648 and 0.669. | SUPPORTED scoped diagnostic result | Frozen signal-audit controls computed without test-label fitting | `paper/figures/e003_final/figure_data__figure_03_pair_dependence.csv`; `results/analysis/e003_signal_audit/e003_signal_audit_summary.json` |
| On fixed measured external surfaces, AUROC changes with the Gold training-negative environment; the largest declared span is 0.193 for corrected TUnA on IntAct. | SUPPORTED scoped transfer result | Same frozen external pairs, six Gold training-negative environments, three seeds, scoring-only with no external selection | `paper/figures/e003_final/figure_data__figure_02_negative_transfer.csv`; `audit/E003_FINAL_SCIENTIFIC_SUMMARY.md` |
| Partner-shuffle results prove that the models never use pair-specific information. | NOT SUPPORTED | The control only tests whether the original A--B association is required for the observed score on the declared surface | `paper/figures/e003_final/figure_data__figure_03_pair_dependence.csv`; `audit/E003_FINAL_REVIEWER_RED_TEAM.md` |
| Bernett or TPPNI is universally better or worse than other negative constructions. | NOT SUPPORTED | Shortcut reduction, in-domain performance, pair dependence, and measured transfer disagree across conditions and are separate estimands | `paper/figures/e003_final/`; `audit/E003_FINAL_SCIENTIFIC_SUMMARY.md` |

## Evidence rules

- A panel hash or runner smoke does not establish a model-performance claim.
- A partial matrix is reported as partial; it is not silently promoted to a
  complete factorial.
- The E001v3 matrix is complete only for the audited 512-residue common cohort;
  its descriptive performance and ranking results must not be projected back
  to the full v2 positive cohort.
- The analysis script must read result artifacts and must not receive manually
  typed scores.
- Independent seeds describe training variability, not independent biological
  datasets.
- `synthetic_unobserved_pair` is an operational construction label and must not
  be rewritten as a verified biological non-interaction.
- E003 supported rows are scoped to the frozen protocol, declared surfaces, and
  machine-readable manifests. The final paper must report corrected evidence,
  including null or contradictory results, rather than preserving a hoped-for
  E001/E002 narrative.
