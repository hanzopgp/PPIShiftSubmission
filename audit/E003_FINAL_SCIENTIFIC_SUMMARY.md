# E003 final scientific summary

Status: `COMPLETE_WITH_SCOPED_CLAIMS` (2026-09-23).

The corrected E003 execution is complete. The results below are read from the
machine-readable manifests; no score is manually entered in this document.

## Execution and provenance

- Branch: `historical-e003-freeze`; executed repository HEAD:
  `withheld-for-anonymous-review`.
- The cluster checkout had deliberate uncommitted execution-script changes;
  the analysis manifest records the executed plotting-script SHA256 and HEAD.
  The scientific result files and figures are therefore artifact-hash bound,
  but the current working tree should be committed before submission if a
  clean source revision is required.
- Protocol freeze SHA256:
  `6c850a3f09bdda349368202dd8482bf1e2c93fad80949e2e296eac2a28837ff1`.
- Corrected score panel SHA256:
  `cdea314e6776cf15a6d02b6fc7011924b9964e3e51cb54c0f1691bc4f2bc6a14`.
- Trace-bearing sampler manifest SHA256:
  `b6da223b7c63a819e8c8fba29cafe48b980824036d2e19e7f836056bb1c6f38e`.
- Four models, six train environments, six test environments, and seeds
  `11/22/33` define the Core-4 matrix.

## Completion gates

| surface | completed units/cells | status |
| --- | ---: | --- |
| Core-4 | 72 training units / 432 score cells | `complete_core_matrix` |
| external scoring-only | 72 training units / 576 registry rows / 8 surfaces | `complete_external_matrix` |
| official DataSAIL sensitivity | 18 training units / 36 score cells | `complete_datasail_sensitivity` |
| signal audit | 510 audited rows across five controls | `PASS` |
| analysis figure integrity | 17 figures, 51 rendered files, 7,272 source values checked | `complete` |
| final paper figure pack | 6 figures, each PNG/PDF/SVG at 300 dpi | generated |

The Core matrix contains 216 new score cells and 216 exact E001v3 score-only
reuse cells. External requests are bound one-to-one to the 72 declared units;
there is no external retraining or external-label selection. The independent
signal audit passes `easy_feature` (18 rows), `protein_only` (36),
`bernett_simple` (18), `broken_pairs` (6), and `core_registry` (432).

## Core-4 results

For the uniform-train/uniform-test reference condition, mean AUPRC across the
three seeds is:

| model | mean AUPRC | seed SD |
| --- | ---: | ---: |
| corrected TUnA | 0.722789 | 0.002830 |
| Frozen-ESM2 pair MLP | 0.696923 | 0.005009 |
| D-SCRIPT-ESM2 | 0.586542 | 0.037644 |
| RAPPPID | 0.518389 | 0.004011 |

Across all 36 train-by-test conditions, the AUPRC winner frequencies are
Frozen-ESM2 `20/36`, corrected TUnA `15/36`, D-SCRIPT-ESM2 `1/36`, and
RAPPPID `0/36`. These are descriptive condition-surface results, not a
universal architecture ranking.

The analysis evaluates 3,780 model-pair/condition comparisons. It records
3,142 no-reversal cases, 290 descriptive reversals, 170 reversals supported by
evaluation uncertainty, and 178 reversals supported by both evaluation and
training-seed stability. The corresponding descriptive reversal rate is
16.88%, while the evaluation-and-seed-supported rate is 4.71%. These rates
are estimands over the declared condition surface, not biological-replicate
significance claims.

## External and DataSAIL results

The external matrix contains eight frozen surfaces: Vidal and IntAct measured
surfaces plus BioGRID/STRING synthetic surfaces for Uniform, Bernett, and
TPPNI. Natural-prevalence IntAct rows remain separate from its deterministic
post-hoc balanced-prevalence diagnostic; the mean natural-prevalence AUPRC is
approximately 0.978 across model/train/seed rows, while the corresponding
post-hoc balanced diagnostic is approximately 0.532--0.545. These values must
not be compared as if they were the same estimand.

Mean normalized excess AP across the measured surfaces is 0.091 for the
Frozen-ESM2 pair MLP, 0.058 for RAPPPID, 0.039 for D-SCRIPT-ESM2, and 0.038
for corrected TUnA. The full surface-by-model table remains authoritative;
these aggregates are descriptive summaries over the declared scoring-only
rows.

The official DataSAIL sensitivity is complete and supplementary. Mean AUPRC
over its 36 cells is 0.528693 for D-SCRIPT-ESM2, 0.582226 for Frozen-ESM2,
and 0.584063 for corrected TUnA. RAPPPID is not part of this official
three-model DataSAIL sensitivity. DataSAIL does not replace the Core-4
factorial.

## Conclusions and boundaries

The strongest defensible conclusion is that, under the frozen E003 protocol,
the choice of negative/candidate environment changes absolute performance and
can change model ordering across the crossed train-by-test surface. The
measured-assay and DataSAIL results provide scoped transfer and split-
sensitivity evidence, not universal robustness evidence.

The following claims remain out of scope:

- synthetic rows are `synthetic_unobserved_pair`, not verified biological
  non-interactions;
- assay non-detection is not biological absence;
- three training seeds are not biological replicates;
- external surfaces are scoring-only and do not establish external selection
  independence beyond the recorded validation-owned contracts;
- the study does not identify a causal architecture mechanism or establish
  general model robustness.

## Authoritative artifacts

- Analysis: `results/analysis/e003_analysis.json`.
- Registry: `results/analysis/e003_result_registry.json`.
- Core, external, and DataSAIL manifests:
  `results/analysis/e003_core_results.json`,
  `results/analysis/e003_external_results.json`, and
  `results/analysis/e003_datasail_results.json`.
- Signal audit: `results/analysis/e003_signal_audit/e003_signal_audit_summary.json`.
- Figure integrity: `results/analysis/e003_figure_integrity_audit.json`.
- Final figures: `paper/figures/e003_final/figure_manifest.json`.
- Claim ledger: `paper/claim_evidence_ledger.md`.
- Execution history: `audit/E003_GOAL_EXECUTION_LOG.md`.
