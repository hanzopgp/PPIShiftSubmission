# E003 final reviewer red-team

Status: `PASS_WITH_SCOPED_CLAIMS` (2026-09-23).

The final gate is closed only for the evidence and estimands explicitly listed
below. It is not a claim of biological-negative validation or universal model
robustness.

| review question | disposition | evidence |
| --- | --- | --- |
| Was the protocol frozen before corrected test scoring? | PASS | `data/manifests/e003_protocol_freeze.yaml`, SHA256 `6c850a3f...28ff1`, and the freeze-before-score execution record |
| Are the corrected panel, sampler trace, config, and result inputs hash-bound? | PASS | `results/analysis/e003_analysis.json` and its input hash set; corrected panel `cdea314e...2bc6a14`, trace manifest `b6da223b...1c6f38e` |
| Are all Core-4 cells present? | PASS | 4 models × 3 seeds × 6 train × 6 test = 432 cells; 72 training units |
| Are reuse and new scores distinguished? | PASS | Core result manifest: 216 new score cells and 216 exact E001v3 score-only reuse cells |
| Is historical SENSE-PPI outside the Core-4 completion gate? | PASS | Core roster and protocol freeze; historical SENSE artifacts remain preserved separately |
| Are checkpoint and threshold choices validation-owned? | PASS | Core, external, and DataSAIL result manifests record validation-owned selection; no external retraining or external-label selection |
| Are Bernett, TPPNI full-pool, and leakage-safe roles separated? | PASS | panel/config manifests, sampler trace manifest, Figure 6 source, and signal audit |
| Are natural-prevalence AP and post-hoc balanced IntAct diagnostics separated? | PASS | `results/analysis/external_metric_summary.csv` and `intact_posthoc_prevalence_sensitivity.csv` |
| Are reversals labelled by their uncertainty scope? | PASS | `results/analysis/reversal_events.json` and `e003_analysis.json`; 3,780 eligible comparisons with separate descriptive/evaluation/seed categories |
| Is DataSAIL supplementary and official? | PASS | complete 36-cell official matrix and `audit/E003_DATASAIL_SENSITIVITY.md` |
| Are figures source-adjacent, hashed, and rendered in all required formats? | PASS | figure integrity audit: 17 figures, 51 rendered files, 17 adjacent sources; final pack: 6 figures × PNG/PDF/SVG at 300 dpi |
| Does the signal audit pass? | PASS | five surfaces PASS: easy-feature, protein-only, Bernett-simple, broken-pairs, and Core registry |
| Are semantic limitations explicit? | PASS | final scientific summary, paper Section 8, and claim ledger preserve `synthetic_unobserved_pair`, assay non-detection, seed, and causal boundaries |

## Remaining submission hygiene

The executed cluster checkout was on branch `historical-e003-freeze` at
HEAD `withheld-for-anonymous-review`, with deliberate uncommitted
execution-script changes. Artifact hashes and the plotting-script SHA256 are
recorded, but the source changes should be committed before a submission that
requires a clean reproducibility revision. This is source-control hygiene, not
an unresolved result or scientific audit failure.

The final verdict is therefore `GO` for the corrected E003 evidence with the
scope and limitations stated above; it is not `GO` for claims of biological
non-interaction, causality, or universal robustness.
