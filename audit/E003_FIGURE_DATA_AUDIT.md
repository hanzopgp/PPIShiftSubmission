# E003 figure/data audit

Status: `PASS` (2026-09-23).

## Analysis figure set

The result-dependent analysis was generated from the complete Core-4,
external, and official DataSAIL manifests. The analysis and verifier produced:

- `paper/figures/e003/`: 17 figures, each with adjacent machine-readable CSV
  or JSON source data;
- PNG, PDF, and SVG for every figure, 51 rendered files in total;
- `results/analysis/e003_figure_hashes.json`;
- `results/analysis/e003_analysis.json` with result/input hashes, metric
  summaries, bootstrap scope, seed stability, reversal categories, and
  interpretation boundaries;
- `results/analysis/e003_figure_integrity_audit.json` with status `complete`.

The verifier checked the exact four-model × three-seed × six-train × six-test
grid (432 Core rows), source adjacency and hashes, PDF/SVG/PNG presence, PNG
resolution of at least 300 dpi, plotting-script provenance, source values
against the registry, sampler-source consistency, and 7,272 plotted source
values. The TPPNI trace-bearing manifest is kept separate from the corrected
score panel but is hash-bound to the same TPPNI pool.

## Final submission pack

`paper/figures/e003_final/figure_manifest.json` is the final director/paper
pack. It contains six figures, 78 source-file/hash entries, and PNG/PDF/SVG
renders at 300 dpi. The pack is generated only from the canonical result
registry and PASS signal-audit outputs; no manually entered scientific values
are used.

## Signal-audit gate

`results/analysis/e003_signal_audit/e003_signal_audit_summary.json` is `PASS`
for every required surface:

| surface | rows | status |
| --- | ---: | --- |
| easy-feature | 18 | PASS |
| protein-only | 36 | PASS |
| Bernett-simple | 18 | PASS |
| broken-pairs | 6 | PASS |
| Core registry | 432 | PASS |

The DataSAIL figure remains supplementary. Measured-assay metrics, synthetic
external surfaces, prevalence-normalized AP, and post-hoc balanced-prevalence
diagnostics remain separate estimands.

## Reproducibility boundary

The analysis manifest records the protocol-freeze, panel, trace-manifest,
config, result-manifest, registry, plotting-script, and source-data hashes.
The cluster checkout was on `historical-e003-freeze` at
`withheld-for-anonymous-review`; the execution scripts had
deliberate uncommitted changes, so those files should be committed before
submission if a clean source revision is required.
