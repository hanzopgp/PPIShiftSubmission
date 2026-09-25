# E003 signal-audit protocol

Status: **FROZEN before any new signal-audit output**  
Protocol identifier: `ppi_shift.e003_signal_audit.v1`  
Freeze date: 2026-09-21 (Europe/Paris)

This document fixes the diagnostic controls for E003. The controls are not a
replacement for the Core-4 evaluation and cannot change model names, panel
semantics, checkpoint selection, or the primary estimand.

## Scope and inputs

The audit uses only the corrected E003 panel contract and artifacts listed in
the Core-4 freeze. The six environments are, in this order:

1. `uniform_partner`
2. `degree_partner_matched`
3. `sequence_nuisance_matched`
4. `esm_partner_matched`
5. `bernett2026_ilp_exact`
6. `tppni2025_scm_cl3_exact_balanced`

The audit consumes the corrected train, validation, and test panels. The
validation and test labels are never used to fit a feature transform,
classifier, or hyperparameter. The diagnostic threshold is selected from the
fit partition only. External measured panels are
optional transfer surfaces and are scored only after the corresponding
machine-readable input manifest passes its hash and semantic checks.

The target remains the panel label `synthetic_unobserved_pair`. It is not a
claim of biological non-interaction. Positive-backbone identifiers, endpoint
identifiers, and panel metadata are retained in every output so that a
diagnostic cannot be mistaken for a biological assay result.

## Fixed split, seeds, and metrics

For every environment, rows in the training panel are grouped by
`positive_row_id` (falling back to the canonical positive-pair key only when
the field is absent). Groups, not rows, are assigned to an 80/20 diagnostic
fit/holdout split by a stable SHA-256 hash of `(seed, group_id)`. A group is
never split across the two partitions. The production diagnostic fit uses
the complete training panel after the split contract has been checked, and
the corrected test panel is scored once. The internal holdout is reported as
an implementation check. The fixed diagnostic threshold is selected by
maximum MCC on the fit partition only; holdout and test labels never select a
threshold or a feature transform.

The fixed seeds are `11`, `22`, and `33`. Classifiers use a fixed
`LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000,
class_weight="balanced")`; no search or early stopping is allowed. Continuous
features are standardized using fit-partition statistics only. The reported
metrics are AUROC, AUPRC, maximum-MCC (diagnostic ranking threshold only),
balanced accuracy at 0.5, Brier score, and log loss when defined. Counts,
prevalence, and the exact feature list are always written beside metrics.

## Predeclared controls

### Easy-feature classifier

The pair-level feature vector is restricted to non-model, sequence- and
annotation-level covariates available in the panel: the two endpoint lengths,
absolute length difference, minimum/maximum length, amino-acid composition
summary (mean absolute composition difference and per-endpoint composition
vectors), endpoint degree summary when present, and the declared nuisance
columns for the relevant environment when present. The fixed columns are
sorted lexicographically before fitting. No Core-4 prediction,
checkpoint score, embedding, pair interaction score, or test-derived feature
is permitted.

A missing optional family is recorded as `missing`, never imputed from the
test panel,
and never silently replaced by a different family. The runner may produce a
sequence/degree result when that is the only complete predeclared family; it
must write a blocker for omitted families.

### Protein-only baselines

Two protein-only controls are fixed:

* `empirical_endpoint_propensity`: smoothed positive rate for each endpoint,
  estimated from training rows only with Beta(1,1) smoothing; a pair score is
  the arithmetic mean of its two endpoint propensities.
* `single_protein_sequence_logistic`: one endpoint at a time is represented by
  its sequence length and amino-acid composition. Each training endpoint
  observation inherits the row label, and duplicate endpoint observations are
  retained with their declared row provenance. The pair score is the mean of
  the two endpoint probabilities. No partner identity, pair hash, endpoint
  product, difference, interaction feature, Core-4 score, or test statistic is
  allowed in this baseline.

These controls are diagnostic only. They quantify endpoint/sequence signal;
they are not interaction predictors and are not eligible for the Core-4
roster.

### Bernett-style simple baseline

When an exact ESM2-150M representation cache is available and its model,
revision, layer, pooling, sequence registry, and hash match the E003
representation contract, the diagnostic uses a fixed low-capacity logistic
classifier on the symmetric pair vector `u + v` and `abs(u - v)`. The
representation is computed per protein and never refit. The classifier has
the same fixed logistic configuration and grouped split as above. If the
exact representation gate fails, the result is `BLOCKED_MISSING_FEATURE`;
the easy-feature or protein-only output must not be relabeled as Bernett.

## Broken-pair construction

Broken pairs are a deterministic diagnostic perturbation of each corrected
test panel. Within each label stratum, rows are stably sorted by
`row_id`, and the right endpoints are permuted by the first deterministic
cyclic shift that minimizes (in order) unchanged pairs, reconstruction of an
original positive pair, and known-positive collisions. The candidate shifts
are tried in increasing order and the selected shift is recorded. If no
valid derangement exists, construction fails closed.

The original panel labels and endpoint marginals must be preserved exactly.
Every output row records its source row, broken-pair flag, permutation shift,
and collision status. A collision means that the broken unordered pair is a
known positive pair in the declared positive set; collisions are counted and
never silently dropped. Broken panels are not substituted for corrected Gold
panels in the primary result registry.

## Failure policy and provenance

Missing seed, duplicate `(model, train_environment, test_environment, seed)`
cell, feature-family omission, incompatible representation, schema mismatch,
non-deterministic broken panel, changed label counts, changed endpoint
marginals, or any test-label-dependent fit is a hard audit failure. A failed
optional surface is recorded as blocked with its exact reason; it is not
converted to a pass by fallback code.

Every CSV/JSON/figure source includes the protocol SHA-256, input path and
SHA-256, code revision, command line, seed, feature family, split manifest,
and status. Scientific values in figures must be read from these
machine-readable outputs; manually typed metrics are forbidden.
