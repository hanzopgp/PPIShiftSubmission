# E003 Bernett 2026 exact-method audit

Date: 2026-09-17 (Europe/Paris)

## Source identity

- Primary paper: [Bernett et al., arXiv:2609.10193](https://arxiv.org/abs/2609.10193).
- Official implementation: [bionetslab/ppi-splitting-pipeline](https://github.com/bionetslab/ppi-splitting-pipeline).
- Pinned revision: `8bf9adb183b702c88db62ff878de482e2f20e4b2`.
- Official sampler: `bin/sample_negatives_ilp.py`.
- Official Nextflow configuration SHA-256:
  `4a9a279410cbb308933988796cc7d43bf35d24f51cdf4e70114f445d47ae1007`.
- Official sampler SHA-256:
  `ffcf164c594f9fb0ecac8cb1aea21ba712542460ec09112bdff96f8104a8df58f`.

The source was cloned and inspected read-only at the pinned revision before the
E003 adapter was written. E002's adapter remains unchanged and is not treated as
this method.

## Fidelity contract

The official process invokes the ILP sampler with `max_candidates = 4 * n_pos`
and, in the pinned Nextflow configuration, all four objective weights equal to
`1.0`: degree, taxon-pair, self-loop, and GO-BP Jaccard. The official solver
defaults used by the E003 record are an open HiGHS-compatible MILP path, a
7,200-second time limit, a 0.01 relative gap, and the declared process thread
count. A solver that returns only a time-limited feasible point is not accepted
as an E003 exact result.

The E003 candidate universe is the global upper triangle over the fixed stage
protein population. Positive identifier pairs, positive exact-sequence pairs,
and both identifier- and sequence-level exclusion-registry pairs are removed.
Self-pairs remain legal candidates unless excluded by the registry, matching the
official sampler's upper-triangle construction. The cap is applied only after
this legal universe is defined.

When the full legal complement exceeds the cap, the adapter records the full
count without enumerating it and applies the official construction policy:

1. force-keep legal self-pairs when the self-loop term is active;
2. sample endpoints with probabilities proportional to positive degree;
3. stratify same- versus cross-taxonomy candidates to the positive edge ratio;
4. prioritize candidates with nonzero GO-BP Jaccard within each stratum; and
5. fill a declared stratum shortfall from the remaining legal pool.

The cap, full-pool count, selected-pool hash, random seed, self-pair count,
taxonomy quotas/fills, annotation coverage, and shortfall state are stored in the
sampler diagnostics. No anchor-local eight-candidate reduction is present.

## Objective implementation

`src/ppi_shift/e003_method_fidelity.py` builds a binary MILP with:

- exactly `round(negative_ratio * n_pos)` selected pairs;
- the official per-protein negative-degree hard cap with `MAX_DEGREE_SLACK=5`;
- absolute residuals for positive degree counts weighted by
  `1/log1p(d_plus)`;
- absolute residuals for global unordered taxon-pair counts;
- the self-loop residual with self-incidence counted once; and
- the absolute difference between positive and selected mean GO-BP Jaccard.

The objective terms are normalized by the official `U` values and multiplied by
the four declared lambda weights. The selected-pair audit recomputes the degree,
taxonomy, self-loop, and Jaccard statistics from the integer solution.

## Differential and regression evidence

The following checks are committed in `tests/test_e003_method_fidelity.py`:

- full legal upper-triangle construction retains legal self-pairs as candidates;
- the active self-loop objective can select zero self-pairs without removing
  them from the candidate universe;
- the `4 * n_pos` cap and weighted/stratified sampling are recorded;
- all four official weights remain active when annotations are supplied;
- repeated fixture runs with the same seed produce identical pairs and audits;
- the result is exactly one global negative per requested positive count; and
- all selected pairs pass the global positive/exclusion registry gate.

These are source-contract and fixture checks, not evidence that the full E003
Gold solve has completed. The full solve remains blocked until annotation input
hashes, solver status, objective/gap, candidate-pool hash, and output hash are
written to the E003 artifact root.

## Explicit non-equivalences with E002

E002 used `bernett_bias_aware_ilp`, one candidate per fixed anchor, a per-anchor
eight-candidate hash/degree cap, and inactive taxonomy/self/GO terms. Those rows
and results remain immutable historical evidence. They are not renamed to
`bernett2026_ilp_exact` and cannot satisfy this audit.

## Current gate

**PASS for the corrected Gold panel materialization and independent panel audit.**
The test legal-pool, cap, objective, annotation, solver, and selected-pair
diagnostics are recorded above; no corrected E003 test score has been selected
from this implementation yet, and external annotation/materialization plus the
protocol freeze remain separate gates.
