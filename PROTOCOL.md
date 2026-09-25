# Confirmatory protocol

Status: **frozen for the E001 Gold core; test panels are frozen in the downstream panel-freeze record**

The protocol remains editable until the explicit freeze commit. After freeze, any change that could plausibly be influenced by final-test performance requires a new experiment version and a documented reason.

## 1. Primary question

For fixed positive PPI rows and a controlled candidate universe, how sensitive are modern sequence-only PPI models to the negative/candidate distribution used during training and evaluation?

For architecture `i`, train-negative environment `j`, and seed `s`:

\[
M_{i,j,s} = \mathrm{Train}(\mathrm{architecture}_i, P_\mathrm{train}, N^\mathrm{train}_j, s)
\]

Every selected checkpoint is then evaluated **without adaptation** on every compatible frozen test-negative environment `k`.

The central object is therefore:

\[
\mathrm{architecture} \times N_\mathrm{train} \times N_\mathrm{test}
\]

with positives, split membership, prevalence, and candidate-universe rules held fixed wherever the comparison is intended to isolate negative-environment effects.

Primary analyses:

- within-model performance shift across test-negative environments;
- within-test-environment effect of changing the train-negative environment;
- model-ranking stability across test-negative environments;
- pairwise model-ranking reversals;
- architecture × negative-environment interaction;
- anchor-fixed partner preference on measured and synthetic candidate sets.

## 2. Claim hierarchy

### Safe claim

Performance and relative ranking of sequence-based PPI models are conditional on candidate/negative construction.

### Strong claim

Apparent PPI generalization can reflect candidate-distribution generalization rather than robust conditional partner compatibility.

The strong claim is allowed only if:

- the effect survives strong leakage controls;
- it is not explained by prevalence alone;
- it repeats across multiple architectures;
- it is visible on at least two independent human evaluation sources or one independent measured-negative source plus one independent physical-PPI source;
- shortcut baselines are included;
- cross-source comparisons use explicit source/assay semantics;
- inherited representation exposure is separated from downstream split leakage.

## 3. Dataset identity and Gold backbone

The canonical historical Gold rows currently materialized for this project are identified by **content hash + split role**, not by the ambiguous historical `v3`/`v4` suffix alone.

Audited materialized splits:

- train / Intra1: 163,192 rows, SHA256 `c3acf8e009e665ee642d31dc2f0ed6d3a702dd1f2be04490233076d00890a3ab`;
- validation / Intra0: 59,260 rows, SHA256 `4d29b51a0c27fe876e5874d3eb1730df7d60e71555438fb01bc5b02945eaab99`;
- test / Intra2: 52,048 rows, SHA256 `50d66f18a2656d7d117be5ac2b79949e97c005c616d5ddff09613d21fe2a4853`.

All three are human in the audited materialized rows, balanced 1:1, have no duplicate/conflicting unordered pairs, and have zero protein overlap across the published train/validation/test splits.

Gold is retained for **published comparability and controlled interventions**. It is not treated as leakage-free ground truth or as the final strict split.

The public Figshare resource currently has a v4 release dated 2025-02-10; its stated v4 change is the addition of the missing Q96PU5 sequence to the auxiliary SwissProt file. Historical papers and internal configs may still cite v3. This provenance ambiguity must be reported, but it does not override the materialized split hashes above.

## 4. Exclusion registries and label firewalls

A single all-knowing known-positive union must **not** be used everywhere. That would allow future OOD labels to influence training-negative construction.

Maintain three explicit registries:

### `train_exclusion_registry`

May contain only positive labels available to the training procedure under the frozen training protocol. Future OOD/test-source labels are forbidden.

Used for:
- training-negative generation;
- training candidate rejection.

### `validation_exclusion_registry`

May contain positive labels legally available when validation panels are materialized, but still no final-test/OOD target labels.

Used for:
- validation-negative generation;
- checkpoint-selection panels.

### `evaluation_positive_union`

A broad, versioned positive union may be used **only after training/validation artifacts are frozen** to construct or audit final evaluation panels and to flag possible false negatives.

It must never feed back into:
- training-negative sampling;
- hyperparameters;
- early stopping;
- calibration fitting;
- model selection.

All three registries must have manifests, source versions, hashes, and row counts.

## 5. Core negative-environment factorial

The main causal factorial uses **anchor-fixed synthetic environments** rather than unrelated negative datasets.

For each positive unordered pair `(A, B+)` in a split:

1. choose the anchor orientation once using a deterministic, versioned rule independent of labels outside that split;
2. keep that anchor fixed across all negative environments;
3. choose the negative partner from one common candidate universe;
4. keep the positive rows, anchor identities, number of negatives, and prevalence identical across compared environments;
5. vary only the negative-partner selection rule.

Primary synthetic environments:

### `uniform_partner`

Uniform unobserved partner from the allowed candidate universe after the split-appropriate exclusion registry is applied.

### `degree_partner_matched`

Partner selected to match a predeclared training-only degree nuisance variable as closely as feasible. Degree is computed only from the graph legally visible in the corresponding split/training stage.

### `sequence_nuisance_matched`

Partner selected using predeclared sequence nuisance variables such as length and amino-acid composition. Exact variables, scaling, tolerances, fallback policy, and insufficiency behavior must be frozen before final-test scoring.

### `esm_partner_matched`

Partner selected by distance between frozen **single-protein** representations of `B+` and candidate `B-`. The encoder revision, layer, pooling rule, metric, candidate pool, tie-breaking, and fallback behavior are frozen in advance.

The first implementation is deliberately a **single-partner representation match**, not a learned joint pair-distribution model.

### Historical `gold_original` zeros

The released Gold zeros remain an important **reference benchmark environment**, but they are not treated as one of the clean anchor-fixed causal interventions because their historical generation contract is not identical to the new sampler skeleton.

### Measured assay outcomes

Explicit assay-negative or assay-non-detection rows are evaluated as their own observation-process environments. They are not forced into the synthetic factorial and are never called universal true non-interactions.

### Deferred extension

A learned joint pair-distribution matcher `q(A,B)` is explicitly deferred until the simpler factorial has been analyzed.

## 6. Validation and checkpoint selection

Hyperparameters and training budgets are fixed before final-test access.

For each train-negative strategy, validation uses the **corresponding validation-negative strategy** generated from validation-legal information. This treats train + validation as one coherent learning environment and avoids selecting all models for one privileged test-like negative distribution.

The checkpoint-selection metric and patience/budget are fixed per architecture, not tuned per final test environment.

No test-negative environment influences checkpoint selection.

## 7. Prevalence control

The primary synthetic factorial uses identical prevalence across negative environments, initially 1:1 positive:negative unless a later pre-freeze power/compute analysis justifies another common ratio.

This is essential because AUPRC changes with prevalence.

Additional evaluations may report:

- source-native prevalence;
- 1:K stress tests;
- nested candidate ranking.

These are separate analyses and must not be compared as if they were the same classification task.

## 8. Leakage controls

Leakage is evaluated on multiple distinct axes rather than compressed into one label.

### Published Gold split

Retained for literature comparability. Protein-disjoint and similarity-reduced according to the published construction, but not assumed to be the strictest possible split.

### DataSAIL sensitivity split

The 2026 DataSAIL addendum explicitly re-evaluated Bernett Gold v4 and reported scaled leakage score `0.3642` for the published Gold split versus `0.0465` for a DataSAIL split. Therefore a DataSAIL-style leakage-minimized split is a required sensitivity control if it can be reproduced from the published v4 data without changing the biological task.

### Common sequence/homology namespace

All main human datasets must be mapped into one project-owned sequence registry and one auditable clustering namespace before cross-source claims are made.

Required novelty strata relative to the legal training set:

1. `pair_novel`: no exact unordered pair overlap;
2. `protein_novel`: neither test protein occurs in training;
3. `exact_sequence_novel`: neither normalized exact sequence occurs in training;
4. `family_i40_novel`: no test protein belongs to an i40 family represented in training;
5. `family_i30_cov80_novel`: no test protein belongs to the strict i30/cov80 family namespace represented in training.

The exact MMseqs2 command, coverage mode, identity interpretation, input sequence registry, software version, and hashes must be recorded.

Cross-source results must always state which novelty stratum they use.

## 9. OOD evaluation surfaces

Human-first candidate sources:

- Vidal/Lambourne Y2H: measured binary assay outcomes; highest-priority observed-negative reference;
- IntAct human binary physical: secondary observed-negative source after conflict/context cleanup;
- BioGRID human multi-validated physical: broad physical OOD positive source with controlled synthetic negatives;
- STRING human physical v12: broad physical-association OOD source with controlled synthetic negatives.

Current audited raw/staging overlaps with Gold are too large to call the unfiltered sources clean OOD tests. OOD claims require explicit `Gold-novel` filtered cohorts from the common sequence/homology namespace.

BioPlex AP-MS remains observation-process/context analysis only, not a headline direct-binary negative benchmark.

HuRI is valuable as an independent human binary-positive source, but network absence is not converted into assay negative without a pair-level screened opportunity record.

Negatome and INMC are non-blocking until verified materialized artifacts exist.

## 10. Vidal/Lambourne measured-negative diagnostic

The audited local Vidal/Lambourne staging currently contains 8,302 unique sequence-resolved pairs:

- 2,563 `binary_assay_positive`;
- 5,739 `explicit_assay_negative`.

Before using it for ranking, audit the original group structure and determine:

- which rows belong to comparable pairwise Y2H retesting;
- anchor/isoform grouping;
- number of anchors with at least one positive and one negative;
- candidate count distribution per anchor;
- orientation/construct consistency;
- exact-sequence and family novelty relative to Gold train.

Where a defensible anchor group exists, primary ranking diagnostics are:

- pairwise win rate `P[s(A,B+) > s(A,B-)]`;
- MRR;
- Hits@k when candidate-group size makes the metric meaningful.

Synthetic candidate environments should also be constructed around the same eligible anchors for direct measured-vs-synthetic comparison.

## 11. Negative-environment distribution audit

Before interpreting neural-model results, quantify how separable the negative environments are using nuisance information alone.

At minimum compare:

- sequence length;
- amino-acid composition;
- training-visible degree where legal;
- exact/cluster reuse;
- frozen single-protein ESM similarity;
- candidate reuse / concentration;
- known-positive collision rate under the evaluation-only positive union.

Train simple environment discriminators using predeclared nuisance feature sets. Their task is to predict **negative-environment identity**, not PPI label.

This audit may support statements such as “ESM-matched synthetic candidates are less distinguishable from assay-observed candidates under these features,” but must not be phrased as proving that a synthetic sampler generates true or natural negatives.

## 12. E001 operational freeze before internal-test scoring

The following contract is frozen before Gold Intra2 test-panel materialization
and before any final-test model score is inspected. The executable configuration
and panel-freeze record carry the corresponding Git and artifact hashes.

### Frozen Gold backbone

| role | file | SHA256 | rows |
|---|---|---|---:|
| train / Intra1 | `/path/to/Anonymous Author/projects/Data/dataset_outputs/gold_21591618_intra/train.csv` | `c3acf8e009e665ee642d31dc2f0ed6d3a702dd1f2be04490233076d00890a3ab` | 163,192 |
| validation / Intra0 | `/path/to/Anonymous Author/projects/Data/dataset_outputs/gold_21591618_intra/val.csv` | `4d29b51a0c27fe876e5874d3eb1730df7d60e71555438fb01bc5b02945eaab99` | 59,260 |
| internal test / Intra2 | `/path/to/Anonymous Author/projects/Data/dataset_outputs/gold_21591618_intra/test.csv` | `50d66f18a2656d7d117be5ac2b79949e97c005c616d5ddff09613d21fe2a4853` | 52,048 |

The positive backbone is fixed at 81,596 / 29,630 / 26,024 rows for
train/validation/test. The primary synthetic prevalence is exactly 1:1.

### Frozen E001 construction

- schema: `ppi_shift.e001_negative_environment.v1`;
- anchor orientation: `sha256_parity:ppi_shift_anchor_v1`;
- samplers: `uniform_partner`, `degree_partner_matched`,
  `sequence_nuisance_matched`, and `esm_partner_matched`;
- one common candidate-universe definition per legal stage, with no self-pair,
  observed-positive, or stage-registry pair;
- training negatives use only `train_exclusion_registry`; validation negatives
  use only `validation_exclusion_registry`; the evaluation positive union is
  evaluation-only and cannot feed backward;
- degree is unique-neighbor degree from legal positive rows only;
- sequence nuisance is standardized `log1p(length)` plus 20 amino-acid
  composition values, scaled from the candidate universe only;
- ESM matching uses the frozen single-protein ESM2 layer-30 cache,
  overlap-corrected residue mean, L2 distance, stable sequence-hash/ID tie
  breaking, and explicit fallback/rejection provenance;
- every environment has exactly one synthetic negative per fixed positive and
  uses the semantic class `synthetic_unobserved_pair`, never a verified
  non-interaction label.

The train and validation panel hashes, candidate-universe hashes, and audit
invariants are recorded in `data/manifests/e001_panel_manifest.yaml` and the
stage-specific `e001_panel_audit_*_manifest.yaml` files. No panel is regenerated
because of a model result.

### Model-selection and scoring boundary

Validation is matched to the training-negative environment. Checkpoint
selection is maximum validation AUPRC with no final-test information. The
native symmetric frozen-ESM2 pair head is the current executable E001 core;
D-SCRIPT-ESM2 and corrected TUnA are smoke-audited adapters pending a scalable
multi-environment runner. All test scoring, including bounded pilots, occurs
only after the explicit test panel-freeze record is written.

## 12. Model inclusion contract

Main models must:

- take protein sequence as the biological input;
- produce one scalar interaction score;
- be realistically retrainable across negative environments and three seeds;
- have auditable training/validation selection;
- avoid unavoidable interaction-specific pretraining that makes the causal train-negative intervention uninterpretable.

Provisional execution order:

1. symmetric frozen-ESM2 pair MLP;
2. D-SCRIPT-ESM2;
3. corrected TUnA/TUnA-R semantics;
4. RAPPPID as the non-pLM comparator.

SENSE-PPI is a high-value diagnostic/alternative if embedding and compute cost is acceptable.

C3PI is a recent Gold diagnostic, but public-code reproducibility concerns must be resolved before clean retraining is used as headline evidence.

Context-only / exposure-confounded comparators include MINT, DSS-PPI, PLM-interact/PPLM-style interaction-pretrained models, X-PAIR, and topology-dependent models.

## 13. Metrics and statistical protocol

### Pair classification

Primary:
- AUPRC;
- AUROC.

Secondary validation-owned threshold metrics:
- MCC;
- balanced accuracy;
- F1;
- precision;
- recall.

### Model-ranking stability

For model rankings across frozen test-negative environments, report:

- Kendall rank correlation with uncertainty/resampling where appropriate;
- pairwise model-order reversals;
- environment-specific best model;
- performance differences on identical rows where possible.

### Partner ranking

Where anchor candidate groups are valid:

- pairwise win rate;
- MRR;
- Hits@k;
- MAP/NDCG only if group semantics justify them.

### Calibration, supplementary

- Brier score;
- ECE / adaptive ECE;
- NLL.

### Uncertainty

- three independent training seeds;
- deterministic bootstrap on fixed evaluation units;
- anchor groups resampled as units for ranking tasks;
- paired comparisons on identical rows whenever possible;
- external target datasets are not treated as IID replicates without justification.

## 14. Shortcut controls

At minimum:

- prevalence;
- length;
- amino-acid composition;
- length + composition;
- train-visible degree only;
- protein identity/lookup where valid;
- simple frozen ESM pair baseline;
- single-partner/marginal controls;
- broken-partner or partner-shuffle control;
- label-shuffled sanity control.

The comparator is the strongest valid shortcut for that surface, not a hand-picked weak baseline.

## 15. Historical artifacts policy

Historical `ExternalPPI`, `HistoricalBenchmark`, and `Anonymous Author` results are provenance and pilot evidence only.

In particular:

- the old projection pipeline paths involving `gold_test_like`, `curated_test_like`, `gold_test`, `curated_test`, or test-derived target statistics must not be reused for confirmatory sampling;
- the historical `0/154` versus `78/154` gate disagreement does not need to be reconciled before the new confirmatory study;
- old D009 results may motivate hypotheses but do not count as headline confirmatory evidence;
- reusable algorithms must be ported minimally and retested under the new row/exclusion contracts.

## 16. Freeze checklist

Before confirmatory training:

- literature novelty audit current through freeze date;
- Gold materialized hashes fixed;
- common sequence registry fixed;
- BioGRID/IntAct/STRING sequence mappings fixed;
- common MMseqs2 namespace fixed;
- DataSAIL control either reproduced or explicitly blocked with reason;
- train/validation/evaluation exclusion registries fixed and hashed;
- anchor orientation rule fixed;
- candidate universes fixed;
- negative generators tested on train/validation only;
- Vidal group/opportunity audit completed;
- final evaluation panels materialized and hashed before model scoring;
- model revisions fixed;
- selection metric and training budget fixed per model;
- metric/statistics code tested;
- claim-valid versus descriptive conditions declared;
- smoke-run wall-clock and VRAM measured.

## 17. E001v2 successor addendum

E001v1 remains the immutable historical freeze. The versioned successor is
defined in `configs/experiments/e001_v2_confirmatory.yaml` and
`audit/negative_environments/E001_V2_DEGREE_DECISION.md`.

E001v2 keeps the train-visible degree reference and repairs only the
degree-matching tie semantics: all legal candidates at the minimum absolute
degree distance are pooled, then selected by a positive-row-specific
hash-to-index draw containing the fixed `positive_row_id`. The global
`degree-target` cache tie key is forbidden.

The primary successor test surface excludes degree because the protein-disjoint
test candidate population is degree-zero under the train-visible graph. Degree
is retained for all training and matched-validation environments and may be
scored on test only as a supplementary diagnostic stress surface. The primary
matrix is `3 models × 4 train environments × 3 primary test environments × 3
seeds`; no test score is valid before the new v2 panel freeze.

## 18. E001v3 common-model-compatible successor

The v2 panel remains immutable. The source-level audit of the pinned external
runtimes found that the integrated D-SCRIPT-ESM2 adapter and D007 ESM2 cache
share a 512-residue native contract, while the pinned Frozen-ESM runtime alone
can accept longer raw sequences. This is a model-compatibility boundary, not a
performance- or retention-based threshold. The auditable decision was therefore
`CASE_B_COMMON_MODEL_COMPATIBLE_SUCCESSOR_REQUIRED`.

E001v3 is defined in
`configs/experiments/e001_v3_common_cohort.yaml` and
`audit/E001_V3_MODEL_LENGTH_CONTRACT.md`:

- schema: `ppi_shift.e001_negative_environment.v3`;
- common length: at most 512 normalized raw amino-acid residues in each member
  of a positive pair;
- overlength positives are excluded before candidate-universe and sampler
  construction, never cropped;
- v1 and v2 identities, panels, and artifacts are not overwritten or mixed;
- the four samplers preserve v2 semantics, fixed anchors, stage-legal common
  candidate universes, 1:1 prevalence, and
  `synthetic_unobserved_pair` semantics.

The resulting retained positive counts are 13,304 train, 11,085 validation,
and 10,111 test rows, from v2 positive backbones of 81,596, 29,630, and 26,024
respectively. Retention is reported as a cohort consequence of the audited
native contract, not as evidence of model quality. The frozen v3 manifest is
`data/manifests/e001_v3_common_cohort_manifest.yaml` with SHA256
`ecb8341a315b9ec57f0e86987af3326d09fbcb999638049466b3b715f6d901a3`.

The common factorial is three models × four train environments × three primary
test environments × three seeds (108 primary cells), plus 36 predeclared
degree-test supplementary cells. Validation selects one checkpoint and
threshold per model/train/seed unit; that checkpoint is reused across all fixed
test environments. Test adaptation, test checkpoint reselection, and native
test cropping are forbidden. E001v2 Frozen-ESM results remain a separately
labeled comparison because the v3 cohort is length-compatible and therefore not
the same positive cohort.

## 19. E002 published-negative and architecture expansion

E002 is a separately versioned successor of the immutable E001v3 common cohort,
defined in `configs/experiments/e002_published_negative_model_expansion.yaml`.
It does not rewrite E001 panels, checkpoints, result registries, or analyses.

The factorial expands to five trainable architectures (the three E001v3
architectures plus clean-init SENSE-PPI and clean-init RAPPPID) and six
environments: the four E001v3 environments, the Bernett bias-aware ILP, and the
TPPNI configuration-model plus Contrastive-L3 environment. Every stage keeps
the exact E001v3 positive backbone, one fixed anchor orientation, one legal
negative per positive, 1:1 prevalence, and `synthetic_unobserved_pair` semantics.
The only new panels are written below a new E002 artifact root; the four old
panels are referenced by exact hash.

The Bernett adapter uses only the published bias-aware objective terms that are
defined by available E001 metadata. Degree is active with the upstream
`1/log1p(d_plus)` weighted absolute residual and the five-degree-slack hard cap.
Taxon-pair and GO-Jaccard terms are explicitly inactive because no new taxon or
GO labels are introduced; self pairs are prohibited by the common candidate
contract. The open HiGHS solver, candidate-network cap, time limit, gap, status,
objective, and selected residuals are recorded. Solver failure is a hard
materialization error; it cannot silently fall back to another sampler.

The TPPNI adapter reconstructs the unipartite configuration probability and
Contrastive-L3 filter inside the legal E001 candidate universe. It uses the
published low-probability N=10m contract, or all legal candidates when the
cohort is smaller, and never substitutes the published pair list. Topology is
train-visible only: train uses retained train positives, while validation/test
use retained train plus validation positives. Construction diagnostics classify
the test environment as `primary_eligible` or
`supplementary_due_to_topology_contract` before any test score is inspected.
Fallbacks, L3 support, protein-overlap degeneracy, and candidate concentration
remain visible if a supplementary surface is still scored.

SENSE-PPI and RAPPPID are retrained from clean initialization; their published
downstream interaction checkpoints are not used. RAPPPID SentencePiece sampling
is disabled for all splits, its tokenizer is trained on the training unit only,
and its 512-token cap is a no-cropping compatibility contract. Any adapter not
proven symmetric scores both orientations and averages them. Checkpoint
selection is maximum matched-validation AUPRC and threshold selection is
validation maximum MCC. Test environments are scored with that same checkpoint
and threshold, without test adaptation or reselection.

The E002 matrix contains 90 training units and 540 core score cells, with the
36 E001v3 training units and 144 E001v3 core cells reused exactly. The planned
primary metric is AUPRC, with AUROC secondary and threshold/calibration metrics
reported only when valid. Paired bootstrap uses at least 2,000 resamples of
`positive_row_id`, shared across models and panels; seed sensitivity is separate.
Supported reversal rate, winner changes, Kendall tau, sensitivity, interaction
residuals, and ranking entropy are descriptive unless the predeclared eligibility
conditions in the E002 freeze are met. Uncorrected p-values are not reported.

External Vidal, IntAct, BioGRID, and STRING surfaces are scored with frozen
internal checkpoints and corresponding Gold validation thresholds. They never
select a checkpoint or threshold and never retrain on external labels. Exact
pair, protein, exact-sequence, MMseqs i40, and MMseqs i30/cov80 novelty states
remain separate, with `unknown` and `unresolved` never promoted to novel.

## 20. E003 Core-4 corrected successor

E003 is the corrected successor of E002 on branch
`historical-e003-freeze`, with the latest
`historical-e003-method-fidelity` commit as its parent. E001, E001v3,
and E002 panels, checkpoints, registries, figures, and SENSE-related artifacts
remain immutable historical evidence.

The final Core-4 roster is exactly:

- `symmetric_frozen_esm2_pair_mlp`;
- `dscript_esm2`;
- `tuna_corrected`;
- `rapppid`.

SENSE-PPI is retained as an optional future/supplementary extension. It is not
part of the Core-4 protocol freeze, required artifact set, core/DAG dependency
graph, DataSAIL or external score roster, postprocessor, figures, or completion
criterion. An existing or pending SENSE ESM2-3B cache job is therefore
scientifically irrelevant to Core-4 completion and is not canceled implicitly.

The Core-4 matrix is `4 × 6 × 6 × 3 = 432` score cells and `72` training
units, using seeds `11`, `22`, and `33`. The six fixed environments are
`uniform_partner`, `degree_partner_matched`, `sequence_nuisance_matched`,
`esm_partner_matched`, `bernett2026_ilp_exact`, and
`tppni2025_scm_cl3_exact_balanced`. For each training unit, validation uses
the matched training-negative environment; the checkpoint is selected by
maximum validation AUPRC and the threshold by matched-validation maximum MCC.
Every selected checkpoint is reused across all six fixed test environments;
test adaptation, checkpoint reselection, threshold fitting, and model/sampler
selection from test outcomes are forbidden.

The corrected Bernett global-pool ILP and exact TPPNI SCM/Contrastive-L3
constructors are retained from the E003 method-fidelity work. All six core
panels retain the balanced 1:1 `synthetic_unobserved_pair` semantics; they are
not claims of biological non-interaction. Vidal assay non-detections, IntAct
measured outcomes, and BioGRID/STRING synthetic surfaces remain distinct
external estimands.

RAPPPID is the core raw-sequence/no-pretrained-PLM architecture. Its frozen
contract is SentencePiece unigram vocabulary 250 fitted on training sequences
only, modified bidirectional AWD-LSTM with embedding Dropout and DropConnect,
the order-sensitive C3 concatenation head, Ranger21, SWA, clean initialization,
and a no-Adam-fallback failure policy. Both direct and swapped pair-order
behavior are recorded without test-driven symmetrization.

The machine-readable pre-score freeze is authoritative at
`data/manifests/e003_protocol_freeze.yaml`; its companion audit is
`audit/E003_PROTOCOL_FREEZE.md`. Corrected scoring may begin only after those
files are committed and pushed on the Core-4 branch.
