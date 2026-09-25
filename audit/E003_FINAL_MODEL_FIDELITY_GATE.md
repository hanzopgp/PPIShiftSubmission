# E003 final model-fidelity gate

Status at freeze audit: **source-fidelity gate recorded; result-publication gate
remains closed until the Core-4 registry is complete and hash-validated**.  
Date: 2026-09-21 (Europe/Paris)  
Roster: `symmetric_frozen_esm2_pair_mlp`, `dscript_esm2`, `tuna_corrected`,
`rapppid`.

This is a model-fidelity gate, not a claim that every scheduled E003 training
unit has finished. A `PASS_MAIN` row is eligible for the final roster only
after its E003 checkpoint, validation selection, prediction files, panel hashes,
and result-registry cells pass the execution audits. `PASS_BASELINE` is a
controlled comparator and must not be described as the exact published model.
Only `PASS_MAIN` model names may be used as published main-model names in the
final E003 claims. `SUPPLEMENTARY_VARIANT` is never promoted by a figure or
summary script.

## Parity matrix

| Model | Source/revision gate | Architecture and representation | Training/checkpoint gate | Data/selection gate | Verdict | Evidence and remaining execution gate |
|---|---|---|---|---|---|---|
| `symmetric_frozen_esm2_pair_mlp` | Local controlled implementation; ESM2-150M representation revision `a695f6045e2e32885fa60af20c13cb35398ce30c`; this is not Richoux and not an exact published Richoux implementation | Symmetric pair MLP over overlap-corrected layer-30 residue-mean vectors, dimension 640; hidden layers 64/32 | Adam/sklearn controlled baseline; E001v3 checkpoint reuse is allowed only for the four exact model-input train projections; Bernett/TPPNI require clean-init E003 training | Corrected E003 train/validation panels; matched validation maximum AUPRC and validation maximum MCC threshold; no test adaptation | `PASS_BASELINE` | E001v3 reuse rows and ESM2 cache contract are recorded in `E003_CHECKPOINT_REUSE_AUDIT.csv`; six-environment E003 scoring is still execution-gated |
| `dscript_esm2` | Pinned source revision `23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8`; upstream D-SCRIPT adapter imported and provenance recorded | `D_SCRIPT_ESM2_150M_contact_CNN`; ESM2-150M token/representation path; source-faithful adapter, not a substitute sequence-only model | E001v3 checkpoint reuse is exact only for the first four model-input-equivalent environments; Bernett/TPPNI are new E003 units; validation-owned epoch/threshold | Corrected panels; external validation AUPRC checkpoint selection; no target/test labels in selection | `PASS_MAIN` | E001 provenance and E003 backend request/partition manifests; backend training artifacts are accepted only after checkpoint/prediction/hash validation |
| `tuna_corrected` | Pinned source revision `b5bda8fee261a4f27821738db995cf5883dcd133`; `official_source_imported=true` in E001 provenance | `official_TUnA_ESM2_150M_transformer_SNGP`; ESM2-150M token cache; corrected adapter | E001v3 checkpoint reuse is exact only for the first four model-input-equivalent environments; Bernett/TPPNI are new E003 units; validation-owned epoch/threshold | Corrected panels; no external/test checkpoint or threshold selection; no test adaptation | `PASS_MAIN` | Source/cache/checkpoint provenance is present; final acceptance waits on corrected E003 backend score manifests |
| `rapppid` | Pinned upstream revision `c3a28be56fb3bd96ccf8cc44ea9145cfcadeaf6c`; no E001v3 RAPPPID checkpoint is eligible | AWD-LSTM twin encoder, two bidirectional layers, 32 hidden units per direction; SentencePiece unigram vocabulary 250; truncation 512; C3 concat head; DropConnect/WeightDrop 0.3; embedding dropout 0.3; head dropout 0.2 | Clean initialization; Ranger21; SWA; no Adam or E002 vanilla BiLSTM fallback; retry path includes modern-PyTorch WeightDrop state handling | Training-only tokenizer fit; corrected train/validation panels; validation maximum AUPRC checkpoint; validation maximum MCC threshold; no test adaptation | `PASS_MAIN` | Architecture and toy/runtime parity are recorded; current failed jobs were the diagnosed WeightDrop state error and the resumable retry fan-out remains execution-gated |
| `senseppi` / ESM2-3B | Historical source/representation is tracked separately; ESM2-3B cache is optional and not part of Core-4 | SENSE-PPI/ESM2-3B is a supplementary variant, not a replacement for any Core-4 model | No submission-critical Core-4 checkpoint or matrix cell may depend on it | Must remain explicitly labeled supplementary in every registry and figure | `SUPPLEMENTARY_VARIANT` | Do not include in main ranking, Core-4 cell counts, or final published model names |

## Common gates

The following parity fields are mandatory in each accepted model run
manifest/checkpoint record:

- exact source revision and adapter/import provenance;
- representation model, revision, layer, pooling, dimension, cache hashes, or
  tokenizer vocabulary/model hash;
- architecture hyperparameters, optimizer, epoch budget, checkpoint selected
  by matched validation AUPRC, and threshold selected by matched validation
  maximum MCC;
- corrected train/validation/test panel paths, row counts, and SHA-256 values;
- seed, training environment, clean-init/reuse state, and explicit
  `test_adaptation: false`, `test_checkpoint_reselection: false`, and
  `test_threshold_selection: false`;
- prediction and checkpoint SHA-256 values, with no duplicate canonical cell.

The controlled frozen baseline is a valid comparator but is never renamed to
an exact published model. The historical E002 RAPPPID approximation is not a
valid E003 RAPPPID result. A missing cache, tokenizer, checkpoint, source
revision, or selection record is `BLOCKED`, not a silent fallback.

## Execution state at audit handoff

- Core design: 4 models × 6 train environments × 6 test environments × 3
  seeds = 432 score cells and 72 training units.
- Exact E001v3 score-only reuse candidates: 36 units (the first four
  environments for the frozen baseline, D-SCRIPT-ESM2, and corrected TUnA).
- New E003 units: 36 units (all six RAPPPID environments plus Bernett/TPPNI
  for the other three models).
- The machine-readable row-level audit is
  [`E003_CHECKPOINT_REUSE_AUDIT.csv`](E003_CHECKPOINT_REUSE_AUDIT.csv), SHA-256
  `f8a5db8bfc3f3e999900afa6878b24cd230a2f0f88dd61b614e18999ed8f531f` at
  creation. It contains 72 unique model × train-environment × seed rows.
- The corrected ESM2 cache, source revisions, corrected panel manifest, and
  validation-owned selection rules are recorded in the requests and manifest
  referenced by the local execution checklist. Pending scheduler jobs are not
  treated as completed scientific results.

## Publication rule

Before final figures or claims are released, the final registry must prove
432 unique Core-4 cells, three seeds per cell, valid checkpoint/prediction
hashes, exact panel hashes, external-label isolation, and complete uncertainty
outputs. Until then the model-fidelity verdicts above describe eligibility and
parity only; they do not turn pending or missing result artifacts into data.
