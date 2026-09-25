# Scientific decision log

Status: **pre-freeze**

This file records decisions that should not be silently revisited after looking at final-test performance. A decision may be changed before freeze if new evidence justifies it; record the change and reason.

## 2026-09-11 — New minimal confirmatory repository

**Decision:** `PPIShift` is the paper implementation. `Anonymous Author`, `ExternalPPI`, and `HistoricalBenchmark` are read-only provenance/pilot sources.

**Rationale:** historical repositories contain valuable assets but also legacy schemas, contradictory validity gates, product/reporting infrastructure, and some test-referencing negative-selection paths.

**Reopen only if:** a missing critical implementation cannot be ported minimally without reproducing old infrastructure.

## 2026-09-11 — Gold identity by hash, not v3/v4 label

**Decision:** the canonical historical Gold rows are identified by their materialized split hashes in `PROTOCOL.md` and `DATASETS.md`.

**Rationale:** internal configs refer to v3 while another inventory calls the source v4; current public Figshare v4 reports only a Q96PU5 auxiliary-sequence addition. The materialized row hashes are unambiguous.

**Reopen only if:** row-level comparison to official v3/v4 proves that the interaction rows or sequences used by the models materially differ.

## 2026-09-11 — Gold published split is comparability, not strict leakage endpoint

**Decision:** retain the published Gold split for literature comparability, but add a modern leakage sensitivity condition rather than calling Gold the strictest split.

**Rationale:** the 2026 DataSAIL addendum reports scaled leakage score 0.3642 for the provided Gold split versus 0.0465 for a DataSAIL split.

**Reopen only if:** DataSAIL reproduction is technically impossible or changes the biological task/population in a way that makes the comparison invalid.

## 2026-09-11 — One common sequence/homology namespace before OOD claims

**Decision:** no external source is called a clean OOD/generalization test until it is mapped to the common PPIShift sequence registry and assigned exact-sequence/i40/i30cov80 novelty relative to training.

**Rationale:** audited external sources already show large Gold overlap, including thousands of exact pairs/proteins and tens of thousands of exact STRING sequence pairs.

**Reopen only if:** an external source has independently proven zero overlap using an equivalently strong audited namespace.

## 2026-09-11 — Three-stage positive-exclusion firewall

**Decision:** maintain separate `train_exclusion_registry`, `validation_exclusion_registry`, and `evaluation_positive_union`.

**Rationale:** using future OOD/test positives to clean training negatives would let target labels influence training-set construction.

**Reopen only if:** a formal protocol can prove that an additional source was available independently of all evaluation design and cannot encode target-specific information.

## 2026-09-11 — Anchor-fixed factorial is the causal core

**Decision:** the clean factorial uses the same positives, deterministic anchor orientation, common candidate-universe definition, common prevalence, and common number of negatives across `uniform_partner`, `degree_partner_matched`, `sequence_nuisance_matched`, and `esm_partner_matched`.

**Rationale:** this isolates the negative-partner selection rule more cleanly than comparing unrelated historical negative datasets.

`gold_original` remains a benchmark reference, not a clean factorial intervention.

**Reopen only if:** one sampler cannot be defined on the same candidate universe without systematic infeasibility; if so, report the broken comparability instead of silently changing the universe.

## 2026-09-11 — ESM matching is an environment, not the novelty claim

**Decision:** the first ESM sampler matches the single partner representation `B+` to candidate `B-` using a frozen protein encoder. It is not presented as a new negative-sampling method.

**Rationale:** similarity/clustering/hard-negative sampling already has substantial prior art. The paper novelty target is the crossed multi-architecture train-negative × test-negative analysis and model-ranking stability.

**Deferred:** learned joint pair-distribution matching `q(A,B)`.

**Reopen only if:** the simple ESM environment is non-informative and the core factorial is already complete enough to justify a methodological extension.

## 2026-09-11 — Prevalence controlled in the main synthetic factorial

**Decision:** all primary synthetic test environments use identical prevalence, provisionally 1:1.

**Rationale:** AUPRC depends on class prevalence; Bernett-style changes that simultaneously alter negative generator and prevalence do not isolate the negative-environment effect.

**Reopen only if:** pre-freeze power/compute analysis motivates another common ratio. The ratio must remain common across compared environments.

## 2026-09-11 — Validation aligned to training environment

**Decision:** each train-negative condition uses the corresponding validation-negative strategy. Hyperparameters and checkpoint-selection rules are fixed per architecture before final-test scoring.

**Rationale:** train + validation should define one learning environment; a single privileged validation-negative distribution would bias model selection toward that environment.

**Reopen only if:** a model's published training contract fundamentally requires another validation setup and a fair common rule cannot be defined.

## 2026-09-11 — Vidal/Lambourne is the primary measured-negative source

**Decision:** prioritize the local sequence-resolved Y2H artifact for measured-negative diagnostics and potential anchor-fixed ranking.

**Audited local facts:** 8,302 unique pairs, 2,563 binary assay positives, 5,739 explicit assay negatives, no unordered pair label conflicts.

**Rationale:** unlike broad database absence or AP-MS non-calls, these rows encode explicit binary assay outcomes.

**Condition:** anchor/group and pairwise-retest semantics must be verified from the raw Data S4 transformation before MRR/Hits@k claims.

## 2026-09-11 — BioGRID/STRING are broad physical OOD positives, not measured binary negatives

**Decision:** use their positives as independent source-shift surfaces after Gold-novel filtering; construct controlled synthetic negative environments around them.

**Rationale:** their physical semantics include broader physical/co-complex evidence and current staging does not provide matched experimental negatives.

## 2026-09-11 — IntAct is secondary measured-negative evidence

**Decision:** use only after context/conflict cleanup and sequence mapping.

**Audited local facts:** 406,110 positive assertions, 764 explicit negative assertions, 73 raw-class conflicting pairs.

**Rationale:** valuable independent source, but the negative class is sparse and pair-level collapsing can destroy assay semantics.

## 2026-09-11 — Core model execution order

**Decision:** first implement/run:

1. symmetric frozen-ESM2 pair MLP;
2. D-SCRIPT-ESM2;
3. corrected TUnA/TUnA-R semantics;
4. RAPPPID as a non-pLM comparator after the first three are stable.

SENSE-PPI is an alternative/diagnostic. C3PI is a high-value recent Gold diagnostic but not a clean core retrain model until its public implementation questions are resolved.

**Rationale:** the first three give rapid coverage of simple pooled, residue-interaction, and attention/SNGP architectures; RAPPPID tests whether the phenomenon survives without pretrained PLM representations.

## 2026-09-11 — Historical result matrices are pilot evidence only

**Decision:** do not spend submission-critical time reconciling the old `0/154` and `78/154` gate surfaces.

**Rationale:** the new study has cleaner contracts and the historical matrices contain contradictory gating plus some old adapter/provenance issues.

**Allowed use:** hypothesis motivation, sanity checks, and supplementary history if clearly labeled exploratory.

## 2026-09-12 — Family-disjoint means unseen families, not unseen family pairs

**Decision:** headline `family_i40_novel` and `family_i30_cov80_novel` require both protein family components to be individually absent from the Gold reference set. Whether an unordered pair of family IDs has appeared before is a separate `family_pair_novel` diagnostic.

**Rationale:** an unseen combination of two previously observed families is not a family-disjoint example. Conflating these quantities would overstate homology novelty.

**Reopen only if:** the paper explicitly changes the estimand from new-family generalization to new-family-pair generalization, in which case both quantities must still be reported separately.

## 2026-09-12 — External evaluation is pair-level, assertion multiplicity is metadata

**Decision:** headline sequence-only OOD evaluation uses one canonical row per unordered protein/sequence pair for STRING, BioGRID, and IntAct. Repeated source assertions remain provenance/evidence-count metadata and do not weight the model metric multiple times.

**Rationale:** a sequence-only model emits one score for one pair. Assertion-level duplication would weight frequently published or multiply curated pairs more strongly and confound source curation intensity with predictive performance.

**IntAct exception:** contradictory positive/negative assertions remain an explicit conflict class or context-specific analysis; they are not silently collapsed to a binary label.

## 2026-09-12 — DataSAIL is a supplementary selection + leakage sensitivity control

**Decision:** do not use the official DataSAIL split as a matched-population primary leakage rung.

**Audited fact:** of 274,500 Gold rows, the official DataSAIL artifact assigns 101,568 to train/validation/test and drops 172,932 (~63%).

**Rationale:** performance changes between published Gold and the DataSAIL subset can reflect both leakage reduction and strong population/data-retention changes.

**Allowed use:** supplementary leakage/selection sensitivity, with retained-population statistics and overlap audits reported explicitly.

## 2026-09-12 — Vidal ranking remains conditionally viable

**Decision:** do not create MRR/Hits from the current 8,302-row local table yet, but do not treat ranking as scientifically ruled out.

**Primary-paper fact:** the Molecular Cell study states that TF isoforms were screened against the human ORFeome and then pairwise tested against all interaction partners for that TF gene, yielding 3,509 successfully tested isoform-resolved pairs. Data S4 defines `FALSE` as a verified non-PPI.

**Condition:** recover the exact paper-defined 3,509-row successfully tested subset and its anchor/candidate semantics from the raw supplement/source analysis. Only that auditable subset may support measured-negative ranking claims.

## 2026-09-12 — Pre-freeze READY gate reverted pending repair

**Decision:** the Codex-generated `E001_DATA_LAYER_READY = YES` is provisionally reverted to `PENDING_REPAIR` until the tracked items in `audit/SCIENTIFIC_REVIEW_PREFREEZE.md` are resolved and regenerated.

**Rationale:** current family-novelty semantics, pair-level OOD materialization, class-stratified IntAct retention, and Vidal 3,509-row reconstruction are decision-critical and affect what E001/OOD claims mean.

**Reopen condition:** scientific re-review of regenerated artifacts.

## 2026-09-14 — Corrected data-layer gate reopened for Gold E001

**Decision:** set `E001_DATA_LAYER_READY = YES` for the Gold factorial backbone
after the corrected family-component novelty audit, explicit
true/false/unknown/unresolved states, canonical pair views, direct DataSAIL
component-overlap audit, outcome-class retention audit, and Vidal re-audit.

**Boundaries:** BioGRID and IntAct remain resolved-subcohort-only OOD surfaces;
DataSAIL remains a supplementary selection plus leakage sensitivity control;
Vidal ranking remains `NO / NOT YET`; only Vidal binary classification is
currently cleared. Synthetic unobserved pairs remain synthetic labels, not
verified biological non-interactions.

**Evidence:** `audit/SCIENTIFIC_REREVIEW_2026-09-14.md`,
`audit/source_class_retention.csv`, and their tracked manifests. This decision
opens the E001 contract and sampler gates; it does not authorize final-test
scoring or any model-selection shortcut.

## 2026-09-14 — E001 negative-environment contract frozen before test scoring

**Decision:** freeze the four anchor-fixed, 1:1 synthetic environments for the
Gold core, with stage-specific exclusion registries and matched validation
construction. Gold Intra2 panels may be materialized only after this decision
and the shortcut audit; they are then immutable.

**Evidence:** `PROTOCOL.md`, `src/ppi_shift/negatives.py`, the E001 contract
tests, and the train/validation panel manifests. The test-freeze record will
add the final Intra2 panel hashes without changing this construction.

**Boundary:** the evaluation positive union may flag evaluation-only candidate
collisions but cannot influence train/validation generation, checkpoint
selection, or sampler definitions. OOD source surfaces and measured assay
classes remain separate from the clean synthetic factorial.

## 2026-09-14 — retain the frozen Intra2 degree concentration as a declared stress surface

**Observation:** after the pre-score freeze, the legal Intra2
`degree_partner_matched` panel used two unique negative partners across 26,024
negative rows (maximum reuse `26,001`, HHI `0.998234`). The uniform,
sequence-nuisance, and ESM panels retained 2,947, 1,721, and 1,836 unique
negative partners respectively.

**Decision:** retain all four frozen panels exactly as generated and label the
degree surface as a concentrated stress environment. It is not silently
discarded or repaired after test materialization. The degree-test result cannot
support an unqualified broad-population claim; any alternate tolerance or
candidate-universe construction must be a new experiment version decided
before scoring.

**Evidence:** `audit/negative_environments/E001_PANEL_FREEZE.md`,
`audit/e001_panel_distribution_test.json`, and the frozen panel manifest.

## 2026-09-14 — E001v2 degree visibility and tie-breaking decision

**Decision:** choose Option B for the successor experiment. Keep the
train-visible degree definition as the primary training and matched-validation
intervention, while treating degree on the protein-disjoint test surface as a
predeclared supplementary/diagnostic stress environment. Repair the sampler in
v2 by pooling all legal candidates at the minimum absolute degree distance and
breaking that tie with a positive-row-specific hash-to-index draw.

**Rationale:** the independent pre-score audit separates a repairable global
`degree-target` cache artifact from intrinsic test degeneracy: all train-visible
test targets and legal candidates have degree zero. Stage-local test degree
would use test labels to define the evaluation statistic and change the
estimand; replacing degree entirely would change the question without evidence
that the current artifact is irreparable.

**Primary boundary:** the v2 primary test environments are uniform, sequence
nuisance, and ESM; degree remains a training-negative intervention and may be
scored only as a separately reported supplementary diagnostic. The decision was
made before v2 materialization and scoring.

**Evidence:** `audit/e001_v1_degree_review.json`,
`audit/negative_environments/E001_V1_DEGREE_REVIEW.md`, and
`audit/negative_environments/E001_V2_DEGREE_DECISION.md`.

## 2026-09-15 — E001v3 common-model-compatible successor

**Decision:** choose CASE B after the source-level and A100 runtime audit. Keep
E001v2 immutable and create a new v3 common cohort at the audited 512-residue
native limit so Frozen-ESM, D-SCRIPT-ESM2, and corrected TUnA can consume the
same primary evaluation surface without model-specific cropping.

**Rationale:** the integrated D-SCRIPT-ESM2 adapter has a fixed 512-position
architecture and the D007 ESM2 cache is generated at the same maximum raw
length. The threshold is therefore a compatibility contract, not a choice
based on performance, retention, or the resulting sample size. Filtering is
pairwise and occurs before candidate-universe and sampler construction.

**Boundary:** v3 retains v2 sampler semantics, fixed positives within each
stage, 1:1 prevalence, and synthetic-unobserved-pair semantics. The large
overlength-positive exclusion is reported as cohort retention and is not
interpreted as a biological or model-quality result. E001v2 Frozen-ESM metrics
and v3 metrics are analyzed separately because their positive cohorts differ.

**Evidence:** `audit/E001_V3_MODEL_LENGTH_CONTRACT.md`,
`audit/E001_V3_NATIVE_LENGTH_TEST.json`,
`audit/E001_V3_PANEL_AUDIT.md`, and
`data/manifests/e001_v3_common_cohort_manifest.yaml`.

## 2026-09-15 — E002 published-negative and architecture expansion freeze plan

**Decision:** create E002 as a new branch and artifact namespace rooted at the
verified E001v3 parent. Keep all four E001v3 panels immutable, add only the
Bernett bias-aware ILP and TPPNI Contrastive-L3 environments, and expand the
retrainable model roster with clean-init SENSE-PPI and RAPPPID.

**Rationale:** the question is whether negative-environment sensitivity and model
ranking stability persist when both the architecture and published negative
construction change. Reusing the exact E001v3 positive/candidate identities
keeps the factorial interpretation narrow and prevents cohort drift.

**Predeclared boundaries:** Bernett is an anchor-fixed legal-candidate adapter
with an explicit open-HiGHS feasibility cap and no silent fallback. Its taxon,
self-loop, and GO terms are inactive unless their required metadata exists.
TPPNI is reconstructed from its published configuration-model/L3 method inside
the legal E001 universe; its role is fixed from construction diagnostics before
test scoring and may be supplementary when topology is degenerate. SENSE-PPI
and RAPPPID use no published downstream interaction checkpoint.

**Evidence:** `configs/experiments/e002_published_negative_model_expansion.yaml`
and the E002 protocol-freeze, compatibility, sampler, and upstream-integration
audits. No E002 claim is promoted from `PENDING` before the corresponding result
cells and paired uncertainty are complete.

## 2026-09-19 — E003 Core-4 final execution path

**Decision:** remove SENSE-PPI from the required E003 core and freeze the
successor on `historical-e003-freeze` with exactly four model families:
`symmetric_frozen_esm2_pair_mlp`, `dscript_esm2`, `tuna_corrected`, and
`rapppid`. The corrected factorial is 432 score cells and 72 training units
over the six fixed negative environments and seeds 11/22/33.

**Rationale:** the paper's central estimand is controlled sensitivity to
training- and evaluation-negative construction, not maximum model count. A
pending or incomplete ESM2-3B SENSE cache is not needed by the four
scientifically complementary architectures and must not block completion.

**Boundary:** all historical SENSE code, audits, jobs, manifests, and artifact
roots remain preserved and explicitly labeled optional future/supplementary
provenance. They are excluded from the Core-4 protocol freeze, request
matrices, DataSAIL/external scoring, postprocessors, figures, and completion
criteria. No Core-4 artifact may silently substitute SENSE or use test results
to alter the roster.

**Evidence:** `configs/experiments/e003_method_fidelity_final.yaml`,
`PROTOCOL.md`, `audit/E003_MODEL_FIDELITY_AUDIT.md`, and the Core-4 protocol
freeze manifest once committed.
