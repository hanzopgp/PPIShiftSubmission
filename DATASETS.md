# Dataset audit

Status: **E001 Gold core panel gate closed; measured/OOD source roles remain
semantically bounded.**

Repository-forensics outputs are under `audit/`. A dataset becomes paper-authoritative only after its exact materialized rows, sequence mapping, hashes, negative semantics, and allowed paper role are fixed in a manifest.

## Required fields

For every source/surface record:

- source name and version;
- primary citation;
- license / redistribution constraints;
- organism / taxon;
- assay / observation process;
- positive semantics;
- zero/negative semantics;
- whether an explicit assayed opportunity universe exists;
- orientation / construct information;
- exact sequence availability;
- assertion count;
- unique pair count;
- positive/zero count;
- duplicate and conflict counts;
- overlap with Gold train/validation/test at pair, protein, exact-sequence, and family levels;
- common MMseqs2 family assignment coverage;
- candidate-ranking eligibility;
- allowed paper roles.

## Canonical negative/observation vocabulary

Use these conceptual classes in manifests and never collapse them silently:

- `synthetic_unobserved_pair`
- `explicit_assay_negative`
- `assay_non_detection`
- `curated_noninteraction`
- `apms_non_call`
- `database_absence_unknown`

`unobserved` is not `non-interacting`.

## Audited core inventory

| Dataset / surface | Audited local facts | Positive semantics | Zero / negative semantics | Provisional paper role | Blocking issue |
|---|---|---|---|---|---|
| Bernett/Reim Gold | Human materialized splits: 163,192 train / 59,260 validation / 52,048 test; balanced; full sequences; zero pair/protein overlap across splits | HIPPIE-derived curated physical PPI | released synthetic benchmark non-edge | canonical training backbone + published comparability | common modern family namespace absent from released rows; historical v3/v4 naming conflict |
| Vidal/Lambourne Y2H | 8,302 unique sequence-resolved pairs; 2,563 assay-positive / 5,739 explicit assay-negative; 1,000 source IDs; no pair conflicts | binary Y2H pairwise positive | explicit assay negative in the recorded Y2H context | highest-priority measured-negative reference + potential anchor ranking surface | recover/verify anchor and pairwise-retest grouping; accession fields blank in staging |
| STRING human physical v12 | 83,544 staging assertions representing 41,772 unique unordered source pairs; raw sequence join exists | STRING physical association/proximity evidence | no native zero in this surface; controlled synthetic environments required | broad physical OOD source | staging accessions/sequences incomplete; very large exact-sequence-pair overlap with Gold before filtering |
| BioGRID 5.0.259 human multi-validated physical | 333,138 assertions; 96,889 unique unordered pairs; 11,640 proteins; UniProt accessions present, sequences absent in staging | curated experimentally supported physical relationship | no native zero in current staging | broad physical OOD source | build frozen accession→sequence mapping and Gold-novel cohorts |
| IntAct human binary physical | 406,874 assertions; 181,961 unique unordered pairs; 406,110 positive assertions / 764 explicit assay-negative assertions; 73 raw-class conflicting pairs | physical/binary interaction assertion under recorded experimental context | explicit negative assertion when annotated | secondary measured-negative reference + OOD | resolve conflicting assertions/context; build sequence mapping; negative class is sparse |
| HuRI | positive/test-space feasibility artifacts exist | systematic human binary Y2H positives | absence is not accepted as assay negative without pair-level opportunity record | optional independent positive source | complete pair-level tested/non-detected opportunity table not materialized |
| BioPlex 3.0 | directed/unfiltered AP-MS feasibility files exist | cell-context AP-MS physical association/co-complex evidence | AP-MS non-call, if defined, is context-specific | supplementary observation-process shift only | no complete failed-opportunity table; not direct binary binding |
| Negatome 2.0 | registry references only; verified raw table not found | n/a | curated/structural noninteraction classes | non-blocking supplementary reference | reacquire and verify if later needed |
| INMC | parser/config references exist; expected pair/sequence files not found | unresolved local artifact | unresolved local artifact | non-blocking until identified/materialized | materialized source absent |
| ProfAff / Proline / DIBS / ELM | materialized historical staging exists | source-specific quantitative/motif/interface positives | source-specific non-calls/absence where present | thesis/supplementary only for this paper | heterogeneous semantics; not needed for core ICLR claim |

## Gold identity

The old repositories disagree on whether the Gold source ID should be called v3 or v4. Do not resolve this by renaming the rows.

PPIShift identifies the canonical materialized rows by split role and SHA256:

| Split | Rows | Positive / zero | Unique proteins | SHA256 |
|---|---:|---:|---:|---|
| train / Intra1 | 163,192 | 81,596 / 81,596 | 4,286 | `c3acf8e009e665ee642d31dc2f0ed6d3a702dd1f2be04490233076d00890a3ab` |
| validation / Intra0 | 59,260 | 29,630 / 29,630 | 3,711 | `4d29b51a0c27fe876e5874d3eb1730df7d60e71555438fb01bc5b02945eaab99` |
| test / Intra2 | 52,048 | 26,024 / 26,024 | 3,022 | `50d66f18a2656d7d117be5ac2b79949e97c005c616d5ddff09613d21fe2a4853` |

Audited rows contain `species1=human` and `species2=human`. The historical `organism: mixed` registry value is stale for these materialized rows.

The `c1_cluster` / `c2_cluster` columns exist but have zero complete assignments in the audited files, so the original KaHIP/CD-HIT provenance is not recoverable as a reusable family namespace from these columns.

The current public Figshare resource is v4. The stated v4 change is addition of the missing Q96PU5 sequence to the auxiliary SwissProt file; historical papers may still cite v3. Preserve this provenance note rather than treating the suffix as the scientific identity.

## E001 Gold negative environments

The four synthetic environments are materialized for train/Intra1,
validation/Intra0, and test/Intra2 with fixed positives, deterministic anchor
orientation, common stage-legal candidate universes, and 1:1 prevalence.
Their exact panel hashes and the final-test freeze boundary are recorded in
[`audit/negative_environments/E001_PANEL_FREEZE.md`](audit/negative_environments/E001_PANEL_FREEZE.md).
The train/validation distribution audit is in
[`audit/negative_environments/GOLD_TRAIN_VALIDATION_AUDIT.md`](audit/negative_environments/GOLD_TRAIN_VALIDATION_AUDIT.md).
The Intra2 degree-matched surface is retained but strongly concentrated; it
must not support an unqualified broad-population claim.

## E001v3 common-model-compatible cohort

E001v3 is a separately versioned successor created after the audited external
runtime length contract established a 512-residue native boundary. It uses the
same Gold positive backbone and exclusion registries as v2, but gives retained
rows new v3 identities and excludes any positive pair whose normalized raw
sequence exceeds 512 residues on either side. No sequence is cropped and the v2
panels remain immutable.

| Split | v2 positives | v3 retained positives | excluded | retention | v3 panel schema |
|---|---:|---:|---:|---:|---|
| train / Intra1 | 81,596 | 13,304 | 68,292 | 16.30% | `ppi_shift.e001_negative_environment.v3` |
| validation / Intra0 | 29,630 | 11,085 | 18,545 | 37.41% | `ppi_shift.e001_negative_environment.v3` |
| test / Intra2 | 26,024 | 10,111 | 15,913 | 38.85% | `ppi_shift.e001_negative_environment.v3` |

Each retained split has four 1:1 synthetic panels with one shared candidate
universe per stage. The exact v3 manifest is
`data/manifests/e001_v3_common_cohort_manifest.yaml` (SHA256
`ecb8341a315b9ec57f0e86987af3326d09fbcb999638049466b3b715f6d901a3`). The
length-retention report and independent post-freeze panel audit are in
`audit/E001_V3_MODEL_LENGTH_CONTRACT.md` and
`audit/E001_V3_PANEL_AUDIT.md`. These rows remain synthetic unobserved pairs;
they are not biological non-interaction labels.

## E002 published-negative environments

E002 retains the exact E001v3 positive rows and four frozen panel references. It
adds two new synthetic environments under a separate artifact root:
`bernett_bias_aware_ilp` and `tppni_contrastive_l3`. Both are generated only from
the legal E001 stage candidate universe and retain
`synthetic_unobserved_pair` semantics. The Bernett adapter records its published
bias terms, open solver status, candidate-network cap, residuals, and any gap.
The TPPNI adapter records configuration-model counts, the legal low-probability
pool, Contrastive-L3 support, fallback rows, topology overlap, and the
pre-score primary/supplementary role.

The E002 model roster adds clean-init SENSE-PPI and RAPPPID. Their published
downstream PPI checkpoints are excluded. External measured and source-specific
surfaces remain separate from the synthetic factorial; no external label is used
for retraining, checkpoint selection, or threshold selection. Full hashes and
source-specific compatibility/novelty states are kept in the E002 manifests and
audits rather than inferred from source names.

## Gold overlap already observed before a unified mapping

The current audit is sufficient to show that raw external sources cannot be called clean OOD surfaces as-is.

### Identifier-level overlap with Gold train

- BioGRID: 11,584 exact unordered pairs; 3,429 proteins.
- IntAct: 12,238 exact unordered pairs; 3,988 proteins.

### Exact-sequence overlap with Gold train where currently measurable

- Vidal: 109 exact sequences and 4 exact sequence pairs.
- STRING raw sequence join: 3,550 exact sequences and 33,571 exact sequence pairs.

These numbers come from different currently available namespaces and are **not** the final homology audit. They establish only that a unified project-owned sequence/homology namespace is mandatory before OOD claims.

## Required Gold-novel cohorts

For each external human evaluation surface, materialize and report nested/related cohorts relative to the legally visible Gold training data:

- `pair_novel`
- `protein_novel`
- `exact_sequence_novel`
- `family_i40_novel`
- `family_i30_cov80_novel`

Every result must state its novelty stratum.

Do not use the phrase “cross-source generalization” without specifying the overlap/homology condition.

## Vidal/Lambourne Y2H

The local staging artifact is currently the strongest observed-negative source in the project:

- 8,302 unique pairs;
- 2,563 `binary_assay_positive`;
- 5,739 `explicit_assay_negative`;
- both sequences available for all rows;
- no unordered pair label conflicts;
- 20 exact-sequence duplicate groups across source identifiers.

Before promoting it to a ranking surface, return to the raw Data S4 transformation and establish:

- which rows correspond to comparable pairwise retesting;
- anchor/isoform grouping;
- number of anchors with at least one positive and one negative;
- candidate-count distribution per anchor;
- construct/orientation consistency;
- Gold exact-sequence and family novelty.

A Y2H negative is interpreted only as an assay-defined negative/non-detection in that experimental context, not as universal biological noninteraction.

## IntAct

The local human-binary staging contains only 764 explicit negative assertions versus 406,110 positive assertions and 73 pair-level raw-class conflicts.

Therefore:

- all negative assertions require context-preserving review before use;
- conflicting pairs must not be silently collapsed;
- IntAct should not be the sole measured-negative headline dataset;
- its strongest role is independent secondary evidence after strict filtering.

## STRING and BioGRID

These are broad physical-interaction sources, not pure direct-binding datasets.

For the main paper they should primarily provide **independent positive-source shift** while negative candidate environments are constructed under the controlled PPIShift sampler contract.

Do not attach synthetic zeros and then describe the resulting dataset as experimentally measured positive/negative interaction outcomes.

## DataSAIL control

The 2026 DataSAIL addendum evaluates Bernett Gold v4 directly and reports a scaled leakage score of 0.3642 for the provided Gold split versus 0.0465 for a DataSAIL split.

Consequences for this project:

- published Gold remains the literature-comparability split;
- it is not described as the strongest achievable leakage control;
- a reproducible DataSAIL sensitivity split should be generated if feasible from the public v4 rows;
- common MMseqs2 novelty strata remain necessary for cross-source family isolation.

## Non-blocking sources

The ICLR core should not wait for:

- Negatome reacquisition;
- INMC recovery;
- BioPlex failed-opportunity reconstruction;
- HuRI absent-edge reconstruction;
- IDR-specific sources.

They may be added only if the core factorial and measured-negative analyses are already stable.
