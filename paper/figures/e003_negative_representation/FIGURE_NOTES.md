# Paper-facing negative-construction figures

These revised candidates use the materialized Gold test panels and the frozen
ESM2-derived source tables from the base pack. Historical figure files remain
in this directory and are not overwritten by the revision builder.

All revised figure titles and legends use these paper-facing labels:
Observed positives; Uniform; Degree-matched*; Matched by sequence; Matched in
ESM2 space; Bernett 2026; TPPNI (balanced).

The degree note is: “*Train-visible degree is degenerate on the protein-disjoint
Gold test cohort; this condition is retained as a diagnostic surface.” The
controlled TPPNI panel is a balanced subset of the reproduced TPPNI pool. The
ESM-matched sampler is constructed using ESM2, so separation in ESM2 space is
partly expected by design. Anchor-based constructions inherit one endpoint from
the common positive backbone.

## Revised candidates

| Figure | Role | Literal quantity | Safe interpretation | Prohibited overinterpretation | Source files |
|---|---|---|---|---|---|
| `figure_endpoint_frequency_concentration` | Main or supplement | Cumulative endpoint-occurrence share versus ranked protein fraction, with observed positives as reference | Relative endpoint concentration can be compared descriptively across constructions | Concentration alone is not evidence of bias or biological implausibility | `e003_negative_representation/endpoint_frequency_with_positive_reference.csv` |
| `figure_pair_and_protein_overlap` | Main | Exact unordered negative-pair Jaccard and resulting protein-set Jaccard on separate scales | Constructions generally select different exact pairs while touching broadly overlapping protein sets | Low exact overlap alone does not establish strongly separated statistical distributions | `e003_negative_representation/pair_overlap.csv`, `e003_negative_representation/endpoint_set_overlap.csv` |
| `figure_pair_esm_similarity_main` | Main | Original-space pair cosine-similarity ECDF | The distributions overlap substantially, but construction shifts pair similarity in frozen ESM2 space | ESM2 similarity is not a PPI probability or biological-negative validation | `e003_negative_representation/pair_esm_similarity.csv` |
| `figure_selected_partner_distance_main` | Main or supplement | Cosine distance from selected negative partner to observed partner for four anchor-based constructions | Partner-selection rules shift distance to the observed partner in original ESM2 space | Bernett 2026 and TPPNI do not define this anchor correspondence | `e003_negative_representation/anchor_candidate_esm_distance.csv` |
| `figure_pair_representation_umap_facets_main` | Supplement or main if space permits | Six facets of one common frozen pair-representation UMAP | Same-coordinate facets make overlap visible qualitatively | UMAP geometry is not quantitative evidence of separation or biological classes | `e003_negative_representation/pair_esm2_umap.csv` |
| `figure_negative_construction_characterization_main` | Recommended compact main composite | Overlap, cosine similarity, and selected-partner cosine distance | Compact descriptive characterization of negative constructions | No model-performance or biological-generalization claim follows | All three source tables above |
| `figure_negative_construction_characterization_with_pair_umap` | Review / supplement | Compact composite adding common pair-representation facets | Adds qualitative representation context | The UMAP panel should not drive quantitative interpretation | `e003_negative_representation/pair_esm2_umap.csv` plus the three source tables above |
| `figure_pair_esm_l2_supplement` | Supplement | Original-space pair L2-distance ECDF | L2 gives a second descriptive metric in frozen ESM2 space | It is not an outcome or realism score | `e003_negative_representation/pair_esm_similarity.csv` |
| `figure_selected_partner_distance_l2_supplement` | Supplement | L2 distance from selected negative partner to observed partner | Companion metric for the four anchor-based constructions | Not defined for Bernett 2026 or TPPNI | `e003_negative_representation/anchor_candidate_esm_distance.csv` |
| `figure_selected_partner_overlap_supplement` | Supplement | Selected-partner-only Jaccard for four anchor-based constructions | Separates selected-partner reuse from shared-anchor endpoint overlap | Do not fabricate this interpretation for global constructions | `e003_negative_representation/selected_partner_overlap.csv` |
| `figure_pair_esm_unique_pair_sensitivity_supplement` | Supplement | Row-weighted versus unique-exact-sequence-pair-weighted ECDFs | Shows whether repeated exact sequence pairs alter descriptive summaries | A later model-performance sensitivity should consider exact-sequence-pair deduplication | `e003_negative_representation/pair_esm_similarity_unique_pair.csv`, `e003_negative_representation/pair_esm_similarity_unique_pair_summary.csv` |

## Common UMAP boundary

All pair-representation facets reuse the immutable common coordinate table
`e003_negative_representation/pair_esm2_umap.csv`; no UMAP is refit and no coordinate is
deleted. The revised main view uses a deterministic robust plotting window and
an inset showing the full coordinate range and retained out-of-window points.
UMAP is qualitative, two-dimensional distances can be distorted, apparent
clusters are not biological classes, and quantitative comparisons use exact
pair overlap or original ESM2 metrics.

## Dataset table

The nonlinear dataset-size graphic is demoted. Use
`tables/dataset_overview.csv`, `.md`, and `.tex` instead. Assay
non-detection, explicit curated negative, and constructed unobserved pair are
kept as distinct evidence classes. The table uses “Lambourne et al. Y2H” for
the external yeast-two-hybrid cohort.

## Reproducibility

Revision provenance, exact source CSV/JSON hashes, package versions, source
tree hash, and the unknown future artifact commit are recorded in
`paper_revision_provenance.json`. The relationship is source tree/configuration →
generated revision artifacts → later artifact commit; no final artifact commit
is claimed to have been known before generation.
