# Literature and novelty ledger

Status: continuously updated until submission.

## Current novelty boundary

The following claims are **not novel**:

- negative sampling affects PPI performance;
- random negatives can induce degree/topology shortcuts;
- train-negative sampling and evaluation-negative sampling serve different purposes;
- replacing one train-negative strategy with another while evaluating on a fixed test set;
- sequence similarity/protein reuse can leak across PPI splits;
- taxonomy can be a shortcut in multi-species PPI data;
- homology-aware, topology-aware, clustering-based, or similarity-based negative sampling;
- similarity-aware biological dataset splitting;
- pretrained protein language models can leak downstream PPI information;
- Gold-style protein-disjoint splitting is not the strongest possible leakage control.

The ESM-matched sampler is therefore **not assumed to be a methodological novelty**. In the first paper it is an experimental environment designed to remove one class of candidate-distribution differences.

The paper must instead test the narrower comparative question:

> Are conclusions about modern sequence-only PPI architectures stable when training-negative and test-negative environments are varied independently while positives, prevalence, candidate-universe rules, leakage controls, source semantics, and model-selection rules are made explicit?

The distinctive analysis target is not merely absolute performance. It includes:

- model-ranking stability across test-negative environments;
- pairwise model-ranking reversals;
- architecture × negative-environment interaction;
- measured-assay versus synthetic candidate environments;
- anchor-fixed partner ranking that removes anchor-only signal;
- repetition under increasingly strict exact-sequence/family/source novelty controls.

If prior art is found that already performs this full combination, the project must pivot before large compute.

## Critical prior art

### Bernett et al., 2026
**Are You Learning Biological Signal or Shortcuts? Auditing and Mitigating Bias in Protein-Protein Interaction Datasets.**
[arXiv:2609.10193](https://arxiv.org/abs/2609.10193).

Closest current novelty threat. Performs a systematic multi-bias audit over PPI databases and introduces similarity-aware splitting plus optimization-based bias-minimizing negative sampling. Their optimization balances multiple nuisance dimensions including degree/taxonomic/functional properties. They also compare matched and more population-like test constructions. The [released Nextflow pipeline](https://github.com/bionetslab/ppi-splitting-pipeline) and associated [Figshare data](https://doi.org/10.6084/m9.figshare.33407437) were checked on 2026-09-15: the implementation centers on a mean-pooled ESM-2/ProtT5 Random Forest baseline, five split/sampler configurations, and balanced versus realistic tests; it does not expose the E001v3 crossed multi-architecture ranking-reversal analysis.

Implication for PPIShift: do not claim first shortcut audit, first bias-aware negative sampler, or first demonstration that changing the negative environment changes performance. Our differentiator must be the crossed multi-architecture train-negative × test-negative experiment and stability of comparative model conclusions under controlled prevalence/leakage/source semantics.

### DataSAIL addendum, 2026
**Addendum: Data splitting against information leakage with DataSAIL.**
Nature Communications 17, 1597. [DOI: 10.1038/s41467-025-67495-w](https://www.nature.com/articles/s41467-025-67495-w). Published 13 February 2026.

Directly re-evaluates the Bernett human Gold PPI benchmark, using Gold v4. The addendum reports scaled leakage score `0.3642` for the provided Gold split and `0.0465` for a DataSAIL split.

Implication for PPIShift: published Gold should be retained for literature comparability but must not be described as the strictest achievable leakage control. A DataSAIL sensitivity split is now a high-priority control, while project-owned MMseqs2 novelty strata remain useful for transparent cross-source family isolation.

### Hallee et al., 2026
**Protein language models are accidental taxonomists.**
BMC Bioinformatics 27, 188. DOI: 10.1186/s12859-026-06491-3.

Demonstrates a negative-sampling-induced taxonomy shortcut in multi-species PPI data. Trains models under normal versus same-species strategic negatives and evaluates on a same-species test environment.

Implication for PPIShift: a train-negative intervention followed by a changed evaluation environment is already established as a diagnostic idea. We must generalize beyond one confounder and one model setting.

### Szymborski & Emad, 2026
**A flaw in using pretrained protein language models in protein-protein interaction inference models.**
Nature Machine Intelligence 8:197-208. DOI: 10.1038/s42256-025-01176-7.

Shows pretrained pLMs can introduce downstream PPI exposure/leakage and that clean downstream splits do not establish exposure-free evaluation.

Implication for PPIShift: generic ESM/ProtT5 models must be described as having inherited sequence-pretraining exposure. This does not invalidate a controlled within-encoder negative-environment intervention, but it limits claims of exposure-free biological generalization. A non-pLM comparator such as RAPPPID is valuable.

### Li et al., 2025
**Negative sampling strategies impact the prediction of scale-free biomolecular network interactions with machine learning.**
BMC Biology 23, 123. DOI: 10.1186/s12915-025-02231-w.

Shows degree-distribution bias from random negatives and proposes degree-distribution-balanced sampling across interaction tasks including PPI.

### Chatterjee et al., 2025
**Topology-driven negative sampling enhances generalizability in protein-protein interaction prediction.**
[Bioinformatics](https://academic.oup.com/bioinformatics/article/41/5/btaf148/8107765). DOI: 10.1093/bioinformatics/btaf148.

Topology-driven hard/credible negative selection and generalization analysis. Importantly, it already compares training with different negative constructions while retaining a fixed evaluation design.

Implication for PPIShift: “change the training negatives and evaluate on a fixed test” is not novel. The crossed train-negative × multiple test-negative environments and architecture-ranking stability are the important distinction.

### Lannelongue et al., 2024
**Pitfalls of machine learning models for protein-protein interaction networks.**
Bioinformatics 40(2): btae012. DOI: 10.1093/bioinformatics/btae012.

B4PPI explicitly treats negative sampling for model training and population-like evaluation as different problems. It uses balanced sampling for training and a different evaluation construction while also studying topology, protein overlap, algorithm choice and cross-species transfer.

Implication for PPIShift: separating training sampling from evaluation sampling is prior art and cannot be the novelty claim.

### Bernett et al., 2024
**Cracking the black box of deep sequence-based protein-protein interaction prediction.**
Briefings in Bioinformatics. DOI: 10.1093/bib/bbae076.

Establishes major leakage, sequence-similarity, network-topology and degree concerns and introduced the human Gold benchmark. Gold positives are HIPPIE-derived and Gold zeros are synthetic benchmark non-edges.

### Reim et al., 2025
**Deep learning models for unbiased sequence-based PPI prediction plateau at an accuracy of 0.65.**
Bioinformatics. DOI: 10.1093/bioinformatics/btaf192.

Controlled architecture comparison on Gold; multiple modern sequence architectures plateau and frozen protein-language-model representations explain much of the apparent improvement.

Implication for PPIShift: Gold is useful precisely because several architectures have already been compared under one fixed environment. We ask whether those comparative conclusions survive environmental interventions.

## Foundational negative-sampling and evaluation papers

### Ben-Hur & Noble, 2006
**Choosing negative examples for the prediction of protein-protein interactions.**
BMC Bioinformatics 7(S1):S2.

Shows non-colocalized negatives can make the classification problem artificially easy because the test distribution no longer reflects intended use.

### Yu et al., 2010
**Simple sequence-based kernels do not predict protein-protein interactions.**
Bioinformatics 26(20):2610-2614.

Shows accuracy estimates depend strongly on dataset construction and negative choice; proposes balanced negative sampling to counter hub/representation bias.

### Park & Marcotte, 2011
**Revisiting the negative example sampling problem for predicting protein-protein interactions.**
Bioinformatics 27(21):3024-3028. DOI: 10.1093/bioinformatics/btr514.

Crucial conceptual distinction: a sampler helpful for training need not be appropriate for unbiased evaluation; critiques using balanced negatives for population-level testing.

### Park & Marcotte, 2012
**Flaws in evaluation schemes for pair-input computational predictions.**
Nature Methods 9:1134-1136. DOI: 10.1038/nmeth.2259.

Establishes the C1/C2/C3-style entity-overlap problem for pair-input learning. Protein-disjoint evaluation and overlap-stratified reporting are controls, not contributions.

### Zhang et al., 2018
**Predicting protein-protein interactions using high-quality non-interacting pairs.**
BMC Bioinformatics 19:525.

Proposes sequence-similarity and network-random-walk negative generators and compares multiple negative constructions.

### Arenavirus-human PPI study, 2025
**Challenges in predicting protein-protein interactions of understudied viruses: Arenavirus-human interactions.**
Computational and Structural Biotechnology Journal 27:5401-5412. DOI: 10.1016/j.csbj.2025.11.037.

Uses a fixed limited positive system with four negative-data constructions and shows large changes in apparent predictive performance. Another direct negative-environment sensitivity precedent, but focused on virus-host transfer rather than modern human sequence-only model ranking under a fully crossed controlled factorial.

### Kesemen et al., 2025
**Clustering-Based Negative Sampling Approaches for Protein-Protein Interaction Prediction.**
CIBB 2024 revised selected papers, published 2025. DOI: 10.1007/978-3-031-89704-7_1.

Reinforces that similarity/clustering-based negative construction is established prior art.

## Dataset / measured-negative context

### Lambourne et al. human TF-isoform Y2H

The local project staging derived from the paper's Data S4 contains sequence-resolved binary assay positives and explicit assay negatives. The experimental design pairwise-tests isoforms against candidate partner sets assembled from the screening/isoform comparison process.

Implication for PPIShift: high-value measured-outcome environment, but not a random sample of all human pairs. Use the assay semantics explicitly and audit anchor/group structure before ranking claims.

### HuRI

Systematic human binary Y2H map and valuable independent positive source. Public network absence should not be turned into a per-pair assay negative without an explicit screened-opportunity/non-detection table.

### BioPlex

AP-MS/co-complex association source. Non-calls are observation-process outcomes, not direct binary non-interactions.

## Model papers under consideration

### Core candidates

- D-SCRIPT, 2021: residue-level symmetric interaction architecture.
- RAPPPID, 2022: raw-sequence twin recurrent architecture; valuable non-pLM comparator.
- SENSE-PPI, 2024: Siamese frozen-ESM2 architecture; important negative-sampling prior art and diagnostic.
- TUnA, 2024: ESM2-based attention architecture with SNGP output; training/inference semantics must be reproduced correctly.
- Richoux-style ESM2 baseline as reconstructed/evaluated by Reim et al., 2025: cheap representation baseline; PPIShift should prefer an explicitly symmetric version for the causal factorial unless exact parity is needed.

### Recent diagnostic

- C3PI, 2025: reports a substantial improvement on Bernett Gold using ProtT5/permutation-convolution machinery. Current public-code training path has reproducibility questions that must be independently resolved before clean retraining is treated as headline evidence.

### Context-only / exposure-confounded

- MINT: interaction-specific large-scale STRING pretraining.
- DSS-PPI: pretrained representations plus graph/topology machinery.
- PLM-interact/PPLM-style pair-pretrained models: interaction-specific exposure and/or prohibitive retraining cost.
- X-PAIR: multitask/interface supervision changes the core task contract.
- topology-dependent models such as Topsy-Turvy: training signal is not limited to the sequence-pair intervention.

## Bounded prior-art closure — 2026-09-15

A primary-source review completed on 2026-09-15 covered negative sampling,
topology/degree matching, similarity-aware splitting, leakage, ranking
stability/reversal, measured Y2H non-detection, and exact/family novelty. The
full search boundary and evidence table are in
`audit/PRIOR_ART_NOVELTY_CLOSURE_2026-09-15.md`.

The review confirms that negative-sampling effects, topology-aware negatives,
leakage-aware splitting, and modern sequence-model comparison are established
lines of work. Bernett et al. (2026) is the closest current threat because it
combines a multi-bias audit with similarity-aware splitting and optimization-
based negative mitigation; Chatterjee et al. (2025) independently establishes
topology-driven negative sampling and generalization analysis; and the 2026
DataSAIL addendum tightens the Gold split. No exact E001v3 combination was
identified in this bounded search, but the project claims only a narrow gap
statement, never `first`, `unique`, or universal novelty.

## Questions that must be answered before protocol freeze

1. Has any PPI paper already crossed **several** training-negative generators with **several** test-negative generators across **several modern sequence-only architectures** while controlling prevalence?
2. Has model-ranking stability/reversal under those environments been quantified as a primary object?
3. Has anchor-fixed matched-partner ranking been combined with explicit measured assay negatives and multiple synthetic environments?
4. Has this full design been repeated under exact-sequence/family/source novelty controls?
5. Can the DataSAIL Gold control be reproduced cleanly enough for a sensitivity experiment without derailing the main factorial?
6. Does C3PI's reported Gold advantage survive a clean implementation audit and negative-environment shift?

If prior art answers the complete combination, pivot before large compute.
