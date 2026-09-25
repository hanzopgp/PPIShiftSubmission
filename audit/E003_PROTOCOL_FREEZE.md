# E003 protocol freeze

> Frozen before any corrected E003 test inference or result inspection.

E003 was motivated by method-fidelity problems discovered during review of immutable E002 evidence. The corrected sampler definitions, model roster, validation-owned selection rules, and uncertainty estimands below were fixed before observing E003 test outcomes.

- Branch: `historical-e003-freeze`
- Base Git SHA: `withheld-for-anonymous-review`
- Corrected source SHA: `withheld-for-anonymous-review`
- Panel manifest: `/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/e003_panel_manifest.yaml` (cdea314e6776cf15a6d02b6fc7011924b9964e3e51cb54c0f1691bc4f2bc6a14)
- Config: `/path/to/PPIShift-e003-core4/configs/experiments/e003_method_fidelity_final.yaml` (78ceac6426cb1094713732c099de35924bc4174c93e7d1af7424b7c92bbd4852)
- Core roster: `symmetric_frozen_esm2_pair_mlp`, `dscript_esm2`, `tuna_corrected`, `rapppid` (432 cells; 72 training units).
- SENSE-PPI is explicitly excluded from Core-4 and retained only as a historical/optional supplementary model; no SENSE cache or job is required.
- Required Core-4 request/runtime manifests:
  - `/path/to/PPIShiftArtifacts/e003/core4_final_paper/backend_matrix/e003_backend_request_manifest.json` (b6460ba64f87fdb5778c9890a8a787bc25eb3b9e63c12bf097e3ae83876e4492)
  - `/path/to/PPIShiftArtifacts/e003/core4_final_paper/python_matrix/e003_python_request_manifest.json` (f642d6730227ed1f1c0917a031264002a26156d765179dd616b3061e51bd89b6)
  - `/path/to/PPIShiftArtifacts/e003/core4_final_paper/frozen_matrix/e003_frozen_request_manifest.json` (9d1d57858d98cb1de3d40b95be4abea414f746c0ac55f2b79f17f6426a17a5a4)
  - `/path/to/PPIShiftArtifacts/e003/core4_final_paper/reuse_matrix/e003_reuse_request_manifest.json` (337348183aa9e2d882c94e2ff82e791307d00c51c1821badbdd56a7de3fbc08b)
  - `/path/to/PPIShiftArtifacts/e001_v3/common_512/audits/e001_v3_run_registry.csv` (fc5e65259ed43148a1982b484f307d3961205649a22cc540258d2482aa04f69e)
  - `/path/to/PPIShiftArtifacts/e003/core4_final_paper/datasail_matrix/datasail_request_manifest.json` (5ad0f5b031a19d9b558eee09a95d0b2402f5417ef3889f70f6a7a72a68f1f601)
  - `/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/e003_datasail_panel_manifest.yaml` (161b2b8e718197c64d7807b50a1807a718594d5d959073380eb9418463303910)
  - `/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/e003_external_panel_manifest.json` (34ff1dfd3c00c7dc53157c89d1dfe104c07e54e9e801b480f88e37005e1364ed)
- Model source revisions, model hyperparameters, sampler definitions, TPPNI topology provenance, and train/validation/test panel hashes are copied into the machine-readable freeze.

## Frozen scientific boundaries

- All six core panels are balanced and retain `synthetic_unobserved_pair` semantics; no synthetic complement pair is called a biological non-interaction.
- Bernett is the new global-pool `bernett2026_ilp_exact` implementation; the historical E002 anchor-fixed adaptation is not reused or renamed in artifacts.
- Paper-faithful TPPNI and leakage-safe TPPNI remain separate supplementary surfaces.
- Checkpoint selection uses only matched validation AUPRC; thresholds use only matched validation maximum MCC.
- External labels cannot affect training, checkpoint selection, threshold selection, or model naming.
- Primary ranking estimates average the three training seeds; evaluation bootstrap and seed stability are reported separately.

## Artifact manifest

The machine-readable freeze is authoritative: `data/manifests/e003_protocol_freeze.yaml`.
Panel, annotation, sampler-pool, Core-4 request/runtime, external, DataSAIL, model, seed, metric, uncertainty, and figure contracts are recorded there with hashes where available.
