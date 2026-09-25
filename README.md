# PPIShift: anonymous ICLR 2027 code submission

This repository contains the code and compact source tables for the ICLR 2027
study of negative-environment shift in sequence-based protein-protein
interaction prediction. The paper source is in `paper/iclr2027/`; the frozen
protocol, dataset semantics, and decision record are in `PROTOCOL.md`,
`DATASETS.md`, and `DECISIONS.md`.

## Contents

- `src/ppi_shift/`: data contracts, negative environments, models, and metrics.
- `scripts/`: E003 panel preparation, model execution, analysis, and figure code.
- `configs/experiments/`: the final E003 configuration and signal audit config.
- `paper/figures/`: compact figure source tables and paper figure PDFs.
- `tests/`: focused reproducibility and scientific-invariant tests.
- `audit/`: protocol freeze, source recovery, and interpretation boundaries.
- `manifest.json`: SHA-256 checksums of the published files.

## Setup and checks

Use Python 3.11 or newer. From the repository root:

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=.:scripts python -m pytest -q tests/test_e003_submission.py tests/test_iclr2027_figures.py
PYTHONPATH=.:scripts python scripts/build_iclr2027_final_figures.py --help
```

Model execution additionally requires the appropriate PyTorch, ESM, DataSAIL,
and external model runtimes. Raw or controlled datasets, checkpoints, embeddings,
and large run artifacts are not redistributed here. Use the cited data sources
and the hashes and provenance in the protocol and audit records to identify the
frozen inputs. Site-specific paths in the historical execution records and
cluster scripts are replaced by `/path/to/...` placeholders; configure them
for your environment before running those workflows.

The source-recovery boundary for historical execution is documented in
`audit/E003_EXECUTION_SOURCE_RECOVERY.md`. The paper's scientific claims are
limited to the recorded frozen runs and should not be inferred from a fresh
execution of this code alone.
