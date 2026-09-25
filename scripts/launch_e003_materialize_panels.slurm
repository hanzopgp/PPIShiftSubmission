#!/usr/bin/env bash
#SBATCH --job-name=e003-panels
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/panels-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/panels-%j.err

set -euo pipefail

REPO=/path/to/PPIShift
ROOT="${E003_ARTIFACT_ROOT:-/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected}"
ANNOTATION_ROOT="${E003_ANNOTATION_ROOT:-$ROOT}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"

cd "$REPO"
export PYTHONPATH=src
exec "$PYTHON_BIN" scripts/materialize_e003_panels.py \
  --e001-config configs/experiments/e001_v3_common_cohort.yaml \
  --e001-manifest data/manifests/e001_v3_common_cohort_manifest.yaml \
  --source-config configs/data/prefreeze_sources.yaml \
  --registry-dir /path/to/PPIShiftArtifacts/exclusion \
  --species "$ANNOTATION_ROOT/annotations/species.tsv" \
  --go-bp "$ANNOTATION_ROOT/annotations/go_annotations.tsv" \
  --output-root "$ROOT/panels_and_samplers" \
  --manifest "$ROOT/e003_panel_manifest.yaml" \
  --repo-root "$REPO" \
  --seed 1729 \
  --tppni-published-n 10000000 \
  --bern-time-limit 7200
