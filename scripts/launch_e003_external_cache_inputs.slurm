#!/usr/bin/env bash
# Build the immutable combined sequence inputs for both external caches.
# This job reads panel manifests only; it performs no model scoring.
#SBATCH --job-name=ppi-e003-ext-inputs
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-inputs-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-inputs-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${PANEL_MANIFEST:?Pass the corrected E003 panel manifest}"
: "${EXTERNAL_MANIFEST:?Pass the corrected E003 external panel manifest}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
INPUT_ROOT="$E003_ARTIFACT_ROOT/external_cache_inputs"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$INPUT_ROOT"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/prepare_e003_external_cache_inputs.py" \
  --panel-manifest "$PANEL_MANIFEST" \
  --external-manifest "$EXTERNAL_MANIFEST" \
  --output-csv "$INPUT_ROOT/sequences.csv" \
  --output-fasta "$INPUT_ROOT/sequences.fasta" \
  --manifest "$INPUT_ROOT/e003_external_cache_input_manifest.json"
