#!/usr/bin/env bash
#SBATCH --job-name=ppi-e003-ext-post
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-post-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-post-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${EXTERNAL_REQUEST_MANIFEST:?Pass the E003 external request manifest}"
: "${EXTERNAL_MANIFEST:?Pass the E003 external panel manifest}"
: "${CORE_RESULTS:?Pass the completed E003 core results}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/postprocess_e003_external.py" \
  --request-manifest "$EXTERNAL_REQUEST_MANIFEST" \
  --external-manifest "$EXTERNAL_MANIFEST" \
  --core-results "$CORE_RESULTS" \
  --output-dir "$E003_ARTIFACT_ROOT/results/external"
