#!/usr/bin/env bash
#SBATCH --job-name=ppi-e003-datasail-post
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/datasail-post-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/datasail-post-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${DATASAIL_REQUEST_MANIFEST:?Pass the DataSAIL request manifest}"
: "${DATASAIL_PANEL_MANIFEST:?Pass the DataSAIL panel manifest}"
: "${CONFIG:?Pass the E003 config}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/postprocess_e003_datasail.py" \
  --request-manifest "$DATASAIL_REQUEST_MANIFEST" \
  --panel-manifest "$DATASAIL_PANEL_MANIFEST" \
  --core-config "$CONFIG" \
  --output-dir "$E003_ARTIFACT_ROOT/results/datasail"
