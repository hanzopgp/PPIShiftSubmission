#!/usr/bin/env bash
#SBATCH --job-name=ppi-e003-core-post
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/core-post-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/core-post-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPI_SHIFT_REPO=/path/to"
: "${E001_REGISTRY:?Pass the immutable E001v3 run registry}"
: "${BACKEND_MANIFEST:?Pass the E003 backend request manifest}"
: "${PYTHON_MANIFEST:?Pass the E003 Python request manifest}"
: "${FROZEN_MANIFEST:?Pass the E003 Frozen request manifest}"
: "${REUSE_MANIFEST:?Pass the E003 reuse request manifest}"
: "${PANEL_MANIFEST:?Pass the frozen E003 panel manifest}"
: "${CONFIG:?Pass the frozen E003 config}"
: "${PROTOCOL_FREEZE:?Pass the immutable E003 protocol freeze}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$E003_ARTIFACT_ROOT/results/core"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/postprocess_e003_core.py" \
  --e001-registry "$E001_REGISTRY" \
  --backend-manifest "$BACKEND_MANIFEST" \
  --python-manifest "$PYTHON_MANIFEST" \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --reuse-manifest "$REUSE_MANIFEST" \
  --panel-manifest "$PANEL_MANIFEST" \
  --config "$CONFIG" \
  --protocol-freeze "$PROTOCOL_FREEZE" \
  --output-dir "$E003_ARTIFACT_ROOT/results/core"
