#!/usr/bin/env bash
# Submit the 72 Core-4 frozen-checkpoint external score jobs after the
# corrected core checkpoint index are complete.
#SBATCH --job-name=ppi-e003-ext-dispatch
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-dispatch-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-dispatch-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${PANEL_MANIFEST:?Pass the corrected E003 panel manifest}"
: "${EXTERNAL_MANIFEST:?Pass the corrected E003 external panel manifest}"
: "${PROTOCOL_FREEZE:?Pass the immutable E003 protocol freeze}"
: "${CORE_INDEX:?Pass the completed E003 checkpoint index}"
: "${CORE_RESULTS:?Pass the completed E003 core results}"
: "${BACKEND_MANIFEST:?Pass the E003 backend request manifest}"
: "${PYTHON_MANIFEST:?Pass the E003 Python request manifest}"
: "${FROZEN_MANIFEST:?Pass the E003 Frozen request manifest}"
: "${REUSE_MANIFEST:?Pass the E003 reuse request manifest}"
: "${EXTERNAL_EMBEDDING_DIR:?Pass the completed external ESM2-150M cache}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
: "${EXTERNAL_LAUNCH_MANIFEST:?Pass the external launch manifest output}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/dispatch_e003_external_scores_staged.py" \
  --repo-root "$PPI_SHIFT_REPO" \
  --panel-manifest "$PANEL_MANIFEST" \
  --external-manifest "$EXTERNAL_MANIFEST" \
  --protocol-freeze "$PROTOCOL_FREEZE" \
  --core-index "$CORE_INDEX" \
  --core-results "$CORE_RESULTS" \
  --backend-manifest "$BACKEND_MANIFEST" \
  --python-manifest "$PYTHON_MANIFEST" \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --reuse-manifest "$REUSE_MANIFEST" \
  --external-embedding-dir "$EXTERNAL_EMBEDDING_DIR" \
  --artifact-root "$E003_ARTIFACT_ROOT" \
  --launch-manifest "$EXTERNAL_LAUNCH_MANIFEST" \
  --state-manifest "$E003_ARTIFACT_ROOT/e003_external_staged_dispatch.json" \
  --max-new-jobs 1
