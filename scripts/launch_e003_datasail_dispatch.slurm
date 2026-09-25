#!/usr/bin/env bash
# Submit the 18 official-DataSAIL score units after panel/matrix preparation.
#SBATCH --job-name=ppi-e003-datasail-dispatch
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/datasail-dispatch-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/datasail-dispatch-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${DATASAIL_PREP_JOB:?Pass the DataSAIL preparation job ID}"
: "${DATASAIL_MATRIX_MANIFEST:?Pass the prepared DataSAIL request manifest}"
: "${DATASAIL_PANEL_MANIFEST:?Pass the prepared DataSAIL panel manifest}"
: "${PROTOCOL_FREEZE:?Pass the immutable E003 protocol freeze}"
: "${CONFIG:?Pass the E003 config}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
: "${DATASAIL_LAUNCH_MANIFEST:?Pass the DataSAIL launch manifest output}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SCHEDULER_DEPENDENCY_ARGS=()
if [[ "${DATASAIL_SKIP_PREP_DEPENDENCY:-0}" == "1" ]]; then
  SCHEDULER_DEPENDENCY_ARGS+=(--no-prepare-dependency)
fi
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/dispatch_e003_datasail_scores_staged.py" \
  --repo-root "$PPI_SHIFT_REPO" \
  --prepare-job "$DATASAIL_PREP_JOB" \
  --matrix-manifest "$DATASAIL_MATRIX_MANIFEST" \
  --panel-manifest "$DATASAIL_PANEL_MANIFEST" \
  --protocol-freeze "$PROTOCOL_FREEZE" \
  --config "$CONFIG" \
  --artifact-root "$E003_ARTIFACT_ROOT" \
  --launch-manifest "$DATASAIL_LAUNCH_MANIFEST" \
  --state-manifest "$E003_ARTIFACT_ROOT/e003_datasail_staged_dispatch.json" \
  --max-new-jobs 1 \
  "${SCHEDULER_DEPENDENCY_ARGS[@]}"
