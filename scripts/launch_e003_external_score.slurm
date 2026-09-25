#!/usr/bin/env bash
# Score one frozen E003 checkpoint across every corrected external surface.
#SBATCH --job-name=ppi-e003-ext-score
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-score-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-score-%j.err

set -euo pipefail
: "${REQUEST:?Pass the E003 external score request}"
: "${CORE_INDEX:?Pass the completed E003 checkpoint index}"
: "${EXTERNAL_EMBEDDING_DIR:?Pass the ESM2-150M external embedding store}"
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src:$PPI_SHIFT_REPO/scripts:/path/to/ExternalPPI/src${PYTHONPATH:+:$PYTHONPATH}"
export PPI_SOTA_TOKEN_CACHE="$E003_ARTIFACT_ROOT/representations/external_token_cache_esm2_t30_150m"
export PPI_SOTA_EXTERNAL_ROOT="${PPI_SOTA_EXTERNAL_ROOT:-/path/to/ExternalPPIArtifacts/external_models/D007}"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/run_e003_external_scores.py" \
  --request "$REQUEST" \
  --core-index "$CORE_INDEX" \
  --embedding-dir "$EXTERNAL_EMBEDDING_DIR"
