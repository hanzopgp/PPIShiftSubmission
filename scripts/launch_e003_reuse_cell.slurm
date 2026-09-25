#!/usr/bin/env bash
#SBATCH --job-name=ppi-e003-reuse
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/reuse-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/reuse-%j.err

set -euo pipefail
: "${REQUEST:?Pass REQUEST=/absolute/path/to/reuse_request.json}"
: "${PPI_SHIFT_REPO:?Pass PPI_SHIFT_REPO=/path/to"
REPO="$PPI_SHIFT_REPO"
PYTHON_BIN=/path/to/miniconda3/envs/bioppi/bin/python3.10
ROOT="${E003_ARTIFACT_ROOT:-/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected}"
export PYTHONPATH="$REPO/src:/path/to/ExternalPPI/src${PYTHONPATH:+:$PYTHONPATH}"
export PPI_SOTA_TOKEN_CACHE="${PPI_SOTA_TOKEN_CACHE:-/path/to/ExternalPPIArtifacts/external_models/D007/token_cache/esm2_t30_150m_a695f604}"
export PPI_SOTA_EXTERNAL_ROOT="${PPI_SOTA_EXTERNAL_ROOT:-/path/to/ExternalPPIArtifacts/external_models/D007}"
mkdir -p "$ROOT/logs"
echo "JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "REQUEST=$REQUEST"
exec "$PYTHON_BIN" "$REPO/scripts/run_e003_reused_e001_scores.py" \
  --request "$REQUEST" --repo-root "$REPO" \
  --embedding-dir /path/to/ExternalPPIArtifacts/embeddings/fresh-gold-a695f604-w1022-o128-v2
