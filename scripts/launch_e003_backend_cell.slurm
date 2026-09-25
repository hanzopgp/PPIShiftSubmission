#!/usr/bin/env bash
#SBATCH --job-name=ppi-e003-backend
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/backend-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/backend-%j.err

set -euo pipefail
: "${REQUEST:?Pass REQUEST=/absolute/path/to/request.json}"
: "${BACKEND_BIN:?Pass the pinned ExternalPPI backend binary}"
ROOT="${E003_ARTIFACT_ROOT:-/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected}"
mkdir -p "$ROOT/logs"
echo "JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "REQUEST=$REQUEST"
echo "HOST=$(hostname)"
nvidia-smi -L
exec "$BACKEND_BIN" --request "$REQUEST"
