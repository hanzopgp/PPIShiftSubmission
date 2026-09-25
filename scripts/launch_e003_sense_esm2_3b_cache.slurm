#!/usr/bin/env bash
# Build the frozen residue-level ESM2-3B cache used by clean-init E003 SENSE-PPI.
# This job performs representation extraction only; it never scores a panel.
# The cache format is the pinned ExternalPPI token-cache contract.
#SBATCH --job-name=ppi-e003-sense-cache
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --array=0-15%8
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/logs/sense-cache-%A_%a.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/logs/sense-cache-%A_%a.err

set -euo pipefail

REPO=/path/to/PPIShift
PYTHON_BIN=/path/to/miniconda3/envs/bioppi/bin/python3.10
PPI_PROJECT=/path/to/ExternalPPI
ROOT="${E003_ARTIFACT_ROOT:-/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected}"
FASTA=$ROOT/annotations/sense_cache_sequences.fasta
REGISTRY_MANIFEST=$ROOT/annotations/sense_cache_registry_manifest.json
OUTPUT=$ROOT/representations/sense_esm2_t36_3b
MODEL=facebook/esm2_t36_3B_UR50D
REVISION=476b639933c8baad5ad09a60ac1a87f987b656fc

test -s "$FASTA"
test -s "$REGISTRY_MANIFEST"
mkdir -p "$OUTPUT" "$ROOT/logs"
export PYTHONPATH="$PPI_PROJECT/src:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
echo "JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-unknown}"
echo "HOST=$(hostname)"
if PPI_PROJECT_COMMIT=$(git -C "$PPI_PROJECT" rev-parse HEAD 2>/dev/null); then
  echo "PPI_PROJECT_COMMIT=$PPI_PROJECT_COMMIT"
else
  echo "PPI_PROJECT_COMMIT=unversioned-source"
fi
echo "CACHE_REGISTRY=$FASTA"
echo "CACHE_REGISTRY_SHA256=$(sha256sum "$FASTA" | awk '{print $1}')"
echo "CACHE_REGISTRY_MANIFEST=$REGISTRY_MANIFEST"
echo "MODEL=$MODEL"
echo "REVISION=$REVISION"
exec "$PYTHON_BIN" -m ppi_project.sota_token_cache \
  --fasta "$FASTA" \
  --output-dir "$OUTPUT" \
  --model-name "$MODEL" \
  --model-revision "$REVISION" \
  --maximum-length 512 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-count 16 \
  --device cuda
