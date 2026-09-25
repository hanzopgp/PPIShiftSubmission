#!/usr/bin/env bash
# Build the E003 external ESM2-150M mean-embedding store and token overlay.
# This job is representation extraction only; no external score is inspected.
#SBATCH --job-name=ppi-e003-ext-150m
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-150m-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/ext-150m-%j.err

set -euo pipefail
: "${CACHE_INPUT_CSV:?Pass the combined E003 cache CSV}"
: "${CACHE_INPUT_FASTA:?Pass the combined E003 cache FASTA}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PPI_PROJECT=/path/to/ExternalPPI
PYTHON_BIN=/path/to/miniconda3/envs/bioppi/bin/python3.10
MEAN_OUTPUT="$E003_ARTIFACT_ROOT/representations/external_esm2_t30_150m"
TOKEN_OUTPUT="$E003_ARTIFACT_ROOT/representations/external_token_cache_esm2_t30_150m"
BASE_TOKEN=/path/to/ExternalPPIArtifacts/external_models/D007/token_cache/esm2_t30_150m_a695f604
REVISION=a695f6045e2e32885fa60af20c13cb35398ce30c
mkdir -p "$E003_ARTIFACT_ROOT/logs"
if [ -e "$MEAN_OUTPUT/summary.json" ] || [ -e "$TOKEN_OUTPUT/summary-shard-000-of-001.json" ]; then
  echo "Refusing to overwrite completed E003 external 150M caches" >&2
  exit 1
fi
export PYTHONPATH="$PPI_PROJECT/src"
export PPI_GIT_COMMIT="$(git -C "$PPI_PROJECT" rev-parse HEAD)"
"$PYTHON_BIN" -m ppi_project.mean_cache build \
  --input "$CACHE_INPUT_CSV" \
  --output-dir "$MEAN_OUTPUT" \
  --model-name facebook/esm2_t30_150M_UR50D \
  --model-revision "$REVISION" \
  --shard-index 0 --shard-count 1 --window-size 1022 --overlap 128 \
  --max-tokens 8192 --max-sequences 16 --device cuda
mkdir -p "$TOKEN_OUTPUT/records"
for source in "$BASE_TOKEN"/records/*.pt; do
  destination="$TOKEN_OUTPUT/records/$(basename "$source")"
  if [ ! -e "$destination" ]; then
    ln -s "$source" "$destination" 2>/dev/null || true
  fi
done
exec "$PYTHON_BIN" -m ppi_project.sota_token_cache \
  --fasta "$CACHE_INPUT_FASTA" \
  --output-dir "$TOKEN_OUTPUT" \
  --model-name facebook/esm2_t30_150M_UR50D \
  --model-revision "$REVISION" \
  --maximum-length 512 --shard-index 0 --shard-count 1 --device cuda
