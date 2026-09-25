#!/usr/bin/env bash
# Build the combined ESM2-3B residue cache used for E003 SENSE external scoring.
# Existing corrected core records are linked into the overlay and validated by
# the pinned cache builder; only missing external sequences are generated.
#SBATCH --job-name=ppi-e003-ext-sense
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --array=0-15%8
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/logs/ext-sense-%A_%a.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected/logs/ext-sense-%A_%a.err

set -euo pipefail
: "${CACHE_INPUT_FASTA:?Pass the combined E003 cache FASTA}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
REPO=/path/to/PPIShift
PPI_PROJECT=/path/to/ExternalPPI
PYTHON_BIN=/path/to/miniconda3/envs/bioppi/bin/python3.10
BASE_CACHE="$E003_ARTIFACT_ROOT/representations/sense_esm2_t36_3b"
OUTPUT="$E003_ARTIFACT_ROOT/representations/sense_esm2_t36_3b_external"
MODEL=facebook/esm2_t36_3B_UR50D
REVISION=476b639933c8baad5ad09a60ac1a87f987b656fc
test -s "$CACHE_INPUT_FASTA"
test -d "$BASE_CACHE/records"
mkdir -p "$OUTPUT/records" "$E003_ARTIFACT_ROOT/logs"
for source in "$BASE_CACHE"/records/*.pt; do
  destination="$OUTPUT/records/$(basename "$source")"
  if [ ! -e "$destination" ]; then
    ln -s "$source" "$destination" 2>/dev/null || true
  fi
done
export PYTHONPATH="$PPI_PROJECT/src:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "$PYTHON_BIN" -m ppi_project.sota_token_cache \
  --fasta "$CACHE_INPUT_FASTA" \
  --output-dir "$OUTPUT" \
  --model-name "$MODEL" \
  --model-revision "$REVISION" \
  --maximum-length 512 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-count 16 \
  --device cuda
