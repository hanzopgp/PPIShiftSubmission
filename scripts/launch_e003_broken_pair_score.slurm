#!/usr/bin/env bash
# Score one selected E003 checkpoint on every deterministic broken-pair panel.
#SBATCH --job-name=ppi-e003-broken
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/broken-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/broken-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${BROKEN_ROOT:?Pass the frozen broken-pair panel root}"
: "${MODEL:?Pass the E003 model name}"
: "${RUN_PATH:?Pass the selected checkpoint run path}"
: "${OUTPUT_DIR:?Pass the signal-audit output directory}"
: "${SEED:?Pass the E003 seed}"
: "${TRAIN_ENVIRONMENT:?Pass the E003 train environment}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"

PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$E003_ARTIFACT_ROOT/representations/external_esm2_t30_150m}"
SOURCE_ROOT="${SOURCE_ROOT:-/path/to/ExternalPPI/src}"
PROTOCOL="${PROTOCOL:-$PPI_SHIFT_REPO/audit/E003_SIGNAL_AUDIT_PROTOCOL.md}"
export PYTHONPATH="$PPI_SHIFT_REPO/src:$PPI_SHIFT_REPO/scripts:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/score_e003_broken_pairs.py" \
  --broken-root "$BROKEN_ROOT" \
  --model "$MODEL" \
  --run-path "$RUN_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --repo-root "$PPI_SHIFT_REPO" \
  --source-root "$SOURCE_ROOT" \
  --embedding-dir "$EMBEDDING_DIR" \
  --protocol "$PROTOCOL" \
  --seed "$SEED" \
  --train-environment "$TRAIN_ENVIRONMENT" \
  --environments "uniform_partner,degree_partner_matched,sequence_nuisance_matched,esm_partner_matched,bernett2026_ilp_exact,tppni2025_scm_cl3_exact_balanced"
