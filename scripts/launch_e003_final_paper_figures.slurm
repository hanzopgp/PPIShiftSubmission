#!/usr/bin/env bash
# Render the result-dependent E003 paper figure pack from hashed inputs only.
# The signal audit and canonical result registry must already be complete.
#SBATCH --job-name=ppi-e003-final-fig
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/final-figures-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/final-figures-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${AUDIT_DIR:?Pass the frozen E003 signal-audit input directory}"
: "${CORE_REGISTRY:?Pass the completed canonical E003 result registry}"
: "${CONFIG:?Pass the frozen E003 config}"
: "${PROTOCOL:?Pass the frozen signal-audit protocol}"
: "${OUTPUT_DIR:?Pass the final paper figure directory}"

PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src:$PPI_SHIFT_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/build_e003_final_paper_figures.py" \
  --audit-dir "$AUDIT_DIR" \
  --core-registry "$CORE_REGISTRY" \
  --config "$CONFIG" \
  --protocol "$PROTOCOL" \
  --repo-root "$PPI_SHIFT_REPO" \
  --output-dir "$OUTPUT_DIR"
