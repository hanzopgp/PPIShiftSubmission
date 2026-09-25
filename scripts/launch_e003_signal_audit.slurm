#!/usr/bin/env bash
# Validate the frozen E003 signal controls after all broken-pair scores exist.
# This job is CPU-only and fail-closed: a non-zero exit means the paper pack is
# not ready for scientific claims.
#SBATCH --job-name=ppi-e003-signal
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/signal-audit-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/signal-audit-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${AUDIT_DIR:?Pass the frozen E003 signal-audit input directory}"
: "${OUTPUT_DIR:?Pass the signal-audit result directory}"
: "${CORE_REGISTRY:?Pass the completed canonical E003 result registry}"
: "${BROKEN_DIR:?Pass the completed broken-pair score directory}"
: "${PROTOCOL:?Pass the frozen signal-audit protocol}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"

PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src:$PPI_SHIFT_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/analyze_e003_signal_audit.py" \
  --audit-dir "$AUDIT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repo-root "$PPI_SHIFT_REPO" \
  --protocol "$PROTOCOL" \
  --core-registry "$CORE_REGISTRY" \
  --broken-dir "$BROKEN_DIR"
