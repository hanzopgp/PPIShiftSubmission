#!/usr/bin/env bash
# Materialize and prepare the predeclared official-DataSAIL sensitivity.
# No model score is read by this job.
#SBATCH --job-name=ppi-e003-datasail-prep
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/datasail-prep-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/datasail-prep-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${PANEL_MANIFEST:?Pass the completed corrected E003 panel manifest}"
: "${PROTOCOL_FREEZE:?Pass the immutable E003 protocol freeze}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
OFFICIAL_MANIFEST="$PPI_SHIFT_REPO/data/manifests/datasail_official_manifest.yaml"
SOURCE_CONFIG="$PPI_SHIFT_REPO/configs/data/prefreeze_sources.yaml"
EMBEDDING_DIR=/path/to/ExternalPPIArtifacts/embeddings/fresh-gold-a695f604-w1022-o128-v2
PANEL_ROOT="$E003_ARTIFACT_ROOT/datasail_panels"
MATRIX_ROOT="$E003_ARTIFACT_ROOT/datasail_matrix"
PANEL_OUTPUT="$E003_ARTIFACT_ROOT/e003_datasail_panel_manifest.yaml"
MATRIX_OUTPUT="$MATRIX_ROOT/datasail_request_manifest.json"
test -s "$PROTOCOL_FREEZE"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$E003_ARTIFACT_ROOT/logs"
"$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/materialize_e003_datasail_panels.py" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --source-config "$SOURCE_CONFIG" \
  --embedding-dir "$EMBEDDING_DIR" \
  --output-root "$PANEL_ROOT" \
  --manifest "$PANEL_OUTPUT"
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/prepare_e003_datasail_matrix.py" \
  --panel-manifest "$PANEL_OUTPUT" \
  --config "$PPI_SHIFT_REPO/configs/experiments/e003_method_fidelity_final.yaml" \
  --embedding-dir "$EMBEDDING_DIR" \
  --output-root "$MATRIX_ROOT" \
  --manifest "$MATRIX_OUTPUT"
