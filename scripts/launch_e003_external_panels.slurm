#!/usr/bin/env bash
#SBATCH --job-name=e003-external
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/external-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/external-%j.err

set -euo pipefail

REPO=/path/to/PPIShift
ROOT="${E003_ARTIFACT_ROOT:-/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
FETCH_DATA="$ROOT/upstream/bernett/bin/fetch_data.py"
HISTORICAL_MANIFEST="$REPO/data/manifests/e002_external_panel_manifest.json"

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/external_annotations" "$ROOT/external_panels"
"$PYTHON_BIN" scripts/prepare_e003_external_annotations.py \
  --historical-manifest "$HISTORICAL_MANIFEST" \
  --fetch-data "$FETCH_DATA" \
  --python "$PYTHON_BIN" \
  --output-dir "$ROOT/external_annotations" \
  --repo-root "$REPO"
exec "$PYTHON_BIN" scripts/materialize_e003_external_panels.py \
  --historical-manifest "$HISTORICAL_MANIFEST" \
  --evaluation-registry /path/to/PPIShiftArtifacts/exclusion/evaluation_positive_union.csv \
  --vidal-panel /path/to/PPIShiftArtifacts/post_e001_v3/vidal/vidal_compatible_panel.csv.gz \
  --source-config configs/data/prefreeze_sources.yaml \
  --novelty-manifest data/manifests/novelty_manifest.yaml \
  --species "$ROOT/external_annotations/species.tsv" \
  --go-bp "$ROOT/external_annotations/go_annotations.tsv" \
  --output-root "$ROOT/external_panels" \
  --manifest "$ROOT/e003_external_panel_manifest.json" \
  --seed 1729 \
  --limit 10111 \
  --bern-time-limit 7200 \
  --tppni-published-n 10000000
