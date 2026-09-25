#!/usr/bin/env bash
# Generate the final E003 summaries and paper figures after every result
# surface is complete. This job consumes result manifests only.
#SBATCH --job-name=ppi-e003-analysis
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/analysis-%j.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/analysis-%j.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPIShift repository root}"
: "${CORE_RESULTS:?Pass the completed E003 core results}"
: "${EXTERNAL_RESULTS:?Pass the completed E003 external results}"
: "${DATASAIL_RESULTS:?Pass the completed DataSAIL results}"
: "${PANEL_MANIFEST:?Pass the frozen E003 panel manifest}"
: "${SAMPLER_TRACE_MANIFEST:?Pass the trace-bearing sampler manifest}"
: "${CONFIG:?Pass the frozen E003 config}"
: "${PROTOCOL_FREEZE:?Pass the immutable E003 protocol freeze}"
: "${E003_ARTIFACT_ROOT:?Pass the corrected E003 artifact root}"
PYTHON_BIN="${PPI_SHIFT_PYTHON:-/path/to/miniconda3/envs/bioppi/bin/python3.10}"
export PYTHONPATH="$PPI_SHIFT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$E003_ARTIFACT_ROOT/results/analysis"
"$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/analyze_e003_results.py" \
  --core-results "$CORE_RESULTS" \
  --external-results "$EXTERNAL_RESULTS" \
  --datasail-results "$DATASAIL_RESULTS" \
  --panel-manifest "$PANEL_MANIFEST" \
  --sampler-trace-manifest "$SAMPLER_TRACE_MANIFEST" \
  --config "$CONFIG" \
  --protocol-freeze "$PROTOCOL_FREEZE" \
  --repo-root "$PPI_SHIFT_REPO" \
  --output-dir "$E003_ARTIFACT_ROOT/results/analysis" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 1729
exec "$PYTHON_BIN" "$PPI_SHIFT_REPO/scripts/verify_e003_figures.py" \
  --analysis "$E003_ARTIFACT_ROOT/results/analysis/e003_analysis.json" \
  --registry "$E003_ARTIFACT_ROOT/results/analysis/e003_result_registry.json" \
  --figure-hashes "$E003_ARTIFACT_ROOT/results/analysis/e003_figure_hashes.json" \
  --repo-root "$PPI_SHIFT_REPO" \
  --output "$E003_ARTIFACT_ROOT/results/analysis/e003_figure_integrity_audit.json"
