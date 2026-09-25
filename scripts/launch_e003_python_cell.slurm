#!/usr/bin/env bash
#SBATCH --job-name=ppi-e003-python
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=lcqb
#SBATCH --qos=normal
#SBATCH --array=0-17%8
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/python-%A_%a.out
#SBATCH --error=/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/python-%A_%a.err

set -euo pipefail
: "${PPI_SHIFT_REPO:?Pass PPI_SHIFT_REPO=/path/to"
REPO="$PPI_SHIFT_REPO"
PYTHON_BIN=/path/to/miniconda3/envs/bioppi/bin/python3.10
ROOT="${E003_ARTIFACT_ROOT:-/path/to/PPIShiftArtifacts/e003/method_fidelity_final_corrected}"
INDEX=$ROOT/python_matrix/request_index.json
test -s "$INDEX"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
request=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["request"])' "$INDEX" "${SLURM_ARRAY_TASK_ID:-0}")
model=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$request")
seed=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$request")
environment=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_environment"])' "$request")
panel_manifest=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["panel_manifest"])' "$request")
config=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"])' "$request")
output_dir=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$request")
test_environments=$($PYTHON_BIN -c 'import json,sys; print(",".join(json.load(open(sys.argv[1])).get("test_environments", [])))' "$request")
echo "JOB_ID=${SLURM_JOB_ID:-unknown} MODEL=$model SEED=$seed TRAIN=$environment"
test "$model" = "rapppid"
exec "$PYTHON_BIN" "$REPO/scripts/run_e003_rapppid_cell.py" \
  --panel-manifest "$panel_manifest" --config "$config" \
  --output-dir "$output_dir" --repo-root "$REPO" \
  --seed "$seed" --train-environment "$environment" \
  --test-environments "$test_environments"
