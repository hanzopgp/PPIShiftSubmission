#!/usr/bin/env bash
set -u

PYTHON_BIN="/path/to/miniconda3/envs/bioppi/bin/python3.10"
DISPATCH="/path/to/PPIShift-e003-core4/scripts/dispatch_e003_broken_pair_scores_staged.py"
STATE="/path/to/PPIShiftArtifacts/e003/core4_final_paper/e003_broken_pair_score_staged.json"
LOG="/path/to/PPIShiftArtifacts/e003/core4_final_paper/logs/broken-dispatch.log"
MAX_ACTIVE_JOBS="${E003_BROKEN_MAX_ACTIVE_JOBS:-19}"
export PYTHONPATH="/path/to/PPIShift-e003-core4/src:/path/to/PPIShift-e003-core4/scripts${PYTHONPATH:+:$PYTHONPATH}"

for _iteration in $(seq 1 180); do
  "$PYTHON_BIN" "$DISPATCH" \
    --core-results /path/to/PPIShiftArtifacts/e003/core4_final_paper/results/core/e003_core_results.json \
    --artifact-root /path/to/PPIShiftArtifacts/e003/core4_final_paper \
    --broken-root /path/to/PPIShiftArtifacts/e003/core4_final_paper/signal_audit/broken_pairs \
    --repo-root /path/to/PPIShift-e003-core4 \
    --script /path/to/PPIShift-e003-core4/scripts/launch_e003_broken_pair_score.slurm \
    --state-manifest "$STATE" \
    --embedding-dir /path/to/PPIShiftArtifacts/e003/core4_final_paper/representations/external_esm2_t30_150m \
    --source-root /path/to/ExternalPPI/src \
    --protocol /path/to/PPIShift-e003-core4/audit/E003_SIGNAL_AUDIT_PROTOCOL.md \
    --max-new-jobs 20 \
    --max-active-jobs "$MAX_ACTIVE_JOBS" \
    --retry-failed >> "$LOG" 2>&1 || true
  complete=$("$PYTHON_BIN" -c 'import json; p=json.load(open("/path/to/PPIShiftArtifacts/e003/core4_final_paper/e003_broken_pair_score_staged.json")); print(p.get("summary", {}).get("complete", 0))')
  if [ "$complete" -ge 72 ]; then
    exit 0
  fi
  sleep 60
done
