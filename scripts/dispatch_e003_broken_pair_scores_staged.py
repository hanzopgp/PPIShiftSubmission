#!/usr/bin/env python3
"""Resume one-unit-at-a-time E003 broken-pair score fan-out."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from e003_staged_fanout import scheduler_state, submit_job, utc_now, write_json_atomic

from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS

CORE_MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
SEEDS = (11, 22, 33)
SCHEMA = "ppi_shift.e003_broken_pair_staged_dispatch.v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON mapping: {path}")
    return value


def _units(core_results: Path, artifact_root: Path) -> list[dict[str, Any]]:
    payload = _json(core_results)
    if payload.get("status") != "complete_core_matrix" or len(payload.get("cells", [])) != 432:
        raise SystemExit("E003 core matrix is incomplete")
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for cell in payload["cells"]:
        key = (str(cell["model"]), int(cell["seed"]), str(cell["train_environment"]))
        prediction_path = Path(str(cell["prediction_path"]))
        current = by_key.setdefault(
            key,
            {
                "model": key[0],
                "seed": key[1],
                "train_environment": key[2],
                "run_path": str(prediction_path.parent),
                "prediction_sha256": str(cell["prediction_sha256"]),
            },
        )
        if current["run_path"] != str(prediction_path.parent):
            raise SystemExit(f"Core run path changed within unit: {key}")
    expected = {
        (model, seed, environment)
        for model in CORE_MODELS
        for seed in SEEDS
        for environment in E003_ENVIRONMENTS
    }
    if set(by_key) != expected:
        raise SystemExit(f"Broken-pair roster mismatch: observed={len(by_key)} expected=72")
    output_dir = artifact_root / "signal_audit" / "broken_scores"
    result = []
    for key in sorted(by_key):
        unit = by_key[key]
        unit["output_dir"] = str(output_dir)
        unit["result_path"] = str(
            output_dir
            / (
                f"broken_scores__{unit['model']}__"
                f"train-{unit['train_environment']}__seed-{unit['seed']}.json"
            )
        )
        result.append(unit)
    return result


def _state(path: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA, "created_utc": utc_now(), "units": units}
    payload = _json(path)
    if payload.get("schema_version") != SCHEMA or payload.get("units") is None:
        raise SystemExit(f"Invalid broken-pair staged state: {path}")
    recorded = [
        {
            key: value
            for key, value in unit.items()
            if key
            in {
                "model",
                "seed",
                "train_environment",
                "run_path",
                "output_dir",
                "result_path",
                "prediction_sha256",
            }
        }
        for unit in payload["units"]
    ]
    expected = [
        {
            key: value
            for key, value in unit.items()
            if key
            in {
                "model",
                "seed",
                "train_environment",
                "run_path",
                "output_dir",
                "result_path",
                "prediction_sha256",
            }
        }
        for unit in units
    ]
    if recorded != expected:
        raise SystemExit("Broken-pair staged immutable unit roster changed")
    return payload


def _is_complete(unit: dict[str, Any]) -> bool:
    path = Path(str(unit["result_path"]))
    if not path.is_file():
        return False
    try:
        payload = _json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("control") == "broken_pair_score" and len(payload.get("results", [])) == 6


def _submit(unit: dict[str, Any], args: argparse.Namespace) -> str:
    environment = {
        "PPI_SHIFT_REPO": str(args.repo_root),
        "BROKEN_ROOT": str(args.broken_root),
        "MODEL": str(unit["model"]),
        "RUN_PATH": str(unit["run_path"]),
        "OUTPUT_DIR": str(unit["output_dir"]),
        "SEED": str(unit["seed"]),
        "TRAIN_ENVIRONMENT": str(unit["train_environment"]),
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
        "EMBEDDING_DIR": str(args.embedding_dir),
        "SOURCE_ROOT": str(args.source_root),
        "PROTOCOL": str(args.protocol),
    }
    return submit_job(args.script, environment, [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-results", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--broken-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--max-new-jobs", type=int, default=1)
    parser.add_argument("--max-active-jobs", type=int, default=20)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.max_new_jobs < 0:
        raise SystemExit("--max-new-jobs must be non-negative")
    if args.max_active_jobs < 1:
        raise SystemExit("--max-active-jobs must be positive")
    units = _units(args.core_results, args.artifact_root)
    payload = _state(args.state_manifest, units)
    entries = payload["units"]
    submitted = 0
    active_jobs = 0
    summary = {"complete": 0, "active": 0, "failed": 0, "missing": 0, "submitted": 0}
    for index, unit in enumerate(entries):
        unit.setdefault("job_id", None)
        unit.setdefault("state", "MISSING")
        if _is_complete(unit):
            unit["state"] = "COMPLETED"
            summary["complete"] += 1
            continue
        job_id = unit.get("job_id")
        if job_id:
            state = scheduler_state(str(job_id))
            unit["state"] = state
            if state.startswith("COMPLETED") and not _is_complete(unit):
                raise SystemExit(f"Broken-pair job completed without its result: {job_id}")
            if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")):
                summary["failed"] += 1
                if not args.retry_failed:
                    continue
                payload.setdefault("retry_history", []).append(
                    {
                        "job_id": str(job_id),
                        "state": state,
                        "model": unit["model"],
                        "seed": unit["seed"],
                        "train_environment": unit["train_environment"],
                        "recorded_utc": utc_now(),
                    }
                )
                unit["retry_of"] = str(job_id)
                unit["job_id"] = None
            else:
                summary["active"] += 1
                active_jobs += 1
                continue
        summary["missing"] += 1
        if submitted >= args.max_new_jobs:
            continue
        if active_jobs >= args.max_active_jobs:
            continue
        try:
            new_job_id = _submit(unit, args)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(json.dumps({"status": "scheduler_capacity_blocked", "error": str(exc)}))
            break
        unit["job_id"] = new_job_id
        unit["state"] = "SUBMITTED"
        unit["submitted_utc"] = utc_now()
        submitted += 1
        active_jobs += 1
        summary["submitted"] += 1
    payload["updated_utc"] = utc_now()
    payload["summary"] = summary
    write_json_atomic(args.state_manifest, payload)
    print(
        json.dumps(
            {"state_manifest": str(args.state_manifest), **summary},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
