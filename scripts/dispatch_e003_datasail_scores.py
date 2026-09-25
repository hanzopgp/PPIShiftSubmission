#!/usr/bin/env python3
"""Submit and record the official-DataSAIL E003 score fan-out."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path

UTC = getattr(timezone, "utc")

BACKEND_BINARIES = {
    "dscript_esm2": (
        "/path/to/ExternalPPIArtifacts/external_models/D007/runtimes/"
        "dscript_esm2/bin/ppi-sota-backend"
    ),
    "tuna_corrected": (
        "/path/to/ExternalPPIArtifacts/external_models/D007/runtimes/tuna/bin/ppi-sota-backend"
    ),
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _submit(script: Path, environment: dict[str, str], dependencies: list[str]) -> str:
    command = ["sbatch", "--parsable", "--qos=normal"]
    if dependencies:
        command.append(f"--dependency=afterok:{':'.join(dependencies)}")
    exports = ["ALL"] + [f"{key}={value}" for key, value in sorted(environment.items())]
    command.extend([f"--export={','.join(exports)}", str(script)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(r"\b(\d+)\b", completed.stdout.strip())
    if match is None:
        raise RuntimeError(f"sbatch returned no job ID for {script}: {completed.stdout!r}")
    return match.group(1)


def _state(job_id: str) -> str:
    live = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if live:
        return live.splitlines()[0].strip()
    completed = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    values = [line.strip().split("|", 1)[0] for line in completed.splitlines() if line.strip()]
    return values[0] if values else "UNKNOWN"


def _dependency_ids(job_ids: list[str]) -> list[str]:
    result: list[str] = []
    for job_id in job_ids:
        state = _state(job_id)
        if state == "COMPLETED":
            continue
        if state == "UNKNOWN":
            raise RuntimeError(f"Cannot establish state for dependency job {job_id}")
        result.append(job_id)
    return result


def _record(
    *,
    unit_id: str,
    action: str,
    job_id: str,
    dependencies: list[str],
    expected_output: str,
    request: dict[str, Any] | None = None,
    array_task_count: int | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": "E003",
        "unit_id": unit_id,
        "action_type": action,
        "slurm_job_id": job_id,
        "dependency_job_ids": dependencies,
        "scheduler_dependency_job_ids": _dependency_ids(dependencies),
        "scheduler_state_at_submission": _state(job_id),
        "expected_output": expected_output,
        "request_path": str(request.get("request", "")) if request else "",
        "request_sha256": str(request.get("request_sha256", "")) if request else "",
        "model": request.get("model") if request else None,
        "seed": request.get("seed") if request else None,
        "train_environment": request.get("train_environment") if request else None,
        "array_task_count": array_task_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prepare-job", required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.launch_manifest.exists():
        raise SystemExit(f"Refusing duplicate DataSAIL launch manifest: {args.launch_manifest}")
    if not args.matrix_manifest.is_file():
        raise SystemExit(f"DataSAIL matrix manifest is missing: {args.matrix_manifest}")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"Protocol freeze is missing: {args.protocol_freeze}")
    payload = _json(args.matrix_manifest)
    if payload.get("schema_version") != "ppi_shift.e003_datasail_request_matrix.v1":
        raise SystemExit("Unexpected DataSAIL request matrix schema")
    backend = payload.get("backend_requests", [])
    frozen = payload.get("frozen_requests", [])
    if len(backend) != 12 or len(frozen) != 6:
        raise SystemExit("Expected 12 backend and 6 Frozen DataSAIL requests")
    dependencies = _dependency_ids([args.prepare_job])
    common = {"E003_ARTIFACT_ROOT": str(args.artifact_root)}
    jobs: list[dict[str, Any]] = []
    compute_ids: list[str] = []
    for request in backend:
        job_id = _submit(
            args.repo_root / "scripts" / "launch_e003_backend_cell.slurm",
            {
                **common,
                "REQUEST": str(request["request"]),
                "BACKEND_BIN": BACKEND_BINARIES[str(request["model"])],
            },
            dependencies,
        )
        compute_ids.append(job_id)
        jobs.append(
            _record(
                unit_id=(
                    f"datasail-backend|{request['model']}|{request['seed']}|"
                    f"{request['train_environment']}"
                ),
                action="datasail_backend_training_unit",
                job_id=job_id,
                dependencies=[args.prepare_job],
                expected_output=str(request["output_dir"]),
                request=request,
            )
        )
    frozen_job = _submit(
        args.repo_root / "scripts" / "launch_e003_frozen_cell.slurm",
        {
            **common,
            "REQUEST_INDEX": str(payload["frozen_index"]),
        },
        dependencies,
    )
    compute_ids.append(frozen_job)
    jobs.append(
        _record(
            unit_id="datasail-frozen|six-units",
            action="datasail_frozen_training_array",
            job_id=frozen_job,
            dependencies=[args.prepare_job],
            expected_output=str(args.artifact_root / "datasail_matrix" / "frozen"),
            array_task_count=6,
        )
    )
    post_job = _submit(
        args.repo_root / "scripts" / "launch_e003_datasail_postprocess.slurm",
        {
            **common,
            "PPI_SHIFT_REPO": str(args.repo_root),
            "DATASAIL_REQUEST_MANIFEST": str(args.matrix_manifest),
            "DATASAIL_PANEL_MANIFEST": str(args.panel_manifest),
            "CONFIG": str(args.config),
        },
        _dependency_ids(compute_ids),
    )
    jobs.append(
        _record(
            unit_id="datasail-postprocess|36-cells",
            action="datasail_postprocess",
            job_id=post_job,
            dependencies=compute_ids,
            expected_output=str(args.artifact_root / "results" / "datasail"),
        )
    )
    output = {
        "schema_version": "ppi_shift.e003_datasail_launch_manifest.v1",
        "experiment_id": "E003",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "repo_root": str(args.repo_root),
        "prepare_job": args.prepare_job,
        "matrix_manifest": {
            "path": str(args.matrix_manifest),
            "sha256": sha256_path(args.matrix_manifest),
        },
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "protocol_freeze": {
            "path": str(args.protocol_freeze),
            "sha256": sha256_path(args.protocol_freeze),
        },
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "artifact_root": str(args.artifact_root),
        "training_units": 18,
        "score_cells": 36,
        "compute_job_ids": compute_ids,
        "datasail_postprocess_job_id": post_job,
        "jobs": jobs,
        "status": "complete_datasail_dag_submitted",
    }
    args.launch_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.launch_manifest.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
