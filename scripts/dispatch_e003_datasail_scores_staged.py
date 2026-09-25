#!/usr/bin/env python3
"""Resume the official-DataSAIL E003 fan-out under a submit-job limit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from e003_staged_fanout import (
    dependency_ids,
    json_mapping,
    parse_job_mapping,
    submit_one_missing,
    utc_now,
    validate_request_entry,
    write_json_atomic,
)

from ppi_shift.artifacts import sha256_path

SCHEMA = "ppi_shift.e003_datasail_staged_dispatch.v1"
MODELS = ("dscript_esm2", "tuna_corrected")
FROZEN_MODEL = "symmetric_frozen_esm2_pair_mlp"
SEEDS = (11, 22, 33)
ENVIRONMENTS = ("uniform_partner", "esm_partner_matched")

BACKEND_BINARIES = {
    "dscript_esm2": (
        "/path/to/ExternalPPIArtifacts/external_models/D007/runtimes/"
        "dscript_esm2/bin/ppi-sota-backend"
    ),
    "tuna_corrected": (
        "/path/to/ExternalPPIArtifacts/external_models/D007/runtimes/tuna/bin/ppi-sota-backend"
    ),
}


def _requests(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    backend = payload.get("backend_requests")
    frozen = payload.get("frozen_requests")
    if not isinstance(backend, list) or not isinstance(frozen, list):
        raise SystemExit("DataSAIL matrix lacks backend_requests/frozen_requests")
    if len(backend) != 12 or len(frozen) != 6:
        raise SystemExit("Expected 12 backend and 6 Frozen DataSAIL requests")
    expected_backend = [
        (model, seed, environment)
        for seed in SEEDS
        for model in MODELS
        for environment in ENVIRONMENTS
    ]
    expected_frozen = [
        (FROZEN_MODEL, seed, environment)
        for seed in SEEDS
        for environment in ENVIRONMENTS
    ]
    for values, expected, label in (
        (backend, expected_backend, "backend"),
        (frozen, expected_frozen, "frozen"),
    ):
        observed = []
        for entry in values:
            if not isinstance(entry, dict):
                raise SystemExit(f"Invalid DataSAIL {label} request entry")
            validate_request_entry(entry)
            observed.append(
                (
                    str(entry.get("model")),
                    int(entry.get("seed")),
                    str(entry.get("train_environment")),
                )
            )
        if observed != expected:
            raise SystemExit(f"DataSAIL {label} request roster/order changed")
    return [dict(entry) for entry in backend], [dict(entry) for entry in frozen]


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
        "job_id": job_id,
        "dependency_job_ids": dependencies,
        "scheduler_dependency_job_ids": dependency_ids(dependencies),
        "expected_output": expected_output,
        "request_path": str(request.get("request", "")) if request else "",
        "request_sha256": str(request.get("request_sha256", "")) if request else "",
        "model": request.get("model") if request else None,
        "seed": request.get("seed") if request else None,
        "train_environment": request.get("train_environment") if request else None,
        "array_task_count": array_task_count,
        "retry_of": None,
    }


def _scheduler_dependencies(args: argparse.Namespace) -> list[str]:
    """Return live scheduler dependencies without erasing preparation provenance."""

    if args.no_prepare_dependency:
        return []
    return [args.prepare_job]


def _immutable(
    args: argparse.Namespace, *, scheduler_dependencies: list[str]
) -> dict[str, Any]:
    return {
        "repo_root": str(args.repo_root),
        "prepare_job": args.prepare_job,
        "scheduler_dependencies": scheduler_dependencies,
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
    }


def _validate_inputs(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "ppi_shift.e003_datasail_request_matrix.v1":
        raise SystemExit("Unexpected DataSAIL request matrix schema")
    if payload.get("panel_manifest_sha256") != sha256_path(args.panel_manifest):
        raise SystemExit("DataSAIL matrix panel hash differs from supplied panel")
    if payload.get("config_sha256") != sha256_path(args.config):
        raise SystemExit("DataSAIL matrix config hash differs from supplied config")
    frozen_index = Path(str(payload.get("frozen_index", "")))
    backend_index = Path(str(payload.get("backend_index", "")))
    if not frozen_index.is_file() or not backend_index.is_file():
        raise SystemExit("DataSAIL request indexes are missing")


def _validate_existing_state(
    state: dict[str, Any],
    immutable: dict[str, Any],
    backend: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> None:
    if state.get("schema_version") != SCHEMA:
        raise SystemExit("Unexpected staged DataSAIL state schema")
    if state.get("immutable_inputs") != immutable:
        raise SystemExit("Staged DataSAIL immutable inputs changed")
    for phase, expected in (("backend", backend), ("frozen", frozen)):
        entries = state.get(phase)
        if not isinstance(entries, list) or len(entries) != len(expected):
            raise SystemExit(f"Staged DataSAIL {phase} state is incomplete")
        for entry, request in zip(entries, expected, strict=True):
            if entry is not None and entry.get("request") != request:
                raise SystemExit(f"Staged DataSAIL {phase} request changed")


def _build_launch_manifest(
    *,
    state: dict[str, Any],
    args: argparse.Namespace,
    immutable: dict[str, Any],
    backend: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
    compute_ids: list[str],
    scheduler_dependencies: list[str],
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for phase, requests, action in (
        ("backend", backend, "datasail_backend_training_unit"),
        ("frozen", frozen, "datasail_frozen_training_unit"),
    ):
        for entry, request in zip(state[phase], requests, strict=True):
            assert entry is not None
            jobs.append(
                _record(
                    unit_id=(
                        f"datasail-{phase}|{request['model']}|{request['seed']}|"
                        f"{request['train_environment']}"
                    ),
                    action=action,
                    job_id=str(entry["job_id"]),
                    dependencies=scheduler_dependencies,
                    expected_output=str(request["output_dir"]),
                    request=request,
                    array_task_count=entry.get("array_task_count"),
                )
            )
    post = state["datasail_postprocess"]
    assert post is not None
    jobs.append(
        _record(
            unit_id="datasail-postprocess|36-cells",
            action="datasail_postprocess",
            job_id=str(post["job_id"]),
            dependencies=compute_ids,
            expected_output=str(args.artifact_root / "results" / "datasail"),
        )
    )
    return {
        "schema_version": "ppi_shift.e003_datasail_launch_manifest.v1",
        "experiment_id": "E003",
        "created_utc": state["created_utc"],
        "repo_root": str(args.repo_root),
        "prepare_job": args.prepare_job,
        "scheduler_dependencies": scheduler_dependencies,
        "matrix_manifest": immutable["matrix_manifest"],
        "panel_manifest": immutable["panel_manifest"],
        "protocol_freeze": immutable["protocol_freeze"],
        "config": immutable["config"],
        "artifact_root": str(args.artifact_root),
        "training_units": 18,
        "score_cells": 36,
        "compute_job_ids": compute_ids,
        "datasail_postprocess_job_id": str(post["job_id"]),
        "jobs": jobs,
        "status": "complete_datasail_dag_submitted",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "matrix-manifest",
        "panel-manifest",
        "protocol-freeze",
        "config",
        "artifact-root",
        "launch-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prepare-job", required=True)
    parser.add_argument(
        "--no-prepare-dependency",
        action="store_true",
        help=(
            "Keep the completed preparation job as provenance but do not submit "
            "new units with an obsolete afterok dependency."
        ),
    )
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--max-new-jobs", type=int, default=1)
    parser.add_argument("--pre-job", action="append", default=[], metavar="NAME=JOB_ID")
    args = parser.parse_args()
    if not args.prepare_job.isdigit():
        raise SystemExit("--prepare-job must be a numeric Slurm job ID")
    if args.max_new_jobs < 1:
        raise SystemExit("--max-new-jobs must be positive")
    state_manifest = args.state_manifest or (
        args.artifact_root / "e003_datasail_staged_dispatch.json"
    )
    if args.launch_manifest.exists():
        raise SystemExit(f"Refusing duplicate DataSAIL launch manifest: {args.launch_manifest}")
    for path in (args.matrix_manifest, args.panel_manifest, args.protocol_freeze, args.config):
        if not path.is_file():
            raise SystemExit(f"Required DataSAIL input is missing: {path}")
    payload = json_mapping(args.matrix_manifest)
    _validate_inputs(args, payload)
    backend, frozen = _requests(payload)
    scheduler_dependencies = _scheduler_dependencies(args)
    immutable = _immutable(args, scheduler_dependencies=scheduler_dependencies)
    if state_manifest.exists():
        state = json_mapping(state_manifest)
        _validate_existing_state(state, immutable, backend, frozen)
    else:
        now = utc_now()
        state = {
            "schema_version": SCHEMA,
            "experiment_id": "E003",
            "created_utc": now,
            "updated_utc": now,
            "immutable_inputs": immutable,
            "pre_jobs": parse_job_mapping(args.pre_job),
            "backend": [None] * len(backend),
            "frozen": [None] * len(frozen),
            "datasail_postprocess": None,
            "status": "partial_datasail_submission",
            "last_scheduler_error": None,
        }
        write_json_atomic(state_manifest, state)
    if state.get("pre_jobs") != parse_job_mapping(args.pre_job):
        raise SystemExit("Staged DataSAIL pre-job mapping changed")

    common = {
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
        "PPI_SHIFT_REPO": str(args.repo_root),
    }
    phases = (
        (
            "backend",
            backend,
            args.repo_root / "scripts" / "launch_e003_backend_cell.slurm",
            "datasail_backend_training_unit",
            False,
            lambda request: {
                **common,
                "REQUEST": str(request["request"]),
                "BACKEND_BIN": BACKEND_BINARIES[str(request["model"])],
            },
        ),
        (
            "frozen",
            frozen,
            args.repo_root / "scripts" / "launch_e003_frozen_cell.slurm",
            "datasail_frozen_training_unit",
            True,
            lambda request: {**common, "REQUEST_INDEX": str(payload["frozen_index"])},
        ),
    )
    submissions = 0
    for phase, requests, script, action, array, environment_for_request in phases:
        while submissions < args.max_new_jobs:
            try:
                entry = submit_one_missing(
                    state=state,
                    phase=phase,
                    requests=requests,
                    script=script,
                    dependencies=scheduler_dependencies,
                    action=action,
                    array=array,
                    environment_for_request=environment_for_request,
                )
            except subprocess.CalledProcessError as error:
                state["last_scheduler_error"] = {
                    "phase": phase,
                    "observed_utc": utc_now(),
                    "returncode": error.returncode,
                    "stdout": (error.stdout or "").strip(),
                    "stderr": (error.stderr or "").strip(),
                }
                state["updated_utc"] = utc_now()
                write_json_atomic(state_manifest, state)
                print(
                    json.dumps(
                        {
                            "status": "scheduler_capacity_blocked",
                            "phase": phase,
                            "state_manifest": str(state_manifest),
                            "stderr": (error.stderr or "").strip(),
                        },
                        sort_keys=True,
                    )
                )
                return 0
            if entry is None:
                break
            submissions += 1
            state["last_scheduler_error"] = None
            state["updated_utc"] = utc_now()
            write_json_atomic(state_manifest, state)
            if submissions >= args.max_new_jobs:
                break
        if submissions >= args.max_new_jobs:
            break

    compute_ids = [
        str(entry["job_id"])
        for phase in ("backend", "frozen")
        for entry in state[phase]
        if entry is not None
    ]
    if (
        len(compute_ids) == len(backend) + len(frozen)
        and state["datasail_postprocess"] is None
        and submissions < args.max_new_jobs
    ):
        from e003_staged_fanout import submit_job

        try:
            post_job = submit_job(
                args.repo_root / "scripts" / "launch_e003_datasail_postprocess.slurm",
                {
                    **common,
                    "DATASAIL_REQUEST_MANIFEST": str(args.matrix_manifest),
                    "DATASAIL_PANEL_MANIFEST": str(args.panel_manifest),
                    "CONFIG": str(args.config),
                },
                dependency_ids(compute_ids),
            )
        except subprocess.CalledProcessError as error:
            state["last_scheduler_error"] = {
                "phase": "postprocess",
                "observed_utc": utc_now(),
                "returncode": error.returncode,
                "stdout": (error.stdout or "").strip(),
                "stderr": (error.stderr or "").strip(),
            }
            state["updated_utc"] = utc_now()
            write_json_atomic(state_manifest, state)
            print(json.dumps({"status": "scheduler_capacity_blocked", "phase": "postprocess"}))
            return 0
        state["datasail_postprocess"] = {
            "job_id": post_job,
            "submitted_utc": utc_now(),
        }
        state["status"] = "complete_datasail_dag_submitted"
        state["updated_utc"] = utc_now()
        write_json_atomic(state_manifest, state)
        submissions += 1

    post = state.get("datasail_postprocess")
    if post is not None and len(compute_ids) == len(backend) + len(frozen):
        launch = _build_launch_manifest(
            state=state,
            args=args,
            immutable=immutable,
            backend=backend,
            frozen=frozen,
            compute_ids=compute_ids,
            scheduler_dependencies=scheduler_dependencies,
        )
        write_json_atomic(args.launch_manifest, launch)
        print(
            json.dumps(
                {"status": launch["status"], "launch_manifest": str(args.launch_manifest)}
            )
        )
    else:
        missing = {
            phase: sum(entry is None for entry in state[phase]) for phase in ("backend", "frozen")
        }
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "new_jobs_submitted": submissions,
                    "missing_compute_units": missing,
                    "state_manifest": str(state_manifest),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
