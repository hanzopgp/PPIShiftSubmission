#!/usr/bin/env python3
"""Resume the corrected E003 external fan-out under a submit-job limit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from e003_staged_fanout import (
    dependency_ids,
    json_mapping,
    submit_job,
    submit_one_missing,
    utc_now,
    validate_request_entry,
    write_json_atomic,
)

from ppi_shift.artifacts import sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS

SCHEMA = "ppi_shift.e003_external_staged_dispatch.v1"
MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
SEEDS = (11, 22, 33)


def _prepare_requests(args: argparse.Namespace, request_manifest: Path) -> None:
    if request_manifest.exists():
        return
    request_root = args.artifact_root / "external_requests"
    command = [
        "/path/to/miniconda3/envs/bioppi/bin/python3.10",
        str(args.repo_root / "scripts" / "prepare_e003_external_requests.py"),
        "--backend-manifest",
        str(args.backend_manifest),
        "--python-manifest",
        str(args.python_manifest),
        "--frozen-manifest",
        str(args.frozen_manifest),
        "--reuse-manifest",
        str(args.reuse_manifest),
        "--external-manifest",
        str(args.external_manifest),
        "--core-index",
        str(args.core_index),
        "--output-root",
        str(request_root),
        "--manifest",
        str(request_manifest),
    ]
    subprocess.run(command, check=True)


def _requests(args: argparse.Namespace, request_manifest: Path) -> list[dict[str, Any]]:
    payload = json_mapping(request_manifest)
    if payload.get("schema_version") != "ppi_shift.e003_external_request_matrix.v1":
        raise SystemExit("Unexpected external request matrix schema")
    if int(payload.get("units", -1)) != 72:
        raise SystemExit("Expected 72 E003 external score requests")
    if payload.get("core_index_sha256") != sha256_path(args.core_index):
        raise SystemExit("External request matrix core-index hash changed")
    external_spec = payload.get("external_manifest")
    if not isinstance(external_spec, dict) or external_spec.get("sha256") != sha256_path(
        args.external_manifest
    ):
        raise SystemExit("External request matrix panel hash changed")
    values = payload.get("requests")
    if not isinstance(values, list) or len(values) != 72:
        raise SystemExit("Expected 72 E003 external score requests")
    expected = sorted(
        (model, seed, environment)
        for model in MODELS
        for seed in SEEDS
        for environment in E003_ENVIRONMENTS
    )
    observed: list[tuple[str, int, str]] = []
    requests: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise SystemExit("Invalid external request entry")
        request = validate_request_entry(value)
        observed.append(
            (
                str(request.get("model")),
                int(request.get("seed")),
                str(request.get("train_environment")),
            )
        )
        requests.append(dict(request))
    if observed != expected:
        raise SystemExit("External request roster/order changed")
    return requests


def _record(
    *,
    unit_id: str,
    action: str,
    job_id: str,
    dependencies: list[str],
    expected_output: str,
    request: dict[str, Any] | None = None,
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
        "array_task_count": None,
        "retry_of": None,
    }


def _immutable(args: argparse.Namespace, request_manifest: Path) -> dict[str, Any]:
    return {
        "repo_root": str(args.repo_root),
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "external_manifest": {
            "path": str(args.external_manifest),
            "sha256": sha256_path(args.external_manifest),
        },
        "protocol_freeze": {
            "path": str(args.protocol_freeze),
            "sha256": sha256_path(args.protocol_freeze),
        },
        "core_index": {"path": str(args.core_index), "sha256": sha256_path(args.core_index)},
        "core_results": {
            "path": str(args.core_results),
            "sha256": sha256_path(args.core_results),
        },
        "backend_manifest": {
            "path": str(args.backend_manifest),
            "sha256": sha256_path(args.backend_manifest),
        },
        "python_manifest": {
            "path": str(args.python_manifest),
            "sha256": sha256_path(args.python_manifest),
        },
        "frozen_manifest": {
            "path": str(args.frozen_manifest),
            "sha256": sha256_path(args.frozen_manifest),
        },
        "reuse_manifest": {
            "path": str(args.reuse_manifest),
            "sha256": sha256_path(args.reuse_manifest),
        },
        "external_request_manifest": {
            "path": str(request_manifest),
            "sha256": sha256_path(request_manifest),
        },
        "external_embedding_dir": str(args.external_embedding_dir),
        "artifact_root": str(args.artifact_root),
    }


def _validate_existing_state(
    state: dict[str, Any], immutable: dict[str, Any], requests: list[dict[str, Any]]
) -> None:
    if state.get("schema_version") != SCHEMA:
        raise SystemExit("Unexpected staged external state schema")
    if state.get("immutable_inputs") != immutable:
        raise SystemExit("Staged external immutable inputs changed")
    entries = state.get("score")
    if not isinstance(entries, list) or len(entries) != len(requests):
        raise SystemExit("Staged external score state is incomplete")
    for entry, request in zip(entries, requests, strict=True):
        if entry is not None and entry.get("request") != request:
            raise SystemExit("Staged external score request changed")


def _build_launch_manifest(
    *,
    state: dict[str, Any],
    args: argparse.Namespace,
    immutable: dict[str, Any],
    request_manifest: Path,
    requests: list[dict[str, Any]],
    score_jobs: list[str],
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for entry, request in zip(state["score"], requests, strict=True):
        assert entry is not None
        jobs.append(
            _record(
                unit_id=(
                    f"external|{request['model']}|{request['seed']}|"
                    f"{request['train_environment']}"
                ),
                action="external_score_bundle",
                job_id=str(entry["job_id"]),
                dependencies=[],
                expected_output=str(request["output_dir"]),
                request=request,
            )
        )
    post = state["external_postprocess"]
    assert post is not None
    jobs.append(
        _record(
            unit_id="external-postprocess|72-bundles",
            action="external_postprocess",
            job_id=str(post["job_id"]),
            dependencies=score_jobs,
            expected_output=str(args.artifact_root / "results" / "external"),
        )
    )
    return {
        "schema_version": "ppi_shift.e003_external_launch_manifest.v1",
        "experiment_id": "E003",
        "created_utc": state["created_utc"],
        "repo_root": str(args.repo_root),
        "panel_manifest": immutable["panel_manifest"],
        "external_manifest": immutable["external_manifest"],
        "core_index": immutable["core_index"],
        "core_results": immutable["core_results"],
        "external_request_manifest": immutable["external_request_manifest"],
        "artifact_root": str(args.artifact_root),
        "training_units": 72,
        "surface_count": 8,
        "score_jobs": score_jobs,
        "external_postprocess_job_id": str(post["job_id"]),
        "jobs": jobs,
        "status": "complete_external_dag_submitted",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "panel-manifest",
        "external-manifest",
        "protocol-freeze",
        "core-index",
        "core-results",
        "backend-manifest",
        "python-manifest",
        "frozen-manifest",
        "reuse-manifest",
        "external-embedding-dir",
        "artifact-root",
        "launch-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--max-new-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.max_new_jobs < 1:
        raise SystemExit("--max-new-jobs must be positive")
    state_manifest = args.state_manifest or (
        args.artifact_root / "e003_external_staged_dispatch.json"
    )
    if args.launch_manifest.exists():
        raise SystemExit(f"Refusing duplicate external launch manifest: {args.launch_manifest}")
    for path in (
        args.panel_manifest,
        args.external_manifest,
        args.protocol_freeze,
        args.core_index,
        args.core_results,
        args.backend_manifest,
        args.python_manifest,
        args.frozen_manifest,
        args.reuse_manifest,
    ):
        if not path.is_file():
            raise SystemExit(f"Required external input is missing: {path}")
    if not args.external_embedding_dir.is_dir():
        raise SystemExit(
            f"External embedding directory is missing: {args.external_embedding_dir}"
        )
    request_manifest = (
        args.artifact_root / "external_requests" / "e003_external_request_manifest.json"
    )
    _prepare_requests(args, request_manifest)
    requests = _requests(args, request_manifest)
    immutable = _immutable(args, request_manifest)
    if state_manifest.exists():
        state = json_mapping(state_manifest)
        _validate_existing_state(state, immutable, requests)
    else:
        now = utc_now()
        state = {
            "schema_version": SCHEMA,
            "experiment_id": "E003",
            "created_utc": now,
            "updated_utc": now,
            "immutable_inputs": immutable,
            "score": [None] * len(requests),
            "external_postprocess": None,
            "status": "partial_external_submission",
            "last_scheduler_error": None,
        }
        write_json_atomic(state_manifest, state)

    common = {
        "PPI_SHIFT_REPO": str(args.repo_root),
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
    }
    submissions = 0
    while submissions < args.max_new_jobs:
        try:
            entry = submit_one_missing(
                state=state,
                phase="score",
                requests=requests,
                script=args.repo_root / "scripts" / "launch_e003_external_score.slurm",
                dependencies=[],
                action="external_score_bundle",
                array=False,
                environment_for_request=lambda request: {
                    **common,
                    "REQUEST": str(request["request"]),
                    "CORE_INDEX": str(args.core_index),
                    "EXTERNAL_EMBEDDING_DIR": str(args.external_embedding_dir),
                },
            )
        except subprocess.CalledProcessError as error:
            state["last_scheduler_error"] = {
                "phase": "score",
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
                        "phase": "score",
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

    score_jobs = [str(entry["job_id"]) for entry in state["score"] if entry is not None]
    if (
        len(score_jobs) == len(requests)
        and state["external_postprocess"] is None
        and submissions < args.max_new_jobs
    ):
        try:
            post_job = submit_job(
                args.repo_root / "scripts" / "launch_e003_external_postprocess.slurm",
                {
                    **common,
                    "EXTERNAL_REQUEST_MANIFEST": str(request_manifest),
                    "EXTERNAL_MANIFEST": str(args.external_manifest),
                    "CORE_RESULTS": str(args.core_results),
                },
                dependency_ids(score_jobs),
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
        state["external_postprocess"] = {
            "job_id": post_job,
            "submitted_utc": utc_now(),
        }
        state["status"] = "complete_external_dag_submitted"
        state["updated_utc"] = utc_now()
        write_json_atomic(state_manifest, state)
        submissions += 1

    post = state.get("external_postprocess")
    if post is not None and len(score_jobs) == len(requests):
        launch = _build_launch_manifest(
            state=state,
            args=args,
            immutable=immutable,
            request_manifest=request_manifest,
            requests=requests,
            score_jobs=score_jobs,
        )
        write_json_atomic(args.launch_manifest, launch)
        print(
            json.dumps(
                {"status": launch["status"], "launch_manifest": str(args.launch_manifest)}
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "new_jobs_submitted": submissions,
                    "missing_score_units": len(requests) - len(score_jobs),
                    "state_manifest": str(state_manifest),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
