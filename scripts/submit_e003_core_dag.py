#!/usr/bin/env python3
"""Submit the corrected E003 core score DAG and record every scheduler edge."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS

UTC = getattr(timezone, "utc")

BACKEND_BINARIES = {
    "dscript_esm2": (
        "/path/to/ExternalPPIArtifacts/external_models/D007/runtimes/"
        "dscript_esm2/bin/ppi-sota-backend"
    ),
    "tuna_corrected": (
        "/path/to/ExternalPPIArtifacts/external_models/D007/runtimes/"
        "tuna/bin/ppi-sota-backend"
    ),
}
CORE_MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
SEEDS = (11, 22, 33)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _submit(
    script: Path,
    environment: dict[str, str],
    dependencies: list[str],
    *,
    array: str | None = None,
) -> str:
    command = ["sbatch", "--parsable", "--qos=normal"]
    if array:
        command.append(f"--array={array}")
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
    """Keep live jobs in Slurm dependencies and audit already-complete jobs."""

    result: list[str] = []
    for job_id in job_ids:
        state = _state(job_id)
        if state == "COMPLETED":
            continue
        if state == "UNKNOWN":
            raise RuntimeError(f"Cannot establish state for dependency job {job_id}")
        result.append(job_id)
    return result


def _manifest_requests(path: Path, schema: str, expected: int) -> list[dict[str, Any]]:
    payload = _json(path)
    if payload.get("schema_version") != schema:
        raise SystemExit(f"Unexpected request manifest schema: {path}")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != expected:
        raise SystemExit(f"Unexpected request count in {path}: {len(requests or [])}")
    return [dict(item) for item in requests]


def _assert_core_request_roster(
    backend: list[dict[str, Any]],
    python: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
    reuse: list[dict[str, Any]],
) -> None:
    def keys(requests: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
        return {
            (str(item["model"]), int(item["seed"]), str(item["train_environment"]))
            for item in requests
        }

    expected_backend = {
        (model, seed, environment)
        for model in ("dscript_esm2", "tuna_corrected")
        for seed in SEEDS
        for environment in (
            "bernett2026_ilp_exact",
            "tppni2025_scm_cl3_exact_balanced",
        )
    }
    expected_python = {
        ("rapppid", seed, environment)
        for seed in SEEDS
        for environment in E003_ENVIRONMENTS
    }
    expected_frozen = {
        ("symmetric_frozen_esm2_pair_mlp", seed, environment)
        for seed in SEEDS
        for environment in (
            "bernett2026_ilp_exact",
            "tppni2025_scm_cl3_exact_balanced",
        )
    }
    expected_reuse = {
        (model, seed, environment)
        for model in CORE_MODELS[:3]
        for seed in SEEDS
        for environment in E003_ENVIRONMENTS[:4]
    }
    observed = {
        "backend": keys(backend),
        "python": keys(python),
        "frozen": keys(frozen),
        "reuse": keys(reuse),
    }
    expected = {
        "backend": expected_backend,
        "python": expected_python,
        "frozen": expected_frozen,
        "reuse": expected_reuse,
    }
    for name in expected:
        if observed[name] != expected[name]:
            raise SystemExit(
                f"E003 Core-4 request roster mismatch in {name}: "
                f"observed={len(observed[name])} expected={len(expected[name])}"
            )


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
        "scheduler_dependency_job_ids": _dependency_ids(dependencies),
        "scheduler_state_at_submission": _state(job_id),
        "expected_output": expected_output,
        "request_path": str(request["request"]) if request else "",
        "request_sha256": str(request.get("request_sha256", "")) if request else "",
        "model": request.get("model") if request else None,
        "seed": request.get("seed") if request else None,
        "train_environment": request.get("train_environment") if request else None,
        "array_task_count": array_task_count,
        "retry_of": None,
        "replacement_of": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--e001-registry", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--pre-job", action="append", default=[], metavar="NAME=JOB_ID")
    parser.add_argument(
        "--reuse-submitted-backend-job",
        action="append",
        default=[],
        metavar="JOB_ID",
        help=(
            "Resume after a partial submission by reusing one already-submitted backend "
            "job ID per backend request, in manifest order."
        ),
    )
    args = parser.parse_args()
    if args.launch_manifest.exists():
        raise SystemExit(f"Refusing duplicate E003 core submission: {args.launch_manifest}")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"Protocol freeze is missing: {args.protocol_freeze}")
    pre_jobs: dict[str, str] = {}
    for value in args.pre_job:
        name, separator, job_id = value.partition("=")
        if not separator or not name or not job_id.isdigit():
            raise SystemExit(f"Invalid --pre-job value: {value!r}")
        pre_jobs[name] = job_id
    if not pre_jobs:
        raise SystemExit("At least one completed-or-pending panel/cache --pre-job is required")
    panel_sha256 = sha256_path(args.panel_manifest)
    config_sha256 = sha256_path(args.config)
    freeze_sha256 = sha256_path(args.protocol_freeze)
    backend = _manifest_requests(
        args.backend_manifest, "ppi_shift.e003_backend_request_matrix.v1", 12
    )
    python = _manifest_requests(
        args.python_manifest, "ppi_shift.e003_python_request_matrix.v1", 18
    )
    frozen = _manifest_requests(
        args.frozen_manifest, "ppi_shift.e003_frozen_esm_request_matrix.v1", 6
    )
    reuse = _manifest_requests(
        args.reuse_manifest, "ppi_shift.e003_reuse_request_matrix.v1", 36
    )
    _assert_core_request_roster(backend, python, frozen, reuse)
    if args.reuse_submitted_backend_job and len(args.reuse_submitted_backend_job) != len(backend):
        raise SystemExit(
            "--reuse-submitted-backend-job must contain exactly one job ID per backend request"
        )
    if any(not value.isdigit() for value in args.reuse_submitted_backend_job):
        raise SystemExit("--reuse-submitted-backend-job values must be numeric Slurm job IDs")
    artifact_root = args.artifact_root
    (artifact_root / "logs").mkdir(parents=True, exist_ok=True)
    pre_dependency_ids = list(pre_jobs.values())
    scheduler_pre_dependencies = _dependency_ids(pre_dependency_ids)
    common = {
        "E003_ARTIFACT_ROOT": str(artifact_root),
        "PPI_SHIFT_REPO": str(args.repo_root),
    }
    jobs: list[dict[str, Any]] = []
    compute_ids: list[str] = []

    for index, request in enumerate(backend):
        model = str(request["model"])
        if args.reuse_submitted_backend_job:
            job_id = args.reuse_submitted_backend_job[index]
        else:
            job_id = _submit(
                args.repo_root / "scripts" / "launch_e003_backend_cell.slurm",
                {
                    **common,
                    "REQUEST": str(request["request"]),
                    "BACKEND_BIN": BACKEND_BINARIES[model],
                },
                scheduler_pre_dependencies,
            )
        compute_ids.append(job_id)
        jobs.append(
            _record(
                unit_id=f"backend|{model}|{request['seed']}|{request['train_environment']}",
                action="backend_training_unit",
                job_id=job_id,
                dependencies=pre_dependency_ids,
                expected_output=str(request["output_dir"]),
                request=request,
            )
        )

    python_job = _submit(
        args.repo_root / "scripts" / "launch_e003_python_cell.slurm",
        common,
        scheduler_pre_dependencies,
        array="0-17%8",
    )
    compute_ids.append(python_job)
    jobs.append(
        _record(
            unit_id="python|rapppid|18-units",
            action="python_training_array",
            job_id=python_job,
            dependencies=pre_dependency_ids,
            expected_output=str(artifact_root / "python_matrix"),
            array_task_count=18,
        )
    )

    frozen_job = _submit(
        args.repo_root / "scripts" / "launch_e003_frozen_cell.slurm",
        common,
        scheduler_pre_dependencies,
        array="0-5%4",
    )
    compute_ids.append(frozen_job)
    jobs.append(
        _record(
            unit_id="frozen|six-new-training-units",
            action="frozen_training_array",
            job_id=frozen_job,
            dependencies=pre_dependency_ids,
            expected_output=str(artifact_root / "frozen_matrix"),
            array_task_count=6,
        )
    )

    for request in reuse:
        job_id = _submit(
            args.repo_root / "scripts" / "launch_e003_reuse_cell.slurm",
            {**common, "REQUEST": str(request["request"])},
            scheduler_pre_dependencies,
        )
        compute_ids.append(job_id)
        jobs.append(
            _record(
                unit_id=(
                    f"reuse|{request['model']}|{request['seed']}|"
                    f"{request['train_environment']}"
                ),
                action="exact_e001_checkpoint_reuse",
                job_id=job_id,
                dependencies=pre_dependency_ids,
                expected_output=str(request["output_dir"]),
                request=request,
            )
        )

    post_dependencies = _dependency_ids(compute_ids)
    post_job = _submit(
        args.repo_root / "scripts" / "launch_e003_core_postprocess.slurm",
        {
            **common,
            "PPI_SHIFT_REPO": str(args.repo_root),
            "E001_REGISTRY": str(args.e001_registry),
            "BACKEND_MANIFEST": str(args.backend_manifest),
            "PYTHON_MANIFEST": str(args.python_manifest),
            "FROZEN_MANIFEST": str(args.frozen_manifest),
            "REUSE_MANIFEST": str(args.reuse_manifest),
            "PANEL_MANIFEST": str(args.panel_manifest),
            "CONFIG": str(args.config),
            "PROTOCOL_FREEZE": str(args.protocol_freeze),
        },
        post_dependencies,
    )
    jobs.append(
        _record(
            unit_id="core-postprocess|432-cells",
            action="core_postprocess",
            job_id=post_job,
            dependencies=compute_ids,
            expected_output=str(artifact_root / "results" / "core"),
        )
    )
    launch = {
        "schema_version": "ppi_shift.e003_core_launch_manifest.v1",
        "experiment_id": "E003",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "repo_root": str(args.repo_root),
        "panel_manifest": {"path": str(args.panel_manifest), "sha256": panel_sha256},
        "config": {"path": str(args.config), "sha256": config_sha256},
        "protocol_freeze": {"path": str(args.protocol_freeze), "sha256": freeze_sha256},
        "e001_registry": {
            "path": str(args.e001_registry),
            "sha256": sha256_path(args.e001_registry),
        },
        "artifact_root": str(artifact_root),
        "pre_jobs": pre_jobs,
        "pre_scheduler_dependencies": scheduler_pre_dependencies,
        "training_units": 72,
        "core_score_cells": 432,
        "jobs": jobs,
        "compute_job_ids": compute_ids,
        "core_postprocess_job_id": post_job,
        "status": "complete_core_dag_submitted",
    }
    args.launch_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.launch_manifest.write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "launch_manifest": str(args.launch_manifest),
                "compute_jobs": len(compute_ids),
                "core_postprocess_job": post_job,
                "status": launch["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
