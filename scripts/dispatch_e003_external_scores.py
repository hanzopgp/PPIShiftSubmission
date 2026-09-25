#!/usr/bin/env python3
"""Submit and record the corrected E003 external scoring fan-out."""

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


def _submit(
    script: Path,
    environment: dict[str, str],
    dependencies: list[str],
) -> str:
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
        "retry_of": None,
    }


def _requests(path: Path, schema: str, expected: int) -> list[dict[str, Any]]:
    payload = _json(path)
    if payload.get("schema_version") != schema:
        raise SystemExit(f"Unexpected request manifest schema: {path}")
    values = payload.get("requests")
    if not isinstance(values, list) or len(values) != expected:
        raise SystemExit(f"Unexpected request count in {path}: {len(values or [])}")
    return [dict(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--core-index", type=Path, required=True)
    parser.add_argument("--core-results", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--external-embedding-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.launch_manifest.exists():
        raise SystemExit(f"Refusing duplicate external launch manifest: {args.launch_manifest}")
    if not args.core_index.is_file() or not args.core_results.is_file():
        raise SystemExit("Completed core index and result manifest are required")
    if not args.external_manifest.is_file() or not args.panel_manifest.is_file():
        raise SystemExit("Corrected panel and external manifests are required")

    request_root = args.artifact_root / "external_requests"
    request_manifest = request_root / "e003_external_request_manifest.json"
    if not request_manifest.exists():
        command = [
            "python3",
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
        # The dispatcher runs inside the cluster environment; using the
        # configured interpreter keeps this preparation independent of PATH.
        command[0] = "/path/to/miniconda3/envs/bioppi/bin/python3.10"
        subprocess.run(command, check=True)
    requests = _requests(request_manifest, "ppi_shift.e003_external_request_matrix.v1", 72)

    common = {
        "PPI_SHIFT_REPO": str(args.repo_root),
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
    }
    score_jobs: list[str] = []
    jobs: list[dict[str, Any]] = []
    for request in requests:
        job_id = _submit(
            args.repo_root / "scripts" / "launch_e003_external_score.slurm",
            {
                **common,
                "REQUEST": str(request["request"]),
                "CORE_INDEX": str(args.core_index),
                "EXTERNAL_EMBEDDING_DIR": str(args.external_embedding_dir),
            },
            [],
        )
        score_jobs.append(job_id)
        jobs.append(
            _record(
                unit_id=(
                    f"external|{request['model']}|{request['seed']}|{request['train_environment']}"
                ),
                action="external_score_bundle",
                job_id=job_id,
                dependencies=[],
                expected_output=str(request["output_dir"]),
                request=request,
            )
        )

    post_job = _submit(
        args.repo_root / "scripts" / "launch_e003_external_postprocess.slurm",
        {
            **common,
            "EXTERNAL_REQUEST_MANIFEST": str(request_manifest),
            "EXTERNAL_MANIFEST": str(args.external_manifest),
            "CORE_RESULTS": str(args.core_results),
        },
        _dependency_ids(score_jobs),
    )
    jobs.append(
        _record(
            unit_id="external-postprocess|72-bundles",
            action="external_postprocess",
            job_id=post_job,
            dependencies=score_jobs,
            expected_output=str(args.artifact_root / "results" / "external"),
        )
    )
    payload = {
        "schema_version": "ppi_shift.e003_external_launch_manifest.v1",
        "experiment_id": "E003",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "repo_root": str(args.repo_root),
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "external_manifest": {
            "path": str(args.external_manifest),
            "sha256": sha256_path(args.external_manifest),
        },
        "core_index": {"path": str(args.core_index), "sha256": sha256_path(args.core_index)},
        "core_results": {
            "path": str(args.core_results),
            "sha256": sha256_path(args.core_results),
        },
        "external_request_manifest": {
            "path": str(request_manifest),
            "sha256": sha256_path(request_manifest),
        },
        "artifact_root": str(args.artifact_root),
        "training_units": 72,
        "surface_count": 8,
        "score_jobs": score_jobs,
        "external_postprocess_job_id": post_job,
        "jobs": jobs,
        "status": "complete_external_dag_submitted",
    }
    args.launch_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.launch_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "launch_manifest": str(args.launch_manifest),
                "score_jobs": len(score_jobs),
                "postprocess": post_job,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
