#!/usr/bin/env python3
"""Submit the official-DataSAIL preparation and dependent score DAG."""

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


def _resolve_prepared_inputs(
    pre_jobs: dict[str, str], panel_manifest: Path, matrix_manifest: Path
) -> tuple[str | None, bool]:
    """Reuse explicitly completed pre-freeze inputs instead of rebuilding them."""

    input_job = pre_jobs.get("datasail_inputs")
    existing_inputs = panel_manifest.exists() or matrix_manifest.exists()
    if input_job is not None:
        if not panel_manifest.is_file() or not matrix_manifest.is_file():
            raise SystemExit(
                "--pre-job datasail_inputs requires both the DataSAIL panel and matrix manifests"
            )
        return input_job, True
    if existing_inputs:
        raise SystemExit(
            "DataSAIL inputs already exist; pass --pre-job datasail_inputs=JOB_ID "
            "to reuse the completed pre-freeze preparation"
        )
    return None, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--pre-job", action="append", default=[], metavar="NAME=JOB_ID")
    args = parser.parse_args()
    if args.submission_manifest.exists():
        raise SystemExit(f"Refusing duplicate DataSAIL submission: {args.submission_manifest}")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"Protocol freeze is missing: {args.protocol_freeze}")
    if not args.panel_manifest.is_file():
        raise SystemExit(f"Panel manifest is missing: {args.panel_manifest}")
    pre_jobs: dict[str, str] = {}
    for value in args.pre_job:
        name, separator, job_id = value.partition("=")
        if not separator or not name or not job_id.isdigit():
            raise SystemExit(f"Invalid --pre-job value: {value!r}")
        pre_jobs[name] = job_id
    matrix_manifest = args.artifact_root / "datasail_matrix" / "datasail_request_manifest.json"
    datasail_panel_manifest = args.artifact_root / "e003_datasail_panel_manifest.yaml"
    prepared_job, inputs_reused = _resolve_prepared_inputs(
        pre_jobs, datasail_panel_manifest, matrix_manifest
    )
    panel_dependencies = [job_id for name, job_id in pre_jobs.items() if name == "panel"]
    common = {
        "PPI_SHIFT_REPO": str(args.repo_root),
        "PANEL_MANIFEST": str(args.panel_manifest),
        "PROTOCOL_FREEZE": str(args.protocol_freeze),
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
    }
    if prepared_job is None:
        prep_job = _submit(
            args.repo_root / "scripts" / "launch_e003_datasail_prepare.slurm",
            common,
            panel_dependencies,
        )
    else:
        prep_job = prepared_job
    launch_manifest = args.artifact_root / "e003_datasail_launch_manifest.json"
    dispatch_job = _submit(
        args.repo_root / "scripts" / "launch_e003_datasail_dispatch.slurm",
        {
            "PPI_SHIFT_REPO": str(args.repo_root),
            "DATASAIL_PREP_JOB": prep_job,
            "DATASAIL_MATRIX_MANIFEST": str(matrix_manifest),
            "DATASAIL_PANEL_MANIFEST": str(datasail_panel_manifest),
            "PROTOCOL_FREEZE": str(args.protocol_freeze),
            "CONFIG": str(args.config),
            "E003_ARTIFACT_ROOT": str(args.artifact_root),
            "DATASAIL_LAUNCH_MANIFEST": str(launch_manifest),
        },
        [prep_job],
    )
    output: dict[str, Any] = {
        "schema_version": "ppi_shift.e003_datasail_submission_manifest.v1",
        "experiment_id": "E003",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "repo_root": str(args.repo_root),
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "protocol_freeze": {
            "path": str(args.protocol_freeze),
            "sha256": sha256_path(args.protocol_freeze),
        },
        "artifact_root": str(args.artifact_root),
        "pre_jobs": pre_jobs,
        "jobs": {
            "datasail_prepare": prep_job,
            "datasail_dispatch": dispatch_job,
        },
        "datasail_inputs_reused": inputs_reused,
        "future_artifacts": {
            "panel_manifest": str(datasail_panel_manifest),
            "request_manifest": str(matrix_manifest),
            "launch_manifest": str(launch_manifest),
        },
        "status": (
            "datasail_input_reuse_and_dispatch_submitted"
            if inputs_reused
            else "datasail_prepare_and_dispatch_submitted"
        ),
    }
    args.submission_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.submission_manifest.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
