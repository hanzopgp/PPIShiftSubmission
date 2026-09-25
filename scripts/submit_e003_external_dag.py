#!/usr/bin/env python3
"""Submit the corrected Core-4 E003 external-cache and score-dispatch DAG."""

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
    values: list[str] = []
    for job_id in job_ids:
        state = _state(job_id)
        if state == "COMPLETED":
            continue
        if state == "UNKNOWN":
            raise RuntimeError(f"Cannot establish state for dependency job {job_id}")
        values.append(job_id)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--core-index", type=Path, required=True)
    parser.add_argument("--core-results", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--pre-job", action="append", default=[], metavar="NAME=JOB_ID")
    args = parser.parse_args()
    if args.submission_manifest.exists():
        raise SystemExit(f"Refusing duplicate external DAG submission: {args.submission_manifest}")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"Protocol freeze is missing: {args.protocol_freeze}")
    if not args.panel_manifest.is_file() or not args.external_manifest.is_file():
        raise SystemExit("Corrected panel and external manifests are required")
    pre_jobs: dict[str, str] = {}
    for value in args.pre_job:
        name, separator, job_id = value.partition("=")
        if not separator or not name or not job_id.isdigit():
            raise SystemExit(f"Invalid --pre-job value: {value!r}")
        pre_jobs[name] = job_id
    panel_dependencies = [
        job_id for name, job_id in pre_jobs.items() if name in {"panel", "external"}
    ]
    cache_dependencies = list(panel_dependencies)
    common = {
        "PPI_SHIFT_REPO": str(args.repo_root),
        "PANEL_MANIFEST": str(args.panel_manifest),
        "EXTERNAL_MANIFEST": str(args.external_manifest),
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
    }
    input_job = _submit(
        args.repo_root / "scripts" / "launch_e003_external_cache_inputs.slurm",
        common,
        _dependency_ids(cache_dependencies),
    )
    cache_input_manifest = (
        args.artifact_root / "external_cache_inputs" / "e003_external_cache_input_manifest.json"
    )
    cache_input_csv = args.artifact_root / "external_cache_inputs" / "sequences.csv"
    cache_input_fasta = args.artifact_root / "external_cache_inputs" / "sequences.fasta"
    cache_150m = _submit(
        args.repo_root / "scripts" / "launch_e003_external_150m_cache.slurm",
        {
            "E003_ARTIFACT_ROOT": str(args.artifact_root),
            "CACHE_INPUT_CSV": str(cache_input_csv),
            "CACHE_INPUT_FASTA": str(cache_input_fasta),
        },
        _dependency_ids([input_job]),
    )
    dispatch_job = _submit(
        args.repo_root / "scripts" / "launch_e003_external_dispatch.slurm",
        {
            "PPI_SHIFT_REPO": str(args.repo_root),
            "PANEL_MANIFEST": str(args.panel_manifest),
            "EXTERNAL_MANIFEST": str(args.external_manifest),
            "CORE_INDEX": str(args.core_index),
            "CORE_RESULTS": str(args.core_results),
            "BACKEND_MANIFEST": str(args.backend_manifest),
            "PYTHON_MANIFEST": str(args.python_manifest),
            "FROZEN_MANIFEST": str(args.frozen_manifest),
            "REUSE_MANIFEST": str(args.reuse_manifest),
            "EXTERNAL_EMBEDDING_DIR": str(
                args.artifact_root / "representations" / "external_esm2_t30_150m"
            ),
            "E003_ARTIFACT_ROOT": str(args.artifact_root),
            "EXTERNAL_LAUNCH_MANIFEST": str(
                args.artifact_root / "e003_external_launch_manifest.json"
            ),
        },
        _dependency_ids([*pre_jobs.values(), cache_150m]),
    )
    payload: dict[str, Any] = {
        "schema_version": "ppi_shift.e003_external_submission_manifest.v1",
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
        "protocol_freeze": {
            "path": str(args.protocol_freeze),
            "sha256": sha256_path(args.protocol_freeze),
        },
        "artifact_root": str(args.artifact_root),
        "pre_jobs": pre_jobs,
        "jobs": {
            "cache_input_manifest": str(cache_input_manifest),
            "external_cache_input_job": input_job,
            "external_150m_cache_job": cache_150m,
            "external_dispatch_job": dispatch_job,
        },
        "status": "complete_core4_external_cache_and_dispatch_submitted",
    }
    args.submission_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.submission_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
