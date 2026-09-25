"""Resume the E003 Core-4 DAG under a one-task Slurm submit limit.

The normal Core-4 launcher submits arrays and is the preferred path.  When a
cluster QOS only permits one additional array element, this script submits
the same immutable request manifests one indexed task at a time and records
each accepted job before attempting the next one.  It never replaces a job
that is already recorded and never submits a postprocess job before every
training unit has an accepted scheduler ID.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from submit_e003_core_dag import (
        _assert_core_request_roster,
        _dependency_ids,
        _json,
        _manifest_requests,
        _record,
        _state,
        _submit,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as a package
    from scripts.submit_e003_core_dag import (
        _assert_core_request_roster,
        _dependency_ids,
        _json,
        _manifest_requests,
        _record,
        _state,
        _submit,
    )

from ppi_shift.artifacts import sha256_path

UTC = getattr(timezone, "utc")
SCHEMA = "ppi_shift.e003_core_staged_submission.v1"


def _parse_pre_jobs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, job_id = value.partition("=")
        if not separator or not name or not job_id.isdigit():
            raise SystemExit(f"Invalid --pre-job value: {value!r}")
        result[name] = job_id
    if not result:
        raise SystemExit("At least one panel --pre-job is required")
    return result


def _parse_retry_targets(values: list[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    allowed = {"python", "frozen", "reuse"}
    for value in values:
        phase, separator, index_text = value.partition(":")
        if not separator or phase not in allowed or not index_text.isdigit():
            raise SystemExit(
                "Invalid --retry-failed value; expected one of "
                "python:N, frozen:N, or reuse:N"
            )
        target = (phase, int(index_text))
        if target in seen:
            raise SystemExit(f"Duplicate --retry-failed target: {value}")
        seen.add(target)
        result.append(target)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _immutable_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "repo_root": str(args.repo_root),
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "protocol_freeze": {
            "path": str(args.protocol_freeze),
            "sha256": sha256_path(args.protocol_freeze),
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
        "e001_registry": {
            "path": str(args.e001_registry),
            "sha256": sha256_path(args.e001_registry),
        },
        "artifact_root": str(args.artifact_root),
    }


def _validate_attempt(
    path: Path,
    backend: list[dict[str, Any]],
    freeze_sha256: str,
) -> list[dict[str, Any]]:
    attempt = _json(path)
    if attempt.get("schema_version") != "ppi_shift.e003_core_submission_attempt.v1":
        raise SystemExit(f"Unexpected partial-submission schema: {path}")
    if attempt.get("protocol_freeze", {}).get("sha256") != freeze_sha256:
        raise SystemExit("Partial submission freeze hash does not match current freeze")
    rows = attempt.get("backend_jobs")
    if not isinstance(rows, list) or len(rows) != len(backend):
        raise SystemExit("Partial submission does not contain all 12 backend jobs")
    result: list[dict[str, Any]] = []
    for expected, observed in zip(backend, rows, strict=True):
        job_id = str(observed.get("job_id", ""))
        if not job_id.isdigit():
            raise SystemExit(f"Invalid accepted backend job ID: {job_id!r}")
        if observed.get("request") != expected.get("request"):
            raise SystemExit("Partial backend request order differs from the frozen manifest")
        if observed.get("request_sha256") != expected.get("request_sha256"):
            raise SystemExit("Partial backend request hash differs from the frozen manifest")
        result.append({"job_id": job_id, "request": expected})
    return result


def _validate_existing_state(
    state: dict[str, Any],
    immutable: dict[str, Any],
    pre_jobs: dict[str, str],
    backend: list[dict[str, Any]],
    submission_attempt_sha256: str,
) -> None:
    if state.get("schema_version") != SCHEMA:
        raise SystemExit("Unexpected staged-submission state schema")
    if state.get("immutable_inputs") != immutable:
        raise SystemExit("Staged-submission immutable inputs changed")
    if state.get("pre_jobs") != pre_jobs:
        raise SystemExit("Staged-submission pre-job mapping changed")
    if state.get("submission_attempt", {}).get("sha256") != submission_attempt_sha256:
        raise SystemExit("Staged-submission partial-attempt hash changed")
    observed = state.get("backend")
    if not isinstance(observed, list) or len(observed) != len(backend):
        raise SystemExit("Staged-submission backend state is incomplete")
    for expected, row in zip(backend, observed, strict=True):
        if row.get("job_id") is None or row.get("request") != expected:
            raise SystemExit("Staged-submission backend state differs from the request manifest")


def _job_is_terminal_success(job_id: str) -> bool:
    state = _state(job_id)
    if state == "UNKNOWN":
        raise SystemExit(f"Cannot establish scheduler state for job {job_id}")
    if state.startswith("COMPLETED"):
        return True
    if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")):
        raise SystemExit(f"Recorded E003 job {job_id} is not recoverable automatically: {state}")
    return False


def _phase_entries(state: dict[str, Any], phase: str) -> list[dict[str, Any] | None]:
    value = state[phase]
    if not isinstance(value, list):
        raise SystemExit(f"Invalid staged-submission phase: {phase}")
    return value


def _submit_one_missing(
    *,
    state: dict[str, Any],
    phase: str,
    requests: list[dict[str, Any]],
    script: Path,
    common: dict[str, str],
    dependencies: list[str],
    action: str,
    array: bool,
) -> dict[str, Any] | None:
    entries = _phase_entries(state, phase)
    for index, entry in enumerate(entries):
        if entry is not None:
            if entry.get("request") != requests[index]:
                raise SystemExit(f"Recorded {phase}[{index}] request changed")
            _job_is_terminal_success(str(entry["job_id"]))
            continue
        request = requests[index]
        environment = {**common}
        if phase == "reuse":
            environment["REQUEST"] = str(request["request"])
        job_id = _submit(
            script,
            environment,
            dependencies,
            array=str(index) if array else None,
        )
        entries[index] = {
            "index": index,
            "job_id": job_id,
            "request": request,
            "action": action,
            "array_task_count": 1 if array else None,
            "submitted_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        return entries[index]
    return None


def _retry_failed_entry(
    *,
    state: dict[str, Any],
    phase: str,
    index: int,
    requests: list[dict[str, Any]],
    script: Path,
    common: dict[str, str],
    dependencies: list[str],
    action: str,
    array: bool,
    reason: str,
) -> dict[str, Any]:
    entries = _phase_entries(state, phase)
    if index < 0 or index >= len(entries):
        raise SystemExit(f"--retry-failed index is out of range: {phase}:{index}")
    entry = entries[index]
    if entry is None:
        raise SystemExit(f"Cannot retry missing {phase}[{index}]; submit it normally")
    request = requests[index]
    if entry.get("request") != request:
        raise SystemExit(f"Recorded {phase}[{index}] request changed")
    failed_job_id = str(entry.get("job_id", ""))
    if not failed_job_id.isdigit():
        raise SystemExit(f"Invalid recorded job ID for {phase}[{index}]")
    failed_state = _state(failed_job_id)
    if failed_state == "UNKNOWN":
        raise SystemExit(f"Cannot establish scheduler state for failed job {failed_job_id}")
    if not failed_state.startswith(
        ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
    ):
        raise SystemExit(
            f"Refusing to retry {phase}[{index}] job {failed_job_id}: {failed_state}"
        )
    environment = {**common}
    if phase == "reuse":
        environment["REQUEST"] = str(request["request"])
    replacement_job_id = _submit(
        script,
        environment,
        dependencies,
        array=str(index) if array else None,
    )
    submitted_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
    state.setdefault("retry_history", []).append(
        {
            "phase": phase,
            "index": index,
            "failed_job_id": failed_job_id,
            "failed_state": failed_state,
            "replacement_job_id": replacement_job_id,
            "request": request,
            "reason": reason,
            "submitted_utc": submitted_utc,
        }
    )
    replacement = {
        "index": index,
        "job_id": replacement_job_id,
        "request": request,
        "action": action,
        "array_task_count": 1 if array else None,
        "submitted_utc": submitted_utc,
        "retry_of": failed_job_id,
        "retry_reason": reason,
    }
    entries[index] = replacement
    return replacement


def _all_compute_ids(state: dict[str, Any]) -> list[str] | None:
    result: list[str] = []
    for phase in ("backend", "python", "frozen", "reuse"):
        entries = _phase_entries(state, phase)
        if any(entry is None for entry in entries):
            return None
        result.extend(str(entry["job_id"]) for entry in entries if entry is not None)
    return result


def _build_launch_manifest(
    *,
    state: dict[str, Any],
    args: argparse.Namespace,
    immutable: dict[str, Any],
    compute_ids: list[str],
) -> dict[str, Any]:
    dependencies = list(state["pre_jobs"].values())
    jobs: list[dict[str, Any]] = []
    for phase in ("backend", "python", "frozen", "reuse"):
        for entry in _phase_entries(state, phase):
            assert entry is not None
            request = entry["request"]
            # The initial backend submission predates the staged-ledger
            # ``action`` field.  Preserve those accepted records and derive
            # the immutable action from their phase when rebuilding the
            # launch manifest; never resubmit the backend units.
            action = entry.get(
                "action",
                {
                    "backend": "backend_training_unit",
                    "python": "python_training_unit",
                    "frozen": "frozen_training_unit",
                    "reuse": "exact_e001_checkpoint_reuse",
                }[phase],
            )
            jobs.append(
                _record(
                    unit_id=(
                        f"{phase}|{request['model']}|{request['seed']}|"
                        f"{request['train_environment']}"
                    ),
                    action=action,
                    job_id=str(entry["job_id"]),
                    dependencies=dependencies,
                    expected_output=str(request["output_dir"]),
                    request=request,
                    array_task_count=entry.get("array_task_count"),
                )
            )
            jobs[-1]["retry_of"] = entry.get("retry_of")
    post_job = str(state["core_postprocess"]["job_id"])
    jobs.append(
        _record(
            unit_id="core-postprocess|432-cells",
            action="core_postprocess",
            job_id=post_job,
            dependencies=compute_ids,
            expected_output=str(args.artifact_root / "results" / "core"),
        )
    )
    return {
        "schema_version": "ppi_shift.e003_core_launch_manifest.v1",
        "experiment_id": "E003",
        "created_utc": state["created_utc"],
        "repo_root": str(args.repo_root),
        "panel_manifest": immutable["panel_manifest"],
        "config": immutable["config"],
        "protocol_freeze": immutable["protocol_freeze"],
        "e001_registry": immutable["e001_registry"],
        "artifact_root": str(args.artifact_root),
        "pre_jobs": state["pre_jobs"],
        "pre_scheduler_dependencies": _dependency_ids(dependencies),
        "training_units": 72,
        "core_score_cells": 432,
        "jobs": jobs,
        "compute_job_ids": compute_ids,
        "core_postprocess_job_id": post_job,
        "status": "complete_core_dag_submitted",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "panel-manifest",
        "config",
        "protocol-freeze",
        "backend-manifest",
        "python-manifest",
        "frozen-manifest",
        "reuse-manifest",
        "e001-registry",
        "artifact-root",
        "state-manifest",
        "launch-manifest",
        "submission-attempt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pre-job", action="append", default=[], metavar="NAME=JOB_ID")
    parser.add_argument(
        "--retry-failed",
        action="append",
        default=[],
        metavar="PHASE:INDEX",
        help="Explicitly retry one terminally failed python, frozen, or reuse unit.",
    )
    parser.add_argument(
        "--retry-reason",
        default="",
        help="Audit reason required when --retry-failed is used.",
    )
    parser.add_argument("--max-new-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.max_new_jobs < 1:
        raise SystemExit("--max-new-jobs must be positive")
    retry_targets = _parse_retry_targets(args.retry_failed)
    if retry_targets and not args.retry_reason.strip():
        raise SystemExit("--retry-reason is required when --retry-failed is used")
    if len(retry_targets) > args.max_new_jobs:
        raise SystemExit("--retry-failed targets exceed --max-new-jobs")
    if args.launch_manifest.exists():
        raise SystemExit(f"Refusing duplicate E003 core launch: {args.launch_manifest}")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"Protocol freeze is missing: {args.protocol_freeze}")
    pre_jobs = _parse_pre_jobs(args.pre_job)
    immutable = _immutable_inputs(args)
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
    accepted_backend = _validate_attempt(
        args.submission_attempt, backend, immutable["protocol_freeze"]["sha256"]
    )
    if args.state_manifest.exists():
        state = _json(args.state_manifest)
        _validate_existing_state(
            state,
            immutable,
            pre_jobs,
            backend,
            sha256_path(args.submission_attempt),
        )
    else:
        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        state = {
            "schema_version": SCHEMA,
            "experiment_id": "E003",
            "created_utc": now,
            "updated_utc": now,
            "immutable_inputs": immutable,
            "submission_attempt": {
                "path": str(args.submission_attempt),
                "sha256": sha256_path(args.submission_attempt),
            },
            "pre_jobs": pre_jobs,
            "backend": accepted_backend,
            "python": [None] * len(python),
            "frozen": [None] * len(frozen),
            "reuse": [None] * len(reuse),
            "core_postprocess": None,
            "retry_history": [],
            "status": "partial_compute_submission",
        }
        _write_json(args.state_manifest, state)

    pre_dependencies = _dependency_ids(list(pre_jobs.values()))
    common = {
        "E003_ARTIFACT_ROOT": str(args.artifact_root),
        "PPI_SHIFT_REPO": str(args.repo_root),
    }
    phase_specs = (
        (
            "python",
            python,
            args.repo_root / "scripts" / "launch_e003_python_cell.slurm",
            "python_training_unit",
            True,
        ),
        (
            "frozen",
            frozen,
            args.repo_root / "scripts" / "launch_e003_frozen_cell.slurm",
            "frozen_training_unit",
            True,
        ),
        (
            "reuse",
            reuse,
            args.repo_root / "scripts" / "launch_e003_reuse_cell.slurm",
            "exact_e001_checkpoint_reuse",
            False,
        ),
    )
    phase_by_name = {
        phase: (requests, script, action, array)
        for phase, requests, script, action, array in phase_specs
    }
    submissions = 0
    for phase, index in retry_targets:
        requests, script, action, array = phase_by_name[phase]
        _retry_failed_entry(
            state=state,
            phase=phase,
            index=index,
            requests=requests,
            script=script,
            common=common,
            dependencies=pre_dependencies,
            action=action,
            array=array,
            reason=args.retry_reason.strip(),
        )
        submissions += 1
        state["last_scheduler_error"] = None
        state["updated_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
        _write_json(args.state_manifest, state)
    for phase, requests, script, action, array in phase_specs:
        while submissions < args.max_new_jobs:
            try:
                entry = _submit_one_missing(
                    state=state,
                    phase=phase,
                    requests=requests,
                    script=script,
                    common=common,
                    dependencies=pre_dependencies,
                    action=action,
                    array=array,
                )
            except subprocess.CalledProcessError as error:
                state["last_scheduler_error"] = {
                    "phase": phase,
                    "observed_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                    "returncode": error.returncode,
                    "stdout": (error.stdout or "").strip(),
                    "stderr": (error.stderr or "").strip(),
                }
                state["updated_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
                _write_json(args.state_manifest, state)
                print(
                    json.dumps(
                        {
                            "status": "scheduler_capacity_blocked",
                            "phase": phase,
                            "state_manifest": str(args.state_manifest),
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
            state["updated_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
            _write_json(args.state_manifest, state)
            if submissions >= args.max_new_jobs:
                break
        if submissions >= args.max_new_jobs:
            break

    compute_ids = _all_compute_ids(state)
    if (
        compute_ids is not None
        and state["core_postprocess"] is None
        and submissions < args.max_new_jobs
    ):
        post_dependencies = _dependency_ids(compute_ids)
        post_job = _submit(
            args.repo_root / "scripts" / "launch_e003_core_postprocess.slurm",
            {
                **common,
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
        state["core_postprocess"] = {
            "job_id": post_job,
            "submitted_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        state["status"] = "complete_core_dag_submitted"
        state["updated_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
        _write_json(args.state_manifest, state)
        submissions += 1

    if state["core_postprocess"] is not None and compute_ids is not None:
        launch = _build_launch_manifest(
            state=state,
            args=args,
            immutable=immutable,
            compute_ids=compute_ids,
        )
        _write_json(args.launch_manifest, launch)
        print(
            json.dumps(
                {"status": launch["status"], "launch_manifest": str(args.launch_manifest)}
            )
        )
    else:
        missing = {
            phase: sum(entry is None for entry in _phase_entries(state, phase))
            for phase in ("python", "frozen", "reuse")
        }
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "new_jobs_submitted": submissions,
                    "missing_compute_units": missing,
                    "state_manifest": str(args.state_manifest),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
