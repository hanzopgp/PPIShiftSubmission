"""Shared fail-closed helpers for QOS-limited E003 fan-out dispatchers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import canonical_json_sha256

UTC = getattr(timezone, "utc")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON mapping: {path}")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def submit_job(
    script: Path,
    environment: dict[str, str],
    dependencies: Sequence[str],
    *,
    array_index: int | None = None,
) -> str:
    command = ["sbatch", "--parsable", "--qos=normal"]
    if dependencies:
        command.append(f"--dependency=afterok:{':'.join(dependencies)}")
    if array_index is not None:
        command.append(f"--array={array_index}")
    exports = ["ALL"] + [f"{key}={value}" for key, value in sorted(environment.items())]
    command.extend([f"--export={','.join(exports)}", str(script)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(r"\b(\d+)\b", completed.stdout.strip())
    if match is None:
        raise RuntimeError(f"sbatch returned no job ID for {script}: {completed.stdout!r}")
    return match.group(1)


def scheduler_state(job_id: str) -> str:
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


def dependency_ids(job_ids: Sequence[str]) -> list[str]:
    result: list[str] = []
    for job_id in job_ids:
        state = scheduler_state(job_id)
        if state == "COMPLETED":
            continue
        if state == "UNKNOWN":
            raise SystemExit(f"Cannot establish scheduler state for dependency job {job_id}")
        result.append(job_id)
    return result


def assert_recorded_job_is_recoverable(job_id: str) -> bool:
    state = scheduler_state(job_id)
    if state == "UNKNOWN":
        raise SystemExit(f"Cannot establish scheduler state for recorded job {job_id}")
    if state.startswith("COMPLETED"):
        return True
    if state.startswith(
        ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
    ):
        raise SystemExit(f"Recorded E003 job {job_id} is not recoverable automatically: {state}")
    return False


def validate_request_entry(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(entry.get("request", "")))
    if not path.is_file():
        raise SystemExit(f"Staged request is missing: {path}")
    payload = json_mapping(path)
    recorded = str(entry.get("request_sha256", ""))
    if not recorded or payload.get("request_sha256") != recorded:
        raise SystemExit(f"Staged request hash field changed: {path}")
    actual = canonical_json_sha256(payload, omit_key="request_sha256")
    if actual != recorded:
        raise SystemExit(f"Staged request canonical hash changed: {path}")
    for key in ("model", "seed", "train_environment"):
        if key in entry and payload.get(key) != entry[key]:
            raise SystemExit(f"Staged request {key} changed: {path}")
    return entry


def submit_one_missing(
    *,
    state: dict[str, Any],
    phase: str,
    requests: list[dict[str, Any]],
    script: Path,
    dependencies: Sequence[str],
    action: str,
    array: bool,
    environment_for_request: Callable[[dict[str, Any]], dict[str, str]],
) -> dict[str, Any] | None:
    entries = state.get(phase)
    if not isinstance(entries, list) or len(entries) != len(requests):
        raise SystemExit(f"Invalid staged-submission phase: {phase}")
    for index, entry in enumerate(entries):
        if entry is not None:
            if entry.get("request") != requests[index]:
                raise SystemExit(f"Recorded {phase}[{index}] request changed")
            assert_recorded_job_is_recoverable(str(entry["job_id"]))
            continue
        request = requests[index]
        job_id = submit_job(
            script,
            environment_for_request(request),
            dependency_ids(dependencies),
            array_index=index if array else None,
        )
        entry = {
            "index": index,
            "job_id": job_id,
            "request": request,
            "action": action,
            "array_task_count": 1 if array else None,
            "submitted_utc": utc_now(),
        }
        entries[index] = entry
        return entry
    return None


def parse_job_mapping(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, job_id = value.partition("=")
        if not separator or not name or not job_id.isdigit():
            raise SystemExit(f"Invalid --pre-job value: {value!r}")
        result[name] = job_id
    return result
