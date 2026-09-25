"""Small shared I/O helpers for the frozen E003 signal-audit runners."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ppi_shift.signal_audit import read_rows, sha256_path

ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
SEEDS = (11, 22, 33)


def parse_csv_values(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("A comma-separated argument cannot be empty")
    return parsed


def panel_path(panel_root: Path, split: str, environment: str) -> Path:
    candidates = (
        panel_root / f"{split}__{environment}.csv.gz",
        panel_root / f"{split}__{environment}.csv",
        panel_root / f"{split}_{environment}.csv.gz",
        panel_root / f"{split}_{environment}.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(panel_root.glob(f"*{split}*{environment}*.csv*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not resolve one {split} panel for {environment} under {panel_root}; "
        f"candidates={matches}"
    )


def load_signal_panel(
    panel_root: Path, split: str, environment: str
) -> tuple[list[dict[str, str]], Path]:
    path = panel_path(panel_root, split, environment)
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Empty E003 panel: {path}")
    observed = {str(row.get("negative_environment", "")) for row in rows}
    if observed != {environment}:
        raise ValueError(f"Panel environment mismatch for {path}: {sorted(observed)}")
    labels = {str(row.get("interaction_label", "")) for row in rows}
    if labels != {"0", "1"}:
        raise ValueError(f"Panel lacks both labels: {path}")
    return rows, path


def git_revision(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty audit CSV: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def metric_row(
    result: dict[str, Any], *, environment: str, control: str | None = None
) -> dict[str, Any]:
    test_metrics = result.get("test_metrics", {})
    holdout_metrics = result.get("holdout_metrics", {})
    row: dict[str, Any] = {
        "control": control or str(result.get("control", "")),
        "environment": environment,
        "seed": int(result["seed"]),
        "status": str(result.get("status", "")),
        "test_labels_used_for_fit": result.get("test_labels_used_for_fit"),
        "feature_count": result.get("feature_count", ""),
        "test_rows": test_metrics.get("rows", ""),
        "test_positive_prevalence": test_metrics.get("positive_prevalence", ""),
        "test_auprc": test_metrics.get("auprc", ""),
        "test_auroc": test_metrics.get("auroc", ""),
        "test_mcc": test_metrics.get("mcc", ""),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", ""),
        "test_brier": test_metrics.get("brier", ""),
        "test_nll": test_metrics.get("nll", ""),
        "holdout_auprc": holdout_metrics.get("auprc", ""),
        "holdout_auroc": holdout_metrics.get("auroc", ""),
        "threshold": result.get("threshold", {}).get("threshold", ""),
    }
    if "reason" in result:
        row["reason"] = result["reason"]
    return row


def base_metadata(repo_root: Path, protocol_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "ppi_shift.e003_signal_audit_output.v1",
        "protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
        "git_revision": git_revision(repo_root),
        "command": " ".join(sys.argv),
    }
