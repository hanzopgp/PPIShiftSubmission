"""Shared frozen-panel runtime helpers for E003 score jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path, write_csv
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_panels import read_e003_panel


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return value


def panel_spec(manifest: dict[str, Any], split: str, environment: str) -> dict[str, Any]:
    if environment not in E003_ENVIRONMENTS:
        raise ValueError(f"Unknown E003 environment: {environment}")
    spec = manifest["stages"][split]["samplers"].get(environment)
    if not isinstance(spec, dict) or spec.get("status") != "complete":
        raise ValueError(f"Missing complete E003 panel: {split}/{environment}")
    return spec


def load_panel(
    manifest: dict[str, Any], split: str, environment: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    spec = panel_spec(manifest, split, environment)
    path = Path(spec["panel_path"])
    actual_hash = sha256_path(path)
    if actual_hash != spec["panel_sha256"]:
        raise ValueError(f"E003 panel hash changed: {path}")
    rows = read_e003_panel(path, environment=environment)
    if len(rows) != int(spec["panel_rows"]):
        raise ValueError(f"E003 panel row count changed: {path}")
    if {str(row.get("sampler_version", "")).split(":", 1)[0] for row in rows} != {
        E003_SCHEMA_VERSION
    }:
        raise ValueError(f"E003 panel schema changed: {path}")
    return rows, {"path": str(path), "sha256": actual_hash, "rows": len(rows)}


def load_all_test_panels(
    manifest: dict[str, Any],
    environments: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    specs: dict[str, dict[str, Any]] = {}
    selected = tuple(environments or manifest.get("environments", E003_ENVIRONMENTS))
    unknown = sorted(set(selected) - set(E003_ENVIRONMENTS))
    if unknown:
        raise ValueError(f"Unknown E003 test environments: {unknown}")
    for environment in selected:
        rows[environment], specs[environment] = load_panel(manifest, "test", environment)
    return rows, specs


def d007_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Translate E003 panel rows to the pinned ExternalPPI backend contract."""

    return [
        {
            "row_id": row["row_id"],
            "left_id": row["anchor_id"],
            "right_id": (
                row["true_partner_id"]
                if row["interaction_label"] == "1"
                else row["candidate_partner_id"]
            ),
            "left_sequence": row["seq1"],
            "right_sequence": row["seq2"],
            "left_sequence_sha256": row["anchor_sequence_sha256"],
            "right_sequence_sha256": (
                row["true_partner_sequence_sha256"]
                if row["interaction_label"] == "1"
                else row["candidate_sequence_sha256"]
            ),
            "label": row["interaction_label"],
        }
        for row in rows
    ]


def write_predictions(path: Path, rows: list[dict[str, str]], scores: Any) -> str:
    write_csv(
        path,
        ("row_id", "positive_row_id", "interaction_label", "score"),
        (
            {
                "row_id": row["row_id"],
                "positive_row_id": row["positive_row_id"],
                "interaction_label": row["interaction_label"],
                "score": format(float(score), ".17g"),
            }
            for row, score in zip(rows, scores, strict=True)
        ),
    )
    return sha256_path(path)


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    return list(iter_delimited(path))


def json_hash(path: Path) -> str:
    """Hash-assert and parse a JSON object used by E003 orchestration."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return sha256_path(path)
