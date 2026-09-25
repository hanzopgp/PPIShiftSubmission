#!/usr/bin/env python3
"""Verify the supplementary official-DataSAIL E003 score matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import canonical_json_sha256, iter_delimited, sha256_path
from ppi_shift.e003_runtime import load_panel, load_yaml
from ppi_shift.metrics import compute_binary_metrics, select_threshold

ENVIRONMENTS = ("uniform_partner", "esm_partner_matched")
METRIC_FIELDS = (
    "rows",
    "positive_prevalence",
    "auprc",
    "auroc",
    "mcc",
    "balanced_accuracy",
    "f1",
    "brier",
    "nll",
    "ece",
    "threshold",
    "threshold_source",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _predictions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing prediction file: {path}")
    return list(iter_delimited(path))


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_path(path) != expected:
        raise SystemExit(f"{label} hash changed or file missing: {path}")


def _assert_metrics_match(
    recorded: dict[str, Any],
    labels: list[int],
    scores: list[float],
    *,
    threshold: float,
    label: str,
) -> None:
    expected = compute_binary_metrics(
        labels,
        scores,
        threshold=threshold,
        threshold_source="matched_validation_max_mcc",
    )
    for field in METRIC_FIELDS:
        if field not in recorded:
            raise SystemExit(f"DataSAIL metric is missing {field}: {label}")
        actual = recorded[field]
        expected_value = expected[field]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), float(expected_value), rel_tol=0.0, abs_tol=5e-12
            ):
                raise SystemExit(f"DataSAIL metric mismatch for {label}/{field}")
        elif actual != expected_value:
            raise SystemExit(f"DataSAIL metric mismatch for {label}/{field}")


def _selected_epoch(payload: dict[str, Any]) -> int | None:
    candidates: list[Any] = [
        payload.get("selected_epoch"),
        payload.get("checkpoint_selection_epoch"),
        payload.get("selection", {}).get("selected_epoch")
        if isinstance(payload.get("selection"), dict)
        else None,
        payload.get("checkpoint", {}).get("selected_epoch")
        if isinstance(payload.get("checkpoint"), dict)
        else None,
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def _backend_unit(
    request: dict[str, Any],
    request_path: Path,
    panel_rows_by_environment: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(request["output_dir"])
    provenance = _json(output / "provenance.json")
    selected_epoch = _selected_epoch(provenance)
    source_revision = str(
        provenance.get("source_revision")
        or provenance.get("git_commit")
        or request.get("source_revision")
        or "unknown_runner_revision"
    )
    validation = _predictions(output / "validation_predictions.csv")
    threshold = select_threshold(
        [int(row["y_true"]) for row in validation],
        [float(row["y_score"]) for row in validation],
    )
    predictions = _predictions(output / "predictions.csv")
    by_id: dict[str, dict[str, str]] = {}
    for row in predictions:
        if row["row_id"] in by_id:
            raise SystemExit(f"DataSAIL backend prediction IDs are not unique: {output}")
        by_id[row["row_id"]] = row
    expected_ids = {
        f"{environment}::{panel_row['row_id']}"
        for environment, panel_rows in panel_rows_by_environment.items()
        for panel_row in panel_rows
    }
    if set(by_id) != expected_ids:
        raise SystemExit(f"DataSAIL backend prediction roster changed: {output}")
    cells: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        panel_rows = panel_rows_by_environment[environment]
        subset = [by_id[f"{environment}::{panel_row['row_id']}"] for panel_row in panel_rows]
        labels = [int(row["y_true"]) for row in subset]
        scores = [float(row["y_score"]) for row in subset]
        expected_labels = [int(row["interaction_label"]) for row in panel_rows]
        if labels != expected_labels:
            raise SystemExit(f"DataSAIL backend prediction labels changed: {output}")
        cells.append(
            {
                "model": request["model"],
                "seed": int(request["seed"]),
                "train_environment": request["train_environment"],
                "test_environment": environment,
                "status": "complete_datasail_sensitivity_unit",
                "metrics": compute_binary_metrics(
                    labels,
                    scores,
                    threshold=float(threshold["threshold"]),
                    threshold_source="matched_validation_max_mcc",
                ),
                "threshold": threshold,
                "prediction_path": str(output / "predictions.csv"),
                "prediction_sha256": sha256_path(output / "predictions.csv"),
                "checkpoint_path": provenance.get("checkpoint_path"),
                "checkpoint_sha256": provenance.get("checkpoint_sha256"),
                "checkpoint_selection_epoch": selected_epoch,
                "source_revision": source_revision,
                "request_path": str(request_path),
                "request_sha256": request["request_sha256"],
            }
        )
    return cells, {
        "model": request["model"],
        "seed": int(request["seed"]),
        "train_environment": request["train_environment"],
        "checkpoint_path": provenance.get("checkpoint_path"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "checkpoint_selection_epoch": selected_epoch,
        "source_revision": source_revision,
        "threshold": threshold,
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
    }


def _frozen_unit(
    request: dict[str, Any],
    request_path: Path,
    panel_rows_by_environment: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(request["output_dir"])
    run = _json(output / "run_manifest.json")
    if run.get("status") != "complete":
        raise SystemExit(f"DataSAIL Frozen unit is incomplete: {output}")
    if tuple(run.get("test_environments", ())) != ENVIRONMENTS:
        raise SystemExit(f"DataSAIL Frozen test roster changed: {output}")
    threshold = run["validation"]["threshold"]
    selected_epoch = _selected_epoch(run)
    source_revision = str(
        run.get("source_revision")
        or run.get("package_revision")
        or run.get("git_commit")
        or "unknown_runner_revision"
    )
    cells: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        result = run["test_results"][environment]
        prediction = run["prediction_files"][environment]
        prediction_path = Path(prediction["path"])
        _assert_hash(prediction_path, prediction["sha256"], "DataSAIL Frozen prediction")
        prediction_rows = _predictions(prediction_path)
        by_id: dict[str, dict[str, str]] = {}
        for row in prediction_rows:
            if row["row_id"] in by_id:
                raise SystemExit(f"DataSAIL Frozen prediction IDs are not unique: {output}")
            by_id[row["row_id"]] = row
        panel_rows = panel_rows_by_environment[environment]
        if set(by_id) != {row["row_id"] for row in panel_rows}:
            raise SystemExit(f"DataSAIL Frozen prediction roster changed: {output}")
        labels = [int(row["interaction_label"]) for row in panel_rows]
        scores = [float(by_id[row["row_id"]]["score"]) for row in panel_rows]
        if any(
            by_id[row["row_id"]].get("interaction_label") != row["interaction_label"]
            for row in panel_rows
        ):
            raise SystemExit(f"DataSAIL Frozen prediction labels changed: {output}")
        _assert_metrics_match(
            result["metrics"],
            labels,
            scores,
            threshold=float(threshold["threshold"]),
            label=str(prediction_path),
        )
        cells.append(
            {
                "model": request["model"],
                "seed": int(request["seed"]),
                "train_environment": request["train_environment"],
                "test_environment": environment,
                "status": "complete_datasail_sensitivity_unit",
                "metrics": result["metrics"],
                "threshold": threshold,
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction["sha256"],
                "checkpoint_path": run["checkpoint"]["path"],
                "checkpoint_sha256": run["checkpoint"]["sha256"],
                "checkpoint_selection_epoch": selected_epoch,
                "source_revision": source_revision,
                "request_path": str(request_path),
                "request_sha256": request["request_sha256"],
            }
        )
    return cells, {
        "model": request["model"],
        "seed": int(request["seed"]),
        "train_environment": request["train_environment"],
        "checkpoint_path": run["checkpoint"]["path"],
        "checkpoint_sha256": run["checkpoint"]["sha256"],
        "checkpoint_selection_epoch": selected_epoch,
        "source_revision": source_revision,
        "threshold": threshold,
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
    }


def _requests(
    payload: dict[str, Any],
    key: str,
    *,
    panel_manifest: Path,
    panel_sha256: str,
    config: Path,
    config_sha256: str,
    require_config: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    values = payload.get(key, [])
    result = []
    for item in values:
        path = Path(item["request"])
        request = _json(path)
        recorded = str(request.get("request_sha256", ""))
        if recorded != item.get("request_sha256"):
            raise SystemExit(f"DataSAIL request hash changed: {path}")
        if not recorded or canonical_json_sha256(request, omit_key="request_sha256") != recorded:
            raise SystemExit(f"DataSAIL request canonical hash is invalid: {path}")
        if (
            Path(str(request.get("panel_manifest", ""))).resolve() != panel_manifest.resolve()
            or request.get("panel_manifest_sha256") != panel_sha256
        ):
            raise SystemExit(f"DataSAIL request panel binding changed: {path}")
        if require_config and (
            Path(str(request.get("config", ""))).resolve() != config.resolve()
            or request.get("config_sha256") != config_sha256
        ):
            raise SystemExit(f"DataSAIL request config binding changed: {path}")
        result.append((path, request))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--core-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = _json(args.request_manifest)
    if payload.get("schema_version") != "ppi_shift.e003_datasail_request_matrix.v1":
        raise SystemExit("Unexpected DataSAIL request matrix schema")
    expected_panel_sha = sha256_path(args.panel_manifest)
    expected_config_sha = sha256_path(args.core_config)
    if (
        Path(str(payload.get("panel_manifest", ""))).resolve() != args.panel_manifest.resolve()
        or payload.get("panel_manifest_sha256") != expected_panel_sha
    ):
        raise SystemExit("DataSAIL matrix panel binding changed")
    if (
        Path(str(payload.get("config", ""))).resolve() != args.core_config.resolve()
        or payload.get("config_sha256") != expected_config_sha
    ):
        raise SystemExit("DataSAIL matrix config binding changed")
    panel_manifest = load_yaml(args.panel_manifest)
    panel_rows_by_environment = {
        environment: load_panel(panel_manifest, "test", environment)[0]
        for environment in ENVIRONMENTS
    }
    request_bindings = {
        "panel_manifest": args.panel_manifest,
        "panel_sha256": expected_panel_sha,
        "config": args.core_config,
        "config_sha256": expected_config_sha,
    }
    backend = _requests(
        payload,
        "backend_requests",
        **request_bindings,
        require_config=False,
    )
    frozen = _requests(
        payload,
        "frozen_requests",
        **request_bindings,
        require_config=True,
    )
    if len(backend) != 12 or len(frozen) != 6:
        raise SystemExit("Expected 12 backend and 6 Frozen DataSAIL training units")
    cells: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for path, request in backend:
        unit_cells, unit = _backend_unit(request, path, panel_rows_by_environment)
        units.append(unit)
        cells.extend(unit_cells)
    for path, request in frozen:
        unit_cells, unit = _frozen_unit(request, path, panel_rows_by_environment)
        units.append(unit)
        cells.extend(unit_cells)
    if len(cells) != 36 or len(units) != 18:
        raise SystemExit(f"DataSAIL matrix incomplete: {len(units)} units, {len(cells)} cells")
    cells.sort(
        key=lambda cell: (
            cell["model"],
            cell["seed"],
            cell["train_environment"],
            cell["test_environment"],
        )
    )
    result = {
        "schema_version": "ppi_shift.e003_datasail_results.v1",
        "experiment_id": "E003",
        "role": "supplementary_official_datasail_sensitivity",
        "request_manifest": {
            "path": str(args.request_manifest),
            "sha256": sha256_path(args.request_manifest),
        },
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "config": {"path": str(args.core_config), "sha256": sha256_path(args.core_config)},
        "training_units": len(units),
        "score_cells": len(cells),
        "units": units,
        "cells": cells,
        "test_adaptation": False,
        "test_cropping": False,
        "status": "complete_datasail_sensitivity",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "e003_datasail_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output), "cells": len(cells), "units": len(units)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
