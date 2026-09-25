#!/usr/bin/env python3
"""Assemble the frozen E003 core matrix after all score units complete."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import canonical_json_sha256, iter_delimited, sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import load_yaml
from ppi_shift.metrics import compute_binary_metrics, select_threshold

MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
SEEDS = (11, 22, 33)
OLD_TRAIN_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
)
NEW_TRAIN_ENVIRONMENTS = (
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
EXPECTED_REUSE_MODELS = ("symmetric_frozen_esm2_pair_mlp", "dscript_esm2", "tuna_corrected")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _read_predictions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(iter_delimited(path))


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    actual = sha256_path(path)
    if actual != expected:
        raise SystemExit(f"{label} hash changed: {path} ({actual} != {expected})")


def _selected_epoch(payload: dict[str, Any]) -> int | None:
    """Extract the validation-selected epoch across runner manifest schemas."""

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


def _request_hash(request: dict[str, Any], request_path: Path) -> str:
    recorded = str(request.get("request_sha256", ""))
    if not recorded:
        raise SystemExit(f"Request lacks request_sha256: {request_path}")
    return recorded


def _source_revision(request: dict[str, Any], provenance: dict[str, Any]) -> str:
    source_revision = provenance.get("source_revision") or request.get("source_revision")
    if not source_revision:
        raise SystemExit(
            "E003 backend output lacks source_revision in both provenance and request summary"
        )
    return str(source_revision)


def _test_environments(request: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(str(value) for value in request.get("test_environments", E003_ENVIRONMENTS))
    if values != E003_ENVIRONMENTS:
        raise SystemExit(f"Training unit does not score the frozen E003 roster: {values}")
    return values


def _request_train_environment(
    request: dict[str, Any], request_path: Path, *, require_new: bool = False
) -> str:
    """Resolve the declared training environment without rewriting the request hash.

    Early E003 backend requests stored this field only in ``panel_contract``.
    Resolve that legacy representation in memory so the immutable request JSON
    and its canonical hash remain unchanged.
    """

    value = request.get("train_environment")
    if value is None and isinstance(request.get("panel_contract"), dict):
        value = request["panel_contract"].get("train_environment")
    environment = str(value or "")
    valid_environments = (
        NEW_TRAIN_ENVIRONMENTS
        if require_new
        else OLD_TRAIN_ENVIRONMENTS + NEW_TRAIN_ENVIRONMENTS
    )
    if environment not in valid_environments:
        raise SystemExit(
            f"Request lacks a valid E003 train environment: {request_path} ({environment!r})"
        )
    return environment


def _d007_unit(
    request: dict[str, Any], request_path: Path, output_dir: Path, train_environment: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environments = _test_environments(request)
    provenance_path = output_dir / "provenance.json"
    provenance = _json(provenance_path)
    source_revision = _source_revision(request, provenance)
    selected_epoch = _selected_epoch(provenance)
    validation = _read_predictions(output_dir / "validation_predictions.csv")
    threshold = select_threshold(
        [int(row["y_true"]) for row in validation],
        [float(row["y_score"]) for row in validation],
    )
    predictions = _read_predictions(output_dir / "predictions.csv")
    swapped = _read_predictions(output_dir / "swap_predictions.csv")
    swapped_by_id = {row["row_id"]: float(row["y_score"]) for row in swapped}
    if len(swapped_by_id) != len(swapped):
        raise SystemExit(f"Duplicate swapped row IDs: {output_dir}")
    checkpoint_path = Path(str(provenance.get("checkpoint_path", "")))
    checkpoint_hash = str(provenance.get("checkpoint_sha256", ""))
    if checkpoint_path and checkpoint_hash:
        _assert_hash(checkpoint_path, checkpoint_hash, "backend checkpoint")
    cells: list[dict[str, Any]] = []
    for environment in environments:
        subset = [
            row
            for row in predictions
            if row["row_id"].split("::", 1)[0] == environment
        ]
        if not subset:
            raise SystemExit(f"D007 output lacks test environment {environment}: {output_dir}")
        if any(row["row_id"] not in swapped_by_id for row in subset):
            raise SystemExit(f"D007 swap output is incomplete: {output_dir}")
        labels = [int(row["y_true"]) for row in subset]
        scores = [float(row["y_score"]) for row in subset]
        cells.append(
            {
                "model": request["model"],
                "seed": int(request["seed"]),
                "train_environment": train_environment,
                "test_environment": environment,
                "status": "complete_new_training_unit",
                "metrics": compute_binary_metrics(
                    labels,
                    scores,
                    threshold=float(threshold["threshold"]),
                    threshold_source="matched_validation_max_mcc",
                ),
                "threshold": threshold,
                "prediction_path": str(output_dir / "predictions.csv"),
                "prediction_sha256": sha256_path(output_dir / "predictions.csv"),
                "swap_max_abs": max(
                    abs(float(row["y_score"]) - swapped_by_id[row["row_id"]])
                    for row in subset
                ),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_selection_epoch": selected_epoch,
                "source_revision": source_revision,
                "request_path": str(request_path),
                "request_sha256": _request_hash(request, request_path),
            }
        )
    unit = {
        "model": request["model"],
        "seed": int(request["seed"]),
        "train_environment": train_environment,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_selection_epoch": selected_epoch,
        "threshold": threshold,
        "source_revision": source_revision,
        "runner_manifest": str(provenance_path),
        "request_path": str(request_path),
        "request_sha256": _request_hash(request, request_path),
    }
    return cells, unit


def _python_unit(
    request: dict[str, Any], request_path: Path, output_dir: Path, train_environment: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environments = _test_environments(request)
    run_path = output_dir / "run_manifest.json"
    run = _json(run_path)
    if run.get("status") != "complete":
        raise SystemExit(f"Python unit is not complete: {run_path}")
    if tuple(run.get("test_environments", ())) != environments:
        raise SystemExit(f"Python unit roster changed: {run_path}")
    threshold = run["validation"]["threshold"]
    checkpoint = run["checkpoint"]
    selected_epoch = _selected_epoch(run)
    source_revision = str(
        run.get("source_revision")
        or run.get("package_revision")
        or run.get("git_commit")
        or "local_clean_init"
    )
    checkpoint_path = Path(checkpoint["path"])
    _assert_hash(checkpoint_path, checkpoint["sha256"], "Python checkpoint")
    cells: list[dict[str, Any]] = []
    for environment in environments:
        result = run["test_results"][environment]
        prediction = run["prediction_files"][environment]
        prediction_path = Path(prediction["path"])
        _assert_hash(prediction_path, prediction["sha256"], "Python prediction")
        cells.append(
            {
                "model": request["model"],
                "seed": int(request["seed"]),
                "train_environment": train_environment,
                "test_environment": environment,
                "status": "complete_new_training_unit",
                "metrics": result["metrics"],
                "threshold": threshold,
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction["sha256"],
                "swap_max_abs": result.get("swap_max_abs"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_selection_epoch": selected_epoch,
                "source_revision": source_revision,
                "request_path": str(request_path),
                "request_sha256": _request_hash(request, request_path),
            }
        )
    return cells, {
        "model": request["model"],
        "seed": int(request["seed"]),
        "train_environment": train_environment,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_selection_epoch": selected_epoch,
        "tokenizer_path": run.get("tokenizer", {}).get("path"),
        "threshold": threshold,
        "source_revision": source_revision,
        "runner_manifest": str(run_path),
        "request_path": str(request_path),
        "request_sha256": _request_hash(request, request_path),
    }


def _reuse_unit(
    request: dict[str, Any],
    request_path: Path,
    output_dir: Path,
    registry_unit: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environments = _test_environments(request)
    run_path = output_dir / "run_manifest.json"
    run = _json(run_path)
    if run.get("status") != "complete" or run.get("retrained"):
        raise SystemExit(f"Reuse unit is not an exact completed score-only unit: {run_path}")
    if Path(str(run.get("e001_run_path"))) != Path(registry_unit["run_path"]):
        raise SystemExit(f"E001 run path changed in reuse output: {run_path}")
    selected_epoch = _selected_epoch(run)
    if selected_epoch is None:
        original_run = Path(str(run.get("e001_run_path", "")))
        for candidate in (
            original_run / "provenance.json",
            original_run / "run_manifest.json",
            original_run / "checkpoint.json",
        ):
            if candidate.is_file():
                selected_epoch = _selected_epoch(_json(candidate))
                if selected_epoch is not None:
                    break
    if tuple(run.get("test_environments", ())) != environments:
        raise SystemExit(f"Reuse unit roster changed: {run_path}")
    checkpoint = run["checkpoint"]
    if checkpoint["sha256"] != registry_unit["checkpoint_sha256"]:
        raise SystemExit(f"E001 checkpoint hash differs from immutable registry: {run_path}")
    _assert_hash(Path(checkpoint["path"]), checkpoint["sha256"], "reused checkpoint")
    cells: list[dict[str, Any]] = []
    for environment in environments:
        prediction = run["prediction_files"][environment]
        _assert_hash(Path(prediction["path"]), prediction["sha256"], "reused prediction")
        result = run["results"][environment]
        cells.append(
            {
                "model": request["model"],
                "seed": int(request["seed"]),
                "train_environment": request["train_environment"],
                "test_environment": environment,
                "status": "complete_reused_e001_checkpoint",
                "metrics": result["metrics"],
                "threshold": run["validation_threshold"],
                "prediction_path": prediction["path"],
                "prediction_sha256": prediction["sha256"],
                "swap_max_abs": run.get("swap_max_abs", {}).get(environment),
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_selection_epoch": selected_epoch,
                "source_revision": "E001v3_exact_checkpoint_reuse",
                "request_path": str(request_path),
                "request_sha256": _request_hash(request, request_path),
            }
        )
    return cells, {
        "model": request["model"],
        "seed": int(request["seed"]),
        "train_environment": request["train_environment"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_selection_epoch": selected_epoch,
        "threshold": run["validation_threshold"],
        "source_revision": "E001v3_exact_checkpoint_reuse",
        "runner_manifest": str(run_path),
        "request_path": str(request_path),
        "request_sha256": _request_hash(request, request_path),
    }


def _manifest_requests(
    path: Path,
    expected_schema: str,
    *,
    panel_manifest: Path,
    panel_sha256: str,
    config: Path,
    config_sha256: str,
    require_config: bool,
    expected_test_panels: dict[str, dict[str, str]],
    expected_train_panels: dict[str, dict[str, str]] | None = None,
    expected_validation_panels: dict[str, dict[str, str]] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    payload = _json(path)
    if payload.get("schema_version") != expected_schema:
        raise SystemExit(f"Unexpected request manifest schema: {path}")
    if (
        Path(str(payload.get("panel_manifest", ""))).resolve() != panel_manifest.resolve()
        or payload.get("panel_manifest_sha256") != panel_sha256
    ):
        raise SystemExit(f"Request matrix panel binding changed: {path}")
    if require_config and (
        Path(str(payload.get("config", ""))).resolve() != config.resolve()
        or payload.get("config_sha256") != config_sha256
    ):
        raise SystemExit(f"Request matrix config binding changed: {path}")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise SystemExit(f"Request manifest has no request list: {path}")

    def assert_test_panels(test_panels: Any, request_path: Path) -> None:
        if not isinstance(test_panels, dict) or set(test_panels) != set(expected_test_panels):
            raise SystemExit(f"Request test-panel binding is incomplete: {request_path}")
        for environment, expected in expected_test_panels.items():
            observed = test_panels.get(environment)
            if not isinstance(observed, dict) or (
                Path(str(observed.get("path", ""))).resolve()
                != Path(expected["path"]).resolve()
                or observed.get("sha256", observed.get("panel_sha256")) != expected["sha256"]
            ):
                raise SystemExit(f"Request test-panel binding changed: {request_path}")

    def assert_request_panel_binding(request: dict[str, Any], request_path: Path) -> None:
        if "panel_manifest" in request:
            if (
                Path(str(request.get("panel_manifest"))).resolve() != panel_manifest.resolve()
                or request.get("panel_manifest_sha256") != panel_sha256
            ):
                raise SystemExit(f"Request panel binding changed: {request_path}")
            return
        if "test_panels" in request:
            assert_test_panels(request.get("test_panels"), request_path)
            return

        summaries = request.get("partition_summaries")
        contract = request.get("panel_contract")
        train_environment = (
            str(request.get("train_environment"))
            if request.get("train_environment") is not None
            else str(contract.get("train_environment", ""))
            if isinstance(contract, dict)
            else ""
        )
        validation_environment = (
            str(contract.get("validation_environment", ""))
            if isinstance(contract, dict)
            else ""
        )
        train_panels = expected_train_panels or expected_test_panels
        validation_panels = expected_validation_panels or expected_test_panels
        expected_train = train_panels.get(train_environment)
        expected_validation = validation_panels.get(validation_environment)
        if (
            not isinstance(summaries, dict)
            or not isinstance(contract, dict)
            or expected_train is None
            or expected_validation is None
            or str(contract.get("train_environment", "")) != train_environment
            or tuple(contract.get("test_environments", ())) != E003_ENVIRONMENTS
        ):
            raise SystemExit(f"Request test-panel binding is incomplete: {request_path}")
        for split in ("train", "validation"):
            summary = summaries.get(split)
            if not isinstance(summary, dict) or (
                summary.get("panel_sha256")
                != (expected_train if split == "train" else expected_validation)["sha256"]
            ):
                raise SystemExit(f"Request test-panel binding changed: {request_path}")
        test_summary = summaries.get("test")
        if not isinstance(test_summary, dict):
            raise SystemExit(f"Request test-panel binding is incomplete: {request_path}")
        assert_test_panels(test_summary.get("source_panels"), request_path)

    result: list[tuple[Path, dict[str, Any]]] = []
    for item in requests:
        request_path = Path(str(item["request"]))
        request = _json(request_path)
        recorded = str(request.get("request_sha256", ""))
        if recorded != item.get("request_sha256"):
            raise SystemExit(f"Request hash differs from matrix manifest: {request_path}")
        if not recorded or canonical_json_sha256(request, omit_key="request_sha256") != recorded:
            raise SystemExit(f"Request canonical hash is invalid: {request_path}")
        assert_request_panel_binding(request, request_path)
        if require_config and (
            Path(str(request.get("config", ""))).resolve() != config.resolve()
            or request.get("config_sha256") != config_sha256
        ):
            raise SystemExit(f"Request config binding changed: {request_path}")
        result.append((request_path, request))
    return result


def _registry_units(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["model"], int(row["seed"]), row["train_environment"])
        if (
            row["model"] in EXPECTED_REUSE_MODELS
            and row["train_environment"] in OLD_TRAIN_ENVIRONMENTS
        ):
            previous = selected.get(key)
            if previous is not None and previous["run_path"] != row["run_path"]:
                raise SystemExit(f"E001 registry has conflicting unit rows: {key}")
            selected[key] = row
    expected = len(EXPECTED_REUSE_MODELS) * len(SEEDS) * len(OLD_TRAIN_ENVIRONMENTS)
    if len(selected) != expected:
        raise SystemExit(f"Expected {expected} E001 reuse units, found {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e001-registry", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty core result root: {args.output_dir}")
    panel_manifest = _json(args.panel_manifest) if args.panel_manifest.suffix == ".json" else None
    if panel_manifest is not None:
        raise SystemExit("E003 panel manifest must be YAML, not JSON")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"Missing protocol freeze: {args.protocol_freeze}")
    # Loading these hashes before any result assembly binds the registry to the
    # immutable pre-score design rather than to runner-local copies.
    protocol_sha256 = sha256_path(args.protocol_freeze)
    config_sha256 = sha256_path(args.config)
    panel_sha256 = sha256_path(args.panel_manifest)
    panel_payload = load_yaml(args.panel_manifest)
    test_samplers = panel_payload.get("stages", {}).get("test", {}).get("samplers", {})
    if not isinstance(test_samplers, dict):
        raise SystemExit("E003 panel manifest lacks test sampler specifications")
    expected_test_panels = {
        environment: {
            "path": str(test_samplers[environment]["panel_path"]),
            "sha256": str(test_samplers[environment]["panel_sha256"]),
        }
        for environment in E003_ENVIRONMENTS
        if environment in test_samplers
    }
    if set(expected_test_panels) != set(E003_ENVIRONMENTS):
        raise SystemExit("E003 panel manifest test roster is incomplete")
    expected_stage_panels: dict[str, dict[str, dict[str, str]]] = {}
    for stage in ("train", "validation"):
        samplers = panel_payload.get("stages", {}).get(stage, {}).get("samplers", {})
        if not isinstance(samplers, dict):
            raise SystemExit(f"E003 panel manifest lacks {stage} sampler specifications")
        expected_stage_panels[stage] = {
            environment: {
                "path": str(samplers[environment]["panel_path"]),
                "sha256": str(samplers[environment]["panel_sha256"]),
            }
            for environment in E003_ENVIRONMENTS
            if environment in samplers
        }
        if set(expected_stage_panels[stage]) != set(E003_ENVIRONMENTS):
            raise SystemExit(f"E003 panel manifest {stage} roster is incomplete")
    registry_units = _registry_units(args.e001_registry)
    backend = _manifest_requests(
        args.backend_manifest,
        "ppi_shift.e003_backend_request_matrix.v1",
        panel_manifest=args.panel_manifest,
        panel_sha256=panel_sha256,
        config=args.config,
        config_sha256=config_sha256,
        require_config=False,
        expected_test_panels=expected_test_panels,
        expected_train_panels=expected_stage_panels["train"],
        expected_validation_panels=expected_stage_panels["validation"],
    )
    python = _manifest_requests(
        args.python_manifest,
        "ppi_shift.e003_python_request_matrix.v1",
        panel_manifest=args.panel_manifest,
        panel_sha256=panel_sha256,
        config=args.config,
        config_sha256=config_sha256,
        require_config=True,
        expected_test_panels=expected_test_panels,
        expected_train_panels=expected_stage_panels["train"],
        expected_validation_panels=expected_stage_panels["validation"],
    )
    frozen = _manifest_requests(
        args.frozen_manifest,
        "ppi_shift.e003_frozen_esm_request_matrix.v1",
        panel_manifest=args.panel_manifest,
        panel_sha256=panel_sha256,
        config=args.config,
        config_sha256=config_sha256,
        require_config=True,
        expected_test_panels=expected_test_panels,
        expected_train_panels=expected_stage_panels["train"],
        expected_validation_panels=expected_stage_panels["validation"],
    )
    reuse = _manifest_requests(
        args.reuse_manifest,
        "ppi_shift.e003_reuse_request_matrix.v1",
        panel_manifest=args.panel_manifest,
        panel_sha256=panel_sha256,
        config=args.config,
        config_sha256=config_sha256,
        require_config=False,
        expected_test_panels=expected_test_panels,
        expected_train_panels=expected_stage_panels["train"],
        expected_validation_panels=expected_stage_panels["validation"],
    )
    cells: list[dict[str, Any]] = []
    checkpoints: dict[tuple[str, int, str], dict[str, Any]] = {}
    cell_keys: set[tuple[str, int, str, str]] = set()
    unit_keys: set[tuple[str, int, str]] = set()

    for request_path, request in reuse:
        key = (request["model"], int(request["seed"]), request["train_environment"])
        if key in unit_keys:
            raise SystemExit(f"Duplicate E003 training unit: {key}")
        registry_key = key
        registry_row = registry_units.get(registry_key)
        if registry_row is None:
            raise SystemExit(f"Reuse request is not covered by E001 registry: {key}")
        unit_cells, checkpoint = _reuse_unit(
            request, request_path, Path(request["output_dir"]), registry_row
        )
        unit_keys.add(key)
        checkpoints[key] = checkpoint
        for cell in unit_cells:
            cell_key = (
                cell["model"],
                cell["seed"],
                cell["train_environment"],
                cell["test_environment"],
            )
            if cell_key in cell_keys:
                raise SystemExit(f"Duplicate E003 core cell: {cell_key}")
            cell_keys.add(cell_key)
            cells.append(cell)

    for manifest_requests, kind in ((backend, "backend"), (python, "python"), (frozen, "frozen")):
        for request_path, request in manifest_requests:
            train_environment = _request_train_environment(
                request,
                request_path,
                require_new=kind in {"backend", "frozen"},
            )
            key = (request["model"], int(request["seed"]), train_environment)
            if key in unit_keys:
                raise SystemExit(f"Duplicate E003 training unit: {key}")
            if kind in {"backend", "frozen"} and key[2] not in NEW_TRAIN_ENVIRONMENTS:
                raise SystemExit(f"New E003 request has an old train environment: {key}")
            if kind == "backend":
                unit_cells, checkpoint = _d007_unit(
                    request, request_path, Path(request["output_dir"]), train_environment
                )
            else:
                unit_cells, checkpoint = _python_unit(
                    request, request_path, Path(request["output_dir"]), train_environment
                )
            unit_keys.add(key)
            checkpoints[key] = checkpoint
            for cell in unit_cells:
                cell_key = (
                    cell["model"],
                    cell["seed"],
                    cell["train_environment"],
                    cell["test_environment"],
                )
                if cell_key in cell_keys:
                    raise SystemExit(f"Duplicate E003 core cell: {cell_key}")
                cell_keys.add(cell_key)
                cells.append(cell)

    expected_units = len(MODELS) * len(SEEDS) * len(E003_ENVIRONMENTS)
    expected_cells = expected_units * len(E003_ENVIRONMENTS)
    if {key[0] for key in unit_keys} != set(MODELS):
        raise SystemExit(
            "Core model roster mismatch: "
            f"observed={sorted({key[0] for key in unit_keys})} expected={list(MODELS)}"
        )
    if len(unit_keys) != expected_units:
        raise SystemExit(
            f"Core training-unit matrix incomplete: {len(unit_keys)} != {expected_units}"
        )
    if len(cells) != expected_cells:
        raise SystemExit(f"Core score matrix incomplete: {len(cells)} != {expected_cells}")
    if len(checkpoints) != expected_units:
        raise SystemExit(f"Checkpoint index incomplete: {len(checkpoints)} != {expected_units}")
    expected_reuse_cells = (
        len(EXPECTED_REUSE_MODELS)
        * len(SEEDS)
        * len(OLD_TRAIN_ENVIRONMENTS)
        * len(E003_ENVIRONMENTS)
    )
    actual_reuse_cells = sum(
        cell["status"] == "complete_reused_e001_checkpoint" for cell in cells
    )
    if actual_reuse_cells != expected_reuse_cells:
        raise SystemExit(f"Unexpected exact-reuse cell count: {actual_reuse_cells}")
    cells.sort(
        key=lambda cell: (
            cell["model"],
            cell["seed"],
            cell["train_environment"],
            cell["test_environment"],
        )
    )
    index_rows = [checkpoints[key] for key in sorted(checkpoints)]
    result = {
        "schema_version": "ppi_shift.e003_core_results.v1",
        "experiment_id": "E003",
        "protocol_freeze": {"path": str(args.protocol_freeze), "sha256": protocol_sha256},
        "config": {"path": str(args.config), "sha256": config_sha256},
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": panel_sha256,
            "schema_version": E003_SCHEMA_VERSION,
        },
        "e001_registry": {
            "path": str(args.e001_registry),
            "sha256": sha256_path(args.e001_registry),
        },
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "environments": list(E003_ENVIRONMENTS),
        "training_units": expected_units,
        "core_score_cells": expected_cells,
        "reused_e001v3_cells": actual_reuse_cells,
        "new_score_cells": expected_cells - actual_reuse_cells,
        "reuse_policy": (
            "exact E001v3 checkpoint/representation/train-unit reuse; "
            "E003 test scoring only"
        ),
        "cells": cells,
        "checkpoint_index": index_rows,
        "test_adaptation": False,
        "test_cropping": False,
        "status": "complete_core_matrix",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "e003_core_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "e003_checkpoint_index.json").write_text(
        json.dumps(
            {
                "schema_version": "ppi_shift.e003_checkpoint_index.v1",
                "protocol_freeze_sha256": protocol_sha256,
                "units": index_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "cells": len(cells),
                "units": len(checkpoints),
                "reused_cells": actual_reuse_cells,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
