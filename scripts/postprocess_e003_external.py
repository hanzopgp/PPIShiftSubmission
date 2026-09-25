#!/usr/bin/env python3
"""Verify and summarize all frozen E003 external score bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import canonical_json_sha256, iter_delimited, sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS
from ppi_shift.metrics import compute_binary_metrics, external_prevalence_metrics

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


def _expected_external_training_units() -> int:
    return len(CORE_MODELS) * len(SEEDS) * len(E003_ENVIRONMENTS)


def _validate_request_matrix(payload: dict[str, Any]) -> list[dict[str, Any]]:
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise SystemExit("E003 external request matrix does not contain a request list")
    expected_units = _expected_external_training_units()
    declared_units = payload.get("units")
    if declared_units != expected_units or len(requests) != expected_units:
        raise SystemExit(
            f"Expected {expected_units} external checkpoint bundles, "
            f"declared={declared_units} found={len(requests)}"
        )
    return requests


def _validate_bundle_request_binding(
    bundle_request: Any, expected_request: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(bundle_request, dict):
        raise SystemExit("External bundle request binding is missing")
    observed_sha256 = str(bundle_request.get("sha256", ""))
    expected_sha256 = str(expected_request.get("request_sha256", ""))
    expected_path = Path(str(expected_request.get("request", ""))).resolve()
    observed_path = Path(str(bundle_request.get("path", ""))).resolve()
    if observed_path == expected_path:
        expected_file_sha256 = sha256_path(expected_path)
        if observed_sha256 == expected_file_sha256:
            return {
                "mode": "exact_external_unit_request",
                "path": str(observed_path),
                "sha256": observed_sha256,
                "hash_semantics": "sha256_file_bytes",
            }
        request_payload = _json(expected_path)
        if (
            observed_sha256 == expected_sha256
            and request_payload.get("request_sha256") == expected_sha256
            and canonical_json_sha256(request_payload, omit_key="request_sha256")
            == expected_sha256
        ):
            return {
                "mode": "exact_external_unit_request",
                "path": str(observed_path),
                "sha256": observed_sha256,
                "hash_semantics": "canonical_json_without_request_sha256",
            }
        raise SystemExit(f"External bundle request binding changed: {observed_path}")

    legacy = _json(observed_path)
    legacy_contract = legacy.get("panel_contract")
    legacy_train_environment = (
        legacy.get("train_environment")
        if isinstance(legacy.get("train_environment"), str)
        else legacy_contract.get("train_environment")
        if isinstance(legacy_contract, dict)
        else None
    )
    if (
        legacy.get("schema_version") != "ppi.sota_backend_request.v1"
        or legacy.get("experiment_id") != "E003"
        or legacy.get("model") != expected_request.get("model")
        or int(legacy.get("seed", -1)) != int(expected_request.get("seed", -2))
        or legacy_train_environment != expected_request.get("train_environment")
        or not isinstance(legacy_contract, dict)
        or set(legacy_contract.get("test_environments", [])) != set(E003_ENVIRONMENTS)
        or legacy.get("test_adaptation") is not False
        or legacy.get("test_checkpoint_reuse_across_environments") is not True
        or legacy.get("test_crop_allowed") is not False
        or legacy.get("selection_status")
        != "validation_auprc_only; validation_max_mcc_threshold; no test adaptation"
        or not observed_sha256
        or observed_sha256
        != canonical_json_sha256(legacy, omit_key="request_sha256")
        or legacy.get("request_sha256") != observed_sha256
    ):
        raise SystemExit(f"External bundle request binding changed: {observed_path}")
    return {
        "mode": "validated_legacy_backend_request",
        "path": str(observed_path),
        "sha256": observed_sha256,
        "legacy_schema_version": legacy.get("schema_version"),
    }


def _balanced_indices(rows: list[dict[str, str]]) -> list[int]:
    positive = [index for index, row in enumerate(rows) if row["label"] == "1"]
    negative = [index for index, row in enumerate(rows) if row["label"] == "0"]
    count = min(len(positive), len(negative))

    def order(index: int) -> str:
        return hashlib.sha256(rows[index]["row_id"].encode("utf-8")).hexdigest()

    selected_positive = sorted(positive, key=order)[:count]
    selected_negative = sorted(negative, key=order)[:count]
    return sorted(selected_positive + selected_negative)


def _surface_metrics_for_audit(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    threshold: float,
    *,
    prevalence_sensitivity_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    result: dict[str, Any] = {
        "natural_prevalence": external_prevalence_metrics(
            labels,
            scores,
            threshold=threshold,
            threshold_source="corresponding_Gold_validation_max_MCC",
        )
    }
    indices = _balanced_indices(rows)
    balanced_rows = [rows[index] for index in indices]
    result["balanced_posthoc_diagnostic"] = {
        "metrics": compute_binary_metrics(
            [int(row["label"]) for row in balanced_rows],
            scores[indices],
            threshold=threshold,
            threshold_source="corresponding_Gold_validation_max_MCC",
        ),
        "selection": {
            "role": "post_hoc_balanced_sensitivity_only",
            "row_order": "sha256(row_id) within each label",
            "positive_rows": sum(row["label"] == "1" for row in balanced_rows),
            "negative_rows": sum(row["label"] == "0" for row in balanced_rows),
            "discarded_rows": len(rows) - len(balanced_rows),
        },
    }
    if prevalence_sensitivity_seeds:
        positive = np.asarray(
            [index for index, row in enumerate(rows) if row["label"] == "1"], dtype=np.int64
        )
        negative = np.asarray(
            [index for index, row in enumerate(rows) if row["label"] == "0"], dtype=np.int64
        )
        if len(positive) < len(negative) or not len(negative):
            raise SystemExit("IntAct post-hoc sensitivity requires at least as many positives")
        replicates: list[dict[str, Any]] = []
        for seed in prevalence_sensitivity_seeds:
            rng = np.random.default_rng(seed)
            selected_positive = rng.choice(positive, size=len(negative), replace=False)
            selected = np.concatenate((selected_positive, negative))
            replicates.append(
                {
                    "seed": int(seed),
                    "positive_rows": int(len(selected_positive)),
                    "negative_rows": int(len(negative)),
                    "metrics": compute_binary_metrics(
                        [int(rows[index]["label"]) for index in selected],
                        scores[selected],
                        threshold=threshold,
                        threshold_source="corresponding_Gold_validation_max_MCC",
                    ),
                }
            )
        result["balanced_prevalence_sensitivity"] = {
            "role": "post_hoc_prevalence_sensitivity_only",
            "sampling": "all explicit negatives plus positives sampled without replacement",
            "seed_set": [int(seed) for seed in prevalence_sensitivity_seeds],
            "positive_rows": int(len(positive)),
            "negative_rows": int(len(negative)),
            "replicates": replicates,
        }
    return result


def _assert_payload_matches(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise SystemExit(f"External metric structure changed: {label}")
        for key in expected:
            _assert_payload_matches(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise SystemExit(f"External metric list changed: {label}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
            _assert_payload_matches(actual_value, expected_value, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=5e-12
        ):
            raise SystemExit(f"External metric mismatch: {label}")
        return
    if actual != expected:
        raise SystemExit(f"External metric mismatch: {label}")


def _audit_surface(
    surface: dict[str, Any],
    spec: dict[str, Any],
    *,
    threshold: float,
    prevalence_sensitivity_seeds: tuple[int, ...] = (),
) -> None:
    panel_path = Path(str(spec["path"]))
    expected_panel_sha = str(spec["sha256"])
    if sha256_path(panel_path) != expected_panel_sha:
        raise SystemExit(f"External panel hash changed: {panel_path}")
    panel_rows = list(iter_delimited(panel_path))
    panel_metadata = surface.get("panel", {})
    if (
        panel_metadata.get("sha256") != expected_panel_sha
        or int(panel_metadata.get("rows", -1)) != len(panel_rows)
    ):
        raise SystemExit(f"External panel provenance changed: {panel_path}")

    prediction = surface.get("prediction", {})
    prediction_path = Path(str(prediction.get("path", "")))
    if sha256_path(prediction_path) != prediction.get("sha256"):
        raise SystemExit(f"External prediction hash changed: {prediction_path}")
    prediction_rows = list(iter_delimited(prediction_path))
    by_id: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        row_id = row.get("row_id", "")
        if not row_id or row_id in by_id:
            raise SystemExit(f"External prediction IDs are not unique: {prediction_path}")
        by_id[row_id] = row
    if set(by_id) != {row["row_id"] for row in panel_rows}:
        raise SystemExit(f"External prediction roster changed: {prediction_path}")
    scores: list[float] = []
    for panel_row in panel_rows:
        prediction_row = by_id[panel_row["row_id"]]
        if prediction_row.get("label") != panel_row.get("label"):
            raise SystemExit(f"External prediction labels changed: {prediction_path}")
        scores.append(float(prediction_row["score"]))
    expected = _surface_metrics_for_audit(
        panel_rows,
        np.asarray(scores, dtype=np.float64),
        threshold,
        prevalence_sensitivity_seeds=prevalence_sensitivity_seeds,
    )
    _assert_payload_matches(surface.get("metrics"), expected, str(prediction_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--core-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requests_payload = _json(args.request_manifest)
    external = _json(args.external_manifest)
    core = _json(args.core_results)
    if core.get("status") != "complete_core_matrix":
        raise SystemExit("Core results are not complete")
    if requests_payload.get("schema_version") != "ppi_shift.e003_external_request_matrix.v1":
        raise SystemExit("Unexpected E003 external request manifest schema")
    requests = _validate_request_matrix(requests_payload)
    expected_external_sha = sha256_path(args.external_manifest)
    if requests_payload.get("external_manifest", {}).get("sha256") != expected_external_sha:
        raise SystemExit("External request matrix is not bound to the supplied external manifest")
    core_index_path = Path(str(requests_payload.get("core_index_path", "")))
    expected_core_index_sha = sha256_path(core_index_path)
    if requests_payload.get("core_index_sha256") != expected_core_index_sha:
        raise SystemExit("External request matrix checkpoint-index hash changed")
    expected_surfaces = {
        f"{source}__{surface}"
        for source, info in external["sources"].items()
        for surface in info["surfaces"]
    }
    bundles: list[dict[str, Any]] = []
    request_binding_audit: list[dict[str, Any]] = []
    for request in requests:
        request_path = Path(request["request"])
        request_payload = _json(request_path)
        request_sha256 = str(request_payload.get("request_sha256", ""))
        if request_sha256 != request.get("request_sha256"):
            raise SystemExit(f"External request hash changed: {request_path}")
        if (
            not request_sha256
            or canonical_json_sha256(request_payload, omit_key="request_sha256")
            != request_sha256
        ):
            raise SystemExit(f"External request canonical hash is invalid: {request_path}")
        if request_payload.get("external_manifest_sha256") != expected_external_sha:
            raise SystemExit(f"External request manifest binding changed: {request_path}")
        if request_payload.get("core_index_sha256") != expected_core_index_sha:
            raise SystemExit(f"External request checkpoint-index binding changed: {request_path}")
        path = Path(request["output_dir"]) / "external_run_manifest.json"
        bundle = _json(path)
        if bundle.get("status") != "complete":
            raise SystemExit(f"External bundle is not complete: {path}")
        if (
            bundle.get("model") != request.get("model")
            or bundle.get("seed") != request.get("seed")
            or bundle.get("train_environment") != request.get("train_environment")
        ):
            raise SystemExit(f"External bundle unit identity changed: {path}")
        if set(bundle.get("surfaces", {})) != expected_surfaces:
            raise SystemExit(f"External surface roster changed in {path}")
        request_binding_audit.append(
            {
                "model": request["model"],
                "seed": request["seed"],
                "train_environment": request["train_environment"],
                **_validate_bundle_request_binding(bundle.get("request"), request),
            }
        )
        if bundle.get("external_manifest", {}).get("sha256") != expected_external_sha:
            raise SystemExit(f"External bundle manifest binding changed: {path}")
        if bundle.get("no_external_retraining") is not True or bundle.get(
            "no_external_selection"
        ) is not True:
            raise SystemExit(f"External bundle violates score-only contract: {path}")
        if bundle.get("threshold_source") != "corresponding_Gold_validation_max_MCC":
            raise SystemExit(f"External bundle threshold provenance changed: {path}")
        threshold = float(bundle["checkpoint"]["threshold"]["threshold"])
        for surface_key, surface in bundle["surfaces"].items():
            source, separator, surface_name = surface_key.partition("__")
            if not separator or source not in external["sources"]:
                raise SystemExit(f"Unknown external surface key: {surface_key}")
            spec = external["sources"][source]["surfaces"].get(surface_name)
            if not isinstance(spec, dict):
                raise SystemExit(f"External surface manifest entry is missing: {surface_key}")
            _audit_surface(
                surface,
                spec,
                threshold=threshold,
                prevalence_sensitivity_seeds=(11, 22, 33)
                if source == "intact" and surface_name == "measured"
                else (),
            )
        if bundle.get("core_index", {}).get("sha256") != sha256_path(
            Path(bundle["core_index"]["path"])
        ):
            raise SystemExit(f"Checkpoint index hash changed in {path}")
        bundles.append(bundle)
    if not bundles:
        raise SystemExit("No external bundles were found")
    result = {
        "schema_version": "ppi_shift.e003_external_results.v1",
        "experiment_id": "E003",
        "request_manifest": {
            "path": str(args.request_manifest),
            "sha256": sha256_path(args.request_manifest),
        },
        "external_manifest": {
            "path": str(args.external_manifest),
            "sha256": sha256_path(args.external_manifest),
        },
        "core_results": {"path": str(args.core_results), "sha256": sha256_path(args.core_results)},
        "training_units": len(bundles),
        "surface_count": len(expected_surfaces),
        "surface_roster": sorted(expected_surfaces),
        "bundles": bundles,
        "request_binding_audit": {
            "modes": {
                mode: sum(item["mode"] == mode for item in request_binding_audit)
                for mode in sorted({item["mode"] for item in request_binding_audit})
            },
            "entries": request_binding_audit,
        },
        "no_external_retraining": True,
        "no_external_selection": True,
        "status": "complete_external_matrix",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "e003_external_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result": str(path), "bundles": len(bundles), "surfaces": len(expected_surfaces)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
