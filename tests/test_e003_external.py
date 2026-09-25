from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppi_shift.artifacts import canonical_json_sha256, sha256_path
from ppi_shift.e003_method_fidelity import BERNETT_ENVIRONMENT, E003_ENVIRONMENTS
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    ExclusionRegistry,
    build_candidate_universe,
)
from scripts.audit_e003_external_panels import main as audit_external_panels
from scripts.materialize_e003_external_panels import (
    _global_rows,
    _positive_pairs,
    _uniform_rows,
)
from scripts.postprocess_e003_external import (
    _audit_surface,
    _expected_external_training_units,
    _validate_bundle_request_binding,
    _validate_request_matrix,
)
from scripts.run_e003_external_scores import _surface_metrics


def _registry(tmp_path: Path) -> ExclusionRegistry:
    path = tmp_path / "registry.csv"
    path.write_text("identifier_pair_key,sequence_pair_key\n", encoding="utf-8")
    return ExclusionRegistry(
        name="toy",
        path=path,
        sha256="0" * 64,
        identifier_pairs=frozenset(),
        sequence_pairs=frozenset(),
    )


def _rows() -> list[dict[str, str]]:
    return [
        {
            "left_id": "P1",
            "right_id": "P2",
            "left_sequence": "AAAA",
            "right_sequence": "CCCC",
            "left_sequence_sha256": "",
            "right_sequence_sha256": "",
            "label": "1",
            "row_id": "positive-1",
            "source_assertion_id": "assertion-1",
            "evidence_count": "1",
        },
        {
            "left_id": "P3",
            "right_id": "P4",
            "left_sequence": "DDDD",
            "right_sequence": "EEEE",
            "left_sequence_sha256": "",
            "right_sequence_sha256": "",
            "label": "1",
            "row_id": "positive-2",
            "source_assertion_id": "assertion-2",
            "evidence_count": "1",
        },
    ]


def test_external_postprocess_uses_the_frozen_core4_request_roster() -> None:
    assert _expected_external_training_units() == 72
    requests = [{"request": str(index)} for index in range(72)]
    assert _validate_request_matrix({"units": 72, "requests": requests}) == requests
    with pytest.raises(SystemExit, match="Expected 72 external checkpoint bundles"):
        _validate_request_matrix({"units": 90, "requests": [{"request": "stale"}] * 90})


def test_external_postprocess_accepts_raw_file_hash_for_exact_request_binding(
    tmp_path: Path,
) -> None:
    request_payload = {"schema_version": "ppi_shift.e003_external_unit_request.v1"}
    request_payload["request_sha256"] = canonical_json_sha256(
        request_payload, omit_key="request_sha256"
    )
    request_path = tmp_path / "external_request.json"
    request_path.write_text(json.dumps(request_payload), encoding="utf-8")
    binding = _validate_bundle_request_binding(
        {"path": str(request_path), "sha256": sha256_path(request_path)},
        {"request": str(request_path), "request_sha256": request_payload["request_sha256"]},
    )
    assert binding["mode"] == "exact_external_unit_request"
    assert binding["hash_semantics"] == "sha256_file_bytes"


def test_external_postprocess_records_only_validated_legacy_backend_binding(
    tmp_path: Path,
) -> None:
    legacy = {
        "schema_version": "ppi.sota_backend_request.v1",
        "experiment_id": "E003",
        "model": "dscript_esm2",
        "seed": 11,
        "panel_contract": {
            "train_environment": "uniform_partner",
            "test_environments": list(E003_ENVIRONMENTS),
        },
        "test_adaptation": False,
        "test_checkpoint_reuse_across_environments": True,
        "test_crop_allowed": False,
        "selection_status": (
            "validation_auprc_only; validation_max_mcc_threshold; no test adaptation"
        ),
    }
    legacy["request_sha256"] = canonical_json_sha256(legacy, omit_key="request_sha256")
    legacy_path = tmp_path / "legacy_backend_request.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    binding = _validate_bundle_request_binding(
        {"path": str(legacy_path), "sha256": legacy["request_sha256"]},
        {
            "request": str(tmp_path / "external_request.json"),
            "request_sha256": "external-request-hash",
            "model": "dscript_esm2",
            "seed": 11,
            "train_environment": "uniform_partner",
        },
    )
    assert binding["mode"] == "validated_legacy_backend_request"


def test_external_baseline_is_new_deterministic_surface(tmp_path: Path) -> None:
    carrier_rows = _rows()
    positives, carriers = _positive_pairs(carrier_rows, "toy")
    universe = build_candidate_universe(
        positives,
        _registry(tmp_path),
        stage="external_toy",
        schema_version=SCHEMA_VERSION_V3,
    )
    rows, diagnostics = _uniform_rows("toy", positives, carriers, universe, seed=1729)
    assert len(rows) == 4
    assert diagnostics["method"].startswith("E003_")
    assert {row["semantic_class"] for row in rows if row["label"] == "0"} == {
        "synthetic_unobserved_pair"
    }


def test_external_global_adapter_does_not_anchor_pair_to_positive(tmp_path: Path) -> None:
    positives, carriers = _positive_pairs(_rows(), "toy")
    universe = build_candidate_universe(
        positives,
        _registry(tmp_path),
        stage="external_toy",
        schema_version=SCHEMA_VERSION_V3,
    )
    pairs = [("P1", "P3"), ("P2", "P4")]
    rows = _global_rows(
        "toy",
        BERNETT_ENVIRONMENT,
        positives,
        carriers,
        universe,
        pairs,
        seed=1729,
    )
    negatives = [row for row in rows if row["label"] == "0"]
    assert len(negatives) == len(positives)
    assert {row["surface"] for row in negatives} == {BERNETT_ENVIRONMENT}
    assert all(row["semantic_class"] == "synthetic_unobserved_pair" for row in negatives)
    positive_pairs = {frozenset((positive.left_id, positive.right_id)) for positive in positives}
    assert all(
        frozenset((row["left_id"], row["right_id"])) not in positive_pairs for row in negatives
    )


def test_intact_posthoc_prevalence_sensitivity_is_seeded_without_selection() -> None:
    rows = [{"row_id": f"p{index}", "label": "1"} for index in range(4)] + [
        {"row_id": f"n{index}", "label": "0"} for index in range(2)
    ]
    result = _surface_metrics(
        rows,
        np.asarray([0.9, 0.8, 0.7, 0.6, 0.4, 0.3]),
        threshold=0.5,
        prevalence_sensitivity_seeds=(11, 22),
    )
    sensitivity = result["balanced_prevalence_sensitivity"]
    assert sensitivity["seed_set"] == [11, 22]
    assert sensitivity["positive_rows"] == 4
    assert sensitivity["negative_rows"] == 2
    assert len(sensitivity["replicates"]) == 2
    assert all(item["positive_rows"] == 2 for item in sensitivity["replicates"])


def test_external_postprocess_recomputes_hash_bound_surface_metrics(tmp_path: Path) -> None:
    panel_rows = [
        {"row_id": "p1", "label": "1"},
        {"row_id": "p2", "label": "1"},
        {"row_id": "n1", "label": "0"},
    ]
    panel_path = tmp_path / "panel.csv"
    panel_path.write_text(
        "row_id,label\n"
        + "\n".join(f"{row['row_id']},{row['label']}" for row in panel_rows)
        + "\n",
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.csv"
    prediction_path.write_text(
        "row_id,label,score\n"
        "p1,1,0.9\n"
        "p2,1,0.8\n"
        "n1,0,0.2\n",
        encoding="utf-8",
    )
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    prediction_sha = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    scores = np.asarray([0.9, 0.8, 0.2])
    surface = {
        "panel": {"sha256": panel_sha, "rows": len(panel_rows)},
        "prediction": {"path": str(prediction_path), "sha256": prediction_sha},
        "metrics": _surface_metrics(panel_rows, scores, 0.5),
    }

    _audit_surface(
        surface,
        {"path": str(panel_path), "sha256": panel_sha},
        threshold=0.5,
    )


def test_external_panel_audit_rechecks_source_semantics(tmp_path: Path, monkeypatch) -> None:
    panels: dict[str, dict[str, dict[str, str]]] = {}
    rows_by_surface = {
        ("biogrid", "uniform_partner"): [
            ("b-pos", "1", "curated_positive_reference"),
            ("b-neg", "0", "synthetic_unobserved_pair"),
        ],
        ("string", "bernett2026_ilp_exact"): [
            ("s-pos", "1", "curated_positive_reference"),
            ("s-neg", "0", "synthetic_unobserved_pair"),
        ],
        ("intact", "measured"): [
            ("i-pos", "1", "physical_positive"),
            ("i-neg", "0", "explicit_assay_negative"),
        ],
        ("vidal", "measured"): [
            ("v-pos", "1", "binary_assay_positive"),
            ("v-neg", "0", "assay_non_detection"),
        ],
    }
    for (source, surface), rows in rows_by_surface.items():
        path = tmp_path / f"{source}_{surface}.csv"
        path.write_text(
            "row_id,label,semantic_class\n"
            + "\n".join(",".join(row) for row in rows)
            + "\n",
            encoding="utf-8",
        )
        panels.setdefault(source, {})[surface] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows),
            "label_counts": {"0": 1, "1": 1},
        }
    manifest = tmp_path / "external_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ppi_shift.e003_external_panel.v1",
                "historical_e002_synthetic_rows_read": False,
                "no_external_retraining": True,
                "no_external_selection": True,
                "sources": {
                    source: {"surfaces": surfaces} for source, surfaces in panels.items()
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_e003_external_panels.py", "--manifest", str(manifest), "--output", str(output)],
    )

    assert audit_external_panels() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
