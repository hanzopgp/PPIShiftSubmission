from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppi_shift.artifacts import canonical_json_sha256, sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS
from scripts.dispatch_e003_datasail_scores_staged import _scheduler_dependencies
from scripts.e003_staged_fanout import submit_one_missing, validate_request_entry
from scripts.postprocess_e003_core import _manifest_requests, _request_train_environment
from scripts.prepare_e003_datasail_matrix import BACKEND_COLUMNS, _partition
from scripts.submit_e003_core_staged import _parse_retry_targets, _retry_failed_entry
from scripts.submit_e003_datasail_dag import _resolve_prepared_inputs


def test_canonical_request_hash_omits_recorded_digest() -> None:
    request = {"model": "senseppi", "seed": 11, "request_sha256": "stale"}
    expected = canonical_json_sha256({"model": "senseppi", "seed": 11})

    assert canonical_json_sha256(request, omit_key="request_sha256") == expected


def test_core_request_matrix_rejects_tampered_panel_binding(tmp_path) -> None:
    panel_manifest = tmp_path / "e003_panel_manifest.yaml"
    config = tmp_path / "e003_config.yaml"
    request_path = tmp_path / "request.json"
    matrix_path = tmp_path / "matrix.json"
    panel_manifest.write_text("panel\n", encoding="utf-8")
    config.write_text("config\n", encoding="utf-8")
    panel_sha256 = sha256_path(panel_manifest)
    config_sha256 = sha256_path(config)
    expected_test_panels = {
        environment: {"path": str(panel_manifest), "sha256": panel_sha256}
        for environment in E003_ENVIRONMENTS
    }
    request = {
        "panel_manifest": str(panel_manifest),
        "panel_manifest_sha256": panel_sha256,
        "config": str(config),
        "config_sha256": config_sha256,
    }
    request["request_sha256"] = canonical_json_sha256(request)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    matrix = {
        "schema_version": "ppi_shift.e003_python_request_matrix.v1",
        "panel_manifest": str(panel_manifest),
        "panel_manifest_sha256": panel_sha256,
        "config": str(config),
        "config_sha256": config_sha256,
        "requests": [
            {
                "request": str(request_path),
                "request_sha256": request["request_sha256"],
            }
        ],
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    assert _manifest_requests(
        matrix_path,
        "ppi_shift.e003_python_request_matrix.v1",
        panel_manifest=panel_manifest,
        panel_sha256=panel_sha256,
        config=config,
        config_sha256=config_sha256,
        require_config=True,
        expected_test_panels=expected_test_panels,
    )[0][0] == request_path

    request["panel_manifest_sha256"] = "tampered"
    request["request_sha256"] = canonical_json_sha256(request, omit_key="request_sha256")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    matrix["requests"][0]["request_sha256"] = request["request_sha256"]
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(SystemExit, match="Request panel binding changed"):
        _manifest_requests(
            matrix_path,
            "ppi_shift.e003_python_request_matrix.v1",
            panel_manifest=panel_manifest,
            panel_sha256=panel_sha256,
            config=config,
            config_sha256=config_sha256,
            require_config=True,
            expected_test_panels=expected_test_panels,
        )


def test_core_backend_request_matrix_validates_partition_panel_bindings(tmp_path) -> None:
    panel_manifest = tmp_path / "e003_panel_manifest.yaml"
    matrix_path = tmp_path / "backend_matrix.json"
    request_path = tmp_path / "backend_request.json"
    panel_manifest.write_text("panel\n", encoding="utf-8")
    panel_sha256 = sha256_path(panel_manifest)
    expected_test_panels = {
        environment: {"path": str(panel_manifest), "sha256": panel_sha256}
        for environment in E003_ENVIRONMENTS
    }
    train_environment = "bernett2026_ilp_exact"
    request = {
        "schema_version": "ppi.sota_backend_request.v1",
        "train_environment": train_environment,
        "panel_contract": {
            "train_environment": train_environment,
            "validation_environment": train_environment,
            "test_environments": list(E003_ENVIRONMENTS),
        },
        "partition_summaries": {
            "train": {"panel_sha256": panel_sha256},
            "validation": {"panel_sha256": panel_sha256},
            "test": {
                "source_panels": {
                    environment: {
                        "path": str(panel_manifest),
                        "panel_sha256": panel_sha256,
                    }
                    for environment in E003_ENVIRONMENTS
                }
            },
        },
    }
    request["request_sha256"] = canonical_json_sha256(request)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    matrix = {
        "schema_version": "ppi_shift.e003_backend_request_matrix.v1",
        "panel_manifest": str(panel_manifest),
        "panel_manifest_sha256": panel_sha256,
        "requests": [
            {"request": str(request_path), "request_sha256": request["request_sha256"]}
        ],
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    assert _manifest_requests(
        matrix_path,
        "ppi_shift.e003_backend_request_matrix.v1",
        panel_manifest=panel_manifest,
        panel_sha256=panel_sha256,
        config=tmp_path / "unused.yaml",
        config_sha256="unused",
        require_config=False,
        expected_test_panels=expected_test_panels,
    )[0][0] == request_path

    request["partition_summaries"]["test"]["source_panels"]["uniform_partner"][
        "panel_sha256"
    ] = "tampered"
    request["request_sha256"] = canonical_json_sha256(request, omit_key="request_sha256")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    matrix["requests"][0]["request_sha256"] = request["request_sha256"]
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(SystemExit, match="Request test-panel binding changed"):
        _manifest_requests(
            matrix_path,
            "ppi_shift.e003_backend_request_matrix.v1",
            panel_manifest=panel_manifest,
            panel_sha256=panel_sha256,
            config=tmp_path / "unused.yaml",
            config_sha256="unused",
            require_config=False,
            expected_test_panels=expected_test_panels,
        )


def test_legacy_backend_request_resolves_train_environment_from_panel_contract(tmp_path) -> None:
    request_path = tmp_path / "seed-11__train-bernett2026_ilp_exact.json"
    request = {
        "schema_version": "ppi.sota_backend_request.v1",
        "panel_contract": {"train_environment": "bernett2026_ilp_exact"},
    }

    assert _request_train_environment(request, request_path) == "bernett2026_ilp_exact"


def test_request_train_environment_rejects_missing_or_old_environment(tmp_path) -> None:
    request_path = tmp_path / "request.json"

    with pytest.raises(SystemExit, match="valid E003 train environment"):
        _request_train_environment({}, request_path)
    with pytest.raises(SystemExit, match="valid E003 train environment"):
        _request_train_environment(
            {"train_environment": "uniform_partner"}, request_path, require_new=True
        )


def test_python_request_can_resolve_an_old_train_environment(tmp_path) -> None:
    assert _request_train_environment(
        {"train_environment": "uniform_partner"}, tmp_path / "request.json"
    ) == "uniform_partner"


def test_datasail_partition_summary_has_backend_cache_fields(tmp_path) -> None:
    rows = [{column: ("0" if column == "label" else "value") for column in BACKEND_COLUMNS}]

    summary = _partition(
        tmp_path / "partition.csv",
        rows,
        {"path": "panel.yaml", "sha256": "panel-hash"},
    )

    assert summary["input_sha256"] == summary["sha256"]
    assert summary["test_crop_allowed"] is False


def test_datasail_dag_reuses_explicit_pre_freeze_inputs(tmp_path) -> None:
    panel = tmp_path / "e003_datasail_panel_manifest.yaml"
    matrix = tmp_path / "datasail_request_manifest.json"
    panel.write_text("schema_version: ppi_shift.e003_method_fidelity.v1\n", encoding="utf-8")
    matrix.write_text("{}\n", encoding="utf-8")

    job_id, reused = _resolve_prepared_inputs(
        {"panel": "103977", "datasail_inputs": "103980"}, panel, matrix
    )

    assert job_id == "103980"
    assert reused is True


def test_datasail_dag_fails_closed_on_implicit_input_rebuild(tmp_path) -> None:
    panel = tmp_path / "e003_datasail_panel_manifest.yaml"
    matrix = tmp_path / "datasail_request_manifest.json"
    panel.write_text("prepared\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="datasail_inputs=JOB_ID"):
        _resolve_prepared_inputs({}, panel, matrix)


def test_datasail_dag_allows_new_preparation_when_inputs_are_absent(tmp_path) -> None:
    job_id, reused = _resolve_prepared_inputs(
        {"panel": "103977"},
        tmp_path / "e003_datasail_panel_manifest.yaml",
        tmp_path / "datasail_request_manifest.json",
    )

    assert job_id is None
    assert reused is False


def test_staged_request_validation_recomputes_canonical_hash(tmp_path) -> None:
    payload = {
        "schema_version": "ppi.sota_backend_request.v1",
        "model": "dscript_esm2",
        "seed": 11,
        "train_environment": "uniform_partner",
    }
    payload["request_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    entry = {
        "request": str(path),
        "request_sha256": payload["request_sha256"],
        "model": payload["model"],
        "seed": payload["seed"],
        "train_environment": payload["train_environment"],
    }
    assert validate_request_entry(entry) == entry

    payload["train_environment"] = "esm_partner_matched"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="canonical hash changed"):
        validate_request_entry(entry)


def test_staged_submit_resumes_first_missing_unit_without_duplicate(
    tmp_path, monkeypatch
) -> None:
    requests = [
        {"request": "first.json", "request_sha256": "a"},
        {"request": "second.json", "request_sha256": "b"},
    ]
    state = {
        "python": [
            {
                "index": 0,
                "job_id": "7001",
                "request": requests[0],
            },
            None,
        ]
    }
    monkeypatch.setattr("scripts.e003_staged_fanout.scheduler_state", lambda job_id: "PENDING")
    monkeypatch.setattr("scripts.e003_staged_fanout.dependency_ids", lambda job_ids: list(job_ids))
    submitted: list[dict[str, object]] = []

    def fake_submit(script, environment, dependencies, *, array_index=None):
        submitted.append(
            {
                "script": script,
                "environment": environment,
                "dependencies": dependencies,
                "array_index": array_index,
            }
        )
        return "7002"

    monkeypatch.setattr("scripts.e003_staged_fanout.submit_job", fake_submit)
    result = submit_one_missing(
        state=state,
        phase="python",
        requests=requests,
        script=tmp_path / "launch.slurm",
        dependencies=["6001"],
        action="python_training_unit",
        array=True,
        environment_for_request=lambda request: {"REQUEST": str(request["request"])},
    )

    assert result is not None
    assert result["job_id"] == "7002"
    assert state["python"][0]["job_id"] == "7001"
    assert state["python"][1]["job_id"] == "7002"
    assert submitted == [
        {
            "script": tmp_path / "launch.slurm",
            "environment": {"REQUEST": "second.json"},
            "dependencies": ["6001"],
            "array_index": 1,
        }
    ]


def test_staged_core_retry_preserves_failed_job_and_request(monkeypatch, tmp_path) -> None:
    request = {
        "model": "rapppid",
        "seed": 11,
        "train_environment": "uniform_partner",
        "request": str(tmp_path / "request.json"),
        "request_sha256": "request-hash",
    }
    state = {
        "python": [{"index": 0, "job_id": "8001", "request": request}],
        "retry_history": [],
    }
    monkeypatch.setattr("scripts.submit_e003_core_staged._state", lambda job_id: "FAILED")
    submitted: list[dict[str, object]] = []

    def fake_submit(script, environment, dependencies, *, array=None):
        submitted.append(
            {
                "script": script,
                "environment": environment,
                "dependencies": dependencies,
                "array": array,
            }
        )
        return "8002"

    monkeypatch.setattr("scripts.submit_e003_core_staged._submit", fake_submit)
    replacement = _retry_failed_entry(
        state=state,
        phase="python",
        index=0,
        requests=[request],
        script=tmp_path / "launch.slurm",
        common={"PPI_SHIFT_REPO": "/repo"},
        dependencies=["7000"],
        action="python_training_unit",
        array=True,
        reason="diagnosed modern PyTorch parameter-registration failure",
    )

    assert replacement["job_id"] == "8002"
    assert replacement["retry_of"] == "8001"
    assert state["python"][0]["job_id"] == "8002"
    assert state["retry_history"][0]["failed_job_id"] == "8001"
    assert state["retry_history"][0]["replacement_job_id"] == "8002"
    assert submitted == [
        {
            "script": tmp_path / "launch.slurm",
            "environment": {"PPI_SHIFT_REPO": "/repo"},
            "dependencies": ["7000"],
            "array": "0",
        }
    ]


def test_staged_core_retry_target_parser_is_explicit() -> None:
    assert _parse_retry_targets(["python:0", "frozen:5"]) == [("python", 0), ("frozen", 5)]
    with pytest.raises(SystemExit, match="Invalid --retry-failed"):
        _parse_retry_targets(["python=0"])


def test_datasail_resume_can_keep_provenance_without_stale_scheduler_dependency() -> None:
    class Args:
        prepare_job = "103980"
        no_prepare_dependency = True

    assert _scheduler_dependencies(Args()) == []


def test_datasail_default_scheduler_dependency_remains_fail_closed() -> None:
    class Args:
        prepare_job = "103980"
        no_prepare_dependency = False

    assert _scheduler_dependencies(Args()) == ["103980"]


def test_downstream_dispatchers_are_wired_to_staged_submitters() -> None:
    root = Path(__file__).parents[1] / "scripts"
    for name, staged_name in (
        ("launch_e003_datasail_dispatch.slurm", "dispatch_e003_datasail_scores_staged.py"),
        ("launch_e003_external_dispatch.slurm", "dispatch_e003_external_scores_staged.py"),
    ):
        source = (root / name).read_text(encoding="utf-8")
        assert staged_name in source
        assert "--state-manifest" in source
        assert "--max-new-jobs 1" in source
    assert "--protocol-freeze" in (
        root / "launch_e003_external_dispatch.slurm"
    ).read_text(encoding="utf-8")
