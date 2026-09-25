from __future__ import annotations

import hashlib
import json

import pytest

from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS
from scripts.freeze_e003_protocol import (
    CORE_MODEL_ROSTER,
    SENSE_CACHE_EMBEDDING_DIMENSION,
    SENSE_CACHE_MAXIMUM_LENGTH,
    SENSE_CACHE_MODEL,
    SENSE_CACHE_REVISION,
    SENSE_CACHE_SHARD_COUNT,
    _panel_hashes,
    _validate_core_model_config,
    _validate_sense_cache_registry_manifest,
    _validate_sense_cache_summaries,
)


def _write_cache_summaries(root, *, mutate=None):
    registry_path = str(root / "sequences.fasta")
    for index in range(SENSE_CACHE_SHARD_COUNT):
        payload = {
            "model_name": SENSE_CACHE_MODEL,
            "model_revision": SENSE_CACHE_REVISION,
            "maximum_sequence_length": SENSE_CACHE_MAXIMUM_LENGTH,
            "shard_index": index,
            "shard_count": SENSE_CACHE_SHARD_COUNT,
            "dtype": "float16",
            "special_tokens_excluded": True,
            "embedding_dimensions": [SENSE_CACHE_EMBEDDING_DIMENSION],
            "assigned_records": 1,
            "registry_path": registry_path,
            "registry_sha256": "a" * 64,
        }
        if mutate is not None:
            mutate(index, payload)
        (root / f"summary-shard-{index:03d}-of-{SENSE_CACHE_SHARD_COUNT:03d}.json").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )


def test_freeze_cache_contract_accepts_all_expected_shards(tmp_path):
    _write_cache_summaries(tmp_path)

    summaries = _validate_sense_cache_summaries(tmp_path)

    assert len(summaries) == SENSE_CACHE_SHARD_COUNT


def test_freeze_cache_contract_rejects_missing_shard(tmp_path):
    _write_cache_summaries(tmp_path)
    (tmp_path / "summary-shard-007-of-016.json").unlink()

    with pytest.raises(SystemExit, match="missing"):
        _validate_sense_cache_summaries(tmp_path)


def test_freeze_cache_contract_rejects_model_mismatch(tmp_path):
    def mutate(index, payload):
        if index == 3:
            payload["model_revision"] = "wrong"

    _write_cache_summaries(tmp_path, mutate=mutate)

    with pytest.raises(SystemExit, match="representation contract"):
        _validate_sense_cache_summaries(tmp_path)


def test_freeze_binds_registry_manifest_to_cache_summaries(tmp_path):
    registry = tmp_path / "sense_cache_sequences.fasta"
    registry.write_text(">" + "a" * 64 + "\nMKT\n", encoding="utf-8")
    _write_cache_summaries(tmp_path)
    registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
    for path in tmp_path.glob("summary-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["registry_path"] = str(registry)
        payload["registry_sha256"] = registry_hash
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest = tmp_path / "registry.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "fasta_path": str(registry),
                "fasta_sha256": registry_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = sorted(tmp_path.glob("summary-*.json"))
    result = _validate_sense_cache_registry_manifest(summaries, manifest)

    assert result["fasta_path"] == str(registry)
    assert result["fasta_sha256"] == registry_hash


def test_freeze_rejects_registry_hash_mismatch(tmp_path):
    registry = tmp_path / "sense_cache_sequences.fasta"
    registry.write_text(">" + "a" * 64 + "\nMKT\n", encoding="utf-8")
    _write_cache_summaries(tmp_path)
    manifest = tmp_path / "registry.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "fasta_path": str(registry),
                "fasta_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="registry manifest"):
        _validate_sense_cache_registry_manifest(
            sorted(tmp_path.glob("summary-*.json")), manifest
        )


def test_freeze_records_hashes_for_all_panel_stages() -> None:
    panel = {
        "stages": {
            stage: {
                "samplers": {
                    environment: {"panel_sha256": f"{stage}-{environment}"}
                    for environment in E003_ENVIRONMENTS
                }
            }
            for stage in ("train", "validation", "test")
        }
    }

    assert set(_panel_hashes(panel)["train"]) == set(E003_ENVIRONMENTS)
    assert _panel_hashes(panel)["test"]["uniform_partner"] == "test-uniform_partner"


def test_freeze_rejects_panel_stage_without_hash() -> None:
    panel = {
        "stages": {
            stage: {
                "samplers": {
                    environment: {"panel_sha256": f"{stage}-{environment}"}
                    for environment in E003_ENVIRONMENTS
                }
            }
            for stage in ("train", "validation", "test")
        }
    }
    del panel["stages"]["test"]["samplers"]["uniform_partner"]["panel_sha256"]

    with pytest.raises(SystemExit, match="panel_sha256"):
        _panel_hashes(panel)


def test_core_freeze_requires_exact_four_model_roster() -> None:
    models, environments = _validate_core_model_config(
        {
            "models": {
                "required": list(CORE_MODEL_ROSTER),
                "excluded_from_core": {
                    "senseppi": {"status": "optional_future_supplementary"}
                },
                "source_revisions": {},
                "fidelity": {},
            },
            "negative_environments": {"core": list(E003_ENVIRONMENTS)},
            "matrix": {
                "models": 4,
                "train_environments": 6,
                "test_environments": 6,
                "seeds": [11, 22, 33],
                "total_core_cells": 432,
                "training_units": 72,
            },
        }
    )

    assert models["required"] == list(CORE_MODEL_ROSTER)
    assert tuple(environments["core"]) == E003_ENVIRONMENTS


def test_core_freeze_rejects_sense_as_required_model() -> None:
    with pytest.raises(SystemExit, match="Core-4 model roster mismatch"):
        _validate_core_model_config(
            {
                "models": {
                    "required": [*CORE_MODEL_ROSTER[:-1], "senseppi"],
                    "excluded_from_core": {
                        "senseppi": {"status": "optional_future_supplementary"}
                    },
                    "source_revisions": {},
                    "fidelity": {},
                },
                "negative_environments": {"core": list(E003_ENVIRONMENTS)},
                "matrix": {
                    "models": 4,
                    "train_environments": 6,
                    "test_environments": 6,
                    "seeds": [11, 22, 33],
                    "total_core_cells": 432,
                    "training_units": 72,
                },
            }
        )
