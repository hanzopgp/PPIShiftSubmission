from __future__ import annotations

import csv
import hashlib
import json

import pytest
import yaml

from ppi_shift.data import sequence_sha256
from scripts.prepare_e003_sense_cache_registry import build_registry


def _panel_manifest(tmp_path, *, long_sequence: bool = False):
    panels = {}
    for stage in ("train", "validation", "test"):
        path = tmp_path / f"{stage}.csv"
        sequence = "ABCDE" if long_sequence else "MKT"
        rows = [
            {
                "interaction_label": "1",
                "seq1": "ACDE",
                "seq2": sequence,
                "anchor_sequence_sha256": sequence_sha256("ACDE"),
                "true_partner_sequence_sha256": sequence_sha256(sequence),
                "anchor_id": "P11111",
                "true_partner_id": "P22222",
            }
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        panels[stage] = {
            "samplers": {
                "uniform_partner": {
                    "panel_path": str(path),
                    "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
        }
    manifest = tmp_path / "panel.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "E003",
                "core_environments": ["uniform_partner"],
                "stages": panels,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_build_registry_uses_hash_ids_and_frozen_panel_sequences(tmp_path):
    panel_manifest = _panel_manifest(tmp_path)
    output = tmp_path / "cache.fasta"
    manifest_path = tmp_path / "registry.json"

    manifest = build_registry(panel_manifest, output, manifest_path)

    expected = sorted(
        (
            (sequence_sha256("ACDE"), "ACDE"),
            (sequence_sha256("MKT"), "MKT"),
        )
    )
    assert output.read_text(encoding="utf-8") == "".join(
        f">{digest}\n{sequence}\n" for digest, sequence in expected
    )
    assert manifest["panel_rows"] == 3
    assert manifest["sequence_count"] == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete"


def test_build_registry_rejects_overlength_panel_sequence(tmp_path):
    panel_manifest = _panel_manifest(tmp_path, long_sequence=True)

    with pytest.raises(ValueError, match="maximum 4"):
        build_registry(
            panel_manifest,
            tmp_path / "cache.fasta",
            tmp_path / "registry.json",
            maximum_length=4,
        )


def test_build_registry_refuses_overwrite(tmp_path):
    panel_manifest = _panel_manifest(tmp_path)
    output = tmp_path / "cache.fasta"
    manifest = tmp_path / "registry.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_registry(panel_manifest, output, manifest)
