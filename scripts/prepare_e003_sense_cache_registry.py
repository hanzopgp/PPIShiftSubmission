"""Build the sequence-hash FASTA required by the pinned E003 SENSE cache."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ppi_shift.artifacts import iter_delimited, sha256_path, write_fasta
from ppi_shift.data import normalize_sequence, sequence_sha256


def build_registry(
    panel_manifest: Path,
    output_fasta: Path,
    output_manifest: Path,
    *,
    maximum_length: int = 512,
) -> dict[str, Any]:
    """Create a deterministic hash-keyed registry from frozen panel rows."""

    if output_fasta.exists() or output_manifest.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing E003 SENSE cache registry: "
            f"{output_fasta}, {output_manifest}"
        )
    if maximum_length < 1:
        raise ValueError(f"maximum_length must be positive: {maximum_length}")

    manifest = yaml.safe_load(panel_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("experiment_id") != "E003":
        raise ValueError("Expected an E003 panel manifest")
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("E003 panel manifest lacks stages")
    declared_core_environments = {
        str(value) for value in manifest.get("core_environments", [])
    }
    if not declared_core_environments:
        raise ValueError("E003 panel manifest lacks core_environments")

    sequences: dict[str, str] = {}
    source_ids_by_hash: dict[str, set[str]] = defaultdict(set)
    panels: list[dict[str, Any]] = []
    panel_rows = 0
    for stage in ("train", "validation", "test"):
        stage_spec = stages.get(stage)
        if not isinstance(stage_spec, dict):
            raise ValueError(f"E003 panel manifest lacks stage: {stage}")
        samplers = stage_spec.get("samplers")
        if not isinstance(samplers, dict) or not samplers:
            raise ValueError(f"E003 panel manifest lacks samplers: {stage}")
        for environment, sampler in sorted(samplers.items()):
            if str(environment) not in declared_core_environments:
                continue
            if not isinstance(sampler, dict):
                raise ValueError(f"Invalid sampler specification: {stage}/{environment}")
            panel_path = Path(str(sampler["panel_path"]))
            expected_panel_hash = str(sampler["panel_sha256"])
            if not panel_path.is_file() or sha256_path(panel_path) != expected_panel_hash:
                raise ValueError(f"Panel is missing or changed: {panel_path}")
            rows = 0
            for row in iter_delimited(panel_path):
                rows += 1
                panel_rows += 1
                label = str(row.get("interaction_label", ""))
                pairs = (
                    ("anchor_sequence_sha256", "seq1", "anchor_id"),
                    (
                        (
                            "true_partner_sequence_sha256"
                            if label == "1"
                            else "candidate_sequence_sha256"
                        ),
                        "seq2",
                        "true_partner_id" if label == "1" else "candidate_partner_id",
                    ),
                )
                for hash_field, sequence_field, source_id_field in pairs:
                    expected_hash = str(row.get(hash_field, ""))
                    raw_sequence = str(row.get(sequence_field, ""))
                    source_id = str(row.get(source_id_field, ""))
                    if not expected_hash or not raw_sequence:
                        raise ValueError(
                            f"Panel row lacks sequence/hash: {panel_path}:{rows}"
                        )
                    sequence = normalize_sequence(raw_sequence)
                    if len(sequence) > maximum_length:
                        raise ValueError(
                            f"Panel sequence exceeds the frozen maximum {maximum_length}: "
                            f"{stage}/{environment}:{rows} length={len(sequence)}"
                        )
                    observed_hash = sequence_sha256(sequence)
                    if observed_hash != expected_hash:
                        raise ValueError(
                            f"Panel sequence hash mismatch at {panel_path}:{rows}: "
                            f"{expected_hash} != {observed_hash}"
                        )
                    previous_sequence = sequences.setdefault(expected_hash, sequence)
                    if previous_sequence != sequence:
                        raise ValueError(f"Sequence hash collision: {expected_hash}")
                    if source_id:
                        source_ids_by_hash[expected_hash].add(source_id)
            panels.append(
                {
                    "stage": stage,
                    "environment": str(environment),
                    "path": str(panel_path),
                    "sha256": expected_panel_hash,
                    "rows": rows,
                }
            )

    records = sorted(sequences.items())
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    count = write_fasta(output_fasta, records)
    if count != len(records):
        raise RuntimeError(f"Wrote {count} records but expected {len(records)}")

    output_manifest_payload: dict[str, Any] = {
        "schema_version": "ppi_shift.e003_sense_cache_registry.v1",
        "status": "complete",
        "panel_manifest": {
            "path": str(panel_manifest),
            "sha256": sha256_path(panel_manifest),
        },
        "core_environments": sorted(declared_core_environments),
        "panels": panels,
        "fasta_path": str(output_fasta),
        "fasta_sha256": sha256_path(output_fasta),
        "sequence_id_namespace": "sha256(normalized_uppercase_sequence)",
        "sequence_count": len(records),
        "panel_rows": panel_rows,
        "duplicate_sequence_groups": sum(
            len(source_ids) > 1 for source_ids in source_ids_by_hash.values()
        ),
        "maximum_sequence_length": maximum_length,
        "sequence_sha256_order": "lexicographic_sequence_sha256",
        "interpretation_boundary": (
            "Deterministic cache-input namespace conversion from hash-verified "
            "frozen panel rows only; no panel row, sequence, label, or annotation "
            "is changed, cropped, or substituted."
        ),
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(output_manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--maximum-length", type=int, default=512)
    args = parser.parse_args()
    manifest = build_registry(
        args.panel_manifest,
        args.output_fasta,
        args.output_manifest,
        maximum_length=args.maximum_length,
    )
    print(
        json.dumps(
            {
                "fasta": manifest["fasta_path"],
                "manifest": str(args.output_manifest),
                "sequence_count": manifest["sequence_count"],
                "panel_rows": manifest["panel_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
