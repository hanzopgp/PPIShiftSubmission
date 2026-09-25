#!/usr/bin/env python3
"""Build one hashed sequence registry for the E003 external model caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ppi_shift.artifacts import iter_delimited, sha256_path

MAX_RAW_SEQUENCE_LENGTH = 512


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _add_sequence(
    sequences: dict[str, str], sequence: str, expected_hash: str, source: str
) -> None:
    normalized = "".join(sequence.split()).upper()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Sequence hash mismatch in {source}: {digest} != {expected_hash}")
    if len(normalized) > MAX_RAW_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence exceeds the frozen length contract in {source}")
    previous = sequences.setdefault(digest, normalized)
    if previous != normalized:
        raise ValueError(f"Sequence hash collision for {digest}")


def _add_core(sequences: dict[str, str], manifest_path: Path) -> dict[str, int]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected E003 panel manifest mapping: {manifest_path}")
    counts: dict[str, int] = {}
    for split, stage in manifest["stages"].items():
        for environment, spec in stage["samplers"].items():
            # Supplementary pools such as the leakage-safe TPPNI pool are
            # intentionally recorded in the manifest but are not materialized
            # evaluation panels. They do not belong in the cache registry.
            panel_path = spec.get("panel_path")
            if panel_path is None:
                continue
            path = Path(panel_path)
            if sha256_path(path) != spec["panel_sha256"]:
                raise ValueError(f"Core panel hash changed: {path}")
            count = 0
            for row in iter_delimited(path):
                _add_sequence(
                    sequences,
                    row["seq1"],
                    row["anchor_sequence_sha256"],
                    str(path),
                )
                right_hash = (
                    row["true_partner_sequence_sha256"]
                    if row["interaction_label"] == "1"
                    else row["candidate_sequence_sha256"]
                )
                _add_sequence(sequences, row["seq2"], right_hash, str(path))
                count += 1
            counts[f"{split}:{environment}"] = count
    return counts


def _add_external(sequences: dict[str, str], manifest_path: Path) -> dict[str, int]:
    manifest = _json(manifest_path)
    counts: dict[str, int] = {}
    for source, source_info in manifest["sources"].items():
        for surface, spec in source_info["surfaces"].items():
            path = Path(spec["path"])
            if sha256_path(path) != spec["sha256"]:
                raise ValueError(f"External panel hash changed: {path}")
            count = 0
            for row in iter_delimited(path):
                _add_sequence(
                    sequences,
                    row["left_sequence"],
                    row["left_sequence_sha256"],
                    str(path),
                )
                _add_sequence(
                    sequences,
                    row["right_sequence"],
                    row["right_sequence_sha256"],
                    str(path),
                )
                count += 1
            counts[f"{source}:{surface}"] = count
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_csv, args.output_fasta, args.manifest):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite cache input: {path}")
    sequences: dict[str, str] = {}
    core_counts = _add_core(sequences, args.panel_manifest)
    external_counts = _add_external(sequences, args.external_manifest)
    if not sequences:
        raise ValueError("No cache sequences were found")
    ordered = sorted(sequences.items())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sequence",))
        writer.writerows((sequence,) for _digest, sequence in ordered)
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.output_fasta.open("w", encoding="utf-8") as handle:
        for digest, sequence in ordered:
            handle.write(f">{digest}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    manifest = {
        "schema_version": "ppi_shift.e003_external_cache_input.v1",
        "experiment_id": "E003",
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "external_manifest": {
            "path": str(args.external_manifest),
            "sha256": sha256_path(args.external_manifest),
        },
        "core_panel_rows": core_counts,
        "external_panel_rows": external_counts,
        "sequence_count": len(ordered),
        "maximum_sequence_length": max(map(lambda item: len(item[1]), ordered)),
        "sequence_hash_order": "sha256(sequence) ascending",
        "csv": {"path": str(args.output_csv), "sha256": sha256_path(args.output_csv)},
        "fasta": {"path": str(args.output_fasta), "sha256": sha256_path(args.output_fasta)},
        "models": {
            "esm2_t30_150m": "mean embedding and D-SCRIPT/TUnA token overlay",
            "esm2_t36_3b": "SENSE-PPI residue cache overlay",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "sequence_count": len(ordered),
                "csv_sha256": manifest["csv"]["sha256"],
                "fasta_sha256": manifest["fasta"]["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
