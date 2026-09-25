#!/usr/bin/env python3
"""Materialize the bounded official-DataSAIL E003 sensitivity panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ppi_shift.artifacts import iter_delimited, sha256_path, write_yaml
from ppi_shift.data import sequence_sha256, unordered_pair_key
from ppi_shift.e003_method_fidelity import E003_SCHEMA_VERSION
from ppi_shift.e003_panels import (
    adapt_anchor_matched_rows_to_e003,
    summarize_e003_panel,
    write_e003_panel,
)
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    EmbeddingStore,
    ExclusionRegistry,
    PositivePair,
    build_candidate_universe,
    generate_negative_rows,
    positive_panel_rows,
)

ENVIRONMENTS = ("uniform_partner", "esm_partner_matched")
STAGES = ("train", "validation", "test")
MAX_RAW_SEQUENCE_LENGTH = 512


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _sequence_index(paths: list[Path]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for path in paths:
        for row in iter_delimited(path):
            left_id = row.get("name1", "").strip()
            right_id = row.get("name2", "").strip()
            left_sequence = row.get("seq1", "")
            right_sequence = row.get("seq2", "")
            if not left_id or not right_id or not left_sequence or not right_sequence:
                continue
            for protein_id, sequence in (
                (left_id, left_sequence),
                (right_id, right_sequence),
            ):
                digest = sequence_sha256(sequence)
                previous = index.get(protein_id)
                if previous is not None and previous[1] != digest:
                    raise ValueError(f"Gold ID maps to incompatible sequences: {protein_id}")
                index[protein_id] = (sequence, digest)
    return index


def _official_rows(path: Path) -> list[dict[str, str]]:
    return list(iter_delimited(path))


def _positive_sets(
    official_rows: list[dict[str, str]],
    sequence_index: dict[str, tuple[str, str]],
) -> tuple[dict[str, list[PositivePair]], set[tuple[str, str]], set[tuple[str, str]], Counter[str]]:
    by_stage: dict[str, list[PositivePair]] = {stage: [] for stage in STAGES}
    all_identifier_pairs: set[tuple[str, str]] = set()
    all_sequence_pairs: set[tuple[str, str]] = set()
    audit: Counter[str] = Counter()
    stage_map = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    for row in official_rows:
        assignment = stage_map.get(row.get("datasail_assignment", ""))
        if assignment is None:
            audit["dropped_or_unassigned"] += 1
            continue
        left_id = row.get("left_id", "")
        right_id = row.get("right_id", "")
        left = sequence_index.get(left_id)
        right = sequence_index.get(right_id)
        if left is None or right is None:
            audit["missing_sequence_mapping"] += 1
            continue
        all_identifier_pairs.add(unordered_pair_key(left_id, right_id))
        all_sequence_pairs.add(unordered_pair_key(left[1], right[1]))
        if row.get("label") != "1":
            audit["official_negative_rows_not_used_for_synthetic_panels"] += 1
            continue
        if len(left[0]) > MAX_RAW_SEQUENCE_LENGTH or len(right[0]) > MAX_RAW_SEQUENCE_LENGTH:
            audit[f"overlength_positive_{assignment}"] += 1
            continue
        positive = PositivePair(
            split=assignment,
            source_row_index=len(by_stage[assignment]),
            dataset_name="datasail_gold_v4",
            left_id=left_id,
            right_id=right_id,
            left_sequence=left[0],
            right_sequence=right[0],
            identity_schema_version=SCHEMA_VERSION_V3,
        )
        by_stage[assignment].append(positive)
        audit[f"retained_positive_{assignment}"] += 1
    return by_stage, all_identifier_pairs, all_sequence_pairs, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty DataSAIL root: {args.output_root}")
    if args.manifest.exists():
        raise SystemExit(f"Refusing to overwrite DataSAIL manifest: {args.manifest}")
    official = _mapping(args.official_manifest)
    mapped = Path(official["mapped_output"]["path"])
    source = _mapping(args.source_config)
    gold_paths = [Path(source["gold"][stage]) for stage in ("train", "validation", "test")]
    sequence_index = _sequence_index(gold_paths)
    official_rows = _official_rows(mapped)
    by_stage, identifier_pairs, sequence_pairs, audit_counts = _positive_sets(
        official_rows, sequence_index
    )
    registry = ExclusionRegistry(
        name="datasail_official_all_positive_registry",
        path=mapped,
        sha256=sha256_path(mapped),
        identifier_pairs=frozenset(identifier_pairs),
        sequence_pairs=frozenset(sequence_pairs),
    )
    embeddings = EmbeddingStore.from_files(
        args.embedding_dir / "embeddings.npy",
        args.embedding_dir / "records.csv",
        args.embedding_dir / "summary.json",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "manifest_type": "ppi_shift.e003_datasail_sensitivity_panels",
        "manifest_version": 1,
        "schema_version": E003_SCHEMA_VERSION,
        "experiment_id": "E003",
        "role": "supplementary_official_datasail_sensitivity",
        "environments": list(ENVIRONMENTS),
        "seeds": [11, 22, 33],
        "official_manifest": {
            "path": str(args.official_manifest),
            "sha256": sha256_path(args.official_manifest),
        },
        "official_mapped_input": {"path": str(mapped), "sha256": sha256_path(mapped)},
        "gold_inputs": [{"path": str(path), "sha256": sha256_path(path)} for path in gold_paths],
        "embedding_store": dict(embeddings.source_hashes),
        "compatibility": {
            "max_raw_sequence_length": MAX_RAW_SEQUENCE_LENGTH,
            "native_cropping": False,
            "official_membership_only": True,
            "official_negative_labels_used": False,
            "synthetic_semantic_class": "synthetic_unobserved_pair",
        },
        "audit_counts": dict(sorted(audit_counts.items())),
        "stages": {},
    }
    for stage in STAGES:
        positives = by_stage[stage]
        if not positives:
            raise SystemExit(f"No compatible DataSAIL positives for {stage}")
        universe = build_candidate_universe(
            positives,
            registry,
            stage=f"datasail_{stage}",
            source_input_hashes={
                "official_manifest": sha256_path(args.official_manifest),
                "official_mapped": sha256_path(mapped),
                "stage_positive_ids": hashlib.sha256(
                    "\n".join(positive.positive_row_id for positive in positives).encode()
                ).hexdigest(),
            },
            schema_version=SCHEMA_VERSION_V3,
        )
        missing_embeddings = embeddings.missing(
            record.sequence_sha256 for record in universe.candidate_pool
        )
        if missing_embeddings:
            raise SystemExit(
                f"DataSAIL ESM coverage missing {len(missing_embeddings)} proteins in {stage}"
            )
        stage_manifest: dict[str, Any] = {
            "positive_rows": len(positives),
            "candidate_universe": universe.manifest_payload(),
            "samplers": {},
        }
        for environment in ENVIRONMENTS:
            negatives = generate_negative_rows(
                positives,
                universe,
                environment,
                seed=args.seed,
                embeddings=embeddings if environment == "esm_partner_matched" else None,
                schema_version=SCHEMA_VERSION_V3,
            )
            rows = adapt_anchor_matched_rows_to_e003(
                positive_panel_rows(
                    positives,
                    universe,
                    environment,
                    seed=args.seed,
                    schema_version=SCHEMA_VERSION_V3,
                )
                + negatives,
                environment=environment,
            )
            path = args.output_root / "panels" / f"{stage}__{environment}.csv.gz"
            write_e003_panel(path, rows)
            stage_manifest["samplers"][environment] = {
                "status": "complete",
                "panel_path": str(path),
                "panel_sha256": sha256_path(path),
                "panel_rows": len(rows),
                "negative_rows": len(negatives),
                "audit": summarize_e003_panel(rows),
            }
        manifest["stages"][stage] = stage_manifest
    manifest_path = args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "stages": {stage: len(by_stage[stage]) for stage in STAGES},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
