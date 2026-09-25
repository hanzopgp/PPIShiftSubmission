#!/usr/bin/env python3
"""Freeze source-specific E002 external panels without using external labels for training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ppi_shift.artifacts import iter_delimited, sha256_path, write_csv
from ppi_shift.data import unordered_pair_key
from ppi_shift.e002_negative import (
    E002_SCHEMA_VERSION,
    _positive_row,
    generate_bernett_negatives,
    generate_tppni_negatives,
    negative_row,
)
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    ExclusionRegistry,
    PositivePair,
    build_candidate_universe,
    build_degree_table,
)
from ppi_shift.novelty import _gold_reference, load_components

UTC = getattr(timezone, "utc")

EXTERNAL_COLUMNS = (
    "row_id",
    "source",
    "surface",
    "left_id",
    "right_id",
    "left_sequence",
    "right_sequence",
    "left_sequence_sha256",
    "right_sequence_sha256",
    "label",
    "semantic_class",
    "novelty_exact_pair",
    "novelty_protein_identity",
    "novelty_exact_sequence",
    "novelty_mmseqs_i40",
    "novelty_mmseqs_i30_cov80",
    "source_assertion_id",
    "evidence_count",
)
SOURCE_IDS = {
    "biogrid": "biogrid_human_multivalidated_physical_5_0_259",
    "string": "string_human_physical_experimental_v12",
    "intact": "intact_human_binary_physical_snapshot",
}
SYNTHETIC_SURFACES = ("uniform_partner", "bernett_bias_aware_ilp", "tppni_contrastive_l3")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _key(left_hash: str, right_hash: str) -> str:
    return "||".join(sorted((left_hash, right_hash)))


def _selection_hash(key: str) -> str:
    return hashlib.sha256(f"ppi_shift.e002.external_source_independent|{key}".encode()).hexdigest()


def _canonical_rows(path: Path, source_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in iter_delimited(path):
        if row.get("source_id") != source_id or row.get("supervised_target") != "1":
            continue
        left = row.get("partner_a_sequence", "")
        right = row.get("partner_b_sequence", "")
        left_hash = row.get("partner_a_sequence_sha256", "")
        right_hash = row.get("partner_b_sequence_sha256", "")
        if not left or not right or len(left) > 512 or len(right) > 512:
            continue
        pair_key = _key(left_hash, right_hash)
        if pair_key in seen:
            continue
        seen.add(pair_key)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _selection_hash(
                _key(row["partner_a_sequence_sha256"], row["partner_b_sequence_sha256"])
            ),
            row["assertion_id"],
        )
    )
    return rows


def _positive_pairs(rows: list[dict[str, str]], source: str, limit: int) -> list[PositivePair]:
    selected = rows[:limit]
    result = []
    for index, row in enumerate(selected):
        left_id = row.get("partner_a_accession") or row["partner_a_source_id"]
        right_id = row.get("partner_b_accession") or row["partner_b_source_id"]
        result.append(
            PositivePair(
                split=f"external_{source}",
                source_row_index=index,
                dataset_name=source,
                left_id=left_id,
                right_id=right_id,
                left_sequence=row["partner_a_sequence"],
                right_sequence=row["partner_b_sequence"],
                identity_schema_version=SCHEMA_VERSION_V3,
            )
        )
    return result


def _external_registry(
    path: Path,
    source_id: str,
    positive_rows: list[dict[str, str]],
    evaluation_registry: Any,
) -> ExclusionRegistry:
    # The frozen evaluation-positive registry already contains the source-positive
    # firewall.  Reuse it directly instead of copying hundreds of thousands of
    # registry rows into three redundant external artifact files.
    identifier_pairs: set[tuple[str, str]] = set(evaluation_registry.identifier_pairs)
    sequence_pairs = set(evaluation_registry.sequence_pairs)
    for row in positive_rows:
        sequence_pairs.add(
            unordered_pair_key(row["partner_a_sequence_sha256"], row["partner_b_sequence_sha256"])
        )
    return ExclusionRegistry(
        name=f"external_{source_id}_positive_registry",
        path=path,
        sha256=evaluation_registry.sha256,
        identifier_pairs=frozenset(identifier_pairs),
        sequence_pairs=frozenset(sequence_pairs),
    )


def _external_row(
    source: str,
    surface: str,
    *,
    row_id: str,
    left_id: str,
    right_id: str,
    left_sequence: str,
    right_sequence: str,
    left_hash: str,
    right_hash: str,
    label: str,
    semantic_class: str,
    assertion_id: str = "",
    evidence_count: str = "",
    novelty: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    novelty = novelty or {}
    return {
        "row_id": row_id,
        "source": source,
        "surface": surface,
        "left_id": left_id,
        "right_id": right_id,
        "left_sequence": left_sequence,
        "right_sequence": right_sequence,
        "left_sequence_sha256": left_hash,
        "right_sequence_sha256": right_hash,
        "label": label,
        "semantic_class": semantic_class,
        "novelty_exact_pair": novelty.get("exact_pair", "unknown"),
        "novelty_protein_identity": novelty.get("protein_identity", "unknown"),
        "novelty_exact_sequence": novelty.get("exact_sequence", "unknown"),
        "novelty_mmseqs_i40": novelty.get("mmseqs_i40", "unknown"),
        "novelty_mmseqs_i30_cov80": novelty.get("mmseqs_i30_cov80", "unknown"),
        "source_assertion_id": assertion_id,
        "evidence_count": evidence_count,
    }


def _status(value: bool | None, *, unresolved: bool = False) -> str:
    if unresolved:
        return "unresolved"
    if value is None:
        return "unknown"
    return "true" if value else "false"


def _novelty_context(source_config_path: Path, novelty_manifest_path: Path) -> dict[str, Any]:
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    novelty_manifest = yaml.safe_load(novelty_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict) or not isinstance(novelty_manifest, dict):
        raise ValueError("Source and novelty manifests must be YAML mappings")
    component_paths = {
        namespace: Path(payload["path"])
        for namespace, payload in novelty_manifest["components"].items()
    }
    for namespace, payload in novelty_manifest["components"].items():
        if sha256_path(component_paths[namespace]) != payload["sha256"]:
            raise ValueError(f"Novelty component hash changed: {namespace}")
    gold_paths = [
        Path(source_config["gold"]["train"]),
        Path(source_config["gold"]["validation"]),
    ]
    components = {namespace: load_components(path) for namespace, path in component_paths.items()}
    reference = _gold_reference(
        "gold_trainval",
        gold_paths,
        components,
    )
    registry_path = Path(novelty_manifest["registry_path"])
    if sha256_path(registry_path) != novelty_manifest["registry_sha256"]:
        raise ValueError("Novelty sequence registry hash changed")
    registry_by_source_id: dict[tuple[str, str], list[dict[str, str]]] = {}
    canonical_by_hash: dict[str, set[str]] = {}
    for row in iter_delimited(registry_path):
        source = row.get("source_dataset", "")
        source_id = row.get("source_protein_id", "")
        if source and source_id:
            registry_by_source_id.setdefault((source, source_id), []).append(row)
        sequence_hash = row.get("sequence_sha256", "")
        canonical = row.get("canonical_accession", "")
        if sequence_hash and canonical and row.get("mapping_status", "").startswith("resolved"):
            canonical_by_hash.setdefault(sequence_hash, set()).add(canonical)
    return {
        "reference": reference,
        "components": components,
        "registry_by_source_id": registry_by_source_id,
        "canonical_by_hash": canonical_by_hash,
        "source_config_sha256": sha256_path(source_config_path),
        "novelty_manifest_sha256": sha256_path(novelty_manifest_path),
        "registry_path": str(registry_path),
        "registry_sha256": sha256_path(registry_path),
        "component_paths": {namespace: str(path) for namespace, path in component_paths.items()},
        "component_hashes": {
            namespace: sha256_path(path) for namespace, path in component_paths.items()
        },
        "gold_paths": [str(path) for path in gold_paths],
        "gold_hashes": [sha256_path(path) for path in gold_paths],
    }


def _resolve_canonical_id(
    context: dict[str, Any], source: str, identifier: str, sequence_hash: str
) -> str | None:
    reference = context["reference"]
    if identifier in reference.proteins:
        return identifier
    rows = context["registry_by_source_id"].get((source, identifier), [])
    candidates = {
        row["canonical_accession"]
        for row in rows
        if row.get("canonical_accession")
        and row.get("mapping_status", "").startswith("resolved")
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    candidates = context["canonical_by_hash"].get(sequence_hash, set())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _novelty_for_external_row(
    row: Mapping[str, Any], context: dict[str, Any]
) -> dict[str, str]:
    reference = context["reference"]
    source = str(row["source"])
    left_id = str(row["left_id"])
    right_id = str(row["right_id"])
    left_hash = str(row["left_sequence_sha256"])
    right_hash = str(row["right_sequence_sha256"])
    left_canonical = _resolve_canonical_id(context, source, left_id, left_hash)
    right_canonical = _resolve_canonical_id(context, source, right_id, right_hash)
    canonical_resolved = bool(left_canonical and right_canonical)
    sequence_resolved = bool(left_hash and right_hash)
    result = {
        "novelty_exact_pair": _status(
            unordered_pair_key(left_canonical, right_canonical)
            not in reference.identifier_pairs
            if canonical_resolved
            else None,
            unresolved=not canonical_resolved,
        ),
        "novelty_protein_identity": _status(
            left_canonical not in reference.proteins and right_canonical not in reference.proteins
            if canonical_resolved
            else None,
            unresolved=not canonical_resolved,
        ),
        "novelty_exact_sequence": _status(
            left_hash not in reference.sequences and right_hash not in reference.sequences
            if sequence_resolved
            else None,
            unresolved=not sequence_resolved,
        ),
    }
    for namespace, components in context["components"].items():
        left_component = components.get(left_hash)
        right_component = components.get(right_hash)
        result[f"novelty_mmseqs_{namespace}"] = _status(
            left_component not in reference.reference_components.get(namespace, set())
            and right_component not in reference.reference_components.get(namespace, set())
            if left_component and right_component
            else None,
            unresolved=not sequence_resolved or not (left_component and right_component),
        )
    return result


def _synthetic_rows(
    source: str,
    positives: list[PositivePair],
    universe: Any,
    surface: str,
    *,
    seed: int,
    bern_candidate_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if surface == "bernett_bias_aware_ilp":
        negatives, diagnostics = generate_bernett_negatives(
            positives,
            universe,
            seed=seed,
            degree_table=build_degree_table(positives),
            candidate_cap_per_anchor=bern_candidate_cap,
        )
    elif surface == "tppni_contrastive_l3":
        negatives, diagnostics = generate_tppni_negatives(
            positives,
            universe,
            topology_positives=positives,
            seed=seed,
        )
    else:
        negatives = []
        diagnostics = {"method": "E002_uniform_hash", "seed": seed}
        for positive in positives:
            candidates = universe.candidate_records(positive)
            ordered = sorted(
                candidates,
                key=lambda candidate: (
                    int.from_bytes(
                        hashlib.sha256(
                            f"{E002_SCHEMA_VERSION}|{surface}|{seed}|{positive.positive_row_id}|{candidate.protein_id}".encode()
                        ).digest(),
                        "big",
                    ),
                    candidate.protein_id,
                ),
            )
            if not ordered:
                raise ValueError(f"No legal external candidate for {positive.positive_row_id}")
            negatives.append(
                negative_row(
                    positive,
                    universe,
                    surface,
                    seed,
                    ordered[0],
                    rank=1,
                    diagnostics=diagnostics,
                )
            )
    positive_rows = [_positive_row(positive, universe, surface, seed) for positive in positives]
    return [
        _external_row(
            source,
            surface,
            row_id=f"{source}:{surface}:positive:{row['positive_row_id']}",
            left_id=row["anchor_id"],
            right_id=row["true_partner_id"],
            left_sequence=row["seq1"],
            right_sequence=row["seq2"],
            left_hash=row["anchor_sequence_sha256"],
            right_hash=row["true_partner_sequence_sha256"],
            label="1",
            semantic_class="curated_positive_reference",
        )
        for row in positive_rows
    ] + [
        _external_row(
            source,
            surface,
            row_id=f"{source}:{surface}:negative:{row['row_id']}",
            left_id=row["anchor_id"],
            right_id=row["candidate_partner_id"],
            left_sequence=row["seq1"],
            right_sequence=row["seq2"],
            left_hash=row["anchor_sequence_sha256"],
            right_hash=row["candidate_sequence_sha256"],
            label="0",
            semantic_class="synthetic_unobserved_pair",
        )
        for row in negatives
    ], diagnostics


def _write_external(
    path: Path, rows: list[dict[str, Any]], novelty_context: dict[str, Any]
) -> dict[str, Any]:
    for row in rows:
        calculated = _novelty_for_external_row(row, novelty_context)
        for field, value in calculated.items():
            if row.get(field, "") in {"", "unknown"}:
                row[field] = value
    rows = sorted(rows, key=lambda row: row["row_id"])
    write_csv(path, EXTERNAL_COLUMNS, rows)
    labels = Counter(row["label"] for row in rows)
    novelty = {
        field: dict(sorted(Counter(row[field] for row in rows).items()))
        for field in EXTERNAL_COLUMNS
        if field.startswith("novelty_")
    }
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "rows": len(rows),
        "label_counts": {key: int(value) for key, value in sorted(labels.items())},
        "semantic_classes": sorted({row["semantic_class"] for row in rows}),
        "novelty_states": novelty,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-positive", type=Path, required=True)
    parser.add_argument("--canonical-negative", type=Path, required=True)
    parser.add_argument("--evaluation-registry", type=Path, required=True)
    parser.add_argument("--vidal-panel", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--novelty-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--limit", type=int, default=10111)
    parser.add_argument("--bern-candidate-cap", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty external output root: {args.output_root}")
    if args.manifest.exists():
        raise SystemExit(f"Refusing to overwrite external manifest: {args.manifest}")
    from ppi_shift.negatives import load_exclusion_registry

    novelty_context = _novelty_context(args.source_config, args.novelty_manifest)
    evaluation_registry = load_exclusion_registry(
        args.evaluation_registry, name="evaluation_positive_union"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "manifest_type": "ppi_shift.e002_external_panels",
        "manifest_version": 1,
        "created_utc": _now(),
        "schema_version": "ppi_shift.e002_external_panel.v1",
        "seed": args.seed,
        "limit": args.limit,
        "input_hashes": {
            "canonical_positive": sha256_path(args.canonical_positive),
            "canonical_negative": sha256_path(args.canonical_negative),
            "evaluation_registry": sha256_path(args.evaluation_registry),
            "vidal_panel": sha256_path(args.vidal_panel),
            "source_config": sha256_path(args.source_config),
            "novelty_manifest": novelty_context["novelty_manifest_sha256"],
            "novelty_registry": novelty_context["registry_sha256"],
            "gold_train": novelty_context["gold_hashes"][0],
            "gold_validation": novelty_context["gold_hashes"][1],
            **{
                f"component_{namespace}": digest
                for namespace, digest in novelty_context["component_hashes"].items()
            },
        },
        "novelty_reference": {
            "reference": "Gold train + validation",
            "gold_paths": novelty_context["gold_paths"],
            "gold_hashes": novelty_context["gold_hashes"],
            "registry_path": novelty_context["registry_path"],
            "registry_sha256": novelty_context["registry_sha256"],
            "component_paths": novelty_context["component_paths"],
            "component_hashes": novelty_context["component_hashes"],
            "state_vocabulary": ["true", "false", "unknown", "unresolved"],
            "unknown_or_unresolved_not_promoted_to_novel": True,
        },
        "no_external_retraining": True,
        "no_external_selection": True,
        "sources": {},
    }

    for source, source_id in SOURCE_IDS.items():
        all_positive = _canonical_rows(args.canonical_positive, source_id)
        selected_rows = all_positive[: args.limit]
        positives = _positive_pairs(selected_rows, source, args.limit)
        registry = _external_registry(
            args.evaluation_registry,
            source_id,
            all_positive,
            evaluation_registry,
        )
        universe = build_candidate_universe(
            positives,
            registry,
            stage=f"external_{source}",
            source_input_hashes={
                "canonical_positive": sha256_path(args.canonical_positive),
                "source_selection_hash": hashlib.sha256(
                    "\n".join(p.positive_row_id for p in positives).encode()
                ).hexdigest(),
            },
            schema_version=SCHEMA_VERSION_V3,
        )
        intact_negative_rows: list[dict[str, str]] = []
        intact_measured_viable = False
        intact_negative_unique_count = 0
        intact_conflicting_selected_count = 0
        if source == "intact":
            negative_candidates: list[dict[str, str]] = []
            negative_keys: set[str] = set()
            for row in iter_delimited(args.canonical_negative):
                if (
                    row.get("source_id") != SOURCE_IDS["intact"]
                    or row.get("supervised_target") != "0"
                ):
                    continue
                if (
                    len(row.get("partner_a_sequence", "")) > 512
                    or len(row.get("partner_b_sequence", "")) > 512
                ):
                    continue
                pair_key = _key(row["partner_a_sequence_sha256"], row["partner_b_sequence_sha256"])
                if pair_key in negative_keys:
                    continue
                negative_keys.add(pair_key)
                negative_candidates.append(row)
            positive_keys = {
                _key(row["partner_a_sequence_sha256"], row["partner_b_sequence_sha256"])
                for row in selected_rows
            }
            intact_negative_rows = [
                row
                for row in negative_candidates
                if _key(row["partner_a_sequence_sha256"], row["partner_b_sequence_sha256"])
                not in positive_keys
            ]
            intact_negative_unique_count = len(negative_candidates)
            intact_conflicting_selected_count = (
                intact_negative_unique_count - len(intact_negative_rows)
            )
            intact_measured_viable = (
                len(positives) >= 200 and len(intact_negative_rows) >= 200
            )
        source_manifest: dict[str, Any] = {
            "source_id": source_id,
            "raw_compatible_positive_rows": len(all_positive),
            "selected_positive_rows": len(positives),
            "selection_rule": (
                "source-independent SHA256(unordered exact-sequence pair key), "
                "then assertion_id"
            ),
            "positive_selection_hash": hashlib.sha256(
                "\n".join(p.positive_row_id for p in positives).encode()
            ).hexdigest(),
            "candidate_universe_sha256": universe.universe_sha256,
            "surfaces": {},
        }
        if source != "intact" or not intact_measured_viable:
            for surface in SYNTHETIC_SURFACES:
                rows, diagnostics = _synthetic_rows(
                    source,
                    positives,
                    universe,
                    surface,
                    seed=args.seed,
                    bern_candidate_cap=args.bern_candidate_cap,
                )
                spec = _write_external(
                    args.output_root / source / f"{surface}.csv", rows, novelty_context
                )
                spec["kind"] = "synthetic"
                spec["sampler_diagnostics"] = diagnostics
                source_manifest["surfaces"][surface] = spec

        if source == "intact":
            measured = [
                _external_row(
                    "intact",
                    "measured",
                    row_id=row["assertion_id"],
                    left_id=row["partner_a_source_id"],
                    right_id=row["partner_b_source_id"],
                    left_sequence=row["partner_a_sequence"],
                    right_sequence=row["partner_b_sequence"],
                    left_hash=row["partner_a_sequence_sha256"],
                    right_hash=row["partner_b_sequence_sha256"],
                    label="0",
                    semantic_class="explicit_assay_negative",
                    assertion_id=row["assertion_id"],
                    evidence_count=row.get("evidence_count", ""),
                )
                for row in intact_negative_rows
            ]
            # Keep deterministic positives, but use canonical source IDs and sequences.
            measured.extend(
                _external_row(
                    "intact",
                    "measured",
                    row_id=row["assertion_id"],
                    left_id=row["partner_a_source_id"],
                    right_id=row["partner_b_source_id"],
                    left_sequence=row["partner_a_sequence"],
                    right_sequence=row["partner_b_sequence"],
                    left_hash=row["partner_a_sequence_sha256"],
                    right_hash=row["partner_b_sequence_sha256"],
                    label="1",
                    semantic_class="physical_positive",
                    assertion_id=row["assertion_id"],
                    evidence_count=row.get("evidence_count", ""),
                )
                for row in selected_rows
            )
            source_manifest["intact_measured_audit"] = {
                "canonical_resolved_positive_rows": len(positives),
                "canonical_resolved_explicit_negative_rows": len(intact_negative_rows),
                "unique_length_compatible_negative_rows": intact_negative_unique_count,
                "viability": intact_measured_viable,
                "minimum_positive_rows": 200,
                "minimum_explicit_negative_rows": 200,
                "conflicting_selected_pair_rows_removed": intact_conflicting_selected_count,
            }
            if intact_measured_viable:
                source_manifest["surfaces"]["measured"] = {
                    **_write_external(
                        args.output_root / source / "measured.csv", measured, novelty_context
                    ),
                    "kind": "measured",
                    "label_semantics": {"1": "physical_positive", "0": "explicit_assay_negative"},
                }
            else:
                source_manifest["intact_measured_audit"]["fallback"] = (
                    "synthetic_cross_source_surfaces_materialized"
                )

        manifest["sources"][source] = source_manifest

    # Vidal is reused verbatim at the row/label level from its frozen E001v3
    # compatible panel; the new file only normalizes it into the external schema.
    vidal_rows = []
    for row in iter_delimited(args.vidal_panel):
        novelty = {
            "exact_sequence": row.get("exact_sequence_novel", "unknown"),
            "mmseqs_i40": row.get("family_i40_novel", "unknown"),
            "mmseqs_i30_cov80": row.get("family_i30_cov80_novel", "unknown"),
        }
        vidal_rows.append(
            _external_row(
                "vidal",
                "measured",
                row_id=row["row_id"],
                left_id=row["left_id"],
                right_id=row["right_id"],
                left_sequence=row["left_sequence"],
                right_sequence=row["right_sequence"],
                left_hash=row["left_sequence_sha256"],
                right_hash=row["right_sequence_sha256"],
                label=row["label"],
                semantic_class=(
                    "binary_assay_positive" if row["label"] == "1" else "assay_non_detection"
                ),
                assertion_id=row.get("assertion_id", ""),
                novelty=novelty,
            )
        )
    manifest["sources"]["vidal"] = {
        "surfaces": {
            "measured": {
                **_write_external(
                    args.output_root / "vidal" / "measured.csv", vidal_rows, novelty_context
                ),
                "kind": "measured_reused_frozen_4031_row_cohort",
            }
        },
        "reused_rows": len(vidal_rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in vidal_rows).items())),
        "no_ranking_metrics": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"manifest": str(args.manifest), "sources": list(manifest["sources"])}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
