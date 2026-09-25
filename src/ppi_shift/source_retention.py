"""Class-stratified sequence-retention audits for external source surfaces.

The audit deliberately counts source classes before any binary pair-view collapse. A
resolved positive row and a resolved explicit-assay-negative row are not interchangeable
observations, and a conflicting IntAct pair is not silently assigned to either class.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import (
    git_commit,
    iter_delimited,
    sha256_path,
    utc_now,
    write_csv,
    write_yaml,
)
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key

RETENTION_COLUMNS = (
    "source",
    "semantic_class",
    "raw_assertion_rows",
    "raw_unique_identifier_pairs",
    "raw_sequence_resolved_rows",
    "raw_sequence_unresolved_rows",
    "raw_sequence_resolved_fraction",
    "raw_unique_sequence_pairs",
    "canonical_nonconflicting_rows",
    "canonical_sequence_resolved_rows",
    "canonical_sequence_unresolved_rows",
    "canonical_sequence_resolved_fraction",
    "canonical_conflicting_pair_rows",
    "mapping_basis",
)


def _registry_index(path: str | Path) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in iter_delimited(path):
        source = row.get("source_dataset", "").strip()
        identifier = row.get("source_protein_id", "").strip()
        digest = row.get("sequence_sha256", "").strip()
        if source and identifier and digest:
            result[(source, identifier)].add(digest)
    return result


def _raw_endpoint_hashes(
    source: str,
    row: dict[str, str],
    registry: dict[tuple[str, str], set[str]],
) -> tuple[str, str]:
    if source == "vidal":
        left = normalize_sequence(row.get("partner_a_sequence", ""))
        right = normalize_sequence(row.get("partner_b_sequence", ""))
        return (
            sequence_sha256(left) if left else "",
            sequence_sha256(right) if right else "",
        )
    left_id = row.get("partner_a_source_id", "").strip()
    right_id = row.get("partner_b_source_id", "").strip()
    left = registry.get((source, left_id), set())
    right = registry.get((source, right_id), set())
    return (
        next(iter(left)) if len(left) == 1 else "",
        next(iter(right)) if len(right) == 1 else "",
    )


def _raw_pair_ids(source: str, row: dict[str, str]) -> tuple[str, str]:
    if source == "vidal":
        left = f"{row.get('partner_a_role', '')}:{row.get('partner_a_source_id', '')}".strip(":")
        right = f"{row.get('partner_b_role', '')}:{row.get('partner_b_source_id', '')}".strip(":")
    else:
        left = row.get("partner_a_source_id", "").strip()
        right = row.get("partner_b_source_id", "").strip()
    return left, right


def _canonical_rows(
    source: str,
    path: str | Path | None,
    registry: dict[tuple[str, str], set[str]],
) -> tuple[dict[str, Counter[str]], int]:
    """Count non-conflicting canonical rows by class and mapping status."""

    if path is None or not Path(path).is_file():
        return {}, 0
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    conflicting = 0
    for row in iter_delimited(path):
        label = row.get("pair_level_label", "").strip()
        if label == "conflicting_assertions":
            conflicting += 1
            continue
        if not label:
            continue
        left = row.get("partner_a_source_id", "").strip()
        right = row.get("partner_b_source_id", "").strip()
        left_hashes = registry.get((source, left), set())
        right_hashes = registry.get((source, right), set())
        resolved = len(left_hashes) == 1 and len(right_hashes) == 1
        counts[label]["rows"] += 1
        counts[label]["resolved"] += int(resolved)
        counts[label]["unresolved"] += int(not resolved)
    return counts, conflicting


def audit_source_class_retention(
    source_paths: dict[str, str | Path],
    registry_path: str | Path,
    *,
    canonical_paths: dict[str, str | Path] | None = None,
    output_path: str | Path,
    manifest_path: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Write raw and canonical sequence-retention counts for IntAct and Vidal."""

    registry = _registry_index(registry_path)
    rows: list[dict[str, Any]] = []
    input_metadata: dict[str, Any] = {
        "registry": {"path": str(registry_path), "sha256": sha256_path(registry_path)},
        "sources": {},
        "canonical_views": {},
    }
    for source in ("intact", "vidal"):
        raw_path = Path(source_paths[source])
        input_metadata["sources"][source] = {
            "path": str(raw_path),
            "sha256": sha256_path(raw_path),
        }
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "rows": 0,
                "pairs": set(),
                "resolved": 0,
                "sequence_pairs": set(),
            }
        )
        for raw in iter_delimited(raw_path):
            semantic = raw.get("observation_class", "").strip() or "unclassified"
            left_id, right_id = _raw_pair_ids(source, raw)
            left_hash, right_hash = _raw_endpoint_hashes(source, raw, registry)
            bucket = grouped[semantic]
            bucket["rows"] += 1
            if left_id and right_id:
                bucket["pairs"].add(unordered_pair_key(left_id, right_id))
            if left_hash and right_hash:
                bucket["resolved"] += 1
                bucket["sequence_pairs"].add(unordered_pair_key(left_hash, right_hash))
        canonical_counts, conflict_rows = _canonical_rows(
            source,
            (canonical_paths or {}).get(source),
            registry,
        )
        canonical_path = (canonical_paths or {}).get(source)
        if canonical_path is not None and Path(canonical_path).is_file():
            input_metadata["canonical_views"][source] = {
                "path": str(canonical_path),
                "sha256": sha256_path(canonical_path),
            }
        for semantic in sorted(grouped):
            bucket = grouped[semantic]
            canonical = canonical_counts.get(semantic, Counter())
            raw_count = int(bucket["rows"])
            resolved = int(bucket["resolved"])
            canonical_count = int(canonical.get("rows", 0))
            canonical_resolved = int(canonical.get("resolved", 0))
            rows.append(
                {
                    "source": source,
                    "semantic_class": semantic,
                    "raw_assertion_rows": raw_count,
                    "raw_unique_identifier_pairs": len(bucket["pairs"]),
                    "raw_sequence_resolved_rows": resolved,
                    "raw_sequence_unresolved_rows": raw_count - resolved,
                    "raw_sequence_resolved_fraction": resolved / raw_count if raw_count else 0.0,
                    "raw_unique_sequence_pairs": len(bucket["sequence_pairs"]),
                    "canonical_nonconflicting_rows": canonical_count,
                    "canonical_sequence_resolved_rows": canonical_resolved,
                    "canonical_sequence_unresolved_rows": canonical_count - canonical_resolved,
                    "canonical_sequence_resolved_fraction": (
                        canonical_resolved / canonical_count if canonical_count else 0.0
                    ),
                    "canonical_conflicting_pair_rows": conflict_rows,
                    "mapping_basis": (
                        "Vidal source sequences hashed directly"
                        if source == "vidal"
                        else "exact source-protein IDs joined to frozen sequence registry"
                    ),
                }
            )

    output = Path(output_path)
    write_csv(output, RETENTION_COLUMNS, rows)
    manifest = {
        "manifest_type": "ppi_shift.source_class_retention",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": git_commit(repo_root),
        "sources": ["intact", "vidal"],
        "input_artifacts": input_metadata,
        "rows": len(rows),
        "output": {"path": str(output), "sha256": sha256_path(output)},
        "semantics": (
            "Counts preserve raw observation classes. Explicit assay negatives and source "
            "positive assertions are not relabeled as universal non-interactions."
        ),
    }
    manifest_file = Path(manifest_path)
    write_yaml(manifest_file, manifest)
    return {"output_path": output, "manifest_path": manifest_file, "manifest": manifest}
