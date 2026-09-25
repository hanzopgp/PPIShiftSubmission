"""Build the project-owned, source-aware human sequence registry.

The registry is intentionally a mapping record rather than a claim that every source
identifier denotes the same biological entity.  Missing and ambiguous mappings remain
visible, while exact sequence hashes provide the common namespace used downstream.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import (
    git_commit,
    iter_delimited,
    iter_fasta,
    sha256_path,
    utc_now,
    write_csv,
    write_fasta,
    write_yaml,
)
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key

REGISTRY_COLUMNS = (
    "source_dataset",
    "source_version",
    "source_protein_id",
    "canonical_accession",
    "sequence",
    "sequence_sha256",
    "taxon_id",
    "mapping_source",
    "mapping_version",
    "mapping_status",
    "ambiguity_status",
    "source_role",
    "source_accession_candidates",
)

DEFAULT_VERSIONS = {
    "gold_train": "gold_21591618_intra",
    "gold_validation": "gold_21591618_intra",
    "gold_test": "gold_21591618_intra",
    "vidal": "Lambourne_2024_Data_S4_local_snapshot",
    "string": "STRING_v12.0",
    "biogrid": "BioGRID_5.0.259",
    "intact": "IntAct_human_binary_snapshot_2026-07-26",
}


@dataclass(frozen=True)
class SnapshotIndex:
    """Exact accession-to-sequence index for a frozen FASTA snapshot."""

    sequences: Mapping[str, str]
    conflicting_ids: frozenset[str] = frozenset()
    path: Path | None = None


@dataclass
class _SourceStats:
    rows: int = 0
    pair_keys: set[tuple[str, str]] = field(default_factory=set)
    protein_ids: set[str] = field(default_factory=set)
    resolved_ids: set[str] = field(default_factory=set)
    ambiguous_ids: set[str] = field(default_factory=set)
    missing_ids: set[str] = field(default_factory=set)
    sequence_hashes: set[str] = field(default_factory=set)
    lengths: list[int] = field(default_factory=list)
    hashes_by_id: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_pair(self, left: str, right: str) -> None:
        if left and right:
            self.pair_keys.add(unordered_pair_key(left, right))

    def add_mapping(self, identifier: str, sequence: str, status: str) -> None:
        if not identifier:
            return
        self.protein_ids.add(identifier)
        if status.startswith("ambiguous"):
            self.ambiguous_ids.add(identifier)
        if sequence and status != "ambiguous_sequence":
            self.resolved_ids.add(identifier)
        elif not sequence:
            self.missing_ids.add(identifier)
        if sequence:
            normalized = normalize_sequence(sequence)
            digest = sequence_sha256(normalized)
            self.sequence_hashes.add(digest)
            self.lengths.append(len(normalized))
            self.hashes_by_id[identifier].add(digest)


@dataclass(frozen=True)
class RegistryBuildResult:
    """Paths and counts emitted by :func:`build_sequence_registry`."""

    manifest_path: Path
    registry_path: Path
    summary_path: Path
    failures_path: Path
    unique_fasta_path: Path
    manifest: Mapping[str, Any]


def load_snapshot(path: str | Path) -> SnapshotIndex:
    """Load an exact-accession FASTA snapshot without isoform fallback."""

    sequences: dict[str, str] = {}
    conflicting: set[str] = set()
    for identifier, raw_sequence in iter_fasta(path):
        sequence = normalize_sequence(raw_sequence)
        previous = sequences.get(identifier)
        if previous is not None and previous != sequence:
            conflicting.add(identifier)
        else:
            sequences[identifier] = sequence
    return SnapshotIndex(
        sequences=sequences,
        conflicting_ids=frozenset(conflicting),
        path=Path(path),
    )


def resolve_snapshot_accession(
    accession: str,
    snapshot: SnapshotIndex,
) -> tuple[str, str, str, str]:
    """Resolve an accession exactly, returning sequence/status/ambiguity/canonical.

    The function deliberately does not try the base accession for ``P12345-2``.  An
    absent or conflicting exact key is therefore an explicit unresolved mapping.
    """

    value = str(accession).strip()
    if not value:
        return "", "missing_sequence", "missing_accession", ""
    if value in snapshot.conflicting_ids:
        return "", "ambiguous_sequence", "incompatible_snapshot_sequences", ""
    sequence = snapshot.sequences.get(value)
    if not sequence:
        return "", "missing_sequence", "exact_accession_not_in_snapshot", ""
    return sequence, "resolved_exact_accession", "none", value


def _read_string_aliases(path: str | Path | None) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    if path is None:
        return aliases
    for row in iter_delimited(path):
        string_id = row.get("string_protein_id", "").strip()
        alias = row.get("alias", "").strip()
        source = row.get("source", "").strip()
        if string_id and alias and source == "UniProt_AC":
            aliases[string_id].append(alias)
    return aliases


def resolve_string_mapping(
    string_id: str,
    sequence: str,
    aliases: Mapping[str, Iterable[str]],
    snapshot: SnapshotIndex | None,
) -> tuple[str, str, str, str, str]:
    """Resolve STRING's optional UniProt alias without discarding its raw sequence."""

    normalized = normalize_sequence(sequence)
    candidates = sorted(set(str(value).strip() for value in aliases.get(string_id, ()) if value))
    if snapshot is None:
        return "", "resolved_source_sequence", "snapshot_unavailable", "", "|".join(candidates)
    compatible = [
        candidate
        for candidate in candidates
        if candidate in snapshot.sequences and snapshot.sequences[candidate] == normalized
    ]
    incompatible = [
        candidate
        for candidate in candidates
        if candidate in snapshot.sequences and snapshot.sequences[candidate] != normalized
    ]
    if len(compatible) == 1 and not incompatible:
        return (
            compatible[0],
            "resolved_source_sequence",
            "none",
            compatible[0],
            "|".join(candidates),
        )
    if len(compatible) > 1:
        return (
            "",
            "ambiguous_accession",
            "multiple_compatible_accessions",
            "",
            "|".join(candidates),
        )
    ambiguity = "alias_sequence_mismatch_or_unavailable" if candidates else "no_uniprot_alias"
    return "", "resolved_source_sequence", ambiguity, "", "|".join(candidates)


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _stats_row(source: str, stats: _SourceStats, input_path: Path) -> dict[str, Any]:
    identifiers_by_hash: dict[str, set[str]] = defaultdict(set)
    for identifier, hashes in stats.hashes_by_id.items():
        for digest in hashes:
            identifiers_by_hash[digest].add(identifier)
    duplicate_groups = sum(len(values) > 1 for values in identifiers_by_hash.values())
    duplicate_mappings = sum(max(0, len(values) - 1) for values in identifiers_by_hash.values())
    resolved = len(stats.resolved_ids)
    proteins = len(stats.protein_ids)
    return {
        "source_dataset": source,
        "assertion_rows": stats.rows,
        "unique_pairs": len(stats.pair_keys),
        "source_proteins": proteins,
        "sequence_resolved_proteins": resolved,
        "mapping_fraction": (resolved / proteins) if proteins else 0.0,
        "ambiguous_mappings": len(stats.ambiguous_ids),
        "missing_mappings": len(stats.missing_ids),
        "unique_sequences": len(stats.sequence_hashes),
        "duplicate_exact_sequence_groups_across_identifiers": duplicate_groups,
        "duplicate_exact_sequence_mappings_across_identifiers": duplicate_mappings,
        "sequence_length_min": min(stats.lengths) if stats.lengths else "",
        "sequence_length_median": _median(stats.lengths) or "",
        "sequence_length_max": max(stats.lengths) if stats.lengths else "",
        "input_path": str(input_path),
        "input_sha256": sha256_path(input_path),
    }


def _add_record(
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
    failures: dict[tuple[str, str, str], dict[str, Any]],
    stats: _SourceStats,
    *,
    source_dataset: str,
    source_version: str,
    source_protein_id: str,
    canonical_accession: str,
    sequence: str,
    taxon_id: str,
    mapping_source: str,
    mapping_version: str,
    mapping_status: str,
    ambiguity_status: str,
    source_role: str,
    source_accession_candidates: str = "",
) -> None:
    normalized = normalize_sequence(sequence)
    digest = sequence_sha256(normalized) if normalized else ""
    stats.add_mapping(source_protein_id, normalized, mapping_status)
    key = (source_dataset, source_protein_id, digest, source_role, mapping_status)
    records[key] = {
        "source_dataset": source_dataset,
        "source_version": source_version,
        "source_protein_id": source_protein_id,
        "canonical_accession": canonical_accession,
        "sequence": normalized,
        "sequence_sha256": digest,
        "taxon_id": taxon_id,
        "mapping_source": mapping_source,
        "mapping_version": mapping_version,
        "mapping_status": mapping_status,
        "ambiguity_status": ambiguity_status,
        "source_role": source_role,
        "source_accession_candidates": source_accession_candidates,
    }
    if mapping_status.startswith("ambiguous") or not normalized:
        failures[(source_dataset, source_protein_id, mapping_status)] = {
            "source_dataset": source_dataset,
            "source_protein_id": source_protein_id,
            "mapping_status": mapping_status,
            "ambiguity_status": ambiguity_status,
            "source_role": source_role,
            "source_accession_candidates": source_accession_candidates,
        }


def _require(row: Mapping[str, str], *columns: str) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _process_gold(
    source_key: str,
    path: Path,
    version: str,
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
    failures: dict[tuple[str, str, str], dict[str, Any]],
    stats: _SourceStats,
) -> None:
    for row in iter_delimited(path):
        _require(row, "name1", "name2", "seq1", "seq2")
        stats.rows += 1
        left, right = row["name1"].strip(), row["name2"].strip()
        stats.add_pair(left, right)
        for identifier, sequence in ((left, row["seq1"]), (right, row["seq2"])):
            _add_record(
                records,
                failures,
                stats,
                source_dataset=source_key,
                source_version=version,
                source_protein_id=identifier,
                canonical_accession=identifier,
                sequence=sequence,
                taxon_id="9606",
                mapping_source="gold_materialized_row",
                mapping_version=version,
                mapping_status="resolved_source_sequence"
                if sequence.strip()
                else "missing_sequence",
                ambiguity_status="none" if sequence.strip() else "empty_sequence",
                source_role=source_key.removeprefix("gold_"),
            )


def _process_vidal(
    path: Path,
    version: str,
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
    failures: dict[tuple[str, str, str], dict[str, Any]],
    stats: _SourceStats,
) -> None:
    required = (
        "partner_a_source_id",
        "partner_b_source_id",
        "partner_a_role",
        "partner_b_role",
        "partner_a_sequence",
        "partner_b_sequence",
    )
    for row in iter_delimited(path):
        _require(row, *required)
        stats.rows += 1
        left = f"{row['partner_a_role']}:{row['partner_a_source_id']}".strip(":")
        right = f"{row['partner_b_role']}:{row['partner_b_source_id']}".strip(":")
        stats.add_pair(left, right)
        for source_id, role, sequence in (
            (left, row["partner_a_role"], row["partner_a_sequence"]),
            (right, row["partner_b_role"], row["partner_b_sequence"]),
        ):
            _add_record(
                records,
                failures,
                stats,
                source_dataset="vidal",
                source_version=version,
                source_protein_id=source_id,
                canonical_accession="",
                sequence=sequence,
                taxon_id=row.get("taxon_a", "9606") or "9606",
                mapping_source="vidal_staging_sequence",
                mapping_version=version,
                mapping_status="resolved_source_sequence"
                if sequence.strip()
                else "missing_sequence",
                ambiguity_status="none" if sequence.strip() else "empty_sequence",
                source_role=role,
            )


def _process_string(
    path: Path,
    fasta_path: Path | None,
    aliases_path: Path | None,
    snapshot: SnapshotIndex | None,
    version: str,
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
    failures: dict[tuple[str, str, str], dict[str, Any]],
    stats: _SourceStats,
) -> None:
    sequences = (
        {
            identifier: normalize_sequence(sequence)
            for identifier, sequence in iter_fasta(fasta_path)
        }
        if fasta_path
        else {}
    )
    aliases = _read_string_aliases(aliases_path)
    snapshot_version = sha256_path(snapshot.path) if snapshot and snapshot.path else "unavailable"
    for row in iter_delimited(path):
        _require(row, "partner_a_source_id", "partner_b_source_id")
        stats.rows += 1
        left, right = row["partner_a_source_id"].strip(), row["partner_b_source_id"].strip()
        stats.add_pair(left, right)
        for source_id in (left, right):
            sequence = sequences.get(source_id, "")
            if sequence:
                canonical, status, ambiguity, _, candidates = resolve_string_mapping(
                    source_id, sequence, aliases, snapshot
                )
            else:
                canonical, status, ambiguity, candidates = (
                    "",
                    "missing_sequence",
                    "raw_sequence_not_found",
                    "",
                )
            _add_record(
                records,
                failures,
                stats,
                source_dataset="string",
                source_version=version,
                source_protein_id=source_id,
                canonical_accession=canonical,
                sequence=sequence,
                taxon_id=row.get("taxon_a", "9606") or "9606",
                mapping_source="STRING_raw_fasta_and_aliases",
                mapping_version=f"{version};snapshot={snapshot_version}",
                mapping_status=status,
                ambiguity_status=ambiguity,
                source_role="physical_positive",
                source_accession_candidates=candidates,
            )


def _process_accession_source(
    source_dataset: str,
    path: Path,
    snapshot: SnapshotIndex | None,
    version: str,
    records: dict[tuple[str, str, str, str, str], dict[str, Any]],
    failures: dict[tuple[str, str, str], dict[str, Any]],
    stats: _SourceStats,
) -> None:
    snapshot_version = sha256_path(snapshot.path) if snapshot and snapshot.path else "unavailable"
    for row in iter_delimited(path):
        _require(row, "partner_a_source_id", "partner_b_source_id")
        stats.rows += 1
        left = row.get("partner_a_accession", "").strip()
        right = row.get("partner_b_accession", "").strip()
        stats.add_pair(left or row["partner_a_source_id"], right or row["partner_b_source_id"])
        for accession, source_id, role in (
            (left, row["partner_a_source_id"].strip(), "partner_a"),
            (right, row["partner_b_source_id"].strip(), "partner_b"),
        ):
            if snapshot is None:
                sequence, status, ambiguity, canonical = (
                    "",
                    "missing_sequence",
                    "snapshot_unavailable",
                    "",
                )
            else:
                sequence, status, ambiguity, canonical = resolve_snapshot_accession(
                    accession, snapshot
                )
            _add_record(
                records,
                failures,
                stats,
                source_dataset=source_dataset,
                source_version=version,
                source_protein_id=source_id or accession,
                canonical_accession=canonical,
                sequence=sequence,
                taxon_id=row.get("taxon_a", "9606") or "9606",
                mapping_source="frozen_human_swissprot_snapshot",
                mapping_version=snapshot_version,
                mapping_status=status,
                ambiguity_status=ambiguity,
                source_role=row.get("observation_class", "") or role,
                source_accession_candidates=accession,
            )


def build_sequence_registry(
    source_paths: Mapping[str, str | Path],
    *,
    string_fasta: str | Path | None,
    string_aliases: str | Path | None,
    swissprot_snapshot: str | Path | None,
    artifact_root: str | Path,
    repo_root: str | Path = ".",
    source_versions: Mapping[str, str] | None = None,
) -> RegistryBuildResult:
    """Materialize D1 registry, summaries, failures, FASTA, and manifest."""

    paths = {key: Path(value) for key, value in source_paths.items()}
    required = {
        "gold_train",
        "gold_validation",
        "gold_test",
        "vidal",
        "string",
        "biogrid",
        "intact",
    }
    missing_keys = sorted(required - paths.keys())
    if missing_keys:
        raise ValueError(f"Missing source paths: {', '.join(missing_keys)}")
    for key in required:
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    if string_fasta is not None and not Path(string_fasta).is_file():
        raise FileNotFoundError(string_fasta)
    if string_aliases is not None and not Path(string_aliases).is_file():
        raise FileNotFoundError(string_aliases)
    if swissprot_snapshot is not None and not Path(swissprot_snapshot).is_file():
        raise FileNotFoundError(swissprot_snapshot)

    versions = dict(DEFAULT_VERSIONS)
    versions.update(source_versions or {})
    snapshot = load_snapshot(swissprot_snapshot) if swissprot_snapshot else None
    records: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    failures: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats: dict[str, _SourceStats] = {key: _SourceStats() for key in required}
    _process_gold(
        "gold_train",
        paths["gold_train"],
        versions["gold_train"],
        records,
        failures,
        stats["gold_train"],
    )
    _process_gold(
        "gold_validation",
        paths["gold_validation"],
        versions["gold_validation"],
        records,
        failures,
        stats["gold_validation"],
    )
    _process_gold(
        "gold_test",
        paths["gold_test"],
        versions["gold_test"],
        records,
        failures,
        stats["gold_test"],
    )
    _process_vidal(paths["vidal"], versions["vidal"], records, failures, stats["vidal"])
    _process_string(
        paths["string"],
        Path(string_fasta) if string_fasta else None,
        Path(string_aliases) if string_aliases else None,
        snapshot,
        versions["string"],
        records,
        failures,
        stats["string"],
    )
    _process_accession_source(
        "biogrid",
        paths["biogrid"],
        snapshot,
        versions["biogrid"],
        records,
        failures,
        stats["biogrid"],
    )
    _process_accession_source(
        "intact", paths["intact"], snapshot, versions["intact"], records, failures, stats["intact"]
    )

    root = Path(artifact_root)
    registry_path = root / "sequence_registry" / "sequence_registry.csv"
    fasta_path = root / "sequence_registry" / "unique_sequences.fasta"
    summary_path = Path(repo_root) / "audit" / "sequence_registry_summary.csv"
    failures_path = Path(repo_root) / "audit" / "sequence_mapping_failures.csv"
    manifest_path = Path(repo_root) / "data" / "manifests" / "sequence_registry_manifest.yaml"
    registry_rows = sorted(
        records.values(), key=lambda row: tuple(str(row[column]) for column in REGISTRY_COLUMNS)
    )
    write_csv(registry_path, REGISTRY_COLUMNS, registry_rows)
    unique_sequences: dict[str, str] = {}
    for row in registry_rows:
        if row["sequence_sha256"] and row["sequence_sha256"] not in unique_sequences:
            unique_sequences[row["sequence_sha256"]] = row["sequence"]
    write_fasta(fasta_path, sorted(unique_sequences.items()))
    summary_columns = (
        "source_dataset",
        "assertion_rows",
        "unique_pairs",
        "source_proteins",
        "sequence_resolved_proteins",
        "mapping_fraction",
        "ambiguous_mappings",
        "missing_mappings",
        "unique_sequences",
        "duplicate_exact_sequence_groups_across_identifiers",
        "duplicate_exact_sequence_mappings_across_identifiers",
        "sequence_length_min",
        "sequence_length_median",
        "sequence_length_max",
        "input_path",
        "input_sha256",
    )
    summary_rows = [_stats_row(key, stats[key], paths[key]) for key in sorted(required)]
    write_csv(summary_path, summary_columns, summary_rows)
    failure_columns = (
        "source_dataset",
        "source_protein_id",
        "mapping_status",
        "ambiguity_status",
        "source_role",
        "source_accession_candidates",
    )
    write_csv(
        failures_path,
        failure_columns,
        sorted(
            failures.values(), key=lambda row: tuple(str(row[column]) for column in failure_columns)
        ),
    )
    input_hashes = {key: sha256_path(paths[key]) for key in sorted(paths)}
    if string_fasta:
        input_hashes["string_fasta"] = sha256_path(string_fasta)
    if string_aliases:
        input_hashes["string_aliases"] = sha256_path(string_aliases)
    if swissprot_snapshot:
        input_hashes["swissprot_snapshot"] = sha256_path(swissprot_snapshot)
    manifest: dict[str, Any] = {
        "manifest_type": "ppi_shift.sequence_registry",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": git_commit(repo_root),
        "artifact_root": str(root),
        "input_artifacts": {
            key: {"path": str(paths[key]), "sha256": input_hashes[key]} for key in sorted(paths)
        },
        "mapping_artifacts": {
            key: {"path": str(value), "sha256": input_hashes[key]}
            for key, value in (
                ("string_fasta", string_fasta),
                ("string_aliases", string_aliases),
                ("swissprot_snapshot", swissprot_snapshot),
            )
            if value is not None
        },
        "outputs": {
            "registry": {
                "path": str(registry_path),
                "sha256": sha256_path(registry_path),
                "rows": len(registry_rows),
            },
            "unique_sequences_fasta": {
                "path": str(fasta_path),
                "sha256": sha256_path(fasta_path),
                "rows": len(unique_sequences),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_path(summary_path),
                "rows": len(summary_rows),
            },
            "failures": {
                "path": str(failures_path),
                "sha256": sha256_path(failures_path),
                "rows": len(failures),
            },
        },
        "source_versions": versions,
        "sequence_normalization": "ppi_shift.data.normalize_sequence",
        "sequence_hash": "sha256(normalized UTF-8 sequence)",
        "source_statistics": {
            key: _stats_row(key, stats[key], paths[key]) for key in sorted(required)
        },
        "command": (
            "scripts/build_sequence_registry.py --config configs/data/prefreeze_sources.yaml"
        ),
    }
    write_yaml(manifest_path, manifest)
    return RegistryBuildResult(
        manifest_path, registry_path, summary_path, failures_path, fasta_path, manifest
    )


def load_registry(path: str | Path) -> list[dict[str, str]]:
    """Load the normally modest registry index for downstream joins."""

    return list(iter_delimited(path))


def registry_json(record: Mapping[str, Any]) -> str:
    """Stable JSON encoding for embedding a mapping record in an audit."""

    return json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
