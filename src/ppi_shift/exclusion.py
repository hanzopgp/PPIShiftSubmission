"""Explicit positive-label firewall registries for train, validation, and evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path, utc_now, write_csv, write_yaml
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key

REGISTRY_COLUMNS = (
    "identifier_pair_key",
    "sequence_pair_key",
    "source_dataset",
    "source_version",
    "source_row_id",
    "identifier_namespace",
    "sequence_mapping_status",
    "semantic_class",
)


class ExclusionPolicyError(ValueError):
    """Raised when a registry receives a source not legal for its stage."""


@dataclass(frozen=True)
class PositiveRecord:
    source_dataset: str
    source_version: str
    left_id: str
    right_id: str
    left_sequence: str
    right_sequence: str
    source_row_id: str
    semantic_class: str
    identifier_namespace: str = ""
    canonical_left: str = ""
    canonical_right: str = ""
    positive: bool = True


@dataclass(frozen=True)
class RegistryBuildResult:
    registry_name: str
    output_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _key_text(values: tuple[str, str]) -> str:
    return "||".join(values) if values[0] and values[1] else ""


def _record_row(record: PositiveRecord) -> dict[str, str]:
    left_id = record.canonical_left or record.left_id
    right_id = record.canonical_right or record.right_id
    identifier_pair = _key_text(unordered_pair_key(left_id, right_id))
    left_sequence = normalize_sequence(record.left_sequence)
    right_sequence = normalize_sequence(record.right_sequence)
    sequence_pair = ""
    if left_sequence and right_sequence:
        sequence_pair = _key_text(
            unordered_pair_key(sequence_sha256(left_sequence), sequence_sha256(right_sequence))
        )
    return {
        "identifier_pair_key": identifier_pair,
        "sequence_pair_key": sequence_pair,
        "source_dataset": record.source_dataset,
        "source_version": record.source_version,
        "source_row_id": record.source_row_id,
        "identifier_namespace": record.identifier_namespace or record.source_dataset,
        "sequence_mapping_status": "resolved" if sequence_pair else "unresolved",
        "semantic_class": record.semantic_class,
    }


def build_registry(
    registry_name: str,
    records: Iterable[PositiveRecord],
    *,
    allowed_sources: set[str],
    output_path: str | Path,
    manifest_path: str | Path,
    component_metadata: dict[str, Any] | None = None,
    repo_root: str | Path = ".",
) -> RegistryBuildResult:
    """Build one stage registry and fail closed on illegal positive sources."""

    materialized = list(records)
    for record in materialized:
        if record.source_dataset not in allowed_sources:
            raise ExclusionPolicyError(
                f"{registry_name} rejects source {record.source_dataset!r}; "
                f"allowed sources are {sorted(allowed_sources)}"
            )
        if not record.positive:
            raise ExclusionPolicyError(f"{registry_name} accepts positive records only")
        if record.semantic_class in {
            "explicit_assay_negative",
            "assay_non_detection",
            "apms_non_call",
        }:
            raise ExclusionPolicyError(f"{registry_name} received non-positive semantics")
    rows = [_record_row(record) for record in materialized]
    rows.sort(key=lambda row: tuple(row[column] for column in REGISTRY_COLUMNS))
    output = Path(output_path)
    write_csv(output, REGISTRY_COLUMNS, rows)
    identifier_keys = {row["identifier_pair_key"] for row in rows if row["identifier_pair_key"]}
    sequence_keys = {row["sequence_pair_key"] for row in rows if row["sequence_pair_key"]}
    rows_by_identifier: dict[str, set[str]] = defaultdict(set)
    rows_by_sequence: dict[str, set[str]] = defaultdict(set)
    semantic_by_sequence: dict[str, set[str]] = defaultdict(set)
    semantic_by_identifier: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["identifier_pair_key"]:
            rows_by_identifier[row["identifier_pair_key"]].add(row["sequence_pair_key"])
            semantic_by_identifier[row["identifier_pair_key"]].add(row["semantic_class"])
        if row["sequence_pair_key"]:
            rows_by_sequence[row["sequence_pair_key"]].add(row["semantic_class"])
            semantic_by_sequence[row["sequence_pair_key"]].add(row["semantic_class"])
    duplicate_rows = len(rows) - len(
        {tuple(row[column] for column in REGISTRY_COLUMNS) for row in rows}
    )
    duplicate_identifier_rows = len(rows) - len(identifier_keys)
    conflicts = sum(
        len(values & {"curated_negative_reference", "explicit_assay_negative", "negative"}) > 0
        and len(
            values
            & {
                "curated_positive_reference",
                "binary_assay_positive",
                "physical_positive",
                "positive",
            }
        )
        > 0
        for values in semantic_by_identifier.values()
    )
    sequence_heterogeneity = sum(len(values - {""}) > 1 for values in semantic_by_sequence.values())
    manifest = {
        "manifest_type": "ppi_shift.positive_exclusion_registry",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "registry_name": registry_name,
        "allowed_sources": sorted(allowed_sources),
        "component_metadata": component_metadata or {},
        "rows": len(rows),
        "unique_identifier_pairs": len(identifier_keys),
        "unique_sequence_pairs": len(sequence_keys),
        "duplicate_rows": duplicate_rows,
        "duplicate_identifier_pair_rows": duplicate_identifier_rows,
        "conflicting_identifier_pairs": conflicts,
        "conflicting_sequence_pairs": conflicts,
        "sequence_pairs_multiple_semantic_classes": sequence_heterogeneity,
        "source_versions": {
            source: sorted(
                {
                    row["source_version"]
                    for row in rows
                    if row["source_dataset"] == source and row["source_version"]
                }
            )
            for source in sorted({row["source_dataset"] for row in rows})
        },
        "semantic_classes": sorted({row["semantic_class"] for row in rows}),
        "output": {"path": str(output), "sha256": sha256_path(output)},
        "firewall": "positive labels only; stage-specific allowed source set is enforced in code",
    }
    manifest_file = Path(manifest_path)
    write_yaml(manifest_file, manifest)
    return RegistryBuildResult(registry_name, output, manifest_file, manifest)


def _registry_mapping(path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in iter_delimited(path):
        key = (row.get("source_dataset", ""), row.get("source_protein_id", ""))
        if not key[0] or not key[1] or not row.get("sequence_sha256", ""):
            continue
        previous = mapping.get(key)
        if previous and previous.get("sequence_sha256") != row.get("sequence_sha256"):
            mapping[key] = {"ambiguous": "true"}
        elif previous is None:
            mapping[key] = row
    return mapping


def _mapped_sequence(
    mapping: dict[tuple[str, str], dict[str, str]],
    source: str,
    identifier: str,
) -> tuple[str, str]:
    row = mapping.get((source, identifier), {})
    if row.get("ambiguous") == "true":
        return "", "ambiguous"
    return row.get("sequence", ""), row.get("canonical_accession", "")


def _gold_records(source: str, path: str | Path) -> list[PositiveRecord]:
    records: list[PositiveRecord] = []
    for row in iter_delimited(path):
        if str(row.get("interaction_label", "")).strip().lower() not in {"1", "true", "positive"}:
            continue
        records.append(
            PositiveRecord(
                source_dataset=source,
                source_version="gold_21591618_intra",
                left_id=row.get("name1", "").strip(),
                right_id=row.get("name2", "").strip(),
                left_sequence=row.get("seq1", ""),
                right_sequence=row.get("seq2", ""),
                source_row_id=f"{row.get('name1', '')}||{row.get('name2', '')}",
                semantic_class="curated_positive_reference",
                identifier_namespace="Gold_UniProt_accession",
            )
        )
    return records


def _external_records(
    source: str,
    path: str | Path,
    mapping: dict[tuple[str, str], dict[str, str]],
) -> list[PositiveRecord]:
    records: list[PositiveRecord] = []
    for row in iter_delimited(path):
        if source == "vidal":
            if row.get("supervised_target", "") != "1":
                continue
            left_id = f"{row.get('partner_a_role', '')}:{row.get('partner_a_source_id', '')}".strip(
                ":"
            )
            right_id = (
                f"{row.get('partner_b_role', '')}:{row.get('partner_b_source_id', '')}".strip(":")
            )
            left_sequence, right_sequence = (
                row.get("partner_a_sequence", ""),
                row.get("partner_b_sequence", ""),
            )
            canonical_left, canonical_right = "", ""
            version = row.get("source_version", "Lambourne_2024_Data_S4_local_snapshot")
            row_id = row.get("assertion_id", "")
            semantic = "binary_assay_positive"
            namespace = "Vidal_source_role_qualified"
        else:
            left_id = row.get("partner_a_source_id", "").strip()
            right_id = row.get("partner_b_source_id", "").strip()
            left_sequence, canonical_left = _mapped_sequence(mapping, source, left_id)
            right_sequence, canonical_right = _mapped_sequence(mapping, source, right_id)
            version = row.get("source_version", "")
            row_id = row.get("assertion_id", row.get("source_record_id", ""))
            semantic = row.get("observation_class", "")
            namespace = f"{source}_source_identifier"
            if semantic != "physical_positive":
                continue
        records.append(
            PositiveRecord(
                source_dataset=source,
                source_version=version,
                left_id=left_id,
                right_id=right_id,
                left_sequence=left_sequence,
                right_sequence=right_sequence,
                source_row_id=row_id,
                semantic_class=semantic,
                identifier_namespace=namespace,
                canonical_left=canonical_left,
                canonical_right=canonical_right,
            )
        )
    return records


def build_exclusion_registries(
    gold_paths: dict[str, str | Path],
    source_paths: dict[str, str | Path],
    *,
    registry_path: str | Path,
    artifact_root: str | Path,
    repo_root: str | Path = ".",
    evaluation_source_paths: dict[str, str | Path] | None = None,
) -> dict[str, RegistryBuildResult]:
    """Build exactly the three protocol registries."""

    mapping = _registry_mapping(registry_path)
    train_records = _gold_records("gold_train", gold_paths["train"])
    validation_records = train_records + _gold_records("gold_validation", gold_paths["validation"])
    evaluation_records = (
        _gold_records("gold_train", gold_paths["train"])
        + _gold_records("gold_validation", gold_paths["validation"])
        + _gold_records("gold_test", gold_paths["test"])
    )
    evaluation_inputs = dict(source_paths)
    if evaluation_source_paths:
        evaluation_inputs.update(evaluation_source_paths)
    for source in ("vidal", "string", "biogrid", "intact"):
        evaluation_records.extend(_external_records(source, evaluation_inputs[source], mapping))
    root = Path(artifact_root) / "exclusion"
    metadata = {
        "gold_train": {
            "path": str(gold_paths["train"]),
            "sha256": sha256_path(gold_paths["train"]),
        },
        "gold_validation": {
            "path": str(gold_paths["validation"]),
            "sha256": sha256_path(gold_paths["validation"]),
        },
        "gold_test": {"path": str(gold_paths["test"]), "sha256": sha256_path(gold_paths["test"])},
        "external_sources": {
            source: {
                "assertion_path": str(source_paths[source]),
                "assertion_sha256": sha256_path(source_paths[source]),
                "evaluation_path": str(evaluation_inputs[source]),
                "evaluation_sha256": sha256_path(evaluation_inputs[source]),
            }
            for source in ("vidal", "string", "biogrid", "intact")
        },
    }
    outputs = {
        "train_exclusion_registry": build_registry(
            "train_exclusion_registry",
            train_records,
            allowed_sources={"gold_train"},
            output_path=root / "train_exclusion_registry.csv",
            manifest_path=Path(repo_root)
            / "data"
            / "manifests"
            / "train_exclusion_registry_manifest.yaml",
            component_metadata=metadata,
            repo_root=repo_root,
        ),
        "validation_exclusion_registry": build_registry(
            "validation_exclusion_registry",
            validation_records,
            allowed_sources={"gold_train", "gold_validation"},
            output_path=root / "validation_exclusion_registry.csv",
            manifest_path=Path(repo_root)
            / "data"
            / "manifests"
            / "validation_exclusion_registry_manifest.yaml",
            component_metadata=metadata,
            repo_root=repo_root,
        ),
        "evaluation_positive_union": build_registry(
            "evaluation_positive_union",
            evaluation_records,
            allowed_sources={
                "gold_train",
                "gold_validation",
                "gold_test",
                "vidal",
                "string",
                "biogrid",
                "intact",
            },
            output_path=root / "evaluation_positive_union.csv",
            manifest_path=Path(repo_root)
            / "data"
            / "manifests"
            / "evaluation_positive_union_manifest.yaml",
            component_metadata=metadata,
            repo_root=repo_root,
        ),
    }
    return outputs
