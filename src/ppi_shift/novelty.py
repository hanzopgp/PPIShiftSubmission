"""Gold-relative novelty audits for source-aware evaluation cohorts."""

# The output manifest contains a single explicit semantics sentence.
# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path, utc_now, write_csv, write_yaml
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key
from ppi_shift.evaluation_views import build_evaluation_pair_views

UNKNOWN = "unknown"
UNRESOLVED = "unresolved"
TRUE = "true"
FALSE = "false"
STATUS_VALUES = (TRUE, FALSE, UNKNOWN, UNRESOLVED)

NOVELTY_FIELDS = (
    "identifier_pair_novel",
    "protein_novel",
    "exact_sequence_novel",
    "exact_sequence_pair_novel",
    "family_i40_novel",
    "family_i30_cov80_novel",
    "family_i40_pair_novel",
    "family_i30_cov80_pair_novel",
)
INDIVIDUAL_FAMILY_FIELDS = ("family_i40_novel", "family_i30_cov80_novel")
FAMILY_PAIR_FIELDS = ("family_i40_pair_novel", "family_i30_cov80_pair_novel")
POSITIVE_LABELS = {"1", "true", "positive", "physical_positive", "binary_assay_positive"}
EXPLICIT_NEGATIVE_LABELS = {
    "0",
    "false",
    "negative",
    "explicit_assay_negative",
    "assay_non_detection",
    "apms_non_call",
}


@dataclass
class GoldReference:
    """Frozen Gold membership at identifier, sequence, and component levels.

    ``family_pairs`` is retained as a compatibility alias for older callers.  New
    code must use ``reference_components`` for individual-component novelty and
    ``reference_component_pairs`` only for the separate pair-of-components diagnostic.
    """

    name: str
    identifier_pairs: set[tuple[str, str]]
    proteins: set[str]
    sequence_pairs: set[tuple[str, str]]
    sequences: set[str]
    family_pairs: dict[str, set[tuple[str, str]]] = dataclass_field(default_factory=dict)
    reference_components: dict[str, set[str]] = dataclass_field(default_factory=dict)
    reference_component_pairs: dict[str, set[tuple[str, str]]] = dataclass_field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        namespaces = (
            set(self.family_pairs)
            | set(self.reference_components)
            | set(self.reference_component_pairs)
        )
        for namespace in namespaces:
            old_pairs = set(self.family_pairs.get(namespace, set()))
            component_pairs = set(self.reference_component_pairs.get(namespace, old_pairs))
            self.reference_component_pairs[namespace] = component_pairs
            self.family_pairs[namespace] = component_pairs
            self.reference_components.setdefault(namespace, set())


@dataclass(frozen=True)
class NoveltyBuildResult:
    summary_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _status(value: bool | None, *, unresolved: bool = False) -> str:
    if unresolved:
        return UNRESOLVED
    if value is None:
        return UNKNOWN
    return TRUE if value else FALSE


def load_components(path: str | Path) -> dict[str, str]:
    """Load sequence-hash to component ID assignments."""

    result: dict[str, str] = {}
    for row in iter_delimited(path):
        identifier = row.get("sequence_sha256", "").strip()
        component = row.get("component_id", "").strip()
        if identifier and component:
            result[identifier] = component
    return result


def _gold_reference(
    name: str,
    paths: list[str | Path],
    components: dict[str, dict[str, str]],
) -> GoldReference:
    identifier_pairs: set[tuple[str, str]] = set()
    proteins: set[str] = set()
    sequence_pairs: set[tuple[str, str]] = set()
    sequences: set[str] = set()
    reference_components: dict[str, set[str]] = {key: set() for key in components}
    reference_component_pairs: dict[str, set[tuple[str, str]]] = {key: set() for key in components}
    for path in paths:
        for row in iter_delimited(path):
            left_id, right_id = row.get("name1", "").strip(), row.get("name2", "").strip()
            left_seq = normalize_sequence(row.get("seq1", ""))
            right_seq = normalize_sequence(row.get("seq2", ""))
            if left_id and right_id:
                identifier_pairs.add(unordered_pair_key(left_id, right_id))
                proteins.update((left_id, right_id))
            if not left_seq or not right_seq:
                continue
            left_hash, right_hash = sequence_sha256(left_seq), sequence_sha256(right_seq)
            sequence_pairs.add(unordered_pair_key(left_hash, right_hash))
            sequences.update((left_hash, right_hash))
            for namespace, assignments in components.items():
                left_component = assignments.get(left_hash)
                right_component = assignments.get(right_hash)
                if left_component:
                    reference_components[namespace].add(left_component)
                if right_component:
                    reference_components[namespace].add(right_component)
                if left_component and right_component:
                    reference_component_pairs[namespace].add(
                        unordered_pair_key(left_component, right_component)
                    )
    return GoldReference(
        name=name,
        identifier_pairs=identifier_pairs,
        proteins=proteins,
        sequence_pairs=sequence_pairs,
        sequences=sequences,
        family_pairs=reference_component_pairs,
        reference_components=reference_components,
        reference_component_pairs=reference_component_pairs,
    )


def _registry_index(registry_path: str | Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in iter_delimited(registry_path):
        index[(row.get("source_dataset", ""), row.get("source_protein_id", ""))].append(row)
    return index


def _mapping(
    index: dict[tuple[str, str], list[dict[str, str]]],
    dataset: str,
    identifier: str,
) -> tuple[str, str, str, str]:
    rows = index.get((dataset, identifier), [])
    hashes = {row.get("sequence_sha256", "") for row in rows if row.get("sequence_sha256", "")}
    if not rows or not hashes:
        return "", UNRESOLVED, "", ""
    statuses = {row.get("mapping_status", "") for row in rows}
    if len(hashes) != 1 or any(status.startswith("ambiguous") for status in statuses):
        return "", UNRESOLVED, "", "ambiguous_or_incompatible_mapping"
    row = next(row for row in rows if row.get("sequence_sha256", ""))
    return (
        next(iter(hashes)),
        "resolved",
        row.get("canonical_accession", "").strip(),
        row.get("mapping_status", ""),
    )


def _row_source_fields(
    source: str, row: dict[str, str]
) -> tuple[str, str, str, str, str, str, str]:
    """Return IDs, sequences, semantics, and a stable row identifier for one source."""

    if source == "vidal":
        left = f"{row.get('partner_a_role', '')}:{row.get('partner_a_source_id', '')}".strip(":")
        right = f"{row.get('partner_b_role', '')}:{row.get('partner_b_source_id', '')}".strip(":")
        return (
            left,
            right,
            row.get("partner_a_sequence", ""),
            row.get("partner_b_sequence", ""),
            row.get("observation_class", ""),
            row.get("label_semantics", ""),
            row.get("assertion_id", row.get("raw_record_id", "")),
        )
    left = row.get("partner_a_source_id", "").strip()
    right = row.get("partner_b_source_id", "").strip()
    return (
        left,
        right,
        row.get("partner_a_sequence", ""),
        row.get("partner_b_sequence", ""),
        row.get("observation_class", ""),
        row.get("label_semantics", ""),
        row.get("assertion_id", row.get("source_record_id", "")),
    )


def _source_label(source: str, row: dict[str, str]) -> str:
    if source == "vidal":
        return row.get("supervised_target", "")
    return row.get("observation_class", "")


def _novelty_for_row(
    source: str,
    row: dict[str, str],
    reference: GoldReference,
    registry: dict[tuple[str, str], list[dict[str, str]]],
    components: dict[str, dict[str, str]],
) -> dict[str, str]:
    left, right, raw_left_seq, raw_right_seq, semantics, label_semantics, row_id = (
        _row_source_fields(source, row)
    )
    left_hash, left_mapping, left_canonical, left_status = _mapping(registry, source, left)
    right_hash, right_mapping, right_canonical, right_status = _mapping(registry, source, right)
    if source == "vidal":
        if raw_left_seq:
            left_hash = sequence_sha256(normalize_sequence(raw_left_seq))
            left_mapping, left_status = "resolved", "source_sequence"
        if raw_right_seq:
            right_hash = sequence_sha256(normalize_sequence(raw_right_seq))
            right_mapping, right_status = "resolved", "source_sequence"

    sequences_resolved = left_mapping == "resolved" and right_mapping == "resolved"
    canonical_resolved = bool(left_canonical and right_canonical)
    identifier_pair: bool | None = None
    protein_novel: bool | None = None
    if canonical_resolved:
        identifier_pair = (
            unordered_pair_key(left_canonical, right_canonical) not in reference.identifier_pairs
        )
        protein_novel = (
            left_canonical not in reference.proteins and right_canonical not in reference.proteins
        )

    sequence_pair_key: tuple[str, str] | None = None
    if sequences_resolved:
        sequence_pair_key = unordered_pair_key(left_hash, right_hash)
    exact_sequence_novel: bool | None = None
    exact_sequence_pair_novel: bool | None = None
    if sequences_resolved:
        exact_sequence_novel = (
            left_hash not in reference.sequences and right_hash not in reference.sequences
        )
        exact_sequence_pair_novel = sequence_pair_key not in reference.sequence_pairs

    row_output = {
        "row_id": row_id,
        "source": source,
        "source_left_id": left,
        "source_right_id": right,
        "canonical_left": left_canonical,
        "canonical_right": right_canonical,
        "label": _source_label(source, row),
        "observation_class": semantics,
        "label_semantics": label_semantics,
        "left_sequence_sha256": left_hash,
        "right_sequence_sha256": right_hash,
        "sequence_mapping_status": "resolved" if sequences_resolved else UNRESOLVED,
        "left_mapping_status": left_status,
        "right_mapping_status": right_status,
        "identifier_pair_novel": _status(identifier_pair, unresolved=not canonical_resolved),
        "exact_sequence_pair_novel": _status(
            exact_sequence_pair_novel, unresolved=not sequences_resolved
        ),
        "pair_novel": _status(identifier_pair, unresolved=not canonical_resolved),
        "protein_novel": _status(protein_novel, unresolved=not canonical_resolved),
        "exact_sequence_novel": _status(exact_sequence_novel, unresolved=not sequences_resolved),
    }
    for namespace, assignments in components.items():
        left_component = assignments.get(left_hash) if sequences_resolved else None
        right_component = assignments.get(right_hash) if sequences_resolved else None
        row_output[f"family_{namespace}_left_component"] = left_component or ""
        row_output[f"family_{namespace}_right_component"] = right_component or ""
        family_novel: bool | None = None
        family_pair_novel: bool | None = None
        if left_component and right_component:
            # This is individual-component novelty.  The pair-of-components check
            # below is intentionally a separate diagnostic and never replaces it.
            family_novel = left_component not in reference.reference_components.get(
                namespace, set()
            ) and right_component not in reference.reference_components.get(namespace, set())
            family_pair_novel = unordered_pair_key(
                left_component, right_component
            ) not in reference.reference_component_pairs.get(namespace, set())
        row_output[f"family_{namespace}_novel"] = _status(
            family_novel,
            unresolved=not sequences_resolved or not (left_component and right_component),
        )
        row_output[f"family_{namespace}_pair_novel"] = _status(
            family_pair_novel,
            unresolved=not sequences_resolved or not (left_component and right_component),
        )
    return row_output


def validate_novelty_rows(rows: list[dict[str, str]]) -> None:
    """Enforce the fail-closed component and exact-sequence invariants."""

    for row in rows:
        for namespace in ("i40", "i30_cov80"):
            field = f"family_{namespace}_novel"
            left = row.get(f"family_{namespace}_left_component", "")
            right = row.get(f"family_{namespace}_right_component", "")
            if row.get("sequence_mapping_status") == "resolved" and row.get(field) != UNRESOLVED:
                if not left or not right:
                    raise ValueError(
                        f"{field} was evaluated without two component assignments for {row.get('row_id')}"
                    )
            if row.get("exact_sequence_novel") == FALSE and row.get(field) == TRUE:
                raise ValueError(
                    f"{field} contradicts exact sequence membership for {row.get('row_id')}"
                )


def _is_positive(row: dict[str, str]) -> bool:
    return row.get("label", "").strip().lower() in POSITIVE_LABELS


def _is_explicit_negative(row: dict[str, str]) -> bool:
    return row.get("label", "").strip().lower() in EXPLICIT_NEGATIVE_LABELS


def _status_counts(
    rows: list[dict[str, str]], field: str, predicate: Any | None = None
) -> dict[str, int]:
    selected = rows if predicate is None else [row for row in rows if predicate(row)]
    counts = Counter(row.get(field, UNKNOWN) for row in selected)
    return {status: counts.get(status, 0) for status in STATUS_VALUES}


def _add_status_counts(target: dict[str, Any], rows: list[dict[str, str]], field: str) -> None:
    subsets = {
        "": lambda row: True,
        "positive_": _is_positive,
        "explicit_negative_": _is_explicit_negative,
    }
    for prefix, predicate in subsets.items():
        counts = _status_counts(rows, field, predicate)
        for status, count in counts.items():
            target[f"{prefix}{field}_{status}_rows"] = count


def _nestedness_row(
    source: str,
    reference: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": source,
        "reference": reference,
        "total_rows": len(rows),
        "sequence_resolved_rows": sum(
            row.get("sequence_mapping_status") == "resolved" for row in rows
        ),
        "complete_component_assignment_rows": sum(
            row.get("family_i40_novel") != UNRESOLVED
            and row.get("family_i30_cov80_novel") != UNRESOLVED
            for row in rows
        ),
    }
    for field in INDIVIDUAL_FAMILY_FIELDS:
        counts = _status_counts(rows, field)
        for status, count in counts.items():
            result[f"{field}_{status}_rows"] = count
    both_known = [
        row
        for row in rows
        if row.get("family_i40_novel") in {TRUE, FALSE}
        and row.get("family_i30_cov80_novel") in {TRUE, FALSE}
    ]
    result["i30_novel_and_i40_novel_rows"] = sum(
        row["family_i30_cov80_novel"] == TRUE and row["family_i40_novel"] == TRUE
        for row in both_known
    )
    result["i30_novel_not_i40_novel_rows"] = sum(
        row["family_i30_cov80_novel"] == TRUE and row["family_i40_novel"] != TRUE
        for row in both_known
    )
    result["i40_novel_not_i30_novel_rows"] = sum(
        row["family_i40_novel"] == TRUE and row["family_i30_cov80_novel"] != TRUE
        for row in both_known
    )
    result["nestedness_known_rows"] = len(both_known)
    result["nestedness_anomaly_rows"] = result["i30_novel_not_i40_novel_rows"]
    result["reverse_direction_rows"] = result["i40_novel_not_i30_novel_rows"]
    for prefix, predicate in (
        ("positive_", _is_positive),
        ("explicit_negative_", _is_explicit_negative),
    ):
        subset = [row for row in both_known if predicate(row)]
        result[f"{prefix}nestedness_known_rows"] = len(subset)
        result[f"{prefix}i30_novel_not_i40_novel_rows"] = sum(
            row["family_i30_cov80_novel"] == TRUE and row["family_i40_novel"] != TRUE
            for row in subset
        )
        result[f"{prefix}i40_novel_not_i30_novel_rows"] = sum(
            row["family_i40_novel"] == TRUE and row["family_i30_cov80_novel"] != TRUE
            for row in subset
        )
    return result


def build_novelty_cohorts(
    source_paths: dict[str, str | Path],
    gold_paths: dict[str, str | Path],
    *,
    registry_path: str | Path,
    component_paths: dict[str, str | Path],
    artifact_root: str | Path,
    repo_root: str | Path = ".",
) -> NoveltyBuildResult:
    """Materialize Gold-relative novelty surfaces and a separate nestedness audit."""

    components = {namespace: load_components(path) for namespace, path in component_paths.items()}
    registry = _registry_index(registry_path)
    reference_paths = {
        "gold_train": [gold_paths["train"]],
        "gold_trainval": [gold_paths["train"], gold_paths["validation"]],
    }
    references = {
        name: _gold_reference(name, paths, components) for name, paths in reference_paths.items()
    }
    view_result = build_evaluation_pair_views(
        source_paths, artifact_root=artifact_root, repo_root=repo_root
    )
    effective_source_paths = dict(source_paths)
    for source, payload in view_result["outputs"].items():
        effective_source_paths[source] = payload["path"]

    output_root = Path(artifact_root) / "novelty"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    nestedness_rows: list[dict[str, Any]] = []
    manifest_outputs: dict[str, Any] = {}
    output_columns = (
        "row_id",
        "source",
        "source_left_id",
        "source_right_id",
        "canonical_left",
        "canonical_right",
        "label",
        "observation_class",
        "label_semantics",
        "left_sequence_sha256",
        "right_sequence_sha256",
        "sequence_mapping_status",
        "left_mapping_status",
        "right_mapping_status",
        "identifier_pair_novel",
        "exact_sequence_pair_novel",
        "pair_novel",
        "protein_novel",
        "exact_sequence_novel",
        "family_i40_left_component",
        "family_i40_right_component",
        "family_i40_novel",
        "family_i40_pair_novel",
        "family_i30_cov80_left_component",
        "family_i30_cov80_right_component",
        "family_i30_cov80_novel",
        "family_i30_cov80_pair_novel",
    )
    for source in sorted(effective_source_paths):
        source_rows = list(iter_delimited(effective_source_paths[source]))
        for reference_name, reference in references.items():
            output_rows = [
                _novelty_for_row(source, row, reference, registry, components)
                for row in source_rows
            ]
            validate_novelty_rows(output_rows)
            output_rows.sort(
                key=lambda row: (row["row_id"], row["source_left_id"], row["source_right_id"])
            )
            output_path = output_root / f"{source}_{reference_name}_novelty.csv"
            write_csv(output_path, output_columns, output_rows)
            manifest_path = output_root / f"{source}_{reference_name}_manifest.yaml"
            manifest = {
                "manifest_type": "ppi_shift.gold_relative_novelty_cohort",
                "manifest_version": 2,
                "created_utc": utc_now(),
                "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
                    repo_root
                ),
                "source": source,
                "reference": reference_name,
                "reference_paths": [str(path) for path in reference_paths[reference_name]],
                "reference_hashes": [sha256_path(path) for path in reference_paths[reference_name]],
                "registry_path": str(registry_path),
                "registry_sha256": sha256_path(registry_path),
                "component_paths": {key: str(value) for key, value in component_paths.items()},
                "component_hashes": {
                    key: sha256_path(value) for key, value in component_paths.items()
                },
                "source_path": str(effective_source_paths[source]),
                "source_sha256": sha256_path(effective_source_paths[source]),
                "source_assertion_path": str(source_paths[source]),
                "source_assertion_sha256": sha256_path(source_paths[source]),
                "semantics": (
                    "status values are true/false/unknown/unresolved; unknown and unresolved are "
                    "never counted as novel; family novelty is individual-component novelty "
                    "(both components unseen in the Gold reference), while family_*_pair_novel is "
                    "a separate pair-of-components diagnostic"
                ),
                "output": {
                    "path": str(output_path),
                    "sha256": sha256_path(output_path),
                    "rows": len(output_rows),
                },
            }
            write_yaml(manifest_path, manifest)
            manifest_outputs[f"{source}:{reference_name}"] = {
                "path": str(output_path),
                "sha256": sha256_path(output_path),
                "rows": len(output_rows),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_path(manifest_path),
            }
            summary: dict[str, Any] = {
                "source": source,
                "reference": reference_name,
                "total_rows": len(output_rows),
                "sequence_resolved_rows": sum(
                    row["sequence_mapping_status"] == "resolved" for row in output_rows
                ),
                "unresolved_rows": sum(
                    any(row.get(field) == UNRESOLVED for field in NOVELTY_FIELDS)
                    for row in output_rows
                ),
                "positive_rows": sum(_is_positive(row) for row in output_rows),
                "explicit_negative_rows": sum(_is_explicit_negative(row) for row in output_rows),
                "output_path": str(output_path),
                "output_sha256": sha256_path(output_path),
            }
            for field in NOVELTY_FIELDS:
                _add_status_counts(summary, output_rows, field)
            for namespace in components:
                summary[f"family_{namespace}_assignment_resolved_rows"] = sum(
                    bool(
                        row.get(f"family_{namespace}_left_component", "")
                        and row.get(f"family_{namespace}_right_component", "")
                    )
                    for row in output_rows
                )
                summary[f"family_{namespace}_assignment_unresolved_rows"] = (
                    len(output_rows) - summary[f"family_{namespace}_assignment_resolved_rows"]
                )
            summary["strictest_all_known_true_rows"] = sum(
                all(
                    row.get(field) == TRUE
                    for field in (
                        "pair_novel",
                        "protein_novel",
                        "exact_sequence_novel",
                        *INDIVIDUAL_FAMILY_FIELDS,
                    )
                )
                for row in output_rows
            )
            summary["explicit_negative_strictest_rows"] = sum(
                _is_explicit_negative(row)
                and all(
                    row.get(field) == TRUE
                    for field in (
                        "pair_novel",
                        "protein_novel",
                        "exact_sequence_novel",
                        *INDIVIDUAL_FAMILY_FIELDS,
                    )
                )
                for row in output_rows
            )
            summary_rows.append(summary)
            nestedness_rows.append(_nestedness_row(source, reference_name, output_rows))

    summary_path = Path(repo_root) / "audit" / "novelty" / "summary.csv"
    summary_columns = tuple(summary_rows[0].keys()) if summary_rows else ("source", "reference")
    write_csv(
        summary_path,
        summary_columns,
        sorted(summary_rows, key=lambda row: (row["source"], row["reference"])),
    )
    nestedness_path = Path(repo_root) / "audit" / "novelty" / "nestedness.csv"
    nestedness_columns = (
        tuple(nestedness_rows[0].keys()) if nestedness_rows else ("source", "reference")
    )
    write_csv(
        nestedness_path,
        nestedness_columns,
        sorted(nestedness_rows, key=lambda row: (row["source"], row["reference"])),
    )
    manifest_path = Path(repo_root) / "data" / "manifests" / "novelty_manifest.yaml"
    manifest = {
        "manifest_type": "ppi_shift.novelty_cohorts",
        "manifest_version": 2,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "gold_references": {
            key: [str(path) for path in value] for key, value in reference_paths.items()
        },
        "gold_reference_hashes": {
            key: [sha256_path(path) for path in value] for key, value in reference_paths.items()
        },
        "registry_path": str(registry_path),
        "registry_sha256": sha256_path(registry_path),
        "components": {
            key: {"path": str(value), "sha256": sha256_path(value)}
            for key, value in component_paths.items()
        },
        "evaluation_pair_views": {
            "manifest_path": str(view_result["manifest_path"]),
            "manifest_sha256": sha256_path(view_result["manifest_path"]),
            "outputs": {
                source: {
                    key: str(value)
                    for key, value in payload.items()
                    if key
                    in {
                        "path",
                        "sha256",
                        "rows",
                        "conflict_path",
                        "conflict_sha256",
                        "conflict_rows",
                    }
                }
                for source, payload in view_result["outputs"].items()
            },
        },
        "outputs": manifest_outputs,
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_path(summary_path),
            "rows": len(summary_rows),
        },
        "nestedness": {
            "path": str(nestedness_path),
            "sha256": sha256_path(nestedness_path),
            "rows": len(nestedness_rows),
            "interpretation": (
                "Both directional differences are reported. The expected strictness direction "
                "must be stated with the chosen component construction; no direction is silently assumed."
            ),
        },
        "command": "scripts/build_novelty_cohorts.py --config configs/data/prefreeze_sources.yaml",
    }
    write_yaml(manifest_path, manifest)
    return NoveltyBuildResult(summary_path, manifest_path, manifest)
