"""Canonical pair-level views for assertion-level external evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path, utc_now, write_csv, write_yaml
from ppi_shift.data import unordered_pair_key

CANONICAL_COLUMNS = (
    "assertion_id",
    "source_id",
    "source_version",
    "raw_record_id",
    "raw_record_sha256",
    "raw_artifact_sha256",
    "transformation_version",
    "assay",
    "observation_class",
    "label_semantics",
    "supervised_target",
    "partner_a_source_id",
    "partner_b_source_id",
    "partner_a_role",
    "partner_b_role",
    "partner_a_gene",
    "partner_b_gene",
    "partner_a_accession",
    "partner_b_accession",
    "partner_a_sequence",
    "partner_b_sequence",
    "partner_a_sequence_sha256",
    "partner_b_sequence_sha256",
    "unordered_sequence_pair_key",
    "taxon_a",
    "taxon_b",
    "evidence_reference",
    "source_context_json",
    "canonical_pair_key",
    "pair_level_label",
    "assertion_count",
    "source_assertion_ids",
    "semantic_classes",
    "canonicalization_status",
)


def _pair_key(row: dict[str, str]) -> tuple[str, str] | None:
    left = row.get("partner_a_accession", "").strip() or row.get("partner_a_source_id", "").strip()
    right = row.get("partner_b_accession", "").strip() or row.get("partner_b_source_id", "").strip()
    if not left or not right:
        return None
    return unordered_pair_key(left, right)


def _sort_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("assertion_id", ""),
        row.get("raw_record_id", ""),
        row.get("partner_a_source_id", ""),
        row.get("partner_b_source_id", ""),
    )


def _canonical_row(group: list[dict[str, str]], key: tuple[str, str]) -> dict[str, Any]:
    ordered = sorted(group, key=_sort_key)
    selected = dict(ordered[0])
    semantic_classes = sorted(
        {
            row.get("observation_class", "").strip()
            for row in group
            if row.get("observation_class", "").strip()
        }
    )
    assertion_ids = sorted(
        {
            row.get("assertion_id", row.get("raw_record_id", "")).strip()
            for row in group
            if row.get("assertion_id", row.get("raw_record_id", "")).strip()
        }
    )
    semantic = semantic_classes[0] if len(semantic_classes) == 1 else "conflicting_assertions"
    selected.update(
        {
            "canonical_pair_key": "||".join(key),
            "pair_level_label": semantic,
            "assertion_count": len(group),
            "source_assertion_ids": "|".join(assertion_ids),
            "semantic_classes": "|".join(semantic_classes),
            "canonicalization_status": (
                "collapsed_same_class_duplicates"
                if len(group) > 1 and len(semantic_classes) == 1
                else "one_assertion"
                if len(group) == 1
                else "conflict_excluded"
            ),
        }
    )
    return selected


def build_evaluation_pair_views(
    source_paths: dict[str, str | Path],
    *,
    artifact_root: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Collapse STRING/BioGRID/IntAct assertions to deterministic pair views.

    Same-class duplicate evidence is retained as an assertion count.  A pair with
    contradictory semantic classes is excluded from the binary headline view and
    written to a separate conflict file.  The raw assertion file remains the
    provenance source recorded in the manifest.
    """

    output_root = Path(artifact_root) / "evaluation_pairs"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for source in sorted(source_paths):
        if source == "vidal":
            continue
        raw_path = Path(source_paths[source])
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        malformed: list[dict[str, str]] = []
        for row in iter_delimited(raw_path):
            key = _pair_key(row)
            if key is None:
                malformed.append(row)
            else:
                groups[key].append(row)

        canonical_rows: list[dict[str, Any]] = []
        conflict_rows: list[dict[str, Any]] = []
        class_counts: Counter[str] = Counter()
        for key in sorted(groups):
            group = groups[key]
            classes = {
                row.get("observation_class", "").strip()
                for row in group
                if row.get("observation_class", "").strip()
            }
            selected = _canonical_row(group, key)
            class_counts.update(classes or {""})
            if len(classes) > 1:
                conflict_rows.append(selected)
            else:
                canonical_rows.append(selected)

        output_path = output_root / f"{source}.csv"
        conflict_path = output_root / f"{source}_conflicts.csv"
        write_csv(output_path, CANONICAL_COLUMNS, canonical_rows)
        write_csv(conflict_path, CANONICAL_COLUMNS, conflict_rows)
        manifest_path = output_root / f"{source}_manifest.yaml"
        manifest = {
            "manifest_type": "ppi_shift.evaluation_pair_view",
            "manifest_version": 1,
            "created_utc": utc_now(),
            "source": source,
            "source_path": str(raw_path),
            "source_sha256": sha256_path(raw_path),
            "source_assertion_rows": sum(len(rows) for rows in groups.values()) + len(malformed),
            "source_pair_groups": len(groups),
            "canonical_rows": len(canonical_rows),
            "conflicting_pair_rows": len(conflict_rows),
            "malformed_rows": len(malformed),
            "same_class_duplicate_assertions_collapsed": sum(
                max(0, len(rows) - 1)
                for key, rows in groups.items()
                if len(
                    {
                        row.get("observation_class", "").strip()
                        for row in rows
                        if row.get("observation_class", "").strip()
                    }
                )
                <= 1
            ),
            "semantic_class_counts": dict(sorted(class_counts.items())),
            "canonicalization": (
                "one row per unordered source-identifier pair; same-class duplicates are counted, "
                "contradictory semantic classes are excluded from the binary view and retained in "
                "the conflict output"
            ),
            "output": {"path": str(output_path), "sha256": sha256_path(output_path)},
            "conflicts": {"path": str(conflict_path), "sha256": sha256_path(conflict_path)},
        }
        write_yaml(manifest_path, manifest)
        outputs[source] = {
            "path": output_path,
            "sha256": sha256_path(output_path),
            "rows": len(canonical_rows),
            "manifest_path": manifest_path,
            "manifest_sha256": sha256_path(manifest_path),
            "conflict_path": conflict_path,
            "conflict_sha256": sha256_path(conflict_path),
            "conflict_rows": len(conflict_rows),
        }

    tracked_path = Path(repo_root) / "data" / "manifests" / "evaluation_pair_views_manifest.yaml"
    tracked = {
        "manifest_type": "ppi_shift.evaluation_pair_views",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "views": {
            source: {
                key: str(value)
                for key, value in payload.items()
                if key.endswith("path") or key in {"sha256", "rows", "conflict_rows"}
            }
            for source, payload in outputs.items()
        },
    }
    write_yaml(tracked_path, tracked)
    return {"outputs": outputs, "manifest_path": tracked_path, "manifest": tracked}
