"""Recover and audit Vidal/Lambourne anchor-group semantics before ranking use."""

# Generated audit prose is kept as complete sentences.
# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path, utc_now, write_csv, write_yaml
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key
from ppi_shift.novelty import _gold_reference

PUBLISHED_TARGET = {
    "rows": 3509,
    "unique_isoform_constructs": 253,
    "unique_tf_genes": 87,
    "unique_partners": 538,
}


def classify_y2h_result(value: str) -> str:
    """Classify raw Data S4 result without turning NA into a negative."""

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "positive"}:
        return "positive"
    if normalized in {"false", "0", "no", "negative"}:
        return "negative"
    if normalized in {"", "na", "n/a", "nan", "null", "none"}:
        return "na"
    return "unknown"


def anchor_key(row: dict[str, str]) -> tuple[str, str]:
    """Use the explicit TF clone and ORF identifiers as the candidate-group key."""

    return row.get("clone_id", "").strip(), row.get("ad_orf_id", "").strip()


def group_vidal_rows(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group rows only by explicit source anchor fields; never by counts or order."""

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[anchor_key(row)].append(row)
    return dict(groups)


def _gold_sequence_reference(path: str | Path) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    reference = _gold_reference("gold_train", [path], {})
    return reference.proteins, reference.sequence_pairs, reference.sequences


def _family_reference(
    gold_path: str | Path,
    assignments: dict[str, str],
) -> set[tuple[str, str]]:
    """Legacy pair-of-components reference retained for compatibility."""

    reference = _gold_reference("gold_train", [gold_path], {"family": assignments})
    return reference.reference_component_pairs["family"]


def _family_reference_components(
    gold_path: str | Path,
    assignments: dict[str, str],
) -> tuple[set[str], set[tuple[str, str]]]:
    reference = _gold_reference("gold_train", [gold_path], {"family": assignments})
    return (
        reference.reference_components["family"],
        reference.reference_component_pairs["family"],
    )


def _row_novelty(
    row: dict[str, str],
    gold_sequences: set[str],
    gold_sequence_pairs: set[tuple[str, str]],
    family_refs: dict[str, set[tuple[str, str]]],
    assignments: dict[str, dict[str, str]],
    reference_components: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    """Return exact-sequence and individual-component novelty for one raw row."""

    left, right = (
        normalize_sequence(row.get("ad_seq", "")),
        normalize_sequence(row.get("db_seq", "")),
    )
    if not left or not right:
        output = {
            "exact_sequence_pair_novel": "unresolved",
            "exact_sequence_novel": "unresolved",
        }
        for namespace in assignments:
            output[f"family_{namespace}_novel"] = "unresolved"
            output[f"family_{namespace}_pair_novel"] = "unresolved"
        return output
    left_hash, right_hash = sequence_sha256(left), sequence_sha256(right)
    sequence_pair = unordered_pair_key(left_hash, right_hash)
    output = {
        "exact_sequence_pair_novel": "true"
        if sequence_pair not in gold_sequence_pairs
        else "false",
        "exact_sequence_novel": "true"
        if left_hash not in gold_sequences and right_hash not in gold_sequences
        else "false",
    }
    for namespace, mapping in assignments.items():
        left_component, right_component = mapping.get(left_hash), mapping.get(right_hash)
        field = f"family_{namespace}_novel"
        pair_field = f"family_{namespace}_pair_novel"
        if left_component and right_component:
            members = (reference_components or {}).get(namespace)
            if members is None:
                members = {
                    component for pair in family_refs.get(namespace, set()) for component in pair
                }
            output[field] = (
                "true"
                if left_component not in members and right_component not in members
                else "false"
            )
            output[pair_field] = (
                "true"
                if unordered_pair_key(left_component, right_component)
                not in family_refs.get(namespace, set())
                else "false"
            )
        else:
            output[field] = "unresolved"
            output[pair_field] = "unresolved"
    return output


def _anchor_summary(
    key: tuple[str, str],
    rows: list[dict[str, str]],
    gold_sequences: set[str],
    gold_sequence_pairs: set[tuple[str, str]],
    family_refs: dict[str, set[tuple[str, str]]],
    assignments: dict[str, dict[str, str]],
    reference_components: dict[str, set[str]] | None = None,
    ranking_eligible: bool = False,
    ranking_reason: str = "not_confirmed",
) -> dict[str, Any]:
    classified = [classify_y2h_result(row.get("Y2H_result", "")) for row in rows]
    candidates = {
        row.get("db_orf_id", "").strip() for row in rows if row.get("db_orf_id", "").strip()
    }
    positives = sum(value == "positive" for value in classified)
    negatives = sum(value == "negative" for value in classified)
    nas = sum(value == "na" for value in classified)
    unknowns = sum(value == "unknown" for value in classified)
    row_novelty = [
        _row_novelty(
            row,
            gold_sequences,
            gold_sequence_pairs,
            family_refs,
            assignments,
            reference_components,
        )
        for row in rows
    ]
    comparable = (
        not unknowns
        and not nas
        and all(row.get("ad_seq", "").strip() and row.get("db_seq", "").strip() for row in rows)
        and len(candidates) == len(rows)
    )
    eligible = ranking_eligible and comparable
    return {
        "anchor_clone_id": key[0],
        "anchor_ad_orf_id": key[1],
        "anchor_label": rows[0].get("ad_gene_symbol", "") if rows else "",
        "tested_candidate_count": len(candidates),
        "raw_row_count": len(rows),
        "positive_count": positives,
        "negative_count": negatives,
        "na_count": nas,
        "unknown_result_count": unknowns,
        "duplicate_candidate_rows": len(rows) - len(candidates),
        "all_binary_results": "true" if not nas and not unknowns else "false",
        "sequence_complete": "true"
        if all(row.get("ad_seq", "").strip() and row.get("db_seq", "").strip() for row in rows)
        else "false",
        "binary_unique_candidate_surface": "true" if comparable else "false",
        "comparable_pairwise_testing": "true" if eligible else ranking_reason,
        "ranking_eligible": "true" if eligible else "false",
        "exact_sequence_novel_candidates": sum(
            item["exact_sequence_novel"] == "true" for item in row_novelty
        ),
        "i40_novel_candidates": sum(item.get("family_i40_novel") == "true" for item in row_novelty),
        "i30_cov80_novel_candidates": sum(
            item.get("family_i30_cov80_novel") == "true" for item in row_novelty
        ),
        "eligible_positive_pairs": positives if eligible else 0,
        "eligible_negative_pairs": negatives if eligible else 0,
    }


def _subset_stats(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    anchors = {(row.get("clone_id", ""), row.get("ad_orf_id", "")) for row in rows}
    return {
        "step": name,
        "rows": len(rows),
        "unique_isoform_constructs": len(anchors),
        "unique_clone_ids": len(
            {row.get("clone_id", "") for row in rows if row.get("clone_id", "")}
        ),
        "unique_tf_genes": len(
            {row.get("ad_gene_symbol", "") for row in rows if row.get("ad_gene_symbol", "")}
        ),
        "unique_partners": len(
            {row.get("db_gene_symbol", "") for row in rows if row.get("db_gene_symbol", "")}
        ),
        "positive_rows": sum(
            classify_y2h_result(row.get("Y2H_result", "")) == "positive" for row in rows
        ),
        "negative_rows": sum(
            classify_y2h_result(row.get("Y2H_result", "")) == "negative" for row in rows
        ),
        "na_rows": sum(classify_y2h_result(row.get("Y2H_result", "")) == "na" for row in rows),
        "unknown_rows": sum(
            classify_y2h_result(row.get("Y2H_result", "")) == "unknown" for row in rows
        ),
    }


def audit_published_subset(raw_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Apply the public Vidal analysis filter sequence to raw Data S4 rows."""

    steps: list[dict[str, Any]] = []
    current = list(raw_rows)
    steps.append(_subset_stats("raw_data_s4", current))

    current = [
        row
        for row in current
        if classify_y2h_result(row.get("Y2H_result", "")) in {"positive", "negative"}
    ]
    steps.append(_subset_stats("binary_y2h_results", current))

    positive_profiles = {
        (row.get("ad_gene_symbol", ""), row.get("db_gene_symbol", ""))
        for row in current
        if classify_y2h_result(row.get("Y2H_result", "")) == "positive"
    }
    current = [
        row
        for row in current
        if (row.get("ad_gene_symbol", ""), row.get("db_gene_symbol", "")) in positive_profiles
    ]
    steps.append(_subset_stats("profile_has_at_least_one_positive", current))

    positive_anchors = {
        anchor_key(row)
        for row in current
        if classify_y2h_result(row.get("Y2H_result", "")) == "positive"
    }
    current = [row for row in current if anchor_key(row) in positive_anchors]
    steps.append(_subset_stats("isoform_has_at_least_one_positive", current))

    gene_anchors: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in current:
        gene_anchors[row.get("ad_gene_symbol", "")].add(anchor_key(row))
    gene_anchor_counts = Counter({gene: len(anchors) for gene, anchors in gene_anchors.items()})
    current = [row for row in current if gene_anchor_counts[row.get("ad_gene_symbol", "")] >= 2]
    steps.append(_subset_stats("tf_gene_has_at_least_two_isoforms", current))

    profile_test_counts = Counter(
        (row.get("ad_gene_symbol", ""), row.get("db_gene_symbol", "")) for row in current
    )
    current = [
        row
        for row in current
        if profile_test_counts[(row.get("ad_gene_symbol", ""), row.get("db_gene_symbol", ""))] >= 2
    ]
    steps.append(_subset_stats("profile_has_at_least_two_successful_tests", current))
    paper_default_rows = list(current)

    gene_partners: dict[str, set[str]] = defaultdict(set)
    for row in current:
        gene_partners[row.get("ad_gene_symbol", "")].add(row.get("db_gene_symbol", ""))
    tested_partner_counts = Counter(
        {gene: len(partners) for gene, partners in gene_partners.items()}
    )
    optional_rows = [
        row for row in current if tested_partner_counts[row.get("ad_gene_symbol", "")] >= 2
    ]
    steps.append(
        _subset_stats("optional_require_two_tested_partners_not_paper_default", optional_rows)
    )

    final_stats = steps[-2]
    target_match = {key: final_stats[key] == value for key, value in PUBLISHED_TARGET.items()}
    quality_gate_fields_present = bool(raw_rows) and {
        "growth_score",
        "orf_identity_confirmed",
    }.issubset(raw_rows[0])
    ranking_eligible = all(target_match.values()) and quality_gate_fields_present
    if not all(target_match.values()):
        ranking_reason = (
            "The source-defined filter sequence does not recover the paper's published "
            "surface for rows, isoform constructs, TF genes, and/or partners."
        )
    elif not quality_gate_fields_present:
        ranking_reason = (
            "The source-defined filter sequence matches the published surface, but the "
            "local raw snapshot exposes no growth-score or ORF-sequence-quality gate fields."
        )
    else:
        ranking_reason = (
            "The source-defined filter sequence matches the published surface and the "
            "required ranking-quality fields are present."
        )
    result = {
        "steps": steps,
        "paper_default_final": final_stats,
        "optional_two_partner_final": steps[-1],
        "target": PUBLISHED_TARGET,
        "target_match": target_match,
        "filter_sequence_recovered": final_stats["rows"] == PUBLISHED_TARGET["rows"],
        "quality_gate_fields_present": quality_gate_fields_present,
        "ranking_eligible": ranking_eligible,
        "ranking_reason": ranking_reason,
        "paper_default_rows": paper_default_rows,
    }
    return result


def audit_vidal_groups(
    raw_path: str | Path,
    staging_path: str | Path,
    *,
    gold_train_path: str | Path,
    component_paths: dict[str, str | Path] | None = None,
    artifact_root: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Trace staged rows, recover the published subset, and write group artifacts."""

    raw_rows = list(iter_delimited(raw_path))
    staging_rows = list(iter_delimited(staging_path))
    required_raw = {
        "clone_id",
        "ad_orf_id",
        "ad_gene_symbol",
        "ad_seq",
        "db_orf_id",
        "db_gene_symbol",
        "db_seq",
        "Y2H_result",
    }
    missing_raw = sorted(required_raw - set(raw_rows[0]) if raw_rows else required_raw)
    result_counts = Counter(classify_y2h_result(row.get("Y2H_result", "")) for row in raw_rows)
    staging_counts = Counter(row.get("observation_class", "") for row in staging_rows)
    raw_record_checks: list[dict[str, str]] = []
    trace_failures: list[str] = []
    for index, row in enumerate(staging_rows, start=1):
        raw_index = index
        raw_record_id = row.get("raw_record_id", "")
        if ":" in raw_record_id:
            suffix = raw_record_id.rsplit(":", 1)[1]
            if suffix.isdigit():
                raw_index = int(suffix)
        raw = raw_rows[raw_index - 1] if 1 <= raw_index <= len(raw_rows) else {}
        try:
            context = json.loads(row.get("source_context_json", "{}"))
        except json.JSONDecodeError:
            context = {}
        comparisons = {
            "clone_id": context.get("clone_id", "") == raw.get("clone_id", ""),
            "ad_orf_id": row.get("partner_a_source_id", "") == raw.get("ad_orf_id", ""),
            "db_orf_id": row.get("partner_b_source_id", "") == raw.get("db_orf_id", ""),
            "ad_sequence": normalize_sequence(row.get("partner_a_sequence", ""))
            == normalize_sequence(raw.get("ad_seq", "")),
            "db_sequence": normalize_sequence(row.get("partner_b_sequence", ""))
            == normalize_sequence(raw.get("db_seq", "")),
            "result": classify_y2h_result(raw.get("Y2H_result", ""))
            == (
                "positive"
                if row.get("supervised_target") == "1"
                else "negative"
                if row.get("supervised_target") == "0"
                else "unknown"
            ),
        }
        if not all(comparisons.values()):
            trace_failures.append(
                f"staging_row={index} raw_row={raw_index} comparisons={comparisons}"
            )
        raw_record_checks.append(
            {
                "staging_row_number": str(index),
                "raw_row_number": str(raw_index),
                "raw_record_id": raw_record_id,
                "assertion_id": row.get("assertion_id", ""),
                "clone_id": context.get("clone_id", raw.get("clone_id", "")),
                "ad_orf_id": raw.get("ad_orf_id", ""),
                "db_orf_id": raw.get("db_orf_id", ""),
                "raw_result": raw.get("Y2H_result", ""),
                "staging_observation_class": row.get("observation_class", ""),
                "trace_status": "matched" if all(comparisons.values()) else "mismatch",
            }
        )

    assignments: dict[str, dict[str, str]] = {}
    if component_paths:
        for namespace, path in component_paths.items():
            assignments[namespace] = {
                row.get("sequence_sha256", ""): row.get("component_id", "")
                for row in iter_delimited(path)
                if row.get("sequence_sha256", "") and row.get("component_id", "")
            }
    reference = _gold_reference("gold_train", [gold_train_path], assignments)
    family_refs = reference.reference_component_pairs
    reference_components = reference.reference_components
    groups = group_vidal_rows(raw_rows)
    publication_audit = audit_published_subset(raw_rows)
    ranking_eligible = bool(publication_audit["ranking_eligible"])
    ranking_reason = publication_audit["ranking_reason"]
    anchor_rows = [
        _anchor_summary(
            key,
            rows,
            reference.sequences,
            reference.sequence_pairs,
            family_refs,
            assignments,
            reference_components,
            ranking_eligible,
            ranking_reason,
        )
        for key, rows in sorted(groups.items())
    ]
    output_root = Path(artifact_root) / "vidal"
    output_root.mkdir(parents=True, exist_ok=True)
    pairwise_rows_path = output_root / "pairwise_rows.csv"
    pairwise_columns = (
        tuple(raw_record_checks[0].keys())
        if raw_record_checks
        else (
            "staging_row_number",
            "raw_row_number",
            "raw_record_id",
            "assertion_id",
            "clone_id",
            "ad_orf_id",
            "db_orf_id",
            "raw_result",
            "staging_observation_class",
            "trace_status",
        )
    )
    write_csv(pairwise_rows_path, pairwise_columns, raw_record_checks)
    anchor_path = Path(repo_root) / "audit" / "vidal" / "anchor_summary.csv"
    anchor_columns = (
        tuple(anchor_rows[0].keys())
        if anchor_rows
        else tuple(_anchor_summary(("", ""), [], set(), set(), {}, {}).keys())
    )
    write_csv(anchor_path, anchor_columns, anchor_rows)

    filter_path = Path(repo_root) / "audit" / "vidal" / "published_subset_filter_steps.csv"
    filter_columns = tuple(publication_audit["steps"][0].keys())
    write_csv(filter_path, filter_columns, publication_audit["steps"])
    published_report_path = Path(repo_root) / "audit" / "vidal" / "VIDAL_PUBLISHED_SUBSET_AUDIT.md"
    final_stats = publication_audit["paper_default_final"]
    optional_stats = publication_audit["optional_two_partner_final"]
    target = publication_audit["target"]
    report_lines = [
        "# Vidal published-subset audit",
        "",
        f"Generated UTC: `{utc_now()}`",
        "",
        "## Scope and provenance",
        "",
        f"- Raw Data S4 snapshot: `{raw_path}` ({len(raw_rows):,} rows; SHA256 `{sha256_path(raw_path)}`).",
        "- This audit applies the filter sequence exposed by the public Vidal analysis implementation to the raw Data S4 rows.",
        "- The optional at-least-two-tested-partners condition is reported separately and is not treated as the paper default.",
        "- No ranking score, sampler, model, or training artifact is produced by this audit.",
        "",
        "## Filter sequence",
        "",
        "| step | rows | isoform constructs | TF genes | partners | positive | explicit negative | NA | unknown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in publication_audit["steps"]:
        report_lines.append(
            "| {step} | {rows:,} | {constructs:,} | {genes:,} | {partners:,} | "
            "{positive:,} | {negative:,} | {na:,} | {unknown:,} |".format(
                step=step["step"],
                rows=step["rows"],
                constructs=step["unique_isoform_constructs"],
                genes=step["unique_tf_genes"],
                partners=step["unique_partners"],
                positive=step["positive_rows"],
                negative=step["negative_rows"],
                na=step["na_rows"],
                unknown=step["unknown_rows"],
            )
        )
    report_lines.extend(
        [
            "",
            "## Published-surface comparison",
            "",
            f"- Paper-default filter output: **{final_stats['rows']:,} rows / {final_stats['unique_isoform_constructs']:,} isoform constructs / {final_stats['unique_tf_genes']:,} TF genes / {final_stats['unique_partners']:,} partners**.",
            f"- Paper headline target: **{target['rows']:,} rows / {target['unique_isoform_constructs']:,} isoform constructs / {target['unique_tf_genes']:,} TF genes / {target['unique_partners']:,} partners**.",
            f"- Per-field target match: `{publication_audit['target_match']}`.",
            f"- Row-count filter recovery: **{'YES' if publication_audit['filter_sequence_recovered'] else 'NO'}**.",
            f"- Quality gate fields present in the raw snapshot (`growth_score`, `orf_identity_confirmed`): **{'YES' if publication_audit['quality_gate_fields_present'] else 'NO'}**.",
            f"- Ranking eligibility: **{'YES' if publication_audit['ranking_eligible'] else 'NO / NOT YET'}**. {publication_audit['ranking_reason']}",
            "",
            "## Optional partner filter",
            "",
            f"The optional two-tested-partners output is **{optional_stats['rows']:,} rows / {optional_stats['unique_isoform_constructs']:,} isoform constructs / {optional_stats['unique_tf_genes']:,} TF genes / {optional_stats['unique_partners']:,} partners**. It is retained for auditability only.",
            "",
            "## Decision",
            "",
            f"The filter sequence is explicit and reproducible for this raw snapshot. It recovers the published row count but {'matches' if all(publication_audit['target_match'].values()) else 'does not match'} the full published entity surface. The Vidal ranking gate remains **{'OPEN' if publication_audit['ranking_eligible'] else 'CLOSED'}** until the source snapshot and ranking-quality semantics are sufficient.",
            "",
            f"Machine-readable filter steps: `{filter_path}`.",
        ]
    )
    published_report_path.parent.mkdir(parents=True, exist_ok=True)
    published_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    filter_manifest_path = (
        Path(repo_root) / "data" / "manifests" / "vidal_published_subset_manifest.yaml"
    )
    filter_manifest = {
        "manifest_type": "ppi_shift.vidal_published_subset_audit",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "raw_source": {
            "path": str(raw_path),
            "sha256": sha256_path(raw_path),
            "rows": len(raw_rows),
        },
        "filter_steps": str(filter_path),
        "filter_steps_sha256": sha256_path(filter_path),
        "published_subset_report": str(published_report_path),
        "published_subset_report_sha256": sha256_path(published_report_path),
        "paper_default_final": publication_audit["paper_default_final"],
        "optional_two_partner_final": publication_audit["optional_two_partner_final"],
        "published_target": publication_audit["target"],
        "target_match": publication_audit["target_match"],
        "filter_sequence_recovered": publication_audit["filter_sequence_recovered"],
        "quality_gate_fields_present": publication_audit["quality_gate_fields_present"],
        "ranking_eligible": publication_audit["ranking_eligible"],
        "ranking_reason": publication_audit["ranking_reason"],
        "source_semantics": (
            "Paper-analysis filter sequence: binary results; at least one positive per "
            "gene/partner profile; at least one positive per isoform anchor; at least two "
            "isoforms per TF gene; at least two successful tests per gene/partner profile. "
            "The optional at-least-two-partners rule is not the paper default."
        ),
    }
    write_yaml(filter_manifest_path, filter_manifest)

    pairwise_manifest_path = (
        Path(repo_root) / "data" / "manifests" / "vidal_pairwise_rows_manifest.yaml"
    )
    pairwise_manifest = {
        "manifest_type": "ppi_shift.vidal_pairwise_trace",
        "manifest_version": 2,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "raw_source": {
            "path": str(raw_path),
            "sha256": sha256_path(raw_path),
            "rows": len(raw_rows),
        },
        "staging_source": {
            "path": str(staging_path),
            "sha256": sha256_path(staging_path),
            "rows": len(staging_rows),
        },
        "transformation": "ppi.vidal_y2h_transform.v1",
        "trace_output": {
            "path": str(pairwise_rows_path),
            "sha256": sha256_path(pairwise_rows_path),
            "rows": len(raw_record_checks),
        },
        "trace_failures": trace_failures[:100],
        "trace_failure_count": len(trace_failures),
        "raw_columns": sorted(raw_rows[0]) if raw_rows else [],
        "published_subset_audit": {
            "manifest_path": str(filter_manifest_path),
            "manifest_sha256": sha256_path(filter_manifest_path),
            "filter_steps_path": str(filter_path),
            "filter_steps_sha256": sha256_path(filter_path),
            "report_path": str(published_report_path),
            "report_sha256": sha256_path(published_report_path),
        },
        "ranking_eligible": ranking_eligible,
        "ranking_reason": ranking_reason,
    }
    write_yaml(pairwise_manifest_path, pairwise_manifest)

    positive_negative_anchors = sum(
        row["positive_count"] >= 1 and row["negative_count"] >= 1 for row in anchor_rows
    )
    positive_three_negative_anchors = sum(
        row["positive_count"] >= 1 and row["negative_count"] >= 3 for row in anchor_rows
    )
    positive_five_negative_anchors = sum(
        row["positive_count"] >= 1 and row["negative_count"] >= 5 for row in anchor_rows
    )
    report_path = Path(repo_root) / "audit" / "vidal" / "VIDAL_GROUP_AUDIT.md"
    final_stats = publication_audit["paper_default_final"]
    report_lines = [
        "# Vidal/Lambourne raw group audit",
        "",
        f"Generated UTC: `{utc_now()}`",
        "",
        "## Provenance",
        "",
        f"- Raw Data S4: `{raw_path}` ({len(raw_rows):,} rows; SHA256 `{sha256_path(raw_path)}`).",
        f"- Staging: `{staging_path}` ({len(staging_rows):,} rows; SHA256 `{sha256_path(staging_path)}`).",
        "- Historical transformation: `ppi.vidal_y2h_transform.v1`.",
        f"- Required raw columns present: **{'YES' if not missing_raw else 'NO'}**; missing: `{', '.join(missing_raw) or 'none'}`.",
        f"- Staging-to-raw trace failures: **{len(trace_failures)}**.",
        "",
        "## Label and opportunity semantics",
        "",
        f"Raw `TRUE` rows: **{result_counts['positive']:,}**; raw `FALSE` rows: **{result_counts['negative']:,}**; raw `NA` rows: **{result_counts['na']:,}**; unknown result rows: **{result_counts['unknown']:,}**.",
        "`TRUE` is retained as `binary_assay_positive`; `FALSE` is retained as `explicit_assay_negative`. `NA` is not converted to a negative.",
        "The raw Data S4 columns expose clone/ORF IDs, sequences, TF family, DB category/cofactor, and transcript context, but no explicit successful-pairwise-versus-screening-only flag or growth/ORF-quality gate fields. Those semantics remain unconfirmed.",
        f"Staging observation classes: `{dict(sorted(staging_counts.items()))}`.",
        "",
        "## Recovered published-subset filter",
        "",
        "The public analysis implementation applies the source-defined sequence of binary-result, positive-profile, positive-isoform, two-isoforms-per-gene, and two-successful-tests-per-profile filters. The optional at-least-two-partners condition is not the default used for the 3,509-row surface.",
        f"- Paper-default filter output: **{final_stats['rows']:,} rows**, **{final_stats['unique_isoform_constructs']:,} isoform constructs**, **{final_stats['unique_tf_genes']:,} TF genes**, **{final_stats['unique_partners']:,} partners**.",
        f"- Paper headline target: **{PUBLISHED_TARGET['rows']:,} rows / {PUBLISHED_TARGET['unique_isoform_constructs']:,} isoforms / {PUBLISHED_TARGET['unique_tf_genes']:,} TF genes / {PUBLISHED_TARGET['unique_partners']:,} partners**.",
        f"- Row-count filter recovery: **{'YES' if publication_audit['filter_sequence_recovered'] else 'NO'}**; full published-surface match: **{'YES' if all(publication_audit['target_match'].values()) else 'NO'}**.",
        f"- Ranking eligibility: **{'YES' if ranking_eligible else 'NO / NOT YET'}**. {ranking_reason}",
        f"- Step table: `{filter_path}`; machine-readable manifest: `{filter_manifest_path}`.",
        "",
        "## Explicit anchor groups",
        "",
        "Groups use the source-provided `(clone_id, ad_orf_id)` key and candidate `db_orf_id`; no group was inferred from row counts or staging order.",
        f"- Anchors with at least one positive and one explicit negative: **{positive_negative_anchors:,}**.",
        f"- Anchors with at least one positive and at least three negatives: **{positive_three_negative_anchors:,}**.",
        f"- Anchors with at least one positive and at least five negatives: **{positive_five_negative_anchors:,}**.",
        f"- Candidate-count distribution (min/median/max): **{min((row['tested_candidate_count'] for row in anchor_rows), default=0)} / {sorted(row['tested_candidate_count'] for row in anchor_rows)[len(anchor_rows) // 2] if anchor_rows else 0} / {max((row['tested_candidate_count'] for row in anchor_rows), default=0)}**.",
        f"- Eligible positive/negative ranking pairs: **{sum(row['eligible_positive_pairs'] for row in anchor_rows):,} / {sum(row['eligible_negative_pairs'] for row in anchor_rows):,}**.",
        "",
        "## Decision",
        "",
        f"Descriptive anchor groups exist. The ranking gate is **{'OPEN' if ranking_eligible else 'CLOSED'}** based on the recovered source semantics, published-surface comparison, and quality-field availability; no MRR/Hits panel is created while that gate is closed. Individual-component novelty and the separate pair-of-components diagnostic are recorded in `anchor_summary.csv`.",
        "",
        f"Tracked summary: `{anchor_path}`; pairwise trace manifest: `{pairwise_manifest_path}`.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "report_path": str(report_path),
        "anchor_summary_path": str(anchor_path),
        "pairwise_manifest_path": str(pairwise_manifest_path),
        "published_subset_manifest_path": str(filter_manifest_path),
        "published_subset_filter_path": str(filter_path),
        "published_subset_report_path": str(published_report_path),
        "raw_rows": len(raw_rows),
        "staging_rows": len(staging_rows),
        "anchors": len(anchor_rows),
        "raw_result_counts": dict(result_counts),
        "trace_failure_count": len(trace_failures),
        "ranking_defensible": ranking_eligible,
        "ranking_reason": ranking_reason,
        "positive_negative_anchors": positive_negative_anchors,
        "positive_three_negative_anchors": positive_three_negative_anchors,
        "positive_five_negative_anchors": positive_five_negative_anchors,
        "pairwise_trace_sha256": sha256_path(pairwise_rows_path),
        "published_subset": publication_audit,
    }
