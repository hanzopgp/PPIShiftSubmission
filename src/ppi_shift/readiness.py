"""Generate the direct D7 pre-freeze readiness decision from tracked evidence."""

# This report intentionally keeps independent gates visible instead of collapsing
# partial source coverage or a proxy execution into one green status.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, utc_now
from ppi_shift.config import load_yaml_config
from ppi_shift.novelty import NOVELTY_FIELDS, STATUS_VALUES


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0))
    except (TypeError, ValueError):
        return 0


def _status_text(row: dict[str, str], field: str) -> str:
    return "/".join(str(_int(row, f"{field}_{status}_rows")) for status in STATUS_VALUES)


def _summary_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("source", ""), row.get("reference", "")): row for row in rows}


def _d3_status_complete(rows: list[dict[str, str]]) -> bool:
    required = {f"{field}_{status}_rows" for field in NOVELTY_FIELDS for status in STATUS_VALUES}
    if len(rows) != 8:
        return False
    for row in rows:
        if not required.issubset(row):
            return False
        for field in NOVELTY_FIELDS:
            if sum(_int(row, f"{field}_{status}_rows") for status in STATUS_VALUES) != _int(
                row, "total_rows"
            ):
                return False
    return True


def _source_resolution(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("source_dataset", ""): row for row in summary_rows}


def _gate_from_mapping(
    source: str,
    resolution: dict[str, dict[str, str]],
    summary_map: dict[tuple[str, str], dict[str, str]],
    canonical_views: dict[str, Any],
) -> tuple[str, str]:
    row = resolution.get(source, {})
    fraction = float(row.get("mapping_fraction", "0") or 0)
    view = canonical_views.get(source, {})
    has_view = bool(view.get("path"))
    source_rows = [
        summary_map[(source, reference)]
        for reference in ("gold_train", "gold_trainval")
        if (source, reference) in summary_map
    ]
    family_complete = len(source_rows) == 2 and all(
        row_value
        and _int(row_value, "family_i40_novel_unresolved_rows") == 0
        and _int(row_value, "family_i30_cov80_novel_unresolved_rows") == 0
        for row_value in source_rows
    )
    if not has_view or len(source_rows) != 2:
        return "NO", "canonical pair view or complete D3 summary is missing"
    if fraction >= 1.0 and family_complete:
        return "YES", "canonical pair view and complete sequence/component coverage"
    if fraction > 0 and source_rows:
        return "RESOLVED_SUBCOHORT_ONLY", (
            f"mapping fraction={fraction:.6f}; unresolved source/component rows remain explicit"
        )
    return "NO", "no defensible resolved source cohort"


def build_readiness_report(repo_root: str | Path = ".") -> Path:
    """Write D7 with independent gates and only genuine core blockers."""

    root = Path(repo_root)
    config = load_yaml_config(root / "configs/data/prefreeze_sources.yaml")
    artifact_root = Path(str(config["artifact_root"]))
    d1 = _yaml(root / "data/manifests/sequence_registry_manifest.yaml")
    d2 = {
        namespace: _yaml(root / "data/manifests" / f"mmseqs_{namespace}_manifest.yaml")
        for namespace in ("i40", "i30_cov80")
    }
    d3_rows = (
        list(iter_delimited(root / "audit/novelty/summary.csv"))
        if (root / "audit/novelty/summary.csv").is_file()
        else []
    )
    d4 = {
        name: _yaml(root / "data/manifests" / f"{name}_manifest.yaml")
        for name in (
            "train_exclusion_registry",
            "validation_exclusion_registry",
            "evaluation_positive_union",
        )
    }
    d5 = _yaml(root / "data/manifests/vidal_pairwise_rows_manifest.yaml")
    d5_published = _yaml(root / "data/manifests/vidal_published_subset_manifest.yaml")
    d6 = _yaml(root / "data/manifests/datasail_official_manifest.yaml")
    d6_repro = _yaml(root / "data/manifests/datasail_reproduction_manifest.yaml")
    d6_result = d6_repro.get("result", {})
    summary_rows = (
        list(iter_delimited(root / "audit/sequence_registry_summary.csv"))
        if (root / "audit/sequence_registry_summary.csv").is_file()
        else []
    )
    resolution = _source_resolution(summary_rows)
    summary_map = _summary_map(d3_rows)
    registry_path = Path(d1.get("outputs", {}).get("registry", {}).get("path", ""))
    d1_available = bool(d1 and registry_path.is_file())
    d2_available = all(
        bool(value and value.get("outputs", {}).get("components")) for value in d2.values()
    )
    common_input = {value.get("input", {}).get("sha256") for value in d2.values() if value}
    same_namespace_universe = len(common_input) == 1 and d2_available
    d4_available = all(
        bool(value and value.get("output", {}).get("sha256")) for value in d4.values()
    )
    d5_available = bool(d5 and d5.get("trace_output") and d5.get("trace_failure_count", 1) == 0)
    official_complete = bool(
        d6
        and d6.get("mapping", {}).get("unmatched_rows") == 0
        and d6.get("mapping", {}).get("ambiguous_rows") == 0
    )
    corrected_overlap = bool(
        d6.get("namespace_overlap")
        and all(
            all(
                f"{namespace}_shared_component_count" in row
                and f"{namespace}_shared_component_protein_fraction_left" in row
                for namespace in ("i40", "i30_cov80")
            )
            for row in d6.get("namespace_overlap", [])
        )
    )
    d3_complete = _d3_status_complete(d3_rows)
    evaluation_views = _yaml(root / "data/manifests/evaluation_pair_views_manifest.yaml")
    canonical_views = evaluation_views.get("views", {}) if evaluation_views else {}
    train_sources = set(d4.get("train_exclusion_registry", {}).get("allowed_sources", []))
    validation_sources = set(d4.get("validation_exclusion_registry", {}).get("allowed_sources", []))
    gold_factorial_ready = bool(
        d1_available
        and d2_available
        and same_namespace_universe
        and d4_available
        and train_sources == {"gold_train"}
        and validation_sources == {"gold_train", "gold_validation"}
        and d3_complete
    )
    datasail_control_ready = official_complete and corrected_overlap
    vidal_ranking_ready = bool(d5_published.get("ranking_eligible", False))
    string_gate, string_reason = _gate_from_mapping(
        "string", resolution, summary_map, canonical_views
    )
    biogrid_gate, biogrid_reason = _gate_from_mapping(
        "biogrid", resolution, summary_map, canonical_views
    )
    intact_gate, intact_reason = _gate_from_mapping(
        "intact", resolution, summary_map, canonical_views
    )
    vidal_classification_ready = bool(d5_available and d5.get("raw_source", {}).get("rows", 0))
    gates = [
        (
            "GOLD_FACTORIAL_READY",
            _yes(gold_factorial_ready),
            "Gold/D2/D4 core evidence and stage firewall",
        ),
        (
            "DATASAIL_CONTROL_READY",
            _yes(datasail_control_ready),
            "official mapping plus direct component overlap audit",
        ),
        (
            "VIDAL_RANKING_READY",
            "YES" if vidal_ranking_ready else "NO / NOT YET",
            d5_published.get(
                "ranking_reason", "published subset or quality semantics are unresolved"
            ),
        ),
        ("OOD_STRING_READY", string_gate, string_reason),
        ("OOD_BIOGRID_READY", biogrid_gate, biogrid_reason),
        ("OOD_INTACT_READY", intact_gate, intact_reason),
        (
            "OOD_VIDAL_CLASSIFICATION_READY",
            _yes(vidal_classification_ready),
            "raw/staging trace and binary classification audit",
        ),
    ]
    blockers: list[str] = []
    if not d1_available:
        blockers.append(
            "D1 common sequence registry is not materialized or its external path is unavailable."
        )
    if not d2_available or not same_namespace_universe:
        blockers.append(
            "D2 i40/i30_cov80 namespaces are not both materialized on one identical sequence universe."
        )
    if not d4_available:
        blockers.append("D4 stage-specific positive-exclusion manifests are incomplete.")
    if not d3_complete:
        blockers.append(
            "D3 novelty summary is missing the four-state counts for every field/source/reference."
        )
    if not official_complete:
        blockers.append(
            "D6 official DataSAIL rows are not fully mapped to Gold-v4 without ambiguity."
        )
    if not corrected_overlap:
        blockers.append(
            "D6 direct individual-component overlap fields are missing from the official control audit."
        )
    if not d5_available:
        blockers.append("D5 Vidal raw/staging trace is missing or has trace failures.")
    core_ready = gold_factorial_ready and datasail_control_ready and d5_available
    output = root / "audit/PREFREEZE_DATA_READINESS.md"
    lines = [
        "# PPIShift pre-freeze data readiness",
        "",
        f"Generated UTC: `{utc_now()}`",
        "",
        "This is a data-substrate gate. It does not authorize negative samplers, embeddings, models, hyperparameter selection, or training.",
        "",
        "## Direct answers",
        "",
        f"1. One common sequence registry available: **{_yes(d1_available)}** (`{registry_path}`; external artifact required).",
        "2. Core-source sequence resolution:",
        "",
        "| source | source proteins | resolved | fraction | ambiguous | missing | unique sequences | input SHA256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('source_dataset', '')} | {row.get('source_proteins', '')} | {row.get('sequence_resolved_proteins', '')} | {row.get('mapping_fraction', '')} | {row.get('ambiguous_mappings', '')} | {row.get('missing_mappings', '')} | {row.get('unique_sequences', '')} | `{row.get('input_sha256', '')}` |"
        )
    lines.extend(
        [
            "",
            f"3. Versioned/frozen BioGRID, IntAct, STRING and pair-level evaluation views: **{_yes(d1_available and bool(d1.get('mapping_artifacts')) and bool(canonical_views))}**.",
            f"4. i40 and i30_cov80 use the same unified sequence universe: **{_yes(same_namespace_universe)}** (input hashes: `{', '.join(sorted(common_input)) or 'missing'}`).",
            "",
            "## Corrected Gold-relative novelty counts",
            "",
            "Each novelty field is reported as `true/false/unknown/unresolved` in that order; counts are rows, and the positive/explicit-negative columns use the same order. `family_i40_pair_novel` and `family_i30_cov80_pair_novel` are separate pair-of-components diagnostics, not the family-novelty headline.",
            "",
            "| source | reference | total | sequence resolved | exact sequence novel | i40 individual T/F/U/R | i30 individual T/F/U/R | i40 pair diagnostic T/F/U/R | i30 pair diagnostic T/F/U/R |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in d3_rows:
        lines.append(
            f"| {row.get('source', '')} | {row.get('reference', '')} | {row.get('total_rows', '')} | {row.get('sequence_resolved_rows', '')} | {_status_text(row, 'exact_sequence_novel')} | {_status_text(row, 'family_i40_novel')} | {_status_text(row, 'family_i30_cov80_novel')} | {_status_text(row, 'family_i40_pair_novel')} | {_status_text(row, 'family_i30_cov80_pair_novel')} |"
        )
    lines.extend(
        [
            "",
            "Positive and explicit-negative status counts for every novelty field are in `audit/novelty/summary.csv`; no unresolved row is counted as novel.",
            f"Nestedness audit: `{root / 'audit/novelty/nestedness.csv'}`. It reports both `i30 novel -> i40 novel` and reverse-direction differences, including positive and explicit-negative subsets; no direction is silently assumed.",
            "",
            "## Vidal and DataSAIL decisions",
            "",
            f"- Vidal published-subset audit: `{root / 'audit/vidal/VIDAL_PUBLISHED_SUBSET_AUDIT.md'}`; filter steps: `{root / 'audit/vidal/published_subset_filter_steps.csv'}`.",
            f"- Vidal ranking gate: **{'YES' if vidal_ranking_ready else 'NO / NOT YET'}**. {d5_published.get('ranking_reason', 'The source semantics are not fully recovered.')}",
            f"- Official DataSAIL mapping: **{_yes(official_complete)}**; direct component-overlap audit: **{_yes(corrected_overlap)}**.",
            f"- Pinned DataSAIL status: **{d6_result.get('status', 'unknown')}**. This is a proxy execution/parsing control, not an official row-for-row reproduction claim and not a prerequisite for the Gold factorial gate.",
            f"- DataSAIL audit report: `{root / 'audit/datasail/DATASAIL_GOLD_AUDIT.md'}`.",
            "",
            "## Independent readiness gates",
            "",
            "| gate | status | evidence/interpretation |",
            "| --- | --- | --- |",
        ]
    )
    for gate, status, reason in gates:
        lines.append(f"| `{gate}` | **{status}** | {reason} |")
    lines.extend(
        [
            "",
            f"Aggregate core data-layer gate: **{'YES' if core_ready else 'NO'}**. IntAct partial coverage and Vidal ranking uncertainty do not block the Gold factorial gate; they remain explicit source-specific gates.",
            "",
            "## What still blocks the aggregate core gate",
            "",
        ]
    )
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None.")
    lines.extend(
        [
            "",
            "No negative sampler, model, training run, merged matrix, MRR/Hits panel, or biological non-interaction claim is created by this audit.",
            "",
            f"Tracked novelty manifest: `{root / 'data/manifests/novelty_manifest.yaml'}`.",
            f"Tracked D6 official manifest: `{root / 'data/manifests/datasail_official_manifest.yaml'}`.",
            f"Configured external artifact root: `{artifact_root}`.",
            "",
            f"E001_DATA_LAYER_READY = {'YES' if core_ready else 'NO'}",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
