#!/usr/bin/env python3
"""Freeze corrected E003 external surfaces without using external labels for training.

The historical E002 external artifact is used only as a frozen source-positive /
explicit-assay-negative row carrier.  None of its synthetic negative rows are
read.  BioGRID and STRING synthetic surfaces are rebuilt from those positive
backbones with the E003 exact samplers; IntAct remains the measured cohort and
Vidal is copied from the frozen assay panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path
from ppi_shift.data import unordered_pair_key
from ppi_shift.e003_method_fidelity import (
    BERNETT_ENVIRONMENT,
    E003_SCHEMA_VERSION,
    TPPNI_BALANCED_ENVIRONMENT,
    BernettConfig,
    generate_bernett_exact_pairs,
    generate_tppni_exact_pool,
    select_tppni_balanced_pairs,
)
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    ExclusionRegistry,
    PositivePair,
    build_candidate_universe,
    load_exclusion_registry,
)
from scripts.materialize_e002_external_panels import (
    SOURCE_IDS,
    _external_row,
    _novelty_context,
    _write_external,
)

MAX_RAW_SEQUENCE_LENGTH = 512
UNIFORM_SURFACE = "uniform_partner"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _pair_hash(left_hash: str, right_hash: str) -> str:
    return "||".join(sorted((left_hash, right_hash)))


def _historical_panel_path(historical_manifest: dict[str, Any], source: str, surface: str) -> Path:
    try:
        payload = historical_manifest["sources"][source]["surfaces"][surface]
        return Path(payload["path"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Historical E002 manifest lacks {source}/{surface}") from error


def _read_historical_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in iter_delimited(path):
        if row.get("label") not in {"0", "1"}:
            continue
        if not row.get("left_sequence") or not row.get("right_sequence"):
            continue
        if (
            len(row["left_sequence"]) > MAX_RAW_SEQUENCE_LENGTH
            or len(row["right_sequence"]) > MAX_RAW_SEQUENCE_LENGTH
        ):
            continue
        rows.append(row)
    return rows


def _source_positive_rows(
    historical_manifest: dict[str, Any], source: str
) -> tuple[Path, list[dict[str, str]]]:
    surface = "measured" if source == "intact" else UNIFORM_SURFACE
    path = _historical_panel_path(historical_manifest, source, surface)
    rows = _read_historical_rows(path)
    positives: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: item["row_id"]):
        if row["label"] != "1":
            continue
        key = _pair_hash(row["left_sequence_sha256"], row["right_sequence_sha256"])
        if key in seen:
            continue
        seen.add(key)
        positives.append(row)
    return path, positives


def _positive_pairs(
    rows: list[dict[str, str]], source: str
) -> tuple[list[PositivePair], dict[str, dict[str, str]]]:
    positives: list[PositivePair] = []
    carriers: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        positive = PositivePair(
            split=f"external_{source}",
            source_row_index=index,
            dataset_name=source,
            left_id=row["left_id"],
            right_id=row["right_id"],
            left_sequence=row["left_sequence"],
            right_sequence=row["right_sequence"],
            identity_schema_version=SCHEMA_VERSION_V3,
        )
        positives.append(positive)
        carriers[positive.positive_row_id] = row
    return positives, carriers


def _external_registry(
    evaluation_registry: ExclusionRegistry,
    positives: list[PositivePair],
    source: str,
) -> ExclusionRegistry:
    identifier_pairs = set(evaluation_registry.identifier_pairs)
    sequence_pairs = set(evaluation_registry.sequence_pairs)
    for positive in positives:
        identifier_pairs.add(unordered_pair_key(positive.left_id, positive.right_id))
        sequence_pairs.add(unordered_pair_key(positive.left_hash, positive.right_hash))
    return ExclusionRegistry(
        name=f"e003_external_{source}_positive_registry",
        path=evaluation_registry.path,
        sha256=evaluation_registry.sha256,
        identifier_pairs=frozenset(identifier_pairs),
        sequence_pairs=frozenset(sequence_pairs),
    )


def _annotation_maps(
    species_path: Path, go_path: Path
) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    species: dict[str, str] = {}
    for row in iter_delimited(species_path):
        protein_id = str(row.get("protein_id", "")).strip()
        if protein_id:
            taxon = str(row.get("taxon_id", "")).strip()
            species[protein_id] = taxon
    go: dict[str, frozenset[str]] = {}
    for row in iter_delimited(go_path):
        protein_id = str(row.get("protein_id", "")).strip()
        if not protein_id:
            continue
        terms = frozenset(
            term.strip() for term in str(row.get("go_bp", "")).split(";") if term.strip()
        )
        go[protein_id] = terms
    # UniProt accessions in IntAct are sometimes carried as ``uniprotkb:ACC``;
    # retain the source identifier in the panel but allow the pinned annotation
    # fetch to be keyed by its bare accession.
    for protein_id, terms in list(go.items()):
        if protein_id.startswith("uniprotkb:"):
            go.setdefault(protein_id.removeprefix("uniprotkb:"), terms)
    return species, go


def _positive_external_row(
    source: str, surface: str, positive: PositivePair, carrier: dict[str, str]
) -> dict[str, Any]:
    return _external_row(
        source,
        surface,
        row_id=f"{source}:{surface}:positive:{positive.positive_row_id}",
        left_id=positive.left_id,
        right_id=positive.right_id,
        left_sequence=positive.left_sequence,
        right_sequence=positive.right_sequence,
        left_hash=positive.left_hash,
        right_hash=positive.right_hash,
        label="1",
        semantic_class="curated_positive_reference",
        assertion_id=carrier.get("source_assertion_id", ""),
        evidence_count=carrier.get("evidence_count", ""),
    )


def _uniform_rows(
    source: str,
    positives: list[PositivePair],
    carriers: dict[str, dict[str, str]],
    universe: Any,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    for positive in sorted(positives, key=lambda item: item.positive_row_id):
        candidates = sorted(
            universe.candidate_records(positive),
            key=lambda candidate: (
                hashlib.sha256(
                    f"{E003_SCHEMA_VERSION}|{UNIFORM_SURFACE}|{seed}|{positive.positive_row_id}|{candidate.protein_id}".encode()
                ).hexdigest(),
                candidate.protein_id,
            ),
        )
        chosen = next(
            (
                candidate
                for candidate in candidates
                if tuple(sorted((positive.left_id, candidate.protein_id))) not in used_pairs
            ),
            None,
        )
        if chosen is None:
            raise ValueError(
                "Uniform external sampler exhausted legal unique pairs for "
                f"{positive.positive_row_id}"
            )
        used_pairs.add(tuple(sorted((positive.left_id, chosen.protein_id))))
        rows.append(
            _positive_external_row(
                source, UNIFORM_SURFACE, positive, carriers[positive.positive_row_id]
            )
        )
        rows.append(
            _external_row(
                source,
                UNIFORM_SURFACE,
                row_id=f"{source}:{UNIFORM_SURFACE}:negative:{positive.positive_row_id}",
                left_id=positive.left_id,
                right_id=chosen.protein_id,
                left_sequence=positive.left_sequence,
                right_sequence=chosen.sequence,
                left_hash=positive.left_hash,
                right_hash=chosen.sequence_sha256,
                label="0",
                semantic_class="synthetic_unobserved_pair",
            )
        )
    return rows, {
        "method": "E003_uniform_partner_deterministic_anchor_baseline",
        "seed": seed,
        "negative_rows": len(positives),
        "unique_negative_pairs": len(used_pairs),
        "negative_semantic_class": "synthetic_unobserved_pair",
    }


def _global_rows(
    source: str,
    surface: str,
    positives: list[PositivePair],
    carriers: dict[str, dict[str, str]],
    universe: Any,
    pairs: list[tuple[str, str]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(positives, key=lambda item: item.positive_row_id)
    if len(ordered) != len(pairs):
        raise ValueError(f"{surface} returned {len(pairs)} pairs for {len(ordered)} positives")
    rows: list[dict[str, Any]] = []
    for positive, pair in zip(ordered, pairs, strict=True):
        left = universe.proteins[pair[0]]
        right = universe.proteins[pair[1]]
        rows.append(
            _positive_external_row(source, surface, positive, carriers[positive.positive_row_id])
        )
        pair_digest = hashlib.sha256(
            f"{E003_SCHEMA_VERSION}|{surface}|{seed}|{pair[0]}|{pair[1]}".encode()
        ).hexdigest()[:32]
        rows.append(
            _external_row(
                source,
                surface,
                row_id=f"{source}:{surface}:negative:{pair_digest}",
                left_id=left.protein_id,
                right_id=right.protein_id,
                left_sequence=left.sequence,
                right_sequence=right.sequence,
                left_hash=left.sequence_sha256,
                right_hash=right.sequence_sha256,
                label="0",
                semantic_class="synthetic_unobserved_pair",
            )
        )
    return rows


def _intact_measured_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_historical_rows(path):
        semantic = "physical_positive" if row["label"] == "1" else "explicit_assay_negative"
        rows.append(
            _external_row(
                "intact",
                "measured",
                row_id=row["row_id"],
                left_id=row["left_id"],
                right_id=row["right_id"],
                left_sequence=row["left_sequence"],
                right_sequence=row["right_sequence"],
                left_hash=row["left_sequence_sha256"],
                right_hash=row["right_sequence_sha256"],
                label=row["label"],
                semantic_class=semantic,
                assertion_id=row.get("source_assertion_id", ""),
                evidence_count=row.get("evidence_count", ""),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-registry", type=Path, required=True)
    parser.add_argument("--vidal-panel", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--novelty-manifest", type=Path, required=True)
    parser.add_argument("--species", type=Path, required=True)
    parser.add_argument("--go-bp", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--limit", type=int, default=10111)
    parser.add_argument("--bern-time-limit", type=float, default=7200.0)
    parser.add_argument("--tppni-published-n", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty E003 external root: {args.output_root}")
    if args.manifest.exists():
        raise SystemExit(f"Refusing to overwrite E003 external manifest: {args.manifest}")

    historical = _json(args.historical_manifest)
    evaluation_registry = load_exclusion_registry(
        args.evaluation_registry, name="evaluation_positive_union"
    )
    novelty_context = _novelty_context(args.source_config, args.novelty_manifest)
    taxonomy, go_bp = _annotation_maps(args.species, args.go_bp)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "manifest_type": "ppi_shift.e003_external_panels",
        "manifest_version": 1,
        "schema_version": "ppi_shift.e003_external_panel.v1",
        "experiment_id": "E003",
        "seed": args.seed,
        "limit": args.limit,
        "historical_e002_synthetic_rows_read": False,
        "no_external_retraining": True,
        "no_external_selection": True,
        "input_hashes": {
            "historical_e002_manifest": sha256_path(args.historical_manifest),
            "evaluation_registry": sha256_path(args.evaluation_registry),
            "vidal_panel": sha256_path(args.vidal_panel),
            "source_config": novelty_context["source_config_sha256"],
            "novelty_manifest": novelty_context["novelty_manifest_sha256"],
            "novelty_registry": novelty_context["registry_sha256"],
            "species": sha256_path(args.species),
            "go_bp": sha256_path(args.go_bp),
        },
        "sampler_revisions": {
            "bernett": {
                "repository": "https://github.com/bionetslab/ppi-splitting-pipeline",
                "revision": "8bf9adb183b702c88db62ff878de482e2f20e4b2",
            },
            "tppni": {
                "repository": "https://github.com/jbusto/UPNA-PPI",
                "revision": "20d1d8225e9d377434669973b7391697be6b4aa2",
            },
        },
        "sources": {},
    }
    for source in ("biogrid", "string"):
        historical_path, source_rows = _source_positive_rows(historical, source)
        selected_rows = source_rows[: args.limit]
        positives, carriers = _positive_pairs(selected_rows, source)
        registry = _external_registry(evaluation_registry, positives, source)
        universe = build_candidate_universe(
            positives,
            registry,
            stage=f"external_{source}",
            source_input_hashes={
                "historical_positive_carrier": sha256_path(historical_path),
                "historical_manifest": sha256_path(args.historical_manifest),
                "source_positive_selection": hashlib.sha256(
                    "\n".join(p.positive_row_id for p in positives).encode()
                ).hexdigest(),
            },
            schema_version=SCHEMA_VERSION_V3,
        )
        missing_taxonomy = sorted(
            protein_id for protein_id in universe.proteins if not taxonomy.get(protein_id)
        )
        source_taxonomy = {
            protein_id: taxonomy.get(protein_id, "9606") or "9606"
            for protein_id in universe.proteins
        }
        source_go = {
            protein_id: go_bp.get(protein_id, frozenset()) for protein_id in universe.proteins
        }
        uniform_rows, uniform_audit = _uniform_rows(
            source, positives, carriers, universe, seed=args.seed
        )
        uniform_path = args.output_root / source / f"{UNIFORM_SURFACE}.csv"
        source_manifest: dict[str, Any] = {
            "source_id": SOURCE_IDS[source],
            "historical_positive_carrier": {
                "path": str(historical_path),
                "sha256": sha256_path(historical_path),
            },
            "selected_positive_rows": len(positives),
            "positive_selection_hash": hashlib.sha256(
                "\n".join(p.positive_row_id for p in positives).encode()
            ).hexdigest(),
            "candidate_universe": universe.manifest_payload(),
            "annotation_coverage": {
                "taxonomy_source": "source_asserted_human_taxon_9606_with_pinned_fetch_override",
                "taxonomy_fetched_rows": len(set(universe.proteins) - set(missing_taxonomy)),
                "taxonomy_missing_filled_human": len(missing_taxonomy),
                "go_bp_rows": sum(bool(terms) for terms in source_go.values()),
                "go_bp_unannotated_or_unresolved": sum(not terms for terms in source_go.values()),
            },
            "surfaces": {},
        }
        source_manifest["surfaces"][UNIFORM_SURFACE] = {
            **_write_external(uniform_path, uniform_rows, novelty_context),
            "kind": "synthetic_baseline",
            "sampler_diagnostics": uniform_audit,
        }
        bernett_pairs, bernett_audit = generate_bernett_exact_pairs(
            positives,
            universe,
            seed=args.seed,
            taxon_by_id=source_taxonomy,
            go_bp_by_id=source_go,
            config=BernettConfig(time_limit_seconds=args.bern_time_limit),
        )
        bernett_rows = _global_rows(
            source,
            BERNETT_ENVIRONMENT,
            positives,
            carriers,
            universe,
            bernett_pairs,
            seed=args.seed,
        )
        bernett_path = args.output_root / source / f"{BERNETT_ENVIRONMENT}.csv"
        source_manifest["surfaces"][BERNETT_ENVIRONMENT] = {
            **_write_external(bernett_path, bernett_rows, novelty_context),
            "kind": "synthetic_exact_published_sampler",
            "sampler_diagnostics": bernett_audit,
        }
        tppni_pool = generate_tppni_exact_pool(
            positives,
            universe,
            topology_positives=positives,
            published_n=args.tppni_published_n,
        )
        tppni_pairs, balanced_audit = select_tppni_balanced_pairs(
            tppni_pool, count=len(positives), seed=args.seed
        )
        tppni_rows = _global_rows(
            source,
            TPPNI_BALANCED_ENVIRONMENT,
            positives,
            carriers,
            universe,
            tppni_pairs,
            seed=args.seed,
        )
        tppni_path = args.output_root / source / f"{TPPNI_BALANCED_ENVIRONMENT}.csv"
        source_manifest["surfaces"][TPPNI_BALANCED_ENVIRONMENT] = {
            **_write_external(tppni_path, tppni_rows, novelty_context),
            "kind": "synthetic_exact_published_sampler_balanced_adaptation",
            "sampler_diagnostics": {**tppni_pool.diagnostics, **balanced_audit},
        }
        manifest["sources"][source] = source_manifest

    intact_path = _historical_panel_path(historical, "intact", "measured")
    intact_rows = _intact_measured_rows(intact_path)
    intact_output = args.output_root / "intact" / "measured.csv"
    manifest["sources"]["intact"] = {
        "source_id": SOURCE_IDS["intact"],
        "historical_measured_carrier": {
            "path": str(intact_path),
            "sha256": sha256_path(intact_path),
        },
        "surfaces": {
            "measured": {
                **_write_external(intact_output, intact_rows, novelty_context),
                "kind": "measured_frozen_cohort",
                "label_semantics": {"1": "physical_positive", "0": "explicit_assay_negative"},
            }
        },
        "prevalence": sum(row["label"] == "1" for row in intact_rows) / max(1, len(intact_rows)),
    }

    vidal_rows: list[dict[str, Any]] = []
    for row in iter_delimited(args.vidal_panel):
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
                novelty={
                    "exact_sequence": row.get("exact_sequence_novel", "unknown"),
                    "mmseqs_i40": row.get("family_i40_novel", "unknown"),
                    "mmseqs_i30_cov80": row.get("family_i30_cov80_novel", "unknown"),
                },
            )
        )
    vidal_output = args.output_root / "vidal" / "measured.csv"
    manifest["sources"]["vidal"] = {
        "surfaces": {
            "measured": {
                **_write_external(vidal_output, vidal_rows, novelty_context),
                "kind": "measured_reused_frozen_4031_row_cohort",
            }
        },
        "reused_rows": len(vidal_rows),
        "no_ranking_metrics": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"manifest": str(args.manifest), "sources": sorted(manifest["sources"])}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
