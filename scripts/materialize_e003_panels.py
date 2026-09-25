"""Materialize the method-faithful E003 six-environment core panels.

The E001v3 positive backbone and registries are read and hash-checked. E001/E002
artifacts are never rewritten. The first four E003 environments reuse the exact
legacy sampler construction only as a new, explicitly versioned reference; the
Bernett and TPPNI environments are constructed by the E003 global samplers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import git_commit, iter_delimited, sha256_path, write_yaml
from ppi_shift.config import config_path, load_yaml_config
from ppi_shift.e003_method_fidelity import (
    BERNETT_ENVIRONMENT,
    E003_SCHEMA_VERSION,
    TPPNI_BALANCED_ENVIRONMENT,
    TPPNI_ENVIRONMENT,
    TPPNI_LEAKAGE_SAFE_ENVIRONMENT,
    BernettConfig,
    generate_bernett_exact_pairs,
    generate_tppni_exact_pool,
    select_tppni_balanced_pairs,
)
from ppi_shift.e003_panels import (
    adapt_anchor_matched_rows_to_e003,
    materialize_global_balanced_panel,
    summarize_e003_panel,
    write_e003_panel,
)
from ppi_shift.model_compatibility import filter_length_compatible_positive_rows
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    EmbeddingStore,
    build_candidate_universe,
    build_degree_table,
    contract_hash,
    generate_negative_rows,
    load_exclusion_registry,
    load_positive_rows,
    positive_panel_rows,
)

UTC = getattr(timezone, "utc")
STAGES = ("train", "validation", "test")
LEGACY_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
)
E003_ENVIRONMENTS = (*LEGACY_ENVIRONMENTS, BERNETT_ENVIRONMENT, TPPNI_BALANCED_ENVIRONMENT)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _registry_name(stage: str) -> str:
    return {
        "train": "train_exclusion_registry",
        "validation": "validation_exclusion_registry",
        "test": "evaluation_positive_union",
    }[stage]


def _read_annotations(
    species_path: Path, go_path: Path, protein_ids: set[str]
) -> tuple[dict[str, str], dict[str, frozenset[str]], dict[str, str]]:
    species: dict[str, str] = {}
    for row in iter_delimited(species_path):
        protein_id = str(row.get("protein_id", "")).strip()
        taxon = str(row.get("taxon_id", "")).strip()
        if protein_id:
            species[protein_id] = taxon
    missing_species = sorted(protein_ids - set(species))
    if missing_species:
        raise SystemExit(
            f"Species annotation is incomplete for {len(missing_species)} E003 proteins; "
            f"first={missing_species[0]}"
        )
    go: dict[str, frozenset[str]] = {}
    for row in iter_delimited(go_path):
        protein_id = str(row.get("protein_id", "")).strip()
        terms = frozenset(
            term.strip()
            for term in str(row.get("go_bp", "")).split(";")
            if term.strip()
        )
        if protein_id:
            go[protein_id] = terms
    # Official loading assigns an empty set to proteins with no retained GO-BP
    # annotation. That is distinct from omitting the annotation input itself.
    go.update({protein_id: frozenset() for protein_id in protein_ids - set(go)})
    return species, go, {
        "species_path": str(species_path),
        "species_sha256": sha256_path(species_path),
        "go_bp_path": str(go_path),
        "go_bp_sha256": sha256_path(go_path),
        "species_coverage": len(protein_ids & set(species)) / max(1, len(protein_ids)),
        "species_unresolved_taxon_count": sum(
            not species.get(protein_id, "") for protein_id in protein_ids
        ),
        "go_bp_nonempty": sum(bool(terms) for terms in go.values()),
    }


def _write_tppni_pool(path: Path, pool: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("rank", "protein1", "protein2", "scm_probability", "l3_count"))
                for rank, (pair, probability, l3) in enumerate(
                    zip(
                        pool.bottom_n_pairs,
                        pool.probabilities,
                        pool.l3_counts,
                        strict=True,
                    ),
                    start=1,
                ):
                    writer.writerow((rank, pair[0], pair[1], format(probability, ".17g"), l3))
    return sha256_path(path)


def _panel_spec(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "complete",
        "panel_rows": len(rows),
        "negative_rows": sum(str(row["interaction_label"]) == "0" for row in rows),
        "panel_path": str(path),
        "panel_sha256": sha256_path(path),
        "audit": summarize_e003_panel(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e001-config", type=Path, required=True)
    parser.add_argument("--e001-manifest", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--species", type=Path, required=True)
    parser.add_argument("--go-bp", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tppni-published-n", type=int, default=10_000_000)
    parser.add_argument("--bern-time-limit", type=float, default=7200.0)
    args = parser.parse_args()
    if args.seed < 0 or args.tppni_published_n < 1:
        raise SystemExit("seed must be nonnegative and TPPNI published_n must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty E003 artifact root: {args.output_root}")
    if args.manifest.exists():
        raise SystemExit(f"Refusing to overwrite E003 panel manifest: {args.manifest}")

    e001_config = load_yaml_config(args.e001_config)
    e001_manifest = load_yaml_config(args.e001_manifest)
    source_config = load_yaml_config(args.source_config)
    if e001_manifest.get("schema_version") != SCHEMA_VERSION_V3:
        raise SystemExit("--e001-manifest must be the frozen E001v3 manifest")
    if e001_config.get("experiment_id") != "E001v3":
        raise SystemExit("--e001-config must be the frozen E001v3 config")

    raw = {
        stage: load_positive_rows(
            config_path(source_config, "gold", stage),
            stage,
            identity_schema_version=SCHEMA_VERSION_V3,
        )
        for stage in STAGES
    }
    retained: dict[str, tuple[Any, ...]] = {}
    universes: dict[str, Any] = {}
    for stage in STAGES:
        selected, _audit = filter_length_compatible_positive_rows(raw[stage], stage=stage)
        expected = int(e001_manifest["stages"][stage]["positive_rows"])
        if len(selected) != expected:
            raise SystemExit(f"E001v3 retained positive count changed for {stage}")
        retained[stage] = tuple(selected)
        registry_name = _registry_name(stage)
        registry = load_exclusion_registry(
            args.registry_dir / f"{registry_name}.csv", name=registry_name
        )
        expected_universe = e001_manifest["stages"][stage]["candidate_universe"]
        source_hashes = dict(expected_universe["source_input_hashes"])
        if source_hashes.get("gold_rows") != sha256_path(config_path(source_config, "gold", stage)):
            raise SystemExit(f"E001 Gold input changed for {stage}")
        if source_hashes.get("exclusion_registry") != registry.sha256:
            raise SystemExit(f"E001 exclusion registry changed for {stage}")
        universe = build_candidate_universe(
            retained[stage],
            registry,
            stage=stage,
            source_input_hashes=source_hashes,
            schema_version=SCHEMA_VERSION_V3,
        )
        if universe.universe_sha256 != expected_universe["sha256"]:
            raise SystemExit(f"Frozen E001 candidate universe changed for {stage}")
        universes[stage] = universe

    all_protein_ids = set().union(*(set(universe.proteins) for universe in universes.values()))
    taxonomy, go_bp, annotation_provenance = _read_annotations(
        args.species.resolve(), args.go_bp.resolve(), all_protein_ids
    )
    embedding_dir = Path(e001_manifest["embedding_store"]["paths"]["embeddings"]).parent
    embeddings = EmbeddingStore.from_files(
        embedding_dir / "embeddings.npy",
        embedding_dir / "records.csv",
        embedding_dir / "summary.json",
    )

    degree_references = {
        "train": retained["train"],
        "validation": retained["train"] + retained["validation"],
        "test": retained["train"] + retained["validation"],
    }
    paper_topology = tuple(retained[stage] for stage in STAGES)
    paper_topology_rows = tuple(item for stage_rows in paper_topology for item in stage_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "manifest_type": "ppi_shift.e003_method_fidelity_panels",
        "manifest_version": 1,
        "created_utc": _now(),
        "git_commit": git_commit(args.repo_root),
        "schema_version": E003_SCHEMA_VERSION,
        "experiment_id": "E003",
        "parent": {
            "experiment_id": "E001v3",
            "config_path": str(args.e001_config),
            "config_sha256": sha256_path(args.e001_config),
            "manifest_path": str(args.e001_manifest),
            "manifest_sha256": sha256_path(args.e001_manifest),
            "manifest_schema": SCHEMA_VERSION_V3,
        },
        "seed": args.seed,
        "environments": list(E003_ENVIRONMENTS),
        "core_environments": list(E003_ENVIRONMENTS),
        "compatibility": {
            "max_raw_sequence_length": 512,
            "native_cropping": False,
            "positive_backbone": "exact E001v3 retained positives",
            "prevalence": "1:1 for balanced core panels",
            "semantic_class": "synthetic_unobserved_pair",
        },
        "annotations": annotation_provenance,
        "topology": {
            "paper_faithful": {
                "rule": "complete_frozen_E003_positive_backbone_before_split",
                "positive_rows": len(paper_topology_rows),
                "positive_backbone_sha256": contract_hash(
                    [positive.positive_row_id for positive in paper_topology_rows]
                ),
            },
            "leakage_safe": {
                "rule": "stage_visible_positive_backbone",
                "stage_references": {
                    stage: len(degree_references[stage]) for stage in STAGES
                },
            },
        },
        "stages": {},
    }

    for stage in STAGES:
        universe = universes[stage]
        positives = retained[stage]
        stage_manifest: dict[str, Any] = {
            "positive_rows": len(positives),
            "positive_backbone_sha256": contract_hash(
                [positive.positive_row_id for positive in positives]
            ),
            "candidate_universe": {
                "id": universe.universe_id,
                "sha256": universe.universe_sha256,
                "source": "exact E001v3 candidate-universe identity; not rewritten",
            },
            "samplers": {},
            "topology": {
                "paper_faithful": len(paper_topology_rows),
                "leakage_safe": len(degree_references[stage]),
            },
        }

        # The four legacy surfaces are copied into a new E003 schema only after
        # exact E001-compatible construction; no E001 file is rewritten.
        nuisance_scaler = None
        for environment in LEGACY_ENVIRONMENTS:
            legacy_negative_rows = generate_negative_rows(
                positives,
                universe,
                environment,
                seed=args.seed,
                degree_table=build_degree_table(degree_references[stage]),
                nuisance_scaler=nuisance_scaler,
                embeddings=embeddings,
                schema_version=SCHEMA_VERSION_V3,
            )
            legacy_positive_rows = positive_panel_rows(
                positives,
                universe,
                environment,
                seed=args.seed,
                schema_version=SCHEMA_VERSION_V3,
            )
            rows = adapt_anchor_matched_rows_to_e003(
                legacy_positive_rows + legacy_negative_rows, environment=environment
            )
            panel_path = args.output_root / "panels" / f"{stage}__{environment}.csv.gz"
            write_e003_panel(panel_path, rows)
            stage_manifest["samplers"][environment] = _panel_spec(panel_path, rows)

        bernett_pairs, bernett_audit = generate_bernett_exact_pairs(
            positives,
            universe,
            seed=args.seed,
            taxon_by_id=taxonomy,
            go_bp_by_id=go_bp,
            config=BernettConfig(time_limit_seconds=args.bern_time_limit),
        )
        bernett_pair_path = args.output_root / "samplers" / f"{stage}__bernett_pairs.json"
        bernett_pair_path.parent.mkdir(parents=True, exist_ok=True)
        bernett_pair_path.write_text(
            json.dumps(
                {"pairs": [list(pair) for pair in bernett_pairs], "audit": bernett_audit},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        bernett_rows = materialize_global_balanced_panel(
            positives,
            universe,
            BERNETT_ENVIRONMENT,
            seed=args.seed,
            pairs=bernett_pairs,
            diagnostics=bernett_audit,
        )
        bernett_panel_path = args.output_root / "panels" / f"{stage}__{BERNETT_ENVIRONMENT}.csv.gz"
        write_e003_panel(bernett_panel_path, bernett_rows)
        bernett_spec = _panel_spec(bernett_panel_path, bernett_rows)
        bernett_spec["pair_artifact"] = {
            "path": str(bernett_pair_path),
            "sha256": sha256_path(bernett_pair_path),
            "audit": bernett_audit,
        }
        stage_manifest["samplers"][BERNETT_ENVIRONMENT] = bernett_spec

        tppni_pool = generate_tppni_exact_pool(
            positives,
            universe,
            topology_positives=paper_topology_rows,
            published_n=args.tppni_published_n,
        )
        tppni_pool_path = args.output_root / "samplers" / f"{stage}__{TPPNI_ENVIRONMENT}.tsv.gz"
        tppni_pool_sha = _write_tppni_pool(tppni_pool_path, tppni_pool)
        tppni_pairs, balanced_audit = select_tppni_balanced_pairs(
            tppni_pool,
            count=len(positives),
            seed=args.seed,
        )
        tppni_rows = materialize_global_balanced_panel(
            positives,
            universe,
            TPPNI_BALANCED_ENVIRONMENT,
            seed=args.seed,
            pairs=tppni_pairs,
            diagnostics={**tppni_pool.diagnostics, **balanced_audit},
        )
        tppni_panel_path = (
            args.output_root / "panels" / f"{stage}__{TPPNI_BALANCED_ENVIRONMENT}.csv.gz"
        )
        write_e003_panel(tppni_panel_path, tppni_rows)
        stage_manifest["samplers"][TPPNI_BALANCED_ENVIRONMENT] = _panel_spec(
            tppni_panel_path, tppni_rows
        )
        stage_manifest["samplers"][TPPNI_BALANCED_ENVIRONMENT]["pool_artifact"] = {
            "path": str(tppni_pool_path),
            "sha256": tppni_pool_sha,
            "diagnostics": tppni_pool.diagnostics,
            "balanced": balanced_audit,
        }

        leakage_pool = generate_tppni_exact_pool(
            positives,
            universe,
            topology_positives=degree_references[stage],
            published_n=args.tppni_published_n,
        )
        leakage_path = (
            args.output_root / "samplers" / f"{stage}__{TPPNI_LEAKAGE_SAFE_ENVIRONMENT}.tsv.gz"
        )
        leakage_sha = _write_tppni_pool(leakage_path, leakage_pool)
        stage_manifest["samplers"][TPPNI_LEAKAGE_SAFE_ENVIRONMENT] = {
            "status": "complete",
            "role": "supplementary_leakage_safe_pool_not_core_panel",
            "pool_path": str(leakage_path),
            "pool_sha256": leakage_sha,
            "diagnostics": {
                **leakage_pool.diagnostics,
                "leakage_safe": True,
            },
        }
        manifest["stages"][stage] = stage_manifest

    write_yaml(args.manifest, manifest)
    print(
        json.dumps(
            {"manifest": str(args.manifest), "artifact_root": str(args.output_root)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
