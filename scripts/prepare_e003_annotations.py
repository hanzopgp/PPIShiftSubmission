"""Fetch and hash Bernett-compatible UniProt annotations for E003."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import git_commit, sha256_path
from ppi_shift.config import config_path, load_yaml_config
from ppi_shift.model_compatibility import filter_length_compatible_positive_rows
from ppi_shift.negatives import SCHEMA_VERSION_V3, load_positive_rows


def _read_species(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["protein_id"].strip(): row["taxon_id"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--e001-manifest", type=Path, required=True)
    parser.add_argument("--fetch-data", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_config = load_yaml_config(args.source_config)
    e001_manifest = load_yaml_config(args.e001_manifest)
    protein_ids: set[str] = set()
    retained_counts: dict[str, int] = {}
    for stage in ("train", "validation", "test"):
        positives = load_positive_rows(
            config_path(source_config, "gold", stage),
            stage,
            identity_schema_version=SCHEMA_VERSION_V3,
        )
        retained, _audit = filter_length_compatible_positive_rows(positives, stage=stage)
        expected = int(e001_manifest["stages"][stage]["positive_rows"])
        if len(retained) != expected:
            raise SystemExit(f"Retained E001v3 count changed for {stage}")
        retained_counts[stage] = len(retained)
        for positive in retained:
            protein_ids.update((positive.left_id, positive.right_id))
    protein_list = args.output_dir / "protein_ids.txt"
    protein_list.write_text("\n".join(sorted(protein_ids)) + "\n", encoding="utf-8")
    fasta_path = args.output_dir / "sequences.fasta"
    go_path = args.output_dir / "go_annotations.tsv"
    species_path = args.output_dir / "species.tsv"
    subprocess.run(
        [
            args.python,
            str(args.fetch_data),
            str(protein_list),
            str(fasta_path),
            str(go_path),
            str(species_path),
        ],
        check=True,
    )
    species = _read_species(species_path)
    missing = sorted(protein_ids - set(species))
    if missing:
        raise SystemExit(f"UniProt annotation fetch missed {len(missing)} proteins: {missing[:3]}")
    unexpected_taxa = sorted(set(species[protein_id] for protein_id in protein_ids))
    manifest: dict[str, Any] = {
        "schema_version": "ppi_shift.e003_bernett_annotations.v1",
        "git_commit": git_commit(args.repo_root),
        "official_source_repository": "https://github.com/bionetslab/ppi-splitting-pipeline",
        "official_source_revision": "8bf9adb183b702c88db62ff878de482e2f20e4b2",
        "official_fetch_script": str(args.fetch_data),
        "official_fetch_script_sha256": sha256_path(args.fetch_data),
        "source_config": str(args.source_config),
        "source_config_sha256": sha256_path(args.source_config),
        "e001_manifest": str(args.e001_manifest),
        "e001_manifest_sha256": sha256_path(args.e001_manifest),
        "retained_positive_rows": retained_counts,
        "protein_count": len(protein_ids),
        "species_taxa": unexpected_taxa,
        "species_unresolved_taxon_count": sum(
            not species.get(protein_id, "") for protein_id in protein_ids
        ),
        "files": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in {
                "protein_ids": protein_list,
                "sequences": fasta_path,
                "go_annotations": go_path,
                "species": species_path,
            }.items()
        },
        "status": "complete",
    }
    manifest_path = args.output_dir / "annotation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
