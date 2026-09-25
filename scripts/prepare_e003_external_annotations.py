#!/usr/bin/env python3
"""Prepare pinned Bernett annotation inputs for the corrected external surfaces."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _panel_path(manifest: dict[str, Any], source: str, surface: str) -> Path:
    return Path(manifest["sources"][source]["surfaces"][surface]["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--fetch-data", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    historical = _json(args.historical_manifest)
    existing_manifest = args.output_dir / "annotation_manifest.json"
    if existing_manifest.is_file():
        existing = _json(existing_manifest)
        files = existing.get("files", {})
        complete = existing.get("status") == "complete" and isinstance(files, dict)
        if complete:
            for item in files.values():
                if not isinstance(item, dict):
                    complete = False
                    break
                path = Path(str(item.get("path", "")))
                if not path.is_file() or sha256_path(path) != item.get("sha256"):
                    complete = False
                    break
        if complete:
            print(json.dumps({"annotation_manifest": str(existing_manifest), "status": "reused"}))
            return 0

    protein_ids: set[str] = set()
    source_paths: dict[str, str] = {}
    for source in ("biogrid", "string", "intact"):
        surface = "measured" if source == "intact" else "uniform_partner"
        path = _panel_path(historical, source, surface)
        source_paths[f"{source}_{surface}"] = str(path)
        for row in iter_delimited(path):
            if row.get("label") != "1":
                continue
            if not row.get("left_sequence") or not row.get("right_sequence"):
                continue
            if len(row.get("left_sequence", "")) > 512 or len(row.get("right_sequence", "")) > 512:
                continue
            protein_ids.update((row["left_id"], row["right_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protein_list = args.output_dir / "protein_ids.txt"
    protein_list.write_text("\n".join(sorted(protein_ids)) + "\n", encoding="utf-8")
    wrapper = args.repo_root / "scripts" / "fetch_e003_external_annotations.py"
    subprocess.run(
        [
            args.python,
            str(wrapper),
            "--official-fetch-script",
            str(args.fetch_data),
            "--protein-list",
            str(protein_list),
            "--fasta",
            str(args.output_dir / "sequences.fasta"),
            "--go",
            str(args.output_dir / "go_annotations.tsv"),
            "--species",
            str(args.output_dir / "species.tsv"),
        ],
        check=True,
    )
    manifest = {
        "schema_version": "ppi_shift.e003_external_bernett_annotations.v1",
        "historical_manifest": str(args.historical_manifest),
        "historical_manifest_sha256": sha256_path(args.historical_manifest),
        "official_fetch_script": str(args.fetch_data),
        "official_fetch_script_sha256": sha256_path(args.fetch_data),
        "fetch_wrapper": str(wrapper),
        "fetch_mode": "pinned_uniprotkb_with_declared_human_taxon_archive_fallback_no_uniparc",
        "source_paths": source_paths,
        "protein_count": len(protein_ids),
        "files": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in {
                "protein_ids": protein_list,
                "sequences": args.output_dir / "sequences.fasta",
                "go_annotations": args.output_dir / "go_annotations.tsv",
                "species": args.output_dir / "species.tsv",
            }.items()
        },
        "taxonomy_fallback": "source panels are human; blank fetched taxon is recorded as 9606",
        "status": "complete",
    }
    output = args.output_dir / "annotation_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
