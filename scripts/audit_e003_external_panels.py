#!/usr/bin/env python3
"""Independently audit corrected E003 external panel identities and semantics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import iter_delimited, sha256_path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _json(args.manifest)
    if manifest.get("schema_version") != "ppi_shift.e003_external_panel.v1":
        raise SystemExit("Unexpected E003 external panel schema")
    if manifest.get("historical_e002_synthetic_rows_read") is not False:
        raise SystemExit("External construction must not read E002 synthetic rows")
    if manifest.get("no_external_retraining") is not True:
        raise SystemExit("External manifest does not assert no retraining")
    if manifest.get("no_external_selection") is not True:
        raise SystemExit("External manifest does not assert no external selection")

    expected_sources = {"biogrid", "string", "intact", "vidal"}
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise SystemExit(f"Unexpected external source roster: {sorted(sources or {})}")

    audit: dict[str, Any] = {
        "schema_version": "ppi_shift.e003_external_panel_audit.v1",
        "manifest": {"path": str(args.manifest), "sha256": sha256_path(args.manifest)},
        "checks": [],
        "sources": {},
        "status": "complete",
    }
    for source, source_payload in sorted(sources.items()):
        surfaces = source_payload.get("surfaces")
        if not isinstance(surfaces, dict) or not surfaces:
            raise SystemExit(f"Missing surfaces for {source}")
        source_audit: dict[str, Any] = {}
        for surface, surface_payload in sorted(surfaces.items()):
            path = Path(str(surface_payload["path"]))
            expected_hash = str(surface_payload["sha256"])
            if not path.is_file() or sha256_path(path) != expected_hash:
                raise SystemExit(f"External panel hash changed or file is missing: {path}")
            rows = list(iter_delimited(path))
            expected_rows = int(surface_payload["rows"])
            if len(rows) != expected_rows:
                raise SystemExit(f"External row count changed: {path}")
            unique_row_ids = len({row["row_id"] for row in rows})
            if unique_row_ids != len(rows):
                raise SystemExit(f"Duplicate external row IDs: {path}")
            labels = Counter(row["label"] for row in rows)
            semantics = Counter(row["semantic_class"] for row in rows)
            if dict(sorted(labels.items())) != dict(surface_payload["label_counts"]):
                raise SystemExit(f"External label counts changed: {path}")
            if source in {"biogrid", "string"}:
                if set(labels) != {"0", "1"} or labels["0"] != labels["1"]:
                    raise SystemExit(f"Synthetic external surface is not balanced: {path}")
                if semantics.get("synthetic_unobserved_pair", 0) != labels["0"]:
                    raise SystemExit(f"Synthetic semantic class changed: {path}")
                if surface not in {
                    "uniform_partner",
                    "bernett2026_ilp_exact",
                    "tppni2025_scm_cl3_exact_balanced",
                }:
                    raise SystemExit(f"Unexpected synthetic surface: {source}/{surface}")
            elif source == "intact":
                if set(semantics) != {"physical_positive", "explicit_assay_negative"}:
                    raise SystemExit(f"IntAct semantics changed: {path}")
            elif source == "vidal":
                if set(semantics) != {"binary_assay_positive", "assay_non_detection"}:
                    raise SystemExit(f"Vidal semantics changed: {path}")
            source_audit[surface] = {
                "path": str(path),
                "sha256": expected_hash,
                "rows": len(rows),
                "label_counts": dict(sorted(labels.items())),
                "semantic_classes": dict(sorted(semantics.items())),
                "unique_row_ids": unique_row_ids,
            }
            audit["checks"].append(f"{source}/{surface}: hash, rows, labels, semantics")
        audit["sources"][source] = source_audit

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": audit["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
