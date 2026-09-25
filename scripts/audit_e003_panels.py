#!/usr/bin/env python3
"""Independently revalidate the corrected E003 panel and sampler artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ppi_shift.artifacts import iter_delimited, sha256_path
from ppi_shift.e003_method_fidelity import (
    BERNETT_ENVIRONMENT,
    E003_ENVIRONMENTS,
    E003_SCHEMA_VERSION,
    TPPNI_BALANCED_ENVIRONMENT,
    TPPNI_ENVIRONMENT,
    TPPNI_LEAKAGE_SAFE_ENVIRONMENT,
)
from ppi_shift.e003_panels import read_e003_panel, summarize_e003_panel

STAGES = ("train", "validation", "test")


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return value


def _pair_rows(path: Path) -> tuple[int, int, int]:
    pairs: set[tuple[str, str]] = set()
    self_pairs = 0
    duplicate_rows = 0
    for row in iter_delimited(path):
        pair = tuple(sorted((row["protein1"], row["protein2"])))
        if pair in pairs:
            duplicate_rows += 1
        pairs.add(pair)
        self_pairs += int(row["protein1"] == row["protein2"])
    return len(pairs), self_pairs, duplicate_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _mapping(args.panel_manifest)
    if manifest.get("schema_version") != E003_SCHEMA_VERSION:
        raise SystemExit("Unexpected E003 panel manifest schema")
    if tuple(manifest.get("environments", ())) != E003_ENVIRONMENTS:
        raise SystemExit("E003 environment roster changed")

    result: dict[str, Any] = {
        "schema_version": "ppi_shift.e003_panel_contract_audit.v1",
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "stages": {},
        "checks": [],
        "status": "complete",
    }
    for stage in STAGES:
        stage_payload: dict[str, Any] = {}
        for environment in E003_ENVIRONMENTS:
            spec = manifest["stages"][stage]["samplers"][environment]
            path = Path(spec["panel_path"])
            if sha256_path(path) != spec["panel_sha256"]:
                raise SystemExit(f"Panel hash changed: {path}")
            rows = read_e003_panel(path, environment=environment)
            if len(rows) != int(spec["panel_rows"]):
                raise SystemExit(f"Panel row count changed: {path}")
            summary = summarize_e003_panel(rows)
            negatives = [row for row in rows if row["interaction_label"] == "0"]
            stage_payload[environment] = {
                "path": str(path),
                "sha256": sha256_path(path),
                "rows": len(rows),
                "summary": summary,
                "negative_self_pairs": sum(
                    row["anchor_id"] == row["candidate_partner_id"] for row in negatives
                ),
                "negative_endpoint_counts": dict(
                    Counter(row["candidate_partner_id"] for row in negatives).most_common(10)
                ),
            }
            result["checks"].append(
                f"{stage}/{environment}: panel hash, schema, balance, and legality"
            )

            if environment == BERNETT_ENVIRONMENT:
                pair_artifact = Path(spec["pair_artifact"]["path"])
                if sha256_path(pair_artifact) != spec["pair_artifact"]["sha256"]:
                    raise SystemExit(f"Bernett pair artifact hash changed: {pair_artifact}")
                payload = json.loads(pair_artifact.read_text(encoding="utf-8"))
                pairs = [tuple(pair) for pair in payload["pairs"]]
                if len(pairs) != len(negatives) or len(set(pairs)) != len(pairs):
                    raise SystemExit(
                        f"Bernett pair artifact count/uniqueness failed: {pair_artifact}"
                    )
                stage_payload[environment]["sampler_audit"] = payload["audit"]
            if environment == TPPNI_BALANCED_ENVIRONMENT:
                pool = spec["pool_artifact"]
                pool_path = Path(pool["path"])
                if sha256_path(pool_path) != pool["sha256"]:
                    raise SystemExit(f"TPPNI pool hash changed: {pool_path}")
                stage_payload[environment]["sampler_audit"] = pool["diagnostics"]
                stage_payload[environment]["balanced_audit"] = pool["balanced"]
        result["stages"][stage] = stage_payload

    result["sampler_roles"] = {
        TPPNI_ENVIRONMENT: "published_method_supplementary_full_pool",
        TPPNI_LEAKAGE_SAFE_ENVIRONMENT: "leakage_safe_supplementary_full_pool",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
