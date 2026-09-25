#!/usr/bin/env python3
"""Prepare exact E001v3 checkpoint-reuse requests for E003 test panels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import load_yaml

MODEL_NAMES = ("symmetric_frozen_esm2_pair_mlp", "dscript_esm2", "tuna_corrected")
SEEDS = (11, 22, 33)
TRAIN_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
)


def _unique_registry_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["model"], int(row["seed"]), row["train_environment"])
        if (
            row["model"] in MODEL_NAMES
            and int(row["seed"]) in SEEDS
            and row["train_environment"] in TRAIN_ENVIRONMENTS
        ):
            selected.setdefault(key, row)
    expected = len(MODEL_NAMES) * len(SEEDS) * len(TRAIN_ENVIRONMENTS)
    if len(selected) != expected:
        raise SystemExit(f"Expected {expected} unique E001v3 training units, found {len(selected)}")
    return [selected[key] for key in sorted(selected)]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e001-registry", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output root: {args.output_root}")
    panel_manifest = load_yaml(args.panel_manifest)
    if panel_manifest.get("schema_version") != E003_SCHEMA_VERSION:
        raise SystemExit("Unexpected E003 panel manifest schema")
    test_panels: dict[str, Any] = {}
    test_specs = panel_manifest.get("stages", {}).get("test", {}).get("samplers", {})
    if not isinstance(test_specs, dict):
        raise SystemExit("E003 panel manifest lacks test sampler specifications")
    for environment in E003_ENVIRONMENTS:
        spec = test_specs.get(environment)
        if not isinstance(spec, dict):
            raise SystemExit(f"E003 panel manifest lacks test environment: {environment}")
        path = Path(str(spec["panel_path"]))
        actual_hash = sha256_path(path)
        if actual_hash != str(spec["panel_sha256"]):
            raise SystemExit(f"E003 panel hash changed: {path}")
        test_panels[environment] = {
            "path": str(path),
            "sha256": actual_hash,
            "rows": int(spec["panel_rows"]),
        }
    requests: list[dict[str, Any]] = []
    for row in _unique_registry_rows(args.e001_registry):
        model = row["model"]
        seed = int(row["seed"])
        train_environment = row["train_environment"]
        request = {
            "schema_version": "ppi_shift.e003_reused_e001_unit_request.v1",
            "experiment_id": "E003",
            "model": model,
            "seed": seed,
            "train_environment": train_environment,
            "e001_run_path": row["run_path"],
            "e001_checkpoint_sha256": row["checkpoint_sha256"],
            "test_environments": list(E003_ENVIRONMENTS),
            "test_panels": test_panels,
            "output_dir": str(args.output_root / model / f"seed-{seed}__train-{train_environment}"),
            "reuse_policy": (
                "exact E001v3 implementation, representation, seed and train panel; "
                "score-only E003 test panels"
            ),
            "test_cropping": False,
            "test_adaptation": False,
        }
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request["request_sha256"] = hashlib.sha256(encoded).hexdigest()
        request_path = (
            args.output_root / "requests" / model / f"seed-{seed}__train-{train_environment}.json"
        )
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        requests.append(
            {
                "model": model,
                "seed": seed,
                "train_environment": train_environment,
                "request": str(request_path),
                "request_sha256": request["request_sha256"],
                "output_dir": request["output_dir"],
                "e001_run_path": row["run_path"],
                "e001_checkpoint_sha256": row["checkpoint_sha256"],
            }
        )
    output = {
        "schema_version": "ppi_shift.e003_reuse_request_matrix.v1",
        "experiment_id": "E003",
        "e001_registry": str(args.e001_registry),
        "e001_registry_sha256": sha256_path(args.e001_registry),
        "panel_manifest": str(args.panel_manifest),
        "panel_manifest_sha256": sha256_path(args.panel_manifest),
        "requests": requests,
        "units": len(requests),
        "interpretation_boundary": (
            "exact checkpoint reuse preparation only; no model result is contained here"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "requests": len(requests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
