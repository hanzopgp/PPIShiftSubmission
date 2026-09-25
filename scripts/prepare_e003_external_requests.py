#!/usr/bin/env python3
"""Prepare one score-only external request for every E003 checkpoint unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS

CORE_MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
SEEDS = (11, 22, 33)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--core-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty external request root: {args.output_root}")
    if args.manifest.exists():
        raise SystemExit(f"Refusing to overwrite external request manifest: {args.manifest}")
    units: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in (
        args.backend_manifest,
        args.python_manifest,
        args.frozen_manifest,
        args.reuse_manifest,
    ):
        for request in _json(path)["requests"]:
            key = (request["model"], int(request["seed"]), request["train_environment"])
            if key in units:
                raise SystemExit(f"Duplicate E003 checkpoint unit: {key}")
            units[key] = request
    expected_keys = {
        (model, seed, environment)
        for model in CORE_MODELS
        for seed in SEEDS
        for environment in E003_ENVIRONMENTS
    }
    if set(units) != expected_keys:
        raise SystemExit(
            "Core-4 external checkpoint roster mismatch: "
            f"observed={len(units)} expected={len(expected_keys)}"
        )
    external = _json(args.external_manifest)
    requests: list[dict[str, Any]] = []
    for model, seed, train_environment in sorted(units):
        request = {
            "schema_version": "ppi_shift.e003_external_unit_request.v1",
            "experiment_id": "E003",
            "model": model,
            "seed": seed,
            "train_environment": train_environment,
            "checkpoint_index_key": f"{model}|{seed}|{train_environment}",
            "core_index_path": str(args.core_index),
            "core_index_sha256": sha256_path(args.core_index),
            "external_manifest": str(args.external_manifest),
            "external_manifest_sha256": sha256_path(args.external_manifest),
            "output_dir": str(args.output_root / model / f"seed-{seed}__train-{train_environment}"),
            "no_external_retraining": True,
            "no_external_selection": True,
            "threshold_source": "corresponding_Gold_validation_max_MCC",
        }
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request["request_sha256"] = hashlib.sha256(encoded).hexdigest()
        path = (
            args.output_root
            / "requests"
            / model
            / f"seed-{seed}__train-{train_environment}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        requests.append(
            {
                "model": model,
                "seed": seed,
                "train_environment": train_environment,
                "request": str(path),
                "request_sha256": request["request_sha256"],
                "output_dir": request["output_dir"],
            }
        )
    output = {
        "schema_version": "ppi_shift.e003_external_request_matrix.v1",
        "experiment_id": "E003",
        "external_manifest": {
            "path": str(args.external_manifest),
            "sha256": sha256_path(args.external_manifest),
        },
        "core_index_path": str(args.core_index),
        "core_index_sha256": sha256_path(args.core_index),
        "units": len(requests),
        "requests": requests,
        "surfaces": {
            source: sorted(info["surfaces"])
            for source, info in external["sources"].items()
        },
        "no_external_retraining": True,
        "no_external_selection": True,
        "interpretation_boundary": "frozen-checkpoint external-score preparation only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "requests": len(requests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
