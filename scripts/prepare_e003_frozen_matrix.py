#!/usr/bin/env python3
"""Prepare clean-init Frozen-ESM2 units for the two new E003 train surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import load_yaml

MODEL = "symmetric_frozen_esm2_pair_mlp"
SEEDS = (11, 22, 33)
NEW_TRAIN_ENVIRONMENTS = (
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
EMBEDDING_DIR = "/path/to/ExternalPPIArtifacts/embeddings/fresh-gold-a695f604-w1022-o128-v2"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output root: {args.output_root}")
    panel_manifest = load_yaml(args.panel_manifest)
    if panel_manifest.get("schema_version") != E003_SCHEMA_VERSION:
        raise SystemExit("Unexpected E003 panel manifest schema")
    args.output_root.mkdir(parents=True, exist_ok=True)
    requests: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    index = 0
    for seed in SEEDS:
        for train_environment in NEW_TRAIN_ENVIRONMENTS:
            request = {
                "schema_version": "ppi_shift.e003_frozen_esm_unit_request.v1",
                "experiment_id": "E003",
                "model": MODEL,
                "seed": seed,
                "train_environment": train_environment,
                "panel_manifest": str(args.panel_manifest),
                "panel_manifest_sha256": sha256_path(args.panel_manifest),
                "config": str(args.config),
                "config_sha256": sha256_path(args.config),
                "embedding_dir": EMBEDDING_DIR,
                "output_dir": str(
                    args.output_root / f"seed-{seed}__train-{train_environment}"
                ),
                "test_environments": list(E003_ENVIRONMENTS),
                "checkpoint_selection": "matched_validation_maximum_auprc",
                "threshold_selection": "matched_validation_maximum_mcc",
                "test_adaptation": False,
                "test_cropping": False,
            }
            encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            request["request_sha256"] = hashlib.sha256(encoded).hexdigest()
            path = (
                args.output_root
                / "requests"
                / f"seed-{seed}__train-{train_environment}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            row = {
                "index": index,
                "request": str(path),
                **{
                    key: request[key]
                    for key in (
                        "model",
                        "seed",
                        "train_environment",
                        "output_dir",
                        "request_sha256",
                    )
                },
            }
            requests.append(row)
            index_rows.append(row)
            index += 1
    index_path = args.output_root / "request_index.json"
    index_path.write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = {
        "schema_version": "ppi_shift.e003_frozen_esm_request_matrix.v1",
        "experiment_id": "E003",
        "panel_manifest": str(args.panel_manifest),
        "panel_manifest_sha256": sha256_path(args.panel_manifest),
        "config": str(args.config),
        "config_sha256": sha256_path(args.config),
        "embedding_dir": EMBEDDING_DIR,
        "model": MODEL,
        "seeds": list(SEEDS),
        "new_train_environments": list(NEW_TRAIN_ENVIRONMENTS),
        "test_environments": list(E003_ENVIRONMENTS),
        "requests": requests,
        "request_index": str(index_path),
        "units": len(requests),
        "interpretation_boundary": (
            "training request preparation only; no model result is contained here"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "requests": len(requests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
