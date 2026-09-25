#!/usr/bin/env python3
"""Prepare the predeclared 36-cell official-DataSAIL sensitivity matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path, write_csv
from ppi_shift.e003_method_fidelity import E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import d007_rows, load_panel, load_yaml

MODELS = {
    "dscript_esm2": "23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8",
    "tuna_corrected": "b5bda8fee261a4f27821738db995cf5883dcd133",
}
FROZEN_MODEL = "symmetric_frozen_esm2_pair_mlp"
SEEDS = (11, 22, 33)
ENVIRONMENTS = ("uniform_partner", "esm_partner_matched")
BACKEND_COLUMNS = (
    "row_id",
    "left_id",
    "right_id",
    "left_sequence",
    "right_sequence",
    "left_sequence_sha256",
    "right_sequence_sha256",
    "label",
)


def _row_ids_hash(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        if index:
            digest.update(b"\n")
        digest.update(row["row_id"].encode())
    return digest.hexdigest()


def _partition(path: Path, rows: list[dict[str, str]], panel: dict[str, Any]) -> dict[str, Any]:
    write_csv(path, BACKEND_COLUMNS, rows)
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "input_sha256": sha256_path(path),
        "rows": len(rows),
        "row_ids_sha256": _row_ids_hash(rows),
        "label_counts": {label: sum(row["label"] == label for row in rows) for label in ("0", "1")},
        "panel_path": panel["path"],
        "panel_sha256": panel["sha256"],
        "test_crop_allowed": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty DataSAIL matrix root: {args.output_root}")
    panel_manifest = load_yaml(args.panel_manifest)
    if panel_manifest.get("schema_version") != E003_SCHEMA_VERSION:
        raise SystemExit("Unexpected E003 DataSAIL panel schema")
    if tuple(panel_manifest.get("environments", ())) != ENVIRONMENTS:
        raise SystemExit("DataSAIL panel roster must contain uniform and ESM environments only")
    args.output_root.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, dict[str, Any]] = {}
    rows_by_key: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "validation", "test"):
        for environment in ENVIRONMENTS:
            rows, panel = load_panel(panel_manifest, split, environment)
            backend_rows = d007_rows(rows)
            key = f"{split}__{environment}"
            rows_by_key[key] = backend_rows
            partitions[key] = _partition(
                args.output_root / "partitions" / f"{key}.csv", backend_rows, panel
            )
    combined_rows: list[dict[str, str]] = []
    for environment in ENVIRONMENTS:
        combined_rows.extend(
            {**row, "row_id": f"{environment}::{row['row_id']}"}
            for row in rows_by_key[f"test__{environment}"]
        )
    partitions["test__combined"] = _partition(
        args.output_root / "partitions" / "test__combined.csv",
        combined_rows,
        {"path": "combined DataSAIL test panels", "sha256": _row_ids_hash(combined_rows)},
    )

    backend_requests: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model, revision in MODELS.items():
            for environment in ENVIRONMENTS:
                request = {
                    "schema_version": "ppi.sota_backend_request.v1",
                    "experiment_id": "E003",
                    "mode": "standardized_leak_free",
                    "datasail_sensitivity": True,
                    "model": model,
                    "adapter": model,
                    "seed": seed,
                    "ablation": "none",
                    "clean_init_downstream_checkpoint": True,
                    "train_environment": environment,
                    "test_environments": list(ENVIRONMENTS),
                    "source_revision": revision,
                    "test_crop_allowed": False,
                    "source_task": f"e003_datasail_train_{environment}",
                    "target_task": "e003_datasail_test_uniform_and_esm",
                    "prepared_partitions": {
                        "train": partitions[f"train__{environment}"]["path"],
                        "validation": partitions[f"validation__{environment}"]["path"],
                        "test": partitions["test__combined"]["path"],
                    },
                    "partition_summaries": {
                        "train": partitions[f"train__{environment}"],
                        "validation": partitions[f"validation__{environment}"],
                        "test": partitions["test__combined"],
                    },
                    "panel_manifest": str(args.panel_manifest),
                    "panel_manifest_sha256": sha256_path(args.panel_manifest),
                    "selection_status": "matched_validation_only; no DataSAIL test adaptation",
                    "test_adaptation": False,
                    "output_dir": str(
                        args.output_root / "backend" / model / f"seed-{seed}__train-{environment}"
                    ),
                }
                encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
                request["request_sha256"] = hashlib.sha256(encoded).hexdigest()
                path = (
                    args.output_root
                    / "backend"
                    / "requests"
                    / model
                    / f"seed-{seed}__train-{environment}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                backend_requests.append(
                    {
                        "request": str(path),
                        "model": model,
                        "seed": seed,
                        "train_environment": environment,
                        "output_dir": request["output_dir"],
                        "request_sha256": request["request_sha256"],
                    }
                )

    frozen_requests: list[dict[str, Any]] = []
    for seed in SEEDS:
        for environment in ENVIRONMENTS:
            request = {
                "schema_version": "ppi_shift.e003_datasail_frozen_request.v1",
                "experiment_id": "E003",
                "datasail_sensitivity": True,
                "model": FROZEN_MODEL,
                "seed": seed,
                "train_environment": environment,
                "test_environments": list(ENVIRONMENTS),
                "panel_manifest": str(args.panel_manifest),
                "panel_manifest_sha256": sha256_path(args.panel_manifest),
                "config": str(args.config),
                "config_sha256": sha256_path(args.config),
                "embedding_dir": str(args.embedding_dir),
                "output_dir": str(
                    args.output_root / "frozen" / f"seed-{seed}__train-{environment}"
                ),
                "checkpoint_selection": "matched_validation_maximum_auprc",
                "threshold_selection": "matched_validation_maximum_mcc",
                "test_adaptation": False,
                "test_cropping": False,
            }
            encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            request["request_sha256"] = hashlib.sha256(encoded).hexdigest()
            path = (
                args.output_root / "frozen" / "requests" / f"seed-{seed}__train-{environment}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            frozen_requests.append(
                {
                    "request": str(path),
                    "model": FROZEN_MODEL,
                    "seed": seed,
                    "train_environment": environment,
                    "output_dir": request["output_dir"],
                    "request_sha256": request["request_sha256"],
                }
            )
    frozen_index = args.output_root / "frozen" / "request_index.json"
    frozen_index.write_text(
        json.dumps(frozen_requests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    backend_index = args.output_root / "backend" / "request_index.json"
    backend_index.parent.mkdir(parents=True, exist_ok=True)
    backend_index.write_text(
        json.dumps(backend_requests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = {
        "schema_version": "ppi_shift.e003_datasail_request_matrix.v1",
        "experiment_id": "E003",
        "panel_manifest": str(args.panel_manifest),
        "panel_manifest_sha256": sha256_path(args.panel_manifest),
        "config": str(args.config),
        "config_sha256": sha256_path(args.config),
        "embedding_dir": str(args.embedding_dir),
        "environments": list(ENVIRONMENTS),
        "seeds": list(SEEDS),
        "backend_requests": backend_requests,
        "frozen_requests": frozen_requests,
        "frozen_index": str(frozen_index),
        "backend_index": str(backend_index),
        "cells": 3 * 2 * 2 * 3,
        "training_units": len(backend_requests) + len(frozen_requests),
        "test_adaptation": False,
        "interpretation_boundary": "supplementary official DataSAIL sensitivity only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "backend_units": len(backend_requests),
                "frozen_units": len(frozen_requests),
                "cells": output["cells"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
