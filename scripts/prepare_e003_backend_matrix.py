#!/usr/bin/env python3
"""Prepare D-SCRIPT/TUnA E003 requests for the two non-reusable train surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import sha256_path, write_csv
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import d007_rows, load_panel, load_yaml

MODELS = {
    "dscript_esm2": "23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8",
    "tuna_corrected": "b5bda8fee261a4f27821738db995cf5883dcd133",
}
SEEDS = (11, 22, 33)
NEW_TRAIN_ENVIRONMENTS = (
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
COMBINED_TEST = "all_e003_test_environments"
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
        digest.update(row["row_id"].encode("utf-8"))
    return digest.hexdigest()


def _write_partition(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    write_csv(path, BACKEND_COLUMNS, rows)
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "input_sha256": sha256_path(path),
        "row_ids_sha256": _row_ids_hash(rows),
        "rows": len(rows),
        "label_counts": {label: sum(row["label"] == label for row in rows) for label in ("0", "1")},
        "test_crop_allowed": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output root: {args.output_root}")
    panel_manifest = load_yaml(args.panel_manifest)
    if panel_manifest.get("schema_version") != E003_SCHEMA_VERSION:
        raise SystemExit("Unexpected E003 panel manifest schema")
    args.output_root.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, dict[str, Any]] = {}
    rows_by_key: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "validation", "test"):
        for environment in E003_ENVIRONMENTS:
            rows, spec = load_panel(panel_manifest, split, environment)
            backend_rows = d007_rows(rows)
            key = f"{split}__{environment}"
            rows_by_key[key] = backend_rows
            partition = _write_partition(
                args.output_root / "partitions" / f"{key}.csv", backend_rows
            )
            partition.update({"panel_path": spec["path"], "panel_sha256": spec["sha256"]})
            partitions[key] = partition

    combined_rows: list[dict[str, str]] = []
    source_panels: dict[str, Any] = {}
    for environment in E003_ENVIRONMENTS:
        source = rows_by_key[f"test__{environment}"]
        combined_rows.extend({**row, "row_id": f"{environment}::{row['row_id']}"} for row in source)
        source_panels[environment] = {
            "path": partitions[f"test__{environment}"]["panel_path"],
            "panel_sha256": partitions[f"test__{environment}"]["panel_sha256"],
            "rows": len(source),
        }
    combined_key = f"test__{COMBINED_TEST}"
    partitions[combined_key] = _write_partition(
        args.output_root / "partitions" / f"{combined_key}.csv", combined_rows
    )
    partitions[combined_key]["source_panels"] = source_panels

    requests: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model, revision in MODELS.items():
            for train_environment in NEW_TRAIN_ENVIRONMENTS:
                train = partitions[f"train__{train_environment}"]
                validation = partitions[f"validation__{train_environment}"]
                test = partitions[combined_key]
                request = {
                    "schema_version": "ppi.sota_backend_request.v1",
                    "experiment_id": "E003",
                    "test_crop_allowed": False,
                    "model": model,
                    "adapter": model,
                    "mode": "standardized_leak_free",
                    "source_task": f"e003_train_{train_environment}",
                    "target_task": f"e003_test_{COMBINED_TEST}",
                    "source_revision": revision,
                    "seed": seed,
                    "ablation": "none",
                    "clean_init_downstream_checkpoint": True,
                    "output_dir": str(
                        args.output_root
                        / f"seed-{seed}"
                        / "runs"
                        / model
                        / f"seed-{seed}__train-{train_environment}__test-{COMBINED_TEST}"
                    ),
                    "prepared_partitions": {
                        "train": train["path"],
                        "validation": validation["path"],
                        "test": test["path"],
                    },
                    "partition_summaries": {
                        split: {
                            key: partition[key]
                            for key in (
                                "path",
                                "sha256",
                                "input_sha256",
                                "row_ids_sha256",
                                "rows",
                                "label_counts",
                                "panel_sha256",
                                "source_panels",
                            )
                            if key in partition
                        }
                        for split, partition in (
                            ("train", train),
                            ("validation", validation),
                            ("test", test),
                        )
                    },
                    "panel_contract": {
                        "schema_version": E003_SCHEMA_VERSION,
                        "train_environment": train_environment,
                        "validation_environment": train_environment,
                        "test_environment": COMBINED_TEST,
                        "test_environments": list(E003_ENVIRONMENTS),
                    },
                    "selection_status": (
                        "validation_auprc_only; validation_max_mcc_threshold; no test adaptation"
                    ),
                    "test_checkpoint_reuse_across_environments": True,
                    "test_adaptation": False,
                }
                encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
                request["request_sha256"] = hashlib.sha256(encoded).hexdigest()
                request_path = (
                    args.output_root
                    / f"seed-{seed}"
                    / "requests"
                    / model
                    / f"seed-{seed}__train-{train_environment}__test-{COMBINED_TEST}.json"
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
                    }
                )
    output = {
        "schema_version": "ppi_shift.e003_backend_request_matrix.v1",
        "experiment_id": "E003",
        "panel_manifest": str(args.panel_manifest),
        "panel_manifest_sha256": sha256_path(args.panel_manifest),
        "models": MODELS,
        "seeds": list(SEEDS),
        "environments": list(E003_ENVIRONMENTS),
        "new_train_environments": list(NEW_TRAIN_ENVIRONMENTS),
        "partitions": partitions,
        "requests": requests,
        "test_cropping": False,
        "test_adaptation": False,
        "interpretation_boundary": "request preparation only; no model result is contained here",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "requests": len(requests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
