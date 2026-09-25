#!/usr/bin/env python3
"""Train one clean-init Frozen-ESM2 pair-MLP E003 unit and score all tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import git_commit, sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import load_all_test_panels, load_panel, load_yaml, write_predictions
from ppi_shift.metrics import compute_binary_metrics, select_threshold
from ppi_shift.models.frozen_esm import FrozenESMPairMLP
from ppi_shift.negatives import EmbeddingStore

MODEL_ID = "symmetric_frozen_esm2_pair_mlp"
REPRESENTATION_REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"


def _checkpoint_payload(model: FrozenESMPairMLP) -> dict[str, Any]:
    classifier = model.model
    scaler = model._feature_scaler
    if scaler is None or not hasattr(classifier, "coefs_"):
        raise RuntimeError("Frozen ESM model did not expose a selected checkpoint")
    return {
        "schema_version": "ppi_shift.e003_frozen_esm2_checkpoint.v1",
        "model_id": MODEL_ID,
        "representation_revision": REPRESENTATION_REVISION,
        "hidden_layers": list(model.model.hidden_layer_sizes),
        "activation": classifier.activation,
        "solver": classifier.solver,
        "learning_rate_init": classifier.learning_rate_init,
        "alpha": classifier.alpha,
        "seed": model.seed,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "coefs": [array.tolist() for array in classifier.coefs_],
        "intercepts": [array.tolist() for array in classifier.intercepts_],
        "n_iter": int(classifier.n_iter_),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--train-environment", choices=E003_ENVIRONMENTS, required=True)
    parser.add_argument(
        "--test-environments",
        default=None,
        help="Comma-separated test environments; defaults to the manifest roster.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    started = time.monotonic()
    manifest = load_yaml(args.panel_manifest)
    config = load_yaml(args.config)
    if manifest.get("schema_version") != E003_SCHEMA_VERSION:
        raise SystemExit("Unexpected E003 panel manifest schema")
    if config.get("experiment_id") != "E003":
        raise SystemExit("Unexpected E003 config")

    train_rows, train_panel = load_panel(manifest, "train", args.train_environment)
    validation_rows, validation_panel = load_panel(manifest, "validation", args.train_environment)
    test_environments = tuple(
        args.test_environments.split(",")
        if args.test_environments
        else manifest.get("environments", E003_ENVIRONMENTS)
    )
    test_rows, test_panels = load_all_test_panels(manifest, test_environments)
    embeddings = EmbeddingStore.from_files(
        args.embedding_dir / "embeddings.npy",
        args.embedding_dir / "records.csv",
        args.embedding_dir / "summary.json",
    )
    model = FrozenESMPairMLP(
        embeddings, seed=args.seed, hidden_layers=(64, 32), max_iter=1
    ).fit_with_validation(train_rows, validation_rows, epochs=args.epochs)
    validation_scores = model.predict(validation_rows)
    validation_labels = [int(row["interaction_label"]) for row in validation_rows]
    threshold = select_threshold(validation_labels, validation_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    checkpoint = _checkpoint_payload(model)
    checkpoint["selection_history"] = model.selection_history
    checkpoint["selected_epoch"] = model.selected_epoch
    checkpoint["selected_validation_auprc"] = model.selected_validation_auprc
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results: dict[str, Any] = {}
    prediction_files: dict[str, Any] = {}
    for environment, rows in test_rows.items():
        scores = model.predict(rows)
        prediction_path = args.output_dir / f"predictions__test-{environment}.csv"
        prediction_hash = write_predictions(prediction_path, rows, scores)
        results[environment] = {
            "metrics": compute_binary_metrics(
                [int(row["interaction_label"]) for row in rows],
                scores,
                threshold=float(threshold["threshold"]),
                threshold_source="matched_validation_max_mcc",
            ),
            "swap_invariant": model.swap_invariant(rows),
        }
        prediction_files[environment] = {
            "path": str(prediction_path),
            "sha256": prediction_hash,
        }
    run_manifest = {
        "schema_version": "ppi_shift.e003_frozen_esm_cell.v1",
        "experiment_id": "E003",
        "model": MODEL_ID,
        "representation_revision": REPRESENTATION_REVISION,
        "seed": args.seed,
        "train_environment": args.train_environment,
        "validation_environment": args.train_environment,
        "test_environments": list(test_environments),
        "git_commit": git_commit(args.repo_root),
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "panels": {"train": train_panel, "validation": validation_panel, "test": test_panels},
        "embedding_store": {
            "metadata": dict(embeddings.metadata),
            "source_paths": dict(embeddings.source_paths),
            "source_hashes": dict(embeddings.source_hashes),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_path(checkpoint_path),
            "selected_epoch": model.selected_epoch,
            "selected_validation_auprc": model.selected_validation_auprc,
        },
        "validation": {
            "metrics": compute_binary_metrics(validation_labels, validation_scores),
            "threshold": threshold,
        },
        "test_results": results,
        "prediction_files": prediction_files,
        "test_adaptation": False,
        "test_cropping": False,
        "status": "complete",
        "wall_time_seconds": time.monotonic() - started,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
