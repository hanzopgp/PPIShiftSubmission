#!/usr/bin/env python3
"""Score frozen E001v3 checkpoints on all frozen E003 test panels."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import git_commit, iter_delimited, sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import d007_rows, write_predictions
from ppi_shift.metrics import compute_binary_metrics, select_threshold
from ppi_shift.models.frozen_esm import FrozenESMPairMLP
from ppi_shift.negatives import EmbeddingStore


def _load_panel(
    path: Path, expected_hash: str, expected_rows: int, environment: str
) -> list[dict[str, str]]:
    actual_hash = sha256_path(path)
    if actual_hash != expected_hash:
        raise ValueError(f"E003 panel hash changed: {path}")
    rows = list(iter_delimited(path))
    if len(rows) != expected_rows or {row["negative_environment"] for row in rows} != {environment}:
        raise ValueError(f"E003 panel contract changed: {path}")
    if {str(row["sampler_version"]).split(":", 1)[0] for row in rows} != {E003_SCHEMA_VERSION}:
        raise ValueError(f"E003 panel schema changed: {path}")
    return rows


def _load_old_threshold(run_path: Path, model: str) -> dict[str, Any]:
    if model == "symmetric_frozen_esm2_pair_mlp":
        payload = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        return payload["training"]["validation_threshold"]
    validation = list(iter_delimited(run_path / "validation_predictions.csv"))
    return select_threshold(
        [int(row["y_true"]) for row in validation],
        [float(row["y_score"]) for row in validation],
    )


def _load_frozen(run_path: Path, embedding_dir: Path, seed: int) -> FrozenESMPairMLP:
    payload = json.loads(_checkpoint_path(run_path).read_text(encoding="utf-8"))
    embeddings = EmbeddingStore.from_files(
        embedding_dir / "embeddings.npy",
        embedding_dir / "records.csv",
        embedding_dir / "summary.json",
    )
    model = FrozenESMPairMLP(
        embeddings,
        seed=seed,
        hidden_layers=tuple(payload["hidden_layers"]),
        max_iter=1,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    classifier = MLPClassifier(
        hidden_layer_sizes=tuple(payload["hidden_layers"]),
        activation=payload["activation"],
        solver=payload["solver"],
        learning_rate_init=payload["learning_rate_init"],
        alpha=payload["alpha"],
        max_iter=1,
        random_state=seed,
    )
    feature_dimension = len(payload["scaler"]["mean"])
    classifier.partial_fit(
        np.zeros((2, feature_dimension), dtype=np.float64),
        np.asarray([0, 1], dtype=np.int64),
        classes=np.asarray([0, 1], dtype=np.int64),
    )
    classifier.coefs_ = [np.asarray(values, dtype=np.float64) for values in payload["coefs"]]
    classifier.intercepts_ = [
        np.asarray(values, dtype=np.float64) for values in payload["intercepts"]
    ]
    classifier.n_iter_ = int(payload.get("n_iter", 1))
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["scaler"]["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scaler"]["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(payload["scaler"]["var"], dtype=np.float64)
    scaler.n_features_in_ = feature_dimension
    scaler.n_samples_seen_ = 2
    model.model = classifier
    model._feature_scaler = scaler
    model._fitted = True
    return model


def _checkpoint_path(run_path: Path) -> Path:
    provenance_path = run_path / "provenance.json"
    if provenance_path.is_file():
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        checkpoint_path = payload.get("checkpoint_path")
    else:
        manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        checkpoint = manifest.get("checkpoint")
        checkpoint_path = checkpoint.get("path") if isinstance(checkpoint, dict) else None
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint path found under {run_path}")
    path = Path(checkpoint_path)
    return path if path.is_absolute() else run_path / path


def _load_dscript(run_path: Path, device: Any) -> tuple[Any, Any]:
    import torch

    module = importlib.import_module("ppi_project.sota_backends.dscript")
    model = module._architecture(640)().to(device)
    payload = torch.load(_checkpoint_path(run_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return module, model


def _load_tuna(run_path: Path, seed: int, device: Any) -> tuple[Any, Any]:
    import torch

    module = importlib.import_module("ppi_project.sota_backends.tuna")
    network, _trainer, tester = module._build_model(seed, device)
    payload = torch.load(_checkpoint_path(run_path), map_location=device, weights_only=False)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return module, tester


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("schema_version") != "ppi_shift.e003_reused_e001_unit_request.v1":
        raise SystemExit("Unexpected E003 reuse request schema")
    started = time.monotonic()
    model_name = request["model"]
    seed = int(request["seed"])
    run_path = Path(request["e001_run_path"])
    if not run_path.is_dir():
        raise SystemExit(f"Frozen E001 run path is unavailable: {run_path}")
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows: dict[str, list[dict[str, str]]] = {}
    for environment in E003_ENVIRONMENTS:
        spec = request["test_panels"][environment]
        test_rows[environment] = _load_panel(
            Path(spec["path"]), spec["sha256"], int(spec["rows"]), environment
        )
    threshold = _load_old_threshold(run_path, model_name)
    swap_max: dict[str, float] = {}
    results: dict[str, Any] = {}
    prediction_files: dict[str, Any] = {}
    if model_name == "symmetric_frozen_esm2_pair_mlp":
        model = _load_frozen(run_path, args.embedding_dir, seed)
        for environment, rows in test_rows.items():
            scores = model.predict(rows)
            path = output_dir / f"predictions__test-{environment}.csv"
            prediction_files[environment] = {
                "path": str(path),
                "sha256": write_predictions(path, rows, scores),
            }
            results[environment] = {
                "metrics": compute_binary_metrics(
                    [int(row["interaction_label"]) for row in rows],
                    scores,
                    threshold=float(threshold["threshold"]),
                    threshold_source="reused_e001_validation_max_mcc",
                ),
                "swap_invariant": model.swap_invariant(rows),
            }
    else:
        import torch

        sys.path.insert(0, "/path/to/ExternalPPI/src")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "dscript_esm2":
            module, model = _load_dscript(run_path, device)
        elif model_name == "tuna_corrected":
            module, model = _load_tuna(run_path, seed, device)
        else:
            raise SystemExit(f"Unsupported E001 reuse model: {model_name}")
        for environment, rows in test_rows.items():
            backend_rows = d007_rows(rows)
            if model_name == "dscript_esm2":
                scores, _logits = module._predict_internal(
                    model, backend_rows, device, original=False
                )
                swapped, _ = module._predict_internal(
                    model, backend_rows, device, original=False, swap=True
                )
            else:
                dataset = module.TokenPairs(backend_rows)
                scores, _ = module._predict(model, dataset, invariant=True)
                swapped, _ = module._predict(model, dataset, swap=True, invariant=True)
            scores = np.asarray(scores, dtype=np.float64)
            swapped = np.asarray(swapped, dtype=np.float64)
            swap_max[environment] = float(np.max(np.abs(scores - swapped), initial=0.0))
            path = output_dir / f"predictions__test-{environment}.csv"
            prediction_files[environment] = {
                "path": str(path),
                "sha256": write_predictions(path, rows, scores),
            }
            results[environment] = {
                "metrics": compute_binary_metrics(
                    [int(row["interaction_label"]) for row in rows],
                    scores,
                    threshold=float(threshold["threshold"]),
                    threshold_source="reused_e001_validation_max_mcc",
                ),
                "swap_invariant": swap_max[environment] <= 1e-6,
            }
    checkpoint_path = (
        run_path / "checkpoint.json"
        if model_name == "symmetric_frozen_esm2_pair_mlp"
        else Path(
            json.loads((run_path / "provenance.json").read_text(encoding="utf-8"))[
                "checkpoint_path"
            ]
        )
    )
    checkpoint_hash = sha256_path(checkpoint_path)
    if checkpoint_hash != request["e001_checkpoint_sha256"]:
        raise ValueError(f"E001 checkpoint hash changed: {checkpoint_path}")
    run_manifest = {
        "schema_version": "ppi_shift.e003_reused_e001_unit.v1",
        "experiment_id": "E003",
        "model": model_name,
        "seed": seed,
        "train_environment": request["train_environment"],
        "test_environments": list(E003_ENVIRONMENTS),
        "git_commit": git_commit(args.repo_root),
        "request": {"path": str(args.request), "sha256": sha256_path(args.request)},
        "e001_run_path": str(run_path),
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "validation_threshold": threshold,
        "prediction_files": prediction_files,
        "results": results,
        "swap_max_abs": swap_max,
        "retrained": False,
        "test_adaptation": False,
        "test_cropping": False,
        "wall_time_seconds": time.monotonic() - started,
        "status": "complete",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
