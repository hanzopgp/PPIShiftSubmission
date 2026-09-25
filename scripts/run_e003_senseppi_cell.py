#!/usr/bin/env python3
"""Train one clean-init paper-faithful SENSE-PPI E003 unit."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import git_commit, sha256_path
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS, E003_SCHEMA_VERSION
from ppi_shift.e003_runtime import load_all_test_panels, load_panel, load_yaml, write_predictions
from ppi_shift.metrics import compute_binary_metrics, select_threshold
from ppi_shift.models.senseppi import SensePPI, architecture_metadata

MODEL_ID = "senseppi"
PACKAGE_REVISION = "0.5.7"
REPRESENTATION_MODEL = "facebook/esm2_t36_3B_UR50D"
REPRESENTATION_REVISION = "476b639933c8baad5ad09a60ac1a87f987b656fc"
INPUT_DIMENSION = 2560
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4


class TokenStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, Any] = {}

    def get(self, sequence_hash: str) -> Any:
        import torch

        cached = self._cache.get(sequence_hash)
        if cached is not None:
            return cached
        path = self.root / "records" / f"{sequence_hash}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen SENSE ESM2-3B residue cache record: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("sequence_sha256") != sequence_hash
            or payload.get("model_name") != REPRESENTATION_MODEL
            or payload.get("model_revision") != REPRESENTATION_REVISION
            or not payload.get("special_tokens_excluded")
        ):
            raise ValueError(f"SENSE representation contract mismatch: {path}")
        embedding = payload["embedding"]
        if tuple(embedding.shape[1:]) != (INPUT_DIMENSION,):
            raise ValueError(f"Unexpected SENSE embedding dimension in {path}: {embedding.shape}")
        self._cache[sequence_hash] = embedding
        return embedding


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _batch_rows(
    rows: list[dict[str, str]], store: TokenStore, indices: np.ndarray, device: Any
) -> tuple[list[Any], list[Any], Any]:
    import torch

    selected = [rows[int(index)] for index in indices]
    left = [
        store.get(row["anchor_sequence_sha256"]).to(device=device, dtype=torch.float32)
        for row in selected
    ]
    right = [
        store.get(
            row["true_partner_sequence_sha256"]
            if row["interaction_label"] == "1"
            else row["candidate_sequence_sha256"]
        ).to(device=device, dtype=torch.float32)
        for row in selected
    ]
    labels = torch.tensor(
        [int(row["interaction_label"]) for row in selected], dtype=torch.float32, device=device
    )
    return left, right, labels


def _predict(
    model: SensePPI, rows: list[dict[str, str]], store: TokenStore, device: Any
) -> np.ndarray:
    import torch

    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(rows)))
            left, right, _labels = _batch_rows(rows, store, indices, device)
            values.append(torch.sigmoid(model.logits(left, right)).cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def _train(
    model: SensePPI,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    store: TokenStore,
    device: Any,
    seed: int,
) -> tuple[dict[str, Any], int, float, list[dict[str, Any]]]:
    import torch
    from sklearn.metrics import average_precision_score

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCEWithLogitsLoss()
    validation_labels = np.asarray(
        [int(row["interaction_label"]) for row in validation_rows], dtype=np.int64
    )
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        model.train()
        indices = np.arange(len(train_rows))
        np.random.default_rng(seed + epoch).shuffle(indices)
        epoch_loss = 0.0
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            left, right, labels = _batch_rows(train_rows, store, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model.logits(left, right), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch_indices)
        validation_scores = _predict(model, validation_rows, store, device)
        auprc = float(average_precision_score(validation_labels, validation_scores))
        record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_rows),
            "validation_auprc": auprc,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if auprc > best_score:
            best_score = auprc
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("SENSE-PPI validation never produced a checkpoint")
    model.load_state_dict(best_state)
    return best_state, best_epoch, best_score, history


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--representation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--train-environment", choices=E003_ENVIRONMENTS, required=True)
    parser.add_argument(
        "--test-environments",
        default=None,
        help="Comma-separated test environments; defaults to the manifest roster.",
    )
    args = parser.parse_args()
    started = time.monotonic()
    _seed_everything(args.seed)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = TokenStore(args.representation_cache)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensePPI(input_dimension=INPUT_DIMENSION)
    best_state, best_epoch, best_auprc, history = _train(
        model, train_rows, validation_rows, store, device, args.seed
    )
    validation_scores = _predict(model, validation_rows, store, device)
    validation_labels = [int(row["interaction_label"]) for row in validation_rows]
    threshold = select_threshold(validation_labels, validation_scores)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": "ppi_shift.e003_senseppi_checkpoint.v1",
            "model_id": MODEL_ID,
            "package_revision": PACKAGE_REVISION,
            "representation_model": REPRESENTATION_MODEL,
            "representation_revision": REPRESENTATION_REVISION,
            "seed": args.seed,
            "architecture": architecture_metadata(
                input_dimension=INPUT_DIMENSION,
                representation_model=REPRESENTATION_MODEL,
                representation_revision=REPRESENTATION_REVISION,
            ),
            "state_dict": best_state,
            "selected_epoch": best_epoch,
            "selected_validation_auprc": best_auprc,
            "selection_history": history,
        },
        checkpoint_path,
    )
    results: dict[str, Any] = {}
    prediction_files: dict[str, Any] = {}
    for environment, rows in test_rows.items():
        scores = _predict(model, rows, store, device)
        prediction_path = args.output_dir / f"predictions__test-{environment}.csv"
        prediction_hash = write_predictions(prediction_path, rows, scores)
        swapped_rows = []
        for row in rows:
            if row["interaction_label"] == "1":
                swapped_rows.append(
                    {
                        **row,
                        "anchor_sequence_sha256": row["true_partner_sequence_sha256"],
                        "true_partner_sequence_sha256": row["anchor_sequence_sha256"],
                    }
                )
            else:
                swapped_rows.append(
                    {
                        **row,
                        "anchor_sequence_sha256": row["candidate_sequence_sha256"],
                        "candidate_sequence_sha256": row["anchor_sequence_sha256"],
                    }
                )
        swapped = _predict(model, swapped_rows, store, device)
        results[environment] = {
            "metrics": compute_binary_metrics(
                [int(row["interaction_label"]) for row in rows],
                scores,
                threshold=float(threshold["threshold"]),
                threshold_source="matched_validation_max_mcc",
            ),
            "swap_invariant": bool(np.allclose(scores, swapped, atol=1e-6, rtol=0.0)),
            "swap_max_abs": float(np.max(np.abs(scores - swapped), initial=0.0)),
        }
        prediction_files[environment] = {"path": str(prediction_path), "sha256": prediction_hash}
    run_manifest = {
        "schema_version": "ppi_shift.e003_senseppi_cell.v1",
        "experiment_id": "E003",
        "model": MODEL_ID,
        "package_revision": PACKAGE_REVISION,
        "representation_model": REPRESENTATION_MODEL,
        "representation_revision": REPRESENTATION_REVISION,
        "seed": args.seed,
        "train_environment": args.train_environment,
        "validation_environment": args.train_environment,
        "test_environments": list(test_environments),
        "git_commit": git_commit(args.repo_root),
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "panels": {"train": train_panel, "validation": validation_panel, "test": test_panels},
        "representation_cache": {"path": str(args.representation_cache)},
        "architecture": architecture_metadata(
            input_dimension=INPUT_DIMENSION,
            representation_model=REPRESENTATION_MODEL,
            representation_revision=REPRESENTATION_REVISION,
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_path(checkpoint_path),
            "selected_epoch": best_epoch,
            "selected_validation_auprc": best_auprc,
        },
        "validation": {
            "metrics": compute_binary_metrics(validation_labels, validation_scores),
            "threshold": threshold,
        },
        "test_results": results,
        "prediction_files": prediction_files,
        "downstream_checkpoint_reused": False,
        "test_adaptation": False,
        "test_cropping": False,
        "device": str(device),
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
