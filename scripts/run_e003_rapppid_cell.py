#!/usr/bin/env python3
"""Train one paper-faithful RAPPPID C3 E003 unit and score all test panels."""

from __future__ import annotations

import argparse
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
from ppi_shift.models.rapppid import RapppidAWD, awd_architecture_metadata

MODEL_ID = "rapppid"
SOURCE_REVISION = "c3a28be56fb3bd96ccf8cc44ea9145cfcadeaf6c"
VOCAB_SIZE = 250
TRUNCATION_LENGTH = 512
BATCH_SIZE = 80
EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001


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


def _train_sentencepiece(rows: list[dict[str, str]], output_dir: Path, seed: int) -> Any:
    import sentencepiece as spm

    sequences = sorted({row["seq1"] for row in rows} | {row["seq2"] for row in rows})
    corpus = output_dir / "train_sequences.txt"
    corpus.write_text("\n".join(sequences) + "\n", encoding="utf-8")
    prefix = output_dir / "rapppid_spm"
    spm.set_random_generator_seed(seed)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="unigram",
        bos_id=-1,
        eos_id=-1,
        pad_id=0,
        unk_id=1,
    )
    corpus.unlink()
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    if processor.vocab_size() != VOCAB_SIZE:
        raise RuntimeError(f"Unexpected SentencePiece vocabulary: {processor.vocab_size()}")
    return processor, prefix.with_suffix(".model")


def _encode_rows(
    rows: list[dict[str, str]], processor: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        encoded_left = np.asarray(processor.encode(row["seq1"], out_type=int), dtype=np.int64)
        encoded_right = np.asarray(processor.encode(row["seq2"], out_type=int), dtype=np.int64)
        if len(encoded_left) > TRUNCATION_LENGTH or len(encoded_right) > TRUNCATION_LENGTH:
            raise ValueError("SentencePiece representation exceeded the frozen <=512 contract")
        left.append(np.pad(encoded_left, (0, TRUNCATION_LENGTH - len(encoded_left))))
        right.append(np.pad(encoded_right, (0, TRUNCATION_LENGTH - len(encoded_right))))
        labels.append(int(row["interaction_label"]))
    return np.asarray(left), np.asarray(right), np.asarray(labels, dtype=np.int64)


def _scores(model: RapppidAWD, left: np.ndarray, right: np.ndarray, device: Any) -> np.ndarray:
    import torch

    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(left), BATCH_SIZE):
            left_tensor = torch.from_numpy(left[start : start + BATCH_SIZE]).to(device)
            right_tensor = torch.from_numpy(right[start : start + BATCH_SIZE]).to(device)
            logits = model.logits(left_tensor, right_tensor)
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def _checkpoint_state(selection_model: Any, target_model: Any) -> dict[str, Any]:
    """Copy an SWA state while dropping transient WeightDrop parameters.

    Recent PyTorch versions can re-register the dropped weight_hh_l0 tensor
    inside an AveragedModel after its first update. That tensor is an
    execution-time view; the target model stores only weight_hh_l0_raw.
    Keep the immutable target roster as the contract for the saved checkpoint
    and fail closed on any other unexpected or missing key.
    """

    selected = selection_model.state_dict()
    target_keys = set(target_model.state_dict())
    unexpected = set(selected) - target_keys
    # The wrapped RNN can expose this execution-time view in either the SWA
    # state or the target state, depending on whether a forward pass has
    # re-registered the dropped parameter.  It is never a checkpoint key.
    transient = {
        key
        for key in (set(selected) | target_keys)
        if key.endswith(".weight_hh_l0")
    }
    unsupported = unexpected - transient
    if unsupported:
        raise RuntimeError(f"Unsupported checkpoint keys: {sorted(unsupported)}")
    missing = (target_keys - set(selected)) - transient
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {sorted(missing)}")
    return {
        key: value.detach().cpu().clone()
        for key, value in selected.items()
        if key not in transient
    }


def _train(
    model: RapppidAWD,
    train_left: np.ndarray,
    train_right: np.ndarray,
    train_labels: np.ndarray,
    validation_left: np.ndarray,
    validation_right: np.ndarray,
    validation_labels: np.ndarray,
    device: Any,
    seed: int,
) -> tuple[dict[str, Any], int, float, list[dict[str, Any]]]:
    import torch
    from torch.optim.swa_utils import AveragedModel

    try:
        from ranger21 import Ranger21
    except ImportError as error:
        raise RuntimeError(
            "RAPPPID E003 requires the pinned upstream Ranger21 optimizer; "
            "no Adam fallback is permitted"
        ) from error

    model.to(device)
    steps_per_epoch = max(1, len(train_left) // BATCH_SIZE)
    optimizer = Ranger21(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_batches_per_epoch=steps_per_epoch,
        num_epochs=EPOCHS,
        warmdown_start_pct=0.72,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    averaged = AveragedModel(model)
    swa_start = max(1, int(np.ceil(0.8 * EPOCHS)))
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        model.train()
        indices = np.arange(len(train_left))
        np.random.default_rng(seed + epoch).shuffle(indices)
        epoch_loss = 0.0
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            left = torch.from_numpy(train_left[batch_indices]).to(device)
            right = torch.from_numpy(train_right[batch_indices]).to(device)
            labels = torch.from_numpy(train_labels[batch_indices].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model.logits(left, right), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch_indices)
        if epoch + 1 >= swa_start:
            averaged.update_parameters(model)
            validation_scores = _scores(averaged.module, validation_left, validation_right, device)
            selection_model = averaged.module
        else:
            validation_scores = _scores(model, validation_left, validation_right, device)
            selection_model = model
        auprc = float(
            __import__(
                "sklearn.metrics", fromlist=["average_precision_score"]
            ).average_precision_score(validation_labels, validation_scores)
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_left),
            "validation_auprc": auprc,
            "swa_active": epoch + 1 >= swa_start,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if auprc > best_score:
            best_score = auprc
            best_epoch = epoch + 1
            best_state = _checkpoint_state(selection_model, model)
    if best_state is None:
        raise RuntimeError("RAPPPID validation never produced a checkpoint")
    loaded = model.load_state_dict(best_state, strict=False)
    transient_load_keys = {
        key
        for key in (*loaded.missing_keys, *loaded.unexpected_keys)
        if key.endswith(".weight_hh_l0")
    }
    missing = set(loaded.missing_keys) - transient_load_keys
    unexpected = set(loaded.unexpected_keys) - transient_load_keys
    if missing or unexpected:
        raise RuntimeError(
            "Sanitized RAPPPID checkpoint did not match the target model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return best_state, best_epoch, best_score, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    processor, tokenizer_path = _train_sentencepiece(train_rows, args.output_dir, args.seed)
    train_left, train_right, train_labels = _encode_rows(train_rows, processor)
    validation_left, validation_right, validation_labels = _encode_rows(validation_rows, processor)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RapppidAWD(
        VOCAB_SIZE,
        embedding_size=64,
        recurrent_layers=2,
        embedding_dropout=0.3,
        weight_dropout=0.3,
        class_head_dropout=0.2,
        truncation_length=TRUNCATION_LENGTH,
        bi_reduce="last",
        class_head_name="concat",
    )
    best_state, best_epoch, best_auprc, history = _train(
        model,
        train_left,
        train_right,
        train_labels,
        validation_left,
        validation_right,
        validation_labels,
        device,
        args.seed,
    )
    validation_scores = _scores(model, validation_left, validation_right, device)
    threshold = select_threshold(validation_labels, validation_scores)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": "ppi_shift.e003_rapppid_checkpoint.v1",
            "model_id": MODEL_ID,
            "source_revision": SOURCE_REVISION,
            "seed": args.seed,
            "architecture": awd_architecture_metadata(),
            "tokenizer_sha256": sha256_path(tokenizer_path),
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
        left, right, labels = _encode_rows(rows, processor)
        scores = _scores(model, left, right, device)
        prediction_path = args.output_dir / f"predictions__test-{environment}.csv"
        prediction_hash = write_predictions(prediction_path, rows, scores)
        swapped = _scores(model, right, left, device)
        results[environment] = {
            "metrics": compute_binary_metrics(
                labels,
                scores,
                threshold=float(threshold["threshold"]),
                threshold_source="matched_validation_max_mcc",
            ),
            "swap_invariant": bool(np.allclose(scores, swapped, atol=1e-6, rtol=0.0)),
            "swap_max_abs": float(np.max(np.abs(scores - swapped), initial=0.0)),
        }
        prediction_files[environment] = {"path": str(prediction_path), "sha256": prediction_hash}
    run_manifest = {
        "schema_version": "ppi_shift.e003_rapppid_cell.v1",
        "experiment_id": "E003",
        "model": MODEL_ID,
        "source_revision": SOURCE_REVISION,
        "seed": args.seed,
        "train_environment": args.train_environment,
        "validation_environment": args.train_environment,
        "test_environments": list(test_environments),
        "git_commit": git_commit(args.repo_root),
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "panels": {"train": train_panel, "validation": validation_panel, "test": test_panels},
        "architecture": awd_architecture_metadata(),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": sha256_path(tokenizer_path),
            "sampling": False,
        },
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
