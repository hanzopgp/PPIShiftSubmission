#!/usr/bin/env python3
"""Score deterministic broken-pair panels with an already selected checkpoint.

The script never fits or selects a model. It reuses a checkpoint and its
training-owned validation threshold, then writes scores and label-preserving
diagnostics for the broken panels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from e003_signal_common import ENVIRONMENTS, base_metadata, parse_csv_values, write_csv, write_json

from ppi_shift.e003_runtime import d007_rows
from ppi_shift.metrics import compute_binary_metrics, select_threshold
from ppi_shift.signal_audit import read_rows, sha256_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_from_run(run_path: Path, model: str) -> dict[str, Any]:
    manifest_path = run_path / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        for candidate in (
            manifest.get("validation", {}).get("threshold"),
            manifest.get("training", {}).get("validation_threshold"),
            manifest.get("validation_threshold"),
        ):
            if isinstance(candidate, dict) and "threshold" in candidate:
                return candidate
    validation_path = run_path / "validation_predictions.csv"
    if validation_path.is_file():
        rows = read_rows(validation_path)
        return select_threshold(
            [int(row["y_true"]) for row in rows],
            [float(row["y_score"]) for row in rows],
        )
    raise FileNotFoundError(f"No training-owned validation threshold found under {run_path}")


def _load_script_module(repo_root: Path, filename: str) -> Any:
    path = repo_root / "scripts" / filename
    specification = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_model_scorer(
    model: str,
    run_path: Path,
    *,
    embedding_dir: Path | None,
    repo_root: Path,
    source_root: Path,
    seed: int,
) -> tuple[Callable[[list[dict[str, str]]], np.ndarray], dict[str, Any]]:
    """Return a score-only adapter for one selected run."""

    threshold = _threshold_from_run(run_path, model)
    reuse = _load_script_module(repo_root, "run_e003_reused_e001_scores.py")
    if model == "symmetric_frozen_esm2_pair_mlp":
        if embedding_dir is None:
            raise ValueError("Frozen-ESM2 broken-pair scoring requires --embedding-dir")
        model_object = reuse._load_frozen(run_path, embedding_dir, seed)
        return lambda rows: np.asarray(model_object.predict(rows), dtype=np.float64), threshold
    sys.path.insert(0, str(source_root))
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model == "dscript_esm2":
        module, model_object = reuse._load_dscript(run_path, device)

        def score(rows: list[dict[str, str]]) -> np.ndarray:
            scores, _ = module._predict_internal(
                model_object, d007_rows(rows), device, original=False
            )
            return np.asarray(scores, dtype=np.float64)

        return score, threshold
    if model == "tuna_corrected":
        module, model_object = reuse._load_tuna(run_path, seed, device)

        def score(rows: list[dict[str, str]]) -> np.ndarray:
            dataset = module.TokenPairs(d007_rows(rows))
            scores, _ = module._predict(model_object, dataset, invariant=True)
            return np.asarray(scores, dtype=np.float64)

        return score, threshold
    if model == "rapppid":
        rapppid_script = _load_script_module(repo_root, "run_e003_rapppid_cell.py")
        import sentencepiece as spm

        from ppi_shift.models.rapppid import RapppidAWD

        manifest = _load_json(run_path / "run_manifest.json")
        checkpoint_path = Path(manifest["checkpoint"]["path"])
        tokenizer_path = Path(manifest["tokenizer"]["path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_path / checkpoint_path
        if not tokenizer_path.is_absolute():
            tokenizer_path = run_path / tokenizer_path
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_object = RapppidAWD(
            250,
            embedding_size=64,
            recurrent_layers=2,
            embedding_dropout=0.3,
            weight_dropout=0.3,
            class_head_dropout=0.2,
            truncation_length=512,
            bi_reduce="last",
            class_head_name="concat",
        )
        model_object.load_state_dict(payload["state_dict"])
        model_object.to(device)
        processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

        def score(rows: list[dict[str, str]]) -> np.ndarray:
            left, right, _labels = rapppid_script._encode_rows(rows, processor)
            return rapppid_script._scores(model_object, left, right, device)

        return score, threshold
    raise ValueError(f"Unsupported broken-pair model: {model}")


def _load_broken_panel(path: Path, environment: str) -> list[dict[str, str]]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Empty broken panel: {path}")
    if {str(row.get("negative_environment", "")) for row in rows} != {environment}:
        raise ValueError(f"Broken panel environment mismatch: {path}")
    if {str(row.get("broken_pair", "")) for row in rows} != {"true"}:
        raise ValueError(f"Broken panel is not marked as broken: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broken-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, default=Path("/path/to/ExternalPPI/src"))
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("audit/E003_SIGNAL_AUDIT_PROTOCOL.md")
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-environment", required=True)
    parser.add_argument("--environments")
    args = parser.parse_args()
    if args.train_environment not in ENVIRONMENTS:
        raise SystemExit(f"Unknown train environment: {args.train_environment}")
    environments = parse_csv_values(args.environments, ENVIRONMENTS)
    scorer, threshold = _load_model_scorer(
        args.model,
        args.run_path,
        embedding_dir=args.embedding_dir,
        repo_root=args.repo_root,
        source_root=args.source_root,
        seed=args.seed,
    )
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    panel_inputs = {}
    for environment in environments:
        candidates = (
            args.broken_root / f"test__{environment}__broken.csv.gz",
            args.broken_root / f"test__{environment}__broken.csv",
        )
        panel_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if panel_path is None:
            raise FileNotFoundError(f"Missing broken panel for {environment}: {candidates}")
        rows = _load_broken_panel(panel_path, environment)
        scores = scorer(rows)
        if len(scores) != len(rows) or not np.all(np.isfinite(scores)):
            raise ValueError(f"Invalid score vector for {environment}")
        labels = [int(row["interaction_label"]) for row in rows]
        summary_rows.append(
            {
                "model": args.model,
                "train_environment": args.train_environment,
                "environment": environment,
                "seed": args.seed,
                "status": "PASS",
                **compute_binary_metrics(
                    labels,
                    scores,
                    threshold=float(threshold["threshold"]),
                    threshold_source="selected_training_validation_threshold",
                ),
                "panel_sha256": sha256_path(panel_path),
            }
        )
        for row, score in zip(rows, scores, strict=True):
            prediction_rows.append(
                {
                    "model": args.model,
                    "train_environment": args.train_environment,
                    "environment": environment,
                    "seed": args.seed,
                    "row_id": row["row_id"],
                    "interaction_label": row["interaction_label"],
                    "y_score": float(score),
                    "broken_pair": row["broken_pair"],
                    "broken_known_positive_collision": row.get(
                        "broken_known_positive_collision", ""
                    ),
                }
            )
        panel_inputs[environment] = {"path": str(panel_path), "sha256": sha256_path(panel_path)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / (
        f"broken_predictions__{args.model}__train-{args.train_environment}__seed-{args.seed}.csv"
    )
    summary_path = args.output_dir / (
        f"broken_metrics__{args.model}__train-{args.train_environment}__seed-{args.seed}.csv"
    )
    write_csv(predictions_path, prediction_rows)
    write_csv(summary_path, summary_rows)
    metadata = base_metadata(args.repo_root, args.protocol)
    metadata.update(
        {
            "control": "broken_pair_score",
            "model": args.model,
            "train_environment": args.train_environment,
            "seed": args.seed,
            "run_path": str(args.run_path),
            "threshold": threshold,
            "panel_inputs": panel_inputs,
            "prediction_file": {
                "path": str(predictions_path),
                "sha256": sha256_path(predictions_path),
            },
            "metrics_file": {"path": str(summary_path), "sha256": sha256_path(summary_path)},
            "results": summary_rows,
        }
    )
    write_json(
        args.output_dir
        / f"broken_scores__{args.model}__train-{args.train_environment}__seed-{args.seed}.json",
        metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
