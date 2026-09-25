#!/usr/bin/env python3
"""Score one frozen E003 checkpoint on every corrected external surface."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import canonical_json_sha256, iter_delimited, sha256_path, write_csv
from ppi_shift.metrics import compute_binary_metrics, external_prevalence_metrics


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    actual = sha256_path(path)
    if actual != expected:
        raise SystemExit(f"{label} hash changed: {path} ({actual} != {expected})")


def _d007_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "row_id": row["row_id"],
            "left_id": row["left_id"],
            "right_id": row["right_id"],
            "left_sequence": row["left_sequence"],
            "right_sequence": row["right_sequence"],
            "left_sequence_sha256": row["left_sequence_sha256"],
            "right_sequence_sha256": row["right_sequence_sha256"],
            "label": row["label"],
        }
        for row in rows
    ]


def _frozen_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "row_id": row["row_id"],
            "positive_row_id": row["row_id"],
            "interaction_label": row["label"],
            "anchor_sequence_sha256": row["left_sequence_sha256"],
            "true_partner_sequence_sha256": (
                row["right_sequence_sha256"] if row["label"] == "1" else ""
            ),
            "candidate_sequence_sha256": (
                row["right_sequence_sha256"] if row["label"] == "0" else ""
            ),
            "seq1": row["left_sequence"],
            "seq2": row["right_sequence"],
        }
        for row in rows
    ]


def _sense_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "row_id": row["row_id"],
            "interaction_label": row["label"],
            "anchor_sequence_sha256": row["left_sequence_sha256"],
            "true_partner_sequence_sha256": (
                row["right_sequence_sha256"] if row["label"] == "1" else ""
            ),
            "candidate_sequence_sha256": (
                row["right_sequence_sha256"] if row["label"] == "0" else ""
            ),
        }
        for row in rows
    ]


def _encode_rapppid(rows: list[dict[str, str]], processor: Any) -> tuple[np.ndarray, np.ndarray]:
    length = 512
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for row in rows:
        left_tokens = np.asarray(
            processor.encode(row["left_sequence"], out_type=int, enable_sampling=False)
        )
        right_tokens = np.asarray(
            processor.encode(row["right_sequence"], out_type=int, enable_sampling=False)
        )
        if len(left_tokens) > length or len(right_tokens) > length:
            raise ValueError("External sequence exceeded RAPPPID's no-cropping length contract")
        left.append(np.pad(left_tokens, (0, length - len(left_tokens))))
        right.append(np.pad(right_tokens, (0, length - len(right_tokens))))
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _balanced_indices(rows: list[dict[str, str]]) -> list[int]:
    positive = [index for index, row in enumerate(rows) if row["label"] == "1"]
    negative = [index for index, row in enumerate(rows) if row["label"] == "0"]
    count = min(len(positive), len(negative))

    def order(index: int) -> str:
        return hashlib.sha256(rows[index]["row_id"].encode("utf-8")).hexdigest()

    selected_positive = sorted(positive, key=order)[:count]
    selected_negative = sorted(negative, key=order)[:count]
    return sorted(selected_positive + selected_negative)


class ExternalScorer:
    """Load one checkpoint once and expose deterministic surface scoring."""

    def __init__(
        self,
        model_name: str,
        checkpoint: Path,
        seed: int,
        *,
        embedding_dir: Path,
        sense_cache: Path | None,
        tokenizer_path: str | None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.checkpoint = checkpoint
        self.device: Any = None
        self.module: Any = None
        self.model: Any = None
        self.tester: Any = None
        self.embedding_model: Any = None
        self.sense_store: Any = None
        self.processor: Any = None
        if model_name == "symmetric_frozen_esm2_pair_mlp":
            sys.path.insert(0, str(Path(__file__).parent))
            from run_e003_reused_e001_scores import _load_frozen

            self.embedding_model = _load_frozen(checkpoint.parent, embedding_dir, seed)
        elif model_name in {"dscript_esm2", "tuna_corrected"}:
            import torch

            sys.path.insert(0, "/path/to/ExternalPPI/src")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            backend_name = model_name.split("_", 1)[0]
            self.module = importlib.import_module(f"ppi_project.sota_backends.{backend_name}")
            if model_name == "dscript_esm2":
                self.model = self.module._architecture(640)().to(self.device)
            else:
                self.model, _trainer, self.tester = self.module._build_model(seed, self.device)
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if model_name == "dscript_esm2":
                self.model.load_state_dict(payload["state_dict"])
                self.model.eval()
            else:
                self.model.load_state_dict(payload["state_dict"])
                self.model.eval()
        elif model_name == "senseppi":
            import torch

            if sense_cache is None:
                raise ValueError("SENSE-PPI external scoring requires an explicit optional cache")
            from ppi_shift.models.senseppi import SensePPI

            sys.path.insert(0, str(Path(__file__).parent))
            from run_e003_senseppi_cell import TokenStore, _predict

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sense_store = TokenStore(sense_cache)
            self.module = _predict
            self.model = SensePPI(input_dimension=2560).to(self.device)
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
        elif model_name == "rapppid":
            import sentencepiece as spm
            import torch

            from ppi_shift.models.rapppid import RapppidAWD

            if not tokenizer_path:
                raise ValueError("RAPPPID checkpoint index lacks tokenizer path")
            self.processor = spm.SentencePieceProcessor(model_file=tokenizer_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = RapppidAWD(
                250,
                embedding_size=64,
                recurrent_layers=2,
                embedding_dropout=0.3,
                weight_dropout=0.3,
                class_head_dropout=0.2,
                truncation_length=512,
                bi_reduce="last",
                class_head_name="concat",
            ).to(self.device)
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
        else:
            raise ValueError(f"Unsupported E003 external model: {model_name}")

    def score(self, rows: list[dict[str, str]]) -> np.ndarray:
        if self.model_name == "symmetric_frozen_esm2_pair_mlp":
            return np.asarray(self.embedding_model.predict(_frozen_rows(rows)), dtype=np.float64)
        if self.model_name == "dscript_esm2":
            d007_rows = _d007_rows(rows)
            scores, _ = self.module._predict_internal(
                self.model, d007_rows, self.device, original=False
            )
            swapped, _ = self.module._predict_internal(
                self.model, d007_rows, self.device, original=False, swap=True
            )
            return 0.5 * (np.asarray(scores) + np.asarray(swapped))
        if self.model_name == "tuna_corrected":
            scores, _ = self.module._predict(
                self.tester, self.module.TokenPairs(_d007_rows(rows)), invariant=True
            )
            return np.asarray(scores, dtype=np.float64)
        if self.model_name == "senseppi":
            return np.asarray(
                self.module(self.model, _sense_rows(rows), self.sense_store, self.device),
                dtype=np.float64,
            )
        import torch

        left, right = _encode_rapppid(rows, self.processor)
        values: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(left), 256):
                logits = self.model.logits(
                    torch.from_numpy(left[start : start + 256]).to(self.device),
                    torch.from_numpy(right[start : start + 256]).to(self.device),
                )
                values.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(values).astype(np.float64, copy=False)


def _write_predictions(path: Path, rows: list[dict[str, str]], scores: np.ndarray) -> str:
    write_csv(
        path,
        ("row_id", "label", "score"),
        (
            {"row_id": row["row_id"], "label": row["label"], "score": format(float(score), ".17g")}
            for row, score in zip(rows, scores, strict=True)
        ),
    )
    return sha256_path(path)


def _surface_metrics(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    threshold: float,
    *,
    prevalence_sensitivity_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    result: dict[str, Any] = {
        "natural_prevalence": external_prevalence_metrics(
            labels,
            scores,
            threshold=threshold,
            threshold_source="corresponding_Gold_validation_max_MCC",
        )
    }
    indices = _balanced_indices(rows)
    balanced_rows = [rows[index] for index in indices]
    balanced_scores = scores[indices]
    result["balanced_posthoc_diagnostic"] = {
        "metrics": compute_binary_metrics(
            [int(row["label"]) for row in balanced_rows],
            balanced_scores,
            threshold=threshold,
            threshold_source="corresponding_Gold_validation_max_MCC",
        ),
        "selection": {
            "role": "post_hoc_balanced_sensitivity_only",
            "row_order": "sha256(row_id) within each label",
            "positive_rows": sum(row["label"] == "1" for row in balanced_rows),
            "negative_rows": sum(row["label"] == "0" for row in balanced_rows),
            "discarded_rows": len(rows) - len(balanced_rows),
        },
    }
    if prevalence_sensitivity_seeds:
        positive = np.asarray(
            [index for index, row in enumerate(rows) if row["label"] == "1"], dtype=np.int64
        )
        negative = np.asarray(
            [index for index, row in enumerate(rows) if row["label"] == "0"], dtype=np.int64
        )
        if len(positive) < len(negative) or not len(negative):
            raise ValueError("IntAct post-hoc sensitivity requires at least as many positives")
        sensitivity_rows: list[dict[str, Any]] = []
        for seed in prevalence_sensitivity_seeds:
            rng = np.random.default_rng(seed)
            selected_positive = rng.choice(positive, size=len(negative), replace=False)
            selected = np.concatenate((selected_positive, negative))
            metrics = compute_binary_metrics(
                [int(rows[index]["label"]) for index in selected],
                scores[selected],
                threshold=threshold,
                threshold_source="corresponding_Gold_validation_max_MCC",
            )
            sensitivity_rows.append(
                {
                    "seed": int(seed),
                    "positive_rows": int(len(selected_positive)),
                    "negative_rows": int(len(negative)),
                    "metrics": metrics,
                }
            )
        result["balanced_prevalence_sensitivity"] = {
            "role": "post_hoc_prevalence_sensitivity_only",
            "sampling": "all explicit negatives plus positives sampled without replacement",
            "seed_set": [int(seed) for seed in prevalence_sensitivity_seeds],
            "positive_rows": int(len(positive)),
            "negative_rows": int(len(negative)),
            "replicates": sensitivity_rows,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--core-index", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument(
        "--sense-cache",
        type=Path,
        help="Optional historical SENSE-PPI cache; never required by the Core-4 roster.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    request = _json(args.request)
    if request.get("schema_version") != "ppi_shift.e003_external_unit_request.v1":
        raise SystemExit("Unexpected E003 external request schema")
    request_sha256 = str(request.get("request_sha256", ""))
    if (
        not request_sha256
        or canonical_json_sha256(request, omit_key="request_sha256") != request_sha256
    ):
        raise SystemExit("E003 external request canonical hash is invalid")
    external_path = Path(request["external_manifest"])
    if request.get("external_manifest_sha256") != sha256_path(external_path):
        raise SystemExit("E003 external manifest hash changed")
    if Path(request["core_index_path"]).resolve() != args.core_index.resolve():
        raise SystemExit("E003 external request core-index path differs from CLI input")
    if request.get("core_index_sha256") != sha256_path(args.core_index):
        raise SystemExit("E003 external checkpoint-index hash changed")
    external = _json(external_path)
    index_payload = _json(args.core_index)
    key = request["checkpoint_index_key"]
    checkpoint = next(
        (
            row
            for row in index_payload["units"]
            if f"{row['model']}|{row['seed']}|{row['train_environment']}" == key
        ),
        None,
    )
    if checkpoint is None:
        raise SystemExit(f"Missing checkpoint-index unit: {key}")
    checkpoint_path = Path(checkpoint["checkpoint_path"])
    _assert_hash(checkpoint_path, checkpoint["checkpoint_sha256"], "E003 checkpoint")
    output_dir = Path(args.output_dir or request["output_dir"])
    if (output_dir / "external_run_manifest.json").exists():
        raise SystemExit(f"Refusing to overwrite completed external bundle: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = ExternalScorer(
        request["model"],
        checkpoint_path,
        int(request["seed"]),
        embedding_dir=args.embedding_dir,
        sense_cache=args.sense_cache,
        tokenizer_path=checkpoint.get("tokenizer_path"),
    )
    started = time.monotonic()
    threshold = float(checkpoint["threshold"]["threshold"])
    surface_results: dict[str, Any] = {}
    for source, source_info in external["sources"].items():
        for surface, spec in source_info["surfaces"].items():
            panel_path = Path(spec["path"])
            _assert_hash(panel_path, spec["sha256"], "E003 external panel")
            rows = list(iter_delimited(panel_path))
            if {row["label"] for row in rows} != {"0", "1"}:
                raise SystemExit(f"External panel is not binary: {panel_path}")
            scores = scorer.score(rows)
            prediction_path = output_dir / source / f"{surface}.csv"
            metric_payload = _surface_metrics(
                rows,
                scores,
                threshold,
                prevalence_sensitivity_seeds=(11, 22, 33)
                if source == "intact" and surface == "measured"
                else (),
            )
            surface_results[f"{source}__{surface}"] = {
                "source": source,
                "surface": surface,
                "panel": {"path": str(panel_path), "sha256": spec["sha256"], "rows": len(rows)},
                "prediction": {
                    "path": str(prediction_path),
                    "sha256": _write_predictions(prediction_path, rows, scores),
                },
                "metrics": metric_payload,
                "semantic_classes": sorted({row["semantic_class"] for row in rows}),
                "novelty_states": {
                    field: dict(sorted(Counter(row[field] for row in rows).items()))
                    for field in (
                        "novelty_exact_pair",
                        "novelty_protein_identity",
                        "novelty_exact_sequence",
                        "novelty_mmseqs_i40",
                        "novelty_mmseqs_i30_cov80",
                    )
                },
            }
    caches = {
        "embedding_dir": str(args.embedding_dir),
        "d007_token_cache_from_environment": True,
    }
    if args.sense_cache is not None:
        caches["historical_optional_sense_cache"] = str(args.sense_cache)
    result = {
        "schema_version": "ppi_shift.e003_external_score_bundle.v1",
        "experiment_id": "E003",
        "model": request["model"],
        "seed": int(request["seed"]),
        "train_environment": request["train_environment"],
        "request": {"path": str(args.request), "sha256": sha256_path(args.request)},
        "external_manifest": {
            "path": str(request["external_manifest"]),
            "sha256": sha256_path(Path(request["external_manifest"])),
        },
        "core_index": {"path": str(args.core_index), "sha256": sha256_path(args.core_index)},
        "checkpoint": checkpoint,
        "caches": caches,
        "surfaces": surface_results,
        "no_external_retraining": True,
        "no_external_selection": True,
        "threshold_source": "corresponding_Gold_validation_max_MCC",
        "wall_time_seconds": time.monotonic() - started,
        "status": "complete",
    }
    (output_dir / "external_run_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_dir / "external_run_manifest.json"),
                "surfaces": len(surface_results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
