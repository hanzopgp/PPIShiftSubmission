"""Minimal frozen-ESM2 pair model used for E001 smoke and baseline runs."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ppi_shift.metrics import compute_binary_metrics, select_threshold
from ppi_shift.negatives import EmbeddingStore

MODEL_ID = "symmetric_frozen_esm2_pair_mlp"
MODEL_VERSION = "ppi_shift.frozen_esm2_pair_mlp.v1"


def _partner_hash(row: Mapping[str, Any]) -> str:
    label = str(row.get("interaction_label", ""))
    if label == "1":
        value = row.get("true_partner_sequence_sha256", "")
    else:
        value = row.get("candidate_sequence_sha256", "")
    if not str(value):
        raise ValueError(f"Panel row lacks the selected partner sequence hash: {row.get('row_id')}")
    return str(value)


def symmetric_pair_features(
    rows: Sequence[Mapping[str, Any]], embeddings: EmbeddingStore
) -> np.ndarray:
    """Encode a pair using sum, absolute difference and product blocks."""

    features = []
    for row in rows:
        anchor_hash = str(row.get("anchor_sequence_sha256", ""))
        partner_hash = _partner_hash(row)
        if anchor_hash not in embeddings.vectors or partner_hash not in embeddings.vectors:
            raise ValueError(f"Embedding coverage missing for row {row.get('row_id')}")
        left = np.asarray(embeddings.vectors[anchor_hash], dtype=np.float64)
        right = np.asarray(embeddings.vectors[partner_hash], dtype=np.float64)
        features.append(np.concatenate((left + right, np.abs(left - right), left * right)))
    if not features:
        raise ValueError("Pair feature extraction requires at least one row")
    return np.asarray(features, dtype=np.float64)


class FrozenESMPairMLP:
    """Small deterministic head over frozen single-protein ESM vectors."""

    def __init__(
        self,
        embeddings: EmbeddingStore,
        *,
        seed: int = 1729,
        hidden_layers: tuple[int, ...] = (64, 32),
        max_iter: int = 20,
    ) -> None:
        self.embeddings = embeddings
        self.seed = seed
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                batch_size="auto",
                max_iter=max_iter,
                random_state=seed,
                early_stopping=False,
                tol=0.0,
            ),
        )
        self._fitted = False
        self._feature_scaler: StandardScaler | None = None
        self.selection_history: list[dict[str, float | int]] = []
        self.selected_epoch: int | None = None
        self.selected_validation_auprc: float | None = None

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> FrozenESMPairMLP:
        labels = np.asarray([int(row["interaction_label"]) for row in rows], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Frozen ESM pair MLP requires both labels")
        self.model.fit(symmetric_pair_features(rows, self.embeddings), labels)
        self._feature_scaler = None
        self._fitted = True
        return self

    def fit_with_validation(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]],
        *,
        epochs: int = 20,
    ) -> FrozenESMPairMLP:
        """Fit fixed epochs and retain the checkpoint selected by validation AUPRC.

        This is the confirmatory path. It deliberately keeps the existing pilot
        ``fit`` method unchanged while making checkpoint selection explicit and
        validation-owned for the full E001v2 runner.
        """

        if epochs < 1:
            raise ValueError("Confirmatory training requires at least one epoch")
        train_labels = np.asarray(
            [int(row["interaction_label"]) for row in train_rows], dtype=np.int64
        )
        validation_labels = np.asarray(
            [int(row["interaction_label"]) for row in validation_rows], dtype=np.int64
        )
        if set(train_labels.tolist()) != {0, 1} or set(validation_labels.tolist()) != {0, 1}:
            raise ValueError("Confirmatory Frozen ESM training and validation require both labels")

        train_features = symmetric_pair_features(train_rows, self.embeddings)
        validation_features = symmetric_pair_features(validation_rows, self.embeddings)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        validation_features = scaler.transform(validation_features)
        classifier = MLPClassifier(
            hidden_layer_sizes=self.model[-1].hidden_layer_sizes,
            activation="relu",
            solver="adam",
            batch_size="auto",
            learning_rate_init=0.001,
            alpha=0.0001,
            max_iter=1,
            random_state=self.seed,
            early_stopping=False,
            tol=0.0,
        )

        best_state: tuple[list[np.ndarray], list[np.ndarray], int, float] | None = None
        history: list[dict[str, float | int]] = []
        classes = np.asarray([0, 1], dtype=np.int64)
        for epoch in range(1, epochs + 1):
            classifier.partial_fit(train_features, train_labels, classes=classes)
            validation_scores = classifier.predict_proba(validation_features)[:, 1]
            validation_auprc = float(average_precision_score(validation_labels, validation_scores))
            history.append({"epoch": epoch, "validation_auprc": validation_auprc})
            if best_state is None or validation_auprc > best_state[3] + 1e-15:
                best_state = (
                    [array.copy() for array in classifier.coefs_],
                    [array.copy() for array in classifier.intercepts_],
                    int(classifier.n_iter_),
                    validation_auprc,
                )

        assert best_state is not None
        classifier.coefs_ = [array.copy() for array in best_state[0]]
        classifier.intercepts_ = [array.copy() for array in best_state[1]]
        classifier.n_iter_ = best_state[2]
        self.model = classifier
        self._feature_scaler = scaler
        self.selection_history = history
        self.selected_epoch = next(
            int(row["epoch"]) for row in history if float(row["validation_auprc"]) == best_state[3]
        )
        self.selected_validation_auprc = best_state[3]
        self._fitted = True
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Frozen ESM pair MLP must be fit before prediction")
        features = symmetric_pair_features(rows, self.embeddings)
        if self._feature_scaler is not None:
            features = self._feature_scaler.transform(features)
        return np.asarray(
            self.model.predict_proba(features)[:, 1],
            dtype=np.float64,
        )

    def swap_invariant(self, rows: Sequence[Mapping[str, Any]]) -> bool:
        swapped = []
        for row in rows:
            copy = dict(row)
            anchor_hash = row.get("anchor_sequence_sha256", "")
            if str(row.get("interaction_label")) == "1":
                partner_hash = row.get("true_partner_sequence_sha256", "")
                copy["anchor_sequence_sha256"] = partner_hash
                copy["true_partner_sequence_sha256"] = anchor_hash
            else:
                partner_hash = row.get("candidate_sequence_sha256", "")
                copy["anchor_sequence_sha256"] = partner_hash
                copy["candidate_sequence_sha256"] = anchor_hash
            copy["seq1"], copy["seq2"] = row.get("seq2", ""), row.get("seq1", "")
            swapped.append(copy)
        return bool(np.allclose(self.predict(rows), self.predict(swapped), rtol=0.0, atol=1e-12))


def smoke_run(
    rows: Sequence[Mapping[str, Any]],
    embeddings: EmbeddingStore,
    *,
    seed: int = 1729,
    max_iter: int = 2,
) -> dict[str, Any]:
    """Fit/predict a bounded smoke model and return runtime/provenance evidence."""

    started = time.monotonic()
    model = FrozenESMPairMLP(embeddings, seed=seed, max_iter=max_iter).fit(rows)
    scores = model.predict(rows)
    labels = [int(row["interaction_label"]) for row in rows]
    threshold = select_threshold(labels, scores)
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "status": "pass",
        "rows": len(rows),
        "embedding_metadata": dict(embeddings.metadata),
        "metrics": compute_binary_metrics(
            labels,
            scores,
            threshold=threshold["threshold"],
            threshold_source="smoke_in_sample_only",
        ),
        "swap_invariant": model.swap_invariant(rows),
        "max_iter": max_iter,
        "wall_time_seconds": time.monotonic() - started,
        "gpu": "not_used_sklearn_cpu_head",
        "vram_gb": 0.0,
        "selection_status": "smoke_only; no checkpoint selected",
    }
