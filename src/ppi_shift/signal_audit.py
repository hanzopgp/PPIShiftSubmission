"""Frozen, label-isolated diagnostic controls for E003.

The functions in this module deliberately keep the diagnostic estimands apart
from the Core-4 score registry.  They accept ordinary row mappings so that
they can be used with the immutable panel CSVs and with small contract
fixtures in tests without importing a model runtime.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ppi_shift.data import sequence_sha256
from ppi_shift.metrics import compute_binary_metrics
from ppi_shift.negatives import AMINO_ACIDS, EmbeddingStore

SIGNAL_AUDIT_SCHEMA_VERSION = "ppi_shift.e003_signal_audit.v1"
BROKEN_PAIR_SCHEMA_VERSION = "ppi_shift.e003_broken_pair.v1"
DEFAULT_SEEDS = (11, 22, 33)


def sha256_path(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV or gzip-compressed CSV as string-valued row mappings."""

    input_path = Path(path)
    opener = gzip.open if input_path.name.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic CSV rows, retaining the first row's column order."""

    if not rows:
        raise ValueError("Cannot write an empty row table")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    def write_csv(handle: Any) -> None:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)

    if output.name.endswith(".gz"):
        with output.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    write_csv(handle)
    else:
        with output.open("w", encoding="utf-8", newline="") as handle:
            write_csv(handle)


def row_label(row: Mapping[str, Any]) -> int:
    """Parse the frozen binary interaction label."""

    value = int(str(row.get("interaction_label", "")))
    if value not in (0, 1):
        raise ValueError(f"Expected a binary interaction label, received {value!r}")
    return value


def row_group_id(row: Mapping[str, Any]) -> str:
    """Return the protocol group identifier for split isolation."""

    group = str(row.get("positive_row_id", "")).strip()
    if group:
        return group
    return canonical_pair_key(
        str(row.get("name1", "")),
        str(row.get("name2", "")),
    )


def canonical_pair_key(left: str, right: str) -> str:
    """Return an orientation-independent pair key."""

    return "||".join(sorted((str(left), str(right))))


def _sequence_hash(sequence: str) -> str:
    return sequence_sha256(str(sequence))


def _stable_order(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()


def grouped_split_indices(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by a stable hash of positive groups, never by individual rows."""

    if not rows:
        raise ValueError("Grouped split requires at least one row")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    groups = sorted({row_group_id(row) for row in rows})
    if len(groups) < 2:
        raise ValueError("Grouped split requires at least two groups")
    ordered = sorted(groups, key=lambda group: _stable_order(seed, group))
    count = min(max(1, round(len(groups) * holdout_fraction)), len(groups) - 1)
    holdout_groups = set(ordered[:count])
    holdout = np.asarray(
        [index for index, row in enumerate(rows) if row_group_id(row) in holdout_groups],
        dtype=np.int64,
    )
    holdout_set = set(holdout.tolist())
    fit = np.asarray(
        [index for index in range(len(rows)) if index not in holdout_set], dtype=np.int64
    )
    if not len(fit) or not len(holdout):
        raise ValueError("Grouped split produced an empty partition")
    if {row_group_id(rows[index]) for index in fit} & {
        row_group_id(rows[index]) for index in holdout
    }:
        raise AssertionError("Grouped split leaked a group")
    return fit, holdout


def _endpoint_vector(sequence: str) -> tuple[np.ndarray, tuple[str, ...]]:
    normalized = str(sequence).strip().upper()
    length = max(1, len(normalized))
    composition = np.asarray(
        [normalized.count(amino) / length for amino in AMINO_ACIDS], dtype=np.float64
    )
    values = np.concatenate(
        (np.asarray((math.log1p(len(normalized)),), dtype=np.float64), composition)
    )
    names = ("log_length",) + tuple(f"composition_{amino}" for amino in AMINO_ACIDS)
    return values, names


def _sort_feature_columns(
    values: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the frozen lexicographic feature-column order."""

    order = sorted(range(len(names)), key=lambda index: str(names[index]))
    return values[..., order], tuple(str(names[index]) for index in order)


def sequence_endpoint_features(sequence: str) -> np.ndarray:
    """Return the public single-protein sequence-only feature vector."""

    return _endpoint_vector(sequence)[0]


def sequence_endpoint_feature_names() -> tuple[str, ...]:
    return _endpoint_vector("A")[1]


def build_positive_degree_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Build endpoint degree from positive training rows only."""

    neighbors: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row_label(row) != 1:
            continue
        left = str(row.get("name1", ""))
        right = str(row.get("name2", ""))
        if left and right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    return {protein: len(values) for protein, values in neighbors.items()}


def pair_easy_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    degree_table: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract the fixed sequence-plus-degree pair feature family."""

    degree_table = degree_table or {}
    output: list[np.ndarray] = []
    names = (
        "left_log_length",
        "right_log_length",
        "absolute_log_length_difference",
        "minimum_log_length",
        "maximum_log_length",
        *tuple(f"left_composition_{amino}" for amino in AMINO_ACIDS),
        *tuple(f"right_composition_{amino}" for amino in AMINO_ACIDS),
        *tuple(f"absolute_composition_difference_{amino}" for amino in AMINO_ACIDS),
        "left_log_degree",
        "right_log_degree",
        "absolute_log_degree_difference",
        "log_degree_product",
    )
    for row in rows:
        left, _ = _endpoint_vector(str(row.get("seq1", "")))
        right, _ = _endpoint_vector(str(row.get("seq2", "")))
        left_degree = math.log1p(int(degree_table.get(str(row.get("name1", "")), 0)))
        right_degree = math.log1p(int(degree_table.get(str(row.get("name2", "")), 0)))
        output.append(
            np.concatenate(
                (
                    left[:1],
                    right[:1],
                    np.asarray((abs(left[0] - right[0]),), dtype=np.float64),
                    np.asarray((min(left[0], right[0]), max(left[0], right[0])), dtype=np.float64),
                    left[1:],
                    right[1:],
                    np.abs(left[1:] - right[1:]),
                    np.asarray(
                        (
                            left_degree,
                            right_degree,
                            abs(left_degree - right_degree),
                            left_degree * right_degree,
                        ),
                        dtype=np.float64,
                    ),
                )
            )
        )
    if not output:
        raise ValueError("Easy-feature extraction requires at least one row")
    return _sort_feature_columns(np.asarray(output, dtype=np.float64), names)


def protein_sequence_features(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], list[tuple[int, int]]]:
    """Expand pair rows into endpoint-only observations.

    The returned provenance tuples identify the source row and endpoint side;
    no partner or pair-derived value is included in the feature matrix.
    """

    values: list[np.ndarray] = []
    provenance: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        for side, field in enumerate(("seq1", "seq2")):
            values.append(sequence_endpoint_features(str(row.get(field, ""))))
            provenance.append((row_index, side))
    if not values:
        raise ValueError("Protein-only extraction requires at least one row")
    names = tuple(f"single_protein_{name}" for name in sequence_endpoint_feature_names())
    sorted_values, sorted_names = _sort_feature_columns(np.asarray(values, dtype=np.float64), names)
    return sorted_values, sorted_names, provenance


def embedding_contract_mismatches(
    store: EmbeddingStore,
    expected: Mapping[str, Any],
) -> list[str]:
    """Return exact representation-contract mismatches for a Bernett control."""

    metadata = store.metadata
    expected_fields = {
        "model_name": str(expected.get("model", "")),
        "model_revision": str(expected.get("revision", "")),
        "embedding_layer": int(expected.get("layer", -1)),
        "pooling": str(expected.get("pooling", "")),
        "embedding_dimension": int(expected.get("dimension", -1)),
    }
    return [
        f"{field}: expected {value!r}, found {metadata.get(field)!r}"
        for field, value in expected_fields.items()
        if metadata.get(field) != value
    ]


def pair_embedding_features(
    rows: Sequence[Mapping[str, Any]],
    embeddings: EmbeddingStore,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the fixed symmetric ``sum`` and ``absolute_difference`` vector."""

    values: list[np.ndarray] = []
    dimension = int(embeddings.metadata["embedding_dimension"])
    for row in rows:
        left_hash = _sequence_hash(str(row.get("seq1", "")))
        right_hash = _sequence_hash(str(row.get("seq2", "")))
        if left_hash not in embeddings.vectors or right_hash not in embeddings.vectors:
            raise ValueError(f"Embedding coverage missing for row {row.get('row_id', '')}")
        left = np.asarray(embeddings.vectors[left_hash], dtype=np.float64)
        right = np.asarray(embeddings.vectors[right_hash], dtype=np.float64)
        values.append(np.concatenate((left + right, np.abs(left - right))))
    if not values:
        raise ValueError("Embedding pair features require at least one row")
    names = tuple(
        [f"esm_sum_{index}" for index in range(dimension)]
        + [f"esm_absolute_difference_{index}" for index in range(dimension)]
    )
    return _sort_feature_columns(np.asarray(values, dtype=np.float64), names)


def _fit_classifier(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[StandardScaler, LogisticRegression]:
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Classifier features and labels are not aligned")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Classifier fit requires both binary labels")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    classifier.fit(scaled, labels)
    return scaler, classifier


def _select_threshold_fast(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Select the exact fixed-rule MCC threshold without quadratic rescoring."""

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if len(labels_array) != len(scores_array) or set(labels_array.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires aligned binary labels")
    positives = int(labels_array.sum())
    negatives = int(len(labels_array) - positives)
    order = np.argsort(scores_array, kind="mergesort")
    ordered_scores = scores_array[order]
    ordered_labels = labels_array[order]
    suffix_positive = np.cumsum(ordered_labels[::-1], dtype=np.int64)[::-1]
    candidates = np.unique(np.concatenate((np.asarray((0.0, 1.0)), scores_array)))
    starts = np.searchsorted(ordered_scores, candidates, side="left")
    valid_starts = starts < len(ordered_scores)
    safe_starts = np.minimum(starts, len(ordered_scores) - 1)
    true_positive = np.where(valid_starts, suffix_positive[safe_starts], 0)
    false_positive = (len(ordered_scores) - starts) - true_positive
    false_negative = positives - true_positive
    true_negative = negatives - false_positive
    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)
    balanced_accuracy = 0.5 * (true_positive / positives + true_negative / negatives)
    best = max(
        range(len(candidates)),
        key=lambda index: (
            float(mcc[index]),
            float(balanced_accuracy[index]),
            -float(candidates[index]),
        ),
    )
    return {
        "method": "validation_max_mcc",
        "threshold": float(candidates[best]),
        "mcc": float(mcc[best]),
        "balanced_accuracy": float(balanced_accuracy[best]),
    }


def _classifier_scores(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    return np.asarray(classifier.predict_proba(scaler.transform(features))[:, 1], dtype=np.float64)


def _coefficient_map(
    classifier: LogisticRegression,
    feature_names: Sequence[str],
) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(feature_names, classifier.coef_[0], strict=True)
    }


def fit_pair_classifier_audit(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    feature_family: str,
    seed: int,
    feature_builder: Any,
    split_feature_builder: Any | None = None,
    full_fit: tuple[StandardScaler, LogisticRegression] | None = None,
) -> dict[str, Any]:
    """Fit a fixed classifier from train rows and evaluate an isolated test panel."""

    if not train_rows or not test_rows:
        raise ValueError("Pair audit requires non-empty train and test panels")
    fit_indices, holdout_indices = grouped_split_indices(train_rows, seed)
    fit_rows = [train_rows[index] for index in fit_indices]
    holdout_rows = [train_rows[index] for index in holdout_indices]
    train_features, feature_names = feature_builder(train_rows)
    test_features, test_feature_names = feature_builder(test_rows)
    if tuple(feature_names) != tuple(test_feature_names):
        raise ValueError("Train and test feature names differ")
    labels = np.asarray([row_label(row) for row in train_rows], dtype=np.int64)
    test_labels = np.asarray([row_label(row) for row in test_rows], dtype=np.int64)
    if split_feature_builder is None:
        fit_features = train_features[fit_indices]
        holdout_features = train_features[holdout_indices]
        split_feature_names = feature_names
    else:
        fit_features, split_feature_names = split_feature_builder(fit_rows, reference_rows=fit_rows)
        holdout_features, holdout_feature_names = split_feature_builder(
            holdout_rows, reference_rows=fit_rows
        )
        if tuple(split_feature_names) != tuple(holdout_feature_names):
            raise ValueError("Split feature names differ")
    if tuple(split_feature_names) != tuple(feature_names):
        raise ValueError("Split and full feature names differ")
    holdout_scaler, holdout_classifier = _fit_classifier(fit_features, labels[fit_indices], seed)
    fit_scores = _classifier_scores(
        holdout_scaler,
        holdout_classifier,
        fit_features,
    )
    holdout_scores = _classifier_scores(
        holdout_scaler,
        holdout_classifier,
        holdout_features,
    )
    threshold = {
        **_select_threshold_fast(labels[fit_indices], fit_scores),
        "method": "training_fit_max_mcc",
    }
    if full_fit is None:
        full_scaler, full_classifier = _fit_classifier(train_features, labels, seed)
    else:
        full_scaler, full_classifier = full_fit
    test_scores = _classifier_scores(full_scaler, full_classifier, test_features)
    holdout_metrics = compute_binary_metrics(
        labels[holdout_indices],
        holdout_scores,
        threshold=float(threshold["threshold"]),
        threshold_source="training_fit_max_mcc",
    )
    test_metrics = compute_binary_metrics(
        test_labels,
        test_scores,
        threshold=float(threshold["threshold"]),
        threshold_source="training_fit_max_mcc",
    )
    return {
        "schema_version": SIGNAL_AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "control": feature_family,
        "seed": int(seed),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "split": {
            "method": "sha256_group_hash",
            "group_key": "positive_row_id",
            "fit_rows": int(len(fit_indices)),
            "holdout_rows": int(len(holdout_indices)),
            "fit_groups": len({row_group_id(train_rows[index]) for index in fit_indices}),
            "holdout_groups": len({row_group_id(train_rows[index]) for index in holdout_indices}),
        },
        "fit_source": "train_panel_only",
        "test_labels_used_for_fit": False,
        "threshold": threshold,
        "holdout_metrics": holdout_metrics,
        "test_metrics": test_metrics,
        "coefficients": _coefficient_map(full_classifier, feature_names),
        "test_scores": test_scores.tolist(),
        "test_row_ids": [str(row.get("row_id", "")) for row in test_rows],
    }


def _expand_endpoint_labels(
    rows: Sequence[Mapping[str, Any]], row_indices: Iterable[int]
) -> tuple[np.ndarray, list[int]]:
    indices = list(row_indices)
    labels: list[int] = []
    for index in indices:
        labels.extend((row_label(rows[index]), row_label(rows[index])))
    return np.asarray(labels, dtype=np.int64), indices


def fit_protein_sequence_audit(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit the endpoint-only sequence baseline with no pair-derived feature."""

    fit_indices, holdout_indices = grouped_split_indices(train_rows, seed)
    train_features, feature_names, _ = protein_sequence_features(train_rows)
    test_features, test_names, _ = protein_sequence_features(test_rows)
    if feature_names != test_names:
        raise ValueError("Protein-only train and test feature names differ")
    train_labels, _ = _expand_endpoint_labels(train_rows, range(len(train_rows)))
    fit_endpoint_indices = np.asarray(
        [2 * index + side for index in fit_indices for side in (0, 1)], dtype=np.int64
    )
    holdout_endpoint_indices = np.asarray(
        [2 * index + side for index in holdout_indices for side in (0, 1)], dtype=np.int64
    )
    holdout_scaler, holdout_classifier = _fit_classifier(
        train_features[fit_endpoint_indices], train_labels[fit_endpoint_indices], seed
    )
    fit_endpoint_scores = _classifier_scores(
        holdout_scaler, holdout_classifier, train_features[fit_endpoint_indices]
    )
    holdout_endpoint_scores = _classifier_scores(
        holdout_scaler, holdout_classifier, train_features[holdout_endpoint_indices]
    )
    threshold = {
        **_select_threshold_fast(train_labels[fit_endpoint_indices], fit_endpoint_scores),
        "method": "training_fit_max_mcc",
    }
    full_scaler, full_classifier = _fit_classifier(train_features, train_labels, seed)
    endpoint_test_scores = _classifier_scores(full_scaler, full_classifier, test_features)
    pair_test_scores = endpoint_test_scores.reshape(-1, 2).mean(axis=1)
    test_labels = np.asarray([row_label(row) for row in test_rows], dtype=np.int64)
    holdout_pair_scores = holdout_endpoint_scores.reshape(-1, 2).mean(axis=1)
    holdout_pair_labels = np.asarray(
        [row_label(train_rows[index]) for index in holdout_indices], dtype=np.int64
    )
    return {
        "schema_version": SIGNAL_AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "control": "single_protein_sequence_logistic",
        "seed": int(seed),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "forbidden_features": [
            "partner_identity",
            "pair_hash",
            "endpoint_product",
            "endpoint_difference",
            "embedding",
            "core4_score",
        ],
        "split": {
            "method": "sha256_group_hash",
            "group_key": "positive_row_id",
            "fit_rows": int(len(fit_indices)),
            "holdout_rows": int(len(holdout_indices)),
        },
        "fit_source": "train_panel_only",
        "test_labels_used_for_fit": False,
        "threshold": threshold,
        "holdout_metrics": compute_binary_metrics(
            holdout_pair_labels,
            holdout_pair_scores,
            threshold=float(threshold["threshold"]),
            threshold_source="training_fit_max_mcc",
        ),
        "test_metrics": compute_binary_metrics(
            test_labels,
            pair_test_scores,
            threshold=float(threshold["threshold"]),
            threshold_source="training_fit_max_mcc",
        ),
        "coefficients": _coefficient_map(full_classifier, feature_names),
        "test_scores": pair_test_scores.tolist(),
        "test_row_ids": [str(row.get("row_id", "")) for row in test_rows],
    }


def _endpoint_propensities(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positives: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row in rows:
        label = row_label(row)
        for field in ("name1", "name2"):
            endpoint = str(row.get(field, ""))
            if not endpoint:
                continue
            totals[endpoint] += 1
            positives[endpoint] += label
    return {endpoint: (positives[endpoint] + 1.0) / (totals[endpoint] + 2.0) for endpoint in totals}


def fit_empirical_endpoint_audit(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Score pairs from smoothed endpoint propensities estimated on training rows."""

    fit_indices, holdout_indices = grouped_split_indices(train_rows, seed)
    fit_rows = [train_rows[index] for index in fit_indices]
    holdout_rows = [train_rows[index] for index in holdout_indices]
    propensity = _endpoint_propensities(fit_rows)

    def score(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                np.mean(
                    [
                        propensity.get(str(row.get("name1", "")), 0.5),
                        propensity.get(str(row.get("name2", "")), 0.5),
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

    fit_scores = score(fit_rows)
    holdout_scores = score(holdout_rows)
    holdout_labels = np.asarray([row_label(row) for row in holdout_rows], dtype=np.int64)
    fit_labels = np.asarray([row_label(row) for row in fit_rows], dtype=np.int64)
    threshold = {
        **_select_threshold_fast(fit_labels, fit_scores),
        "method": "training_fit_max_mcc",
    }
    full_propensity = _endpoint_propensities(train_rows)
    test_scores = np.asarray(
        [
            np.mean(
                [
                    full_propensity.get(str(row.get("name1", "")), 0.5),
                    full_propensity.get(str(row.get("name2", "")), 0.5),
                ]
            )
            for row in test_rows
        ],
        dtype=np.float64,
    )
    test_labels = np.asarray([row_label(row) for row in test_rows], dtype=np.int64)
    return {
        "schema_version": SIGNAL_AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "control": "empirical_endpoint_propensity",
        "seed": int(seed),
        "feature_names": [
            "left_endpoint_smoothed_positive_rate",
            "right_endpoint_smoothed_positive_rate",
        ],
        "feature_count": 2,
        "smoothing": {"alpha": 1.0, "beta": 1.0},
        "split": {
            "method": "sha256_group_hash",
            "group_key": "positive_row_id",
            "fit_rows": len(fit_rows),
            "holdout_rows": len(holdout_rows),
        },
        "fit_source": "train_panel_only",
        "test_labels_used_for_fit": False,
        "threshold": threshold,
        "holdout_metrics": compute_binary_metrics(
            holdout_labels,
            holdout_scores,
            threshold=float(threshold["threshold"]),
            threshold_source="training_fit_max_mcc",
        ),
        "test_metrics": compute_binary_metrics(
            test_labels,
            test_scores,
            threshold=float(threshold["threshold"]),
            threshold_source="training_fit_max_mcc",
        ),
        "test_scores": test_scores.tolist(),
        "test_row_ids": [str(row.get("row_id", "")) for row in test_rows],
    }


def _right_endpoint(row: Mapping[str, Any]) -> tuple[str, str, str]:
    name = str(row.get("name2", ""))
    sequence = str(row.get("seq2", ""))
    return name, sequence, _sequence_hash(sequence)


def _pair_sequence_key(left_sequence: str, right_sequence: str) -> str:
    return canonical_pair_key(_sequence_hash(left_sequence), _sequence_hash(right_sequence))


def _known_positive_rights(known_positive_pairs: set[str]) -> dict[str, set[str]]:
    """Index unordered known-positive sequence pairs by either endpoint."""

    indexed: dict[str, set[str]] = defaultdict(set)
    for pair in known_positive_pairs:
        left, separator, right = str(pair).partition("||")
        if not separator or not left or not right:
            raise ValueError(f"Malformed known positive pair key: {pair!r}")
        indexed[left].add(right)
        indexed[right].add(left)
    return indexed


def _cyclic_shift_objective_counts(
    rows: Sequence[Mapping[str, Any]], ordered: Sequence[int], known_positive_pairs: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Count unchanged and known-positive collisions for every cyclic shift.

    The implementation is sparse in repeated endpoint/known-pair categories,
    but computes the same objective as the literal all-shifts definition.
    """

    size = len(ordered)
    unchanged = np.zeros(size, dtype=np.int64)
    collisions = np.zeros(size, dtype=np.int64)
    right_endpoints = [_right_endpoint(rows[index]) for index in ordered]

    positions_by_endpoint: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    positions_by_hash: dict[str, list[int]] = defaultdict(list)
    for position, endpoint in enumerate(right_endpoints):
        positions_by_endpoint[endpoint].append(position)
        positions_by_hash[endpoint[2]].append(position)

    for positions in positions_by_endpoint.values():
        for source_position in positions:
            for target_position in positions:
                shift = (target_position - source_position) % size
                if shift:
                    unchanged[shift] += 1

    known_by_endpoint = _known_positive_rights(known_positive_pairs)
    for source_position, row_index in enumerate(ordered):
        left_hash = _sequence_hash(str(rows[row_index].get("seq1", "")))
        for right_hash in sorted(known_by_endpoint.get(left_hash, ())):
            for target_position in positions_by_hash.get(right_hash, ()):
                shift = (target_position - source_position) % size
                if shift:
                    collisions[shift] += 1
    return unchanged, collisions


def build_broken_pair_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_positive_pairs: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct the deterministic label-stratified broken-pair diagnostic."""

    if not rows:
        raise ValueError("Broken-pair construction requires rows")
    known = set(
        known_positive_pairs
        or {
            _pair_sequence_key(str(row.get("seq1", "")), str(row.get("seq2", "")))
            for row in rows
            if row_label(row) == 1
        }
    )
    output = [dict(row) for row in rows]
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[row_label(row)].append(index)
    selected_shifts: dict[str, int] = {}
    row_audits: list[dict[str, Any]] = []
    for label, indices in sorted(by_label.items()):
        ordered = sorted(indices, key=lambda index: str(rows[index].get("row_id", "")))
        if len(ordered) < 2:
            raise ValueError(f"No valid derangement exists for label {label} with one row")
        rights = [_right_endpoint(rows[index]) for index in ordered]
        unchanged_counts, collision_counts = _cyclic_shift_objective_counts(rows, ordered, known)
        shift = min(
            range(1, len(ordered)),
            key=lambda candidate: (
                int(unchanged_counts[candidate]),
                int(collision_counts[candidate]),
                candidate,
            ),
        )
        unchanged = []
        collisions = []
        selected_shifts[str(label)] = shift
        for position, index in enumerate(ordered):
            source = rows[index]
            right_name, right_sequence, right_hash = rights[(position + shift) % len(rights)]
            updated = output[index]
            original_right = _right_endpoint(source)
            unchanged.append(original_right == (right_name, right_sequence, right_hash))
            collisions.append(
                _pair_sequence_key(str(source.get("seq1", "")), right_sequence) in known
            )
            updated["name2"] = right_name
            updated["seq2"] = right_sequence
            updated["broken_candidate_partner_id"] = right_name
            updated["broken_candidate_sequence_sha256"] = right_hash
            updated["broken_original_right_endpoint"] = original_right[0]
            updated["broken_original_right_sequence_sha256"] = original_right[2]
            if label == 1:
                updated["true_partner_id"] = str(
                    source.get("true_partner_id", "") or original_right[0]
                )
                updated["true_partner_sequence_sha256"] = str(
                    source.get("true_partner_sequence_sha256", "") or original_right[2]
                )
            else:
                updated["candidate_partner_id"] = right_name
                updated["candidate_sequence_sha256"] = right_hash
            updated["broken_pair"] = "true"
            updated["broken_source_row_id"] = str(source.get("row_id", ""))
            updated["broken_permutation_shift"] = str(shift)
            updated["broken_known_positive_collision"] = str(collisions[position]).lower()
            updated["broken_pair_schema_version"] = BROKEN_PAIR_SCHEMA_VERSION
            row_audits.append(
                {
                    "row_id": str(source.get("row_id", "")),
                    "label": label,
                    "shift": shift,
                    "unchanged": unchanged[position],
                    "known_positive_collision": collisions[position],
                }
            )
    original_labels = Counter(row_label(row) for row in rows)
    broken_labels = Counter(row_label(row) for row in output)
    original_rights = Counter(_right_endpoint(row)[2] for row in rows)
    broken_rights = Counter(_right_endpoint(row)[2] for row in output)
    if original_labels != broken_labels:
        raise AssertionError("Broken-pair construction changed label counts")
    if original_rights != broken_rights:
        raise AssertionError("Broken-pair construction changed endpoint marginals")
    audit = {
        "schema_version": BROKEN_PAIR_SCHEMA_VERSION,
        "status": "PASS",
        "rows": len(output),
        "label_counts": dict(sorted(original_labels.items())),
        "selected_shifts": selected_shifts,
        "unchanged_rows": sum(int(item["unchanged"]) for item in row_audits),
        "known_positive_collisions": sum(
            int(item["known_positive_collision"]) for item in row_audits
        ),
        "endpoint_marginals_preserved": True,
        "label_marginals_preserved": True,
        "row_audit": row_audits,
    }
    return output, audit
