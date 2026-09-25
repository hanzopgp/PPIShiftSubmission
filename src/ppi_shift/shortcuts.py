"""Nuisance-only diagnostics for separating negative environments.

These classifiers predict environment identity among synthetic negative rows. They are
diagnostics, not PPI predictors, and use anchor-grouped train/test partitions so the same
positive anchor cannot occur on both sides of the audit split.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ppi_shift.negatives import AMINO_ACIDS, EmbeddingStore, canonical_json

SHORTCUTS = (
    "prevalence",
    "length",
    "composition",
    "length_composition",
    "train_positive_degree",
    "protein_lookup",
    "simple_esm_pair",
    "broken_partner",
)
ENVIRONMENT_FEATURE_SETS = (
    "length",
    "composition",
    "length_composition",
    "degree",
    "length_composition_degree",
    "esm_distance",
    "all_nuisance",
)


def _sequence_vector(sequence: str) -> np.ndarray:
    normalized = str(sequence).strip().upper()
    counts = np.asarray(
        [normalized.count(amino) / max(1, len(normalized)) for amino in AMINO_ACIDS],
        dtype=np.float64,
    )
    return np.concatenate(([math.log1p(len(normalized))], counts))


def _row_sequence_vectors(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return _sequence_vector(str(row.get("seq1", ""))), _sequence_vector(str(row.get("seq2", "")))


def _pair_sequence_features(row: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    left, right = _row_sequence_vectors(row)
    names = ("partner_log_length",) + tuple(f"partner_composition_{amino}" for amino in AMINO_ACIDS)
    return np.concatenate((right, np.abs(left - right))), names + tuple(
        f"anchor_partner_absdiff_{name}" for name in names
    )


def _degree_features(
    row: Mapping[str, Any], degree_table: Mapping[str, int]
) -> tuple[np.ndarray, tuple[str, ...]]:
    anchor = math.log1p(int(degree_table.get(str(row.get("anchor_id", "")), 0)))
    candidate = math.log1p(int(degree_table.get(str(row.get("candidate_partner_id", "")), 0)))
    return np.asarray((anchor, candidate, abs(anchor - candidate), anchor * candidate)), (
        "anchor_log_degree",
        "candidate_log_degree",
        "degree_absdiff",
        "degree_product",
    )


def _esm_features(
    row: Mapping[str, Any], embeddings: EmbeddingStore
) -> tuple[np.ndarray, tuple[str, ...]]:
    left_hash = str(row.get("anchor_sequence_sha256", ""))
    right_hash = str(row.get("candidate_sequence_sha256", ""))
    if left_hash not in embeddings.vectors or right_hash not in embeddings.vectors:
        raise ValueError(f"Embedding feature coverage missing for row {row.get('row_id')}")
    left = np.asarray(embeddings.vectors[left_hash], dtype=np.float64)
    right = np.asarray(embeddings.vectors[right_hash], dtype=np.float64)
    difference = float(np.linalg.norm(left - right))
    cosine = float(np.dot(left, right) / max(1e-12, np.linalg.norm(left) * np.linalg.norm(right)))
    return np.asarray((difference, cosine)), ("esm_l2_distance", "esm_cosine")


def environment_features(
    rows: Sequence[Mapping[str, Any]],
    feature_set: str,
    *,
    degree_table: Mapping[str, int] | None = None,
    embeddings: EmbeddingStore | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract a predeclared nuisance feature matrix for environment classification."""

    if feature_set not in ENVIRONMENT_FEATURE_SETS:
        raise ValueError(f"Unknown environment feature set: {feature_set!r}")
    feature_rows: list[np.ndarray] = []
    feature_names: tuple[str, ...] = ()
    for row in rows:
        sequence, sequence_names = _pair_sequence_features(row)
        degree, degree_names = _degree_features(row, degree_table or {})
        if feature_set == "length":
            values, names = sequence[:1], sequence_names[:1]
        elif feature_set == "composition":
            values, names = sequence[1:21], sequence_names[1:21]
        elif feature_set == "length_composition":
            values, names = sequence, sequence_names
        elif feature_set == "degree":
            values, names = degree, degree_names
        elif feature_set == "length_composition_degree":
            values, names = np.concatenate((sequence, degree)), sequence_names + degree_names
        else:
            if embeddings is None:
                raise ValueError(f"{feature_set} requires a frozen single-protein embedding store")
            esm, esm_names = _esm_features(row, embeddings)
            if feature_set == "esm_distance":
                values, names = esm, esm_names
            else:
                values, names = (
                    np.concatenate((sequence, degree, esm)),
                    (sequence_names + degree_names + esm_names),
                )
        feature_rows.append(values)
        feature_names = names
    if not feature_rows:
        raise ValueError("Environment audit requires at least one negative row")
    return np.asarray(feature_rows, dtype=np.float64), feature_names


def _group_split(
    rows: Sequence[Mapping[str, Any]], seed: int, test_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({str(row.get("positive_row_id", row.get("row_id", ""))) for row in rows})
    if len(groups) < 2:
        raise ValueError("Environment audit requires at least two anchor groups")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(),
    )
    test_count = min(max(1, round(len(groups) * test_fraction)), len(groups) - 1)
    test_groups = set(ordered[:test_count])
    test_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if str(row.get("positive_row_id", "")) in test_groups
        ],
        dtype=np.int64,
    )
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if index not in set(test_indices.tolist())],
        dtype=np.int64,
    )
    return train_indices, test_indices


def fit_environment_discriminator(
    rows: Sequence[Mapping[str, Any]],
    feature_set: str,
    *,
    seed: int = 1729,
    test_fraction: float = 0.25,
    degree_table: Mapping[str, int] | None = None,
    embeddings: EmbeddingStore | None = None,
) -> dict[str, Any]:
    """Fit a grouped multinomial logistic discriminator and return audit metadata."""

    negative_rows = [row for row in rows if str(row.get("interaction_label", "0")) == "0"]
    labels = np.asarray([str(row.get("negative_environment", "")) for row in negative_rows])
    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2:
        raise ValueError("Environment audit requires at least two negative environments")
    features, feature_names = environment_features(
        negative_rows,
        feature_set,
        degree_table=degree_table,
        embeddings=embeddings,
    )
    train_indices, test_indices = _group_split(negative_rows, seed, test_fraction)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
    )
    estimator.fit(features[train_indices], labels[train_indices])
    predicted = estimator.predict(features[test_indices])
    probabilities = estimator.predict_proba(features[test_indices])
    classes = [str(value) for value in estimator.classes_]
    test_labels = labels[test_indices]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(test_labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)),
        "macro_f1": float(f1_score(test_labels, predicted, average="macro")),
    }
    if len(classes) > 2 and set(test_labels.tolist()) == set(classes):
        metrics["macro_ovr_auroc"] = float(
            roc_auc_score(test_labels, probabilities, multi_class="ovr", labels=classes)
        )
    matrix = confusion_matrix(test_labels, predicted, labels=classes).tolist()
    classifier = estimator[-1]
    coefficients = {
        class_name: {
            feature_name: float(value)
            for feature_name, value in zip(feature_names, coefficient, strict=True)
        }
        for class_name, coefficient in zip(classes, classifier.coef_, strict=True)
    }
    return {
        "schema_version": "ppi_shift.environment_discriminator.v1",
        "feature_set": feature_set,
        "feature_names": list(feature_names),
        "seed": seed,
        "group_split": "positive_row_id_hash_partition",
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_groups": int(len({str(negative_rows[i]["positive_row_id"]) for i in train_indices})),
        "test_groups": int(len({str(negative_rows[i]["positive_row_id"]) for i in test_indices})),
        "environment_counts": dict(sorted(Counter(labels.tolist()).items())),
        "test_environment_counts": dict(sorted(Counter(test_labels.tolist()).items())),
        "classes": classes,
        "metrics": metrics,
        "confusion_matrix": {"labels": classes, "values": matrix},
        "standardized_coefficients": coefficients,
        "interpretation_boundary": (
            "Nuisance separability is descriptive evidence about these generated surfaces; "
            "it does not establish that synthetic rows are true biological negatives."
        ),
    }


def audit_environment_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 1729,
    degree_table: Mapping[str, int] | None = None,
    embeddings: EmbeddingStore | None = None,
) -> dict[str, Any]:
    results = {
        feature_set: fit_environment_discriminator(
            rows,
            feature_set,
            seed=seed,
            degree_table=degree_table,
            embeddings=embeddings,
        )
        for feature_set in ENVIRONMENT_FEATURE_SETS
        if feature_set != "esm_distance" or embeddings is not None
    }
    return {
        "schema_version": "ppi_shift.environment_shortcut_audit.v1",
        "seed": seed,
        "feature_sets": results,
        "environment_counts": dict(
            sorted(
                Counter(
                    str(row.get("negative_environment", ""))
                    for row in rows
                    if str(row.get("interaction_label", "0")) == "0"
                ).items()
            )
        ),
        "feature_contract": (
            "sequence length and composition, legal positive-graph degree, and frozen "
            "single-protein ESM distance; no labels from validation/test/OOD are features"
        ),
    }


def audit_json(result: Mapping[str, Any]) -> str:
    return canonical_json(result)
