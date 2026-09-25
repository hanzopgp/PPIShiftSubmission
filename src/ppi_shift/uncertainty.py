"""Paired-unit bootstrap helpers for confirmatory PPI evaluation.

The E001 panels contain one positive row and one synthetic negative row for
each ``positive_row_id``.  These helpers make that identity explicit: a
bootstrap draw samples positive-row units, then applies the same multiplicity
to both rows and to every model evaluated on the panel.  This is deliberately
separate from :func:`ppi_shift.metrics.bootstrap_metric`, whose historical
IID row bootstrap is not suitable for paired panel rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class PairedUnitIndex:
    """Deterministic mapping from panel rows to sorted paired units."""

    unit_ids: tuple[str, ...]
    row_to_unit: np.ndarray

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)


def validate_paired_units(unit_ids: Sequence[str], labels: Sequence[int]) -> PairedUnitIndex:
    """Validate one positive and one negative row for every paired unit.

    Row order is intentionally irrelevant.  The returned unit order is lexical
    and therefore stable across model prediction files and operating systems.
    """

    ids = np.asarray([str(value) for value in unit_ids], dtype=object)
    values = np.asarray(labels, dtype=np.int64)
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values) or not len(ids):
        raise ValueError("Paired-unit inputs must be aligned non-empty vectors")
    if any(not value for value in ids):
        raise ValueError("Paired-unit identifiers must be non-empty")
    if not set(values.tolist()).issubset({0, 1}):
        raise ValueError("Paired-unit labels must be binary")

    by_unit: dict[str, list[int]] = {}
    for unit, label in zip(ids.tolist(), values.tolist(), strict=True):
        by_unit.setdefault(str(unit), []).append(int(label))
    expected_units = tuple(sorted(by_unit))
    for unit in expected_units:
        unit_labels = by_unit[unit]
        if len(unit_labels) != 2 or sorted(unit_labels) != [0, 1]:
            raise ValueError(
                f"Paired unit {unit!r} must have exactly one positive and one negative row"
            )
    lookup = {unit: index for index, unit in enumerate(expected_units)}
    row_to_unit = np.asarray([lookup[str(value)] for value in ids], dtype=np.int64)
    return PairedUnitIndex(expected_units, row_to_unit)


def make_paired_bootstrap_indices(n_units: int, replicates: int, seed: int) -> np.ndarray:
    """Return a deterministic ``(replicates, n_units)`` unit-resampling matrix."""

    if n_units < 1 or replicates < 1:
        raise ValueError("n_units and replicates must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(replicates, n_units), dtype=np.int64)


def bootstrap_weights(indices: np.ndarray, n_units: int) -> np.ndarray:
    """Convert sampled unit indices to per-unit multiplicities."""

    draws = np.asarray(indices, dtype=np.int64)
    if (
        draws.ndim != 2
        or draws.shape[1] != n_units
        or np.any(draws < 0)
        or np.any(draws >= n_units)
    ):
        raise ValueError("Bootstrap indices have an invalid shape or unit id")
    weights = np.zeros((draws.shape[0], n_units), dtype=np.int64)
    for row, sampled in enumerate(draws):
        weights[row] = np.bincount(sampled, minlength=n_units)
    return weights


def make_anchor_cluster_bootstrap_weights(
    anchor_ids: Sequence[str], paired_units: PairedUnitIndex, replicates: int, seed: int
) -> np.ndarray:
    """Sample anchors with replacement and apply each anchor count to its units."""

    anchors = np.asarray([str(value) for value in anchor_ids], dtype=object)
    if len(anchors) != len(paired_units.row_to_unit) or any(not value for value in anchors):
        raise ValueError("Anchor ids must align with non-empty paired-panel rows")
    unit_anchors: dict[int, set[str]] = {}
    for unit, anchor in zip(paired_units.row_to_unit.tolist(), anchors.tolist(), strict=True):
        unit_anchors.setdefault(int(unit), set()).add(str(anchor))
    if any(len(values) != 1 for values in unit_anchors.values()):
        raise ValueError("Every paired unit must belong to exactly one anchor cluster")
    ordered_anchors = tuple(sorted({next(iter(values)) for values in unit_anchors.values()}))
    anchor_lookup = {anchor: index for index, anchor in enumerate(ordered_anchors)}
    unit_to_anchor = np.asarray(
        [anchor_lookup[next(iter(unit_anchors[index]))] for index in range(paired_units.n_units)],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(ordered_anchors), size=(replicates, len(ordered_anchors)), dtype=np.int64
    )
    anchor_weights = np.zeros((replicates, len(ordered_anchors)), dtype=np.int64)
    for row, values in enumerate(sampled):
        anchor_weights[row] = np.bincount(values, minlength=len(ordered_anchors))
    return anchor_weights[:, unit_to_anchor]


@dataclass(frozen=True)
class _SortedRows:
    labels: np.ndarray
    unit_ids: np.ndarray
    starts: np.ndarray
    positive_by_group: np.ndarray


def _prepare_sorted_rows(
    labels: np.ndarray, scores: np.ndarray, unit_ids: np.ndarray
) -> _SortedRows:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64, copy=False)
    sorted_units = unit_ids[order]
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    )
    positive = np.add.reduceat(sorted_labels, starts)
    return _SortedRows(sorted_labels, sorted_units, starts, positive)


def _weighted_metric_rows(sorted_rows: _SortedRows, row_weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != len(sorted_rows.labels):
        raise ValueError("Row weights must align with sorted rows")
    weighted_labels = weights * sorted_rows.labels
    positive_by_group = np.add.reduceat(weighted_labels, sorted_rows.starts)
    total_by_group = np.add.reduceat(weights, sorted_rows.starts)
    total_positive = float(positive_by_group.sum())
    total_negative = float(total_by_group.sum() - total_positive)
    if total_positive <= 0 or total_negative <= 0:
        raise ValueError("Weighted binary metrics require both labels")
    cumulative_positive = np.cumsum(positive_by_group)
    cumulative_total = np.cumsum(total_by_group)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    auprc = float(np.sum(precision * positive_by_group) / total_positive)
    negative_by_group = total_by_group - positive_by_group
    negative_after = total_negative - np.cumsum(negative_by_group)
    auroc = float(
        np.sum(positive_by_group * (negative_after + 0.5 * negative_by_group))
        / (total_positive * total_negative)
    )
    return auprc, auroc


def weighted_paired_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    row_to_unit: Sequence[int],
    unit_weights: Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute weighted AUPRC/AUROC while keeping paired-unit weights intact."""

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    unit_array = np.asarray(row_to_unit, dtype=np.int64)
    if (
        label_array.ndim != 1
        or score_array.shape != label_array.shape
        or unit_array.shape != label_array.shape
    ):
        raise ValueError("Metric inputs must be aligned one-dimensional vectors")
    if not len(label_array) or not set(label_array.tolist()).issubset({0, 1}):
        raise ValueError("Metric inputs must be non-empty and binary")
    if np.any(~np.isfinite(score_array)) or np.any(unit_array < 0):
        raise ValueError("Scores must be finite and unit ids non-negative")
    n_units = int(unit_array.max()) + 1
    weights = (
        np.ones(n_units, dtype=np.float64)
        if unit_weights is None
        else np.asarray(unit_weights, dtype=np.float64)
    )
    if weights.ndim != 1 or len(weights) != n_units or np.any(weights < 0):
        raise ValueError("Unit weights have an invalid shape or value")
    sorted_rows = _prepare_sorted_rows(label_array, score_array, unit_array)
    row_weights = weights[sorted_rows.unit_ids]
    auprc, auroc = _weighted_metric_rows(sorted_rows, row_weights)
    return {"auprc": auprc, "auroc": auroc}


def paired_bootstrap_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    row_to_unit: Sequence[int],
    unit_weights: np.ndarray,
    *,
    chunk_size: int = 64,
) -> dict[str, np.ndarray | float]:
    """Compute paired-bootstrap AUPRC/AUROC distributions.

    ``unit_weights`` must be shared by every model comparison on a panel.  The
    implementation sorts each score vector once and evaluates weighted
    precision-recall and Mann-Whitney areas in chunks to bound memory.
    """

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    unit_array = np.asarray(row_to_unit, dtype=np.int64)
    weights = np.asarray(unit_weights, dtype=np.float64)
    if (
        label_array.ndim != 1
        or score_array.shape != label_array.shape
        or unit_array.shape != label_array.shape
    ):
        raise ValueError("Metric inputs must be aligned one-dimensional vectors")
    if weights.ndim != 2 or weights.shape[1] != int(unit_array.max()) + 1:
        raise ValueError("Bootstrap weights do not match the unit mapping")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    sorted_rows = _prepare_sorted_rows(label_array, score_array, unit_array)
    row_unit_ids = sorted_rows.unit_ids
    result = np.empty((len(weights), 2), dtype=np.float64)
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        row_weights = weights[start:stop][:, row_unit_ids]
        weighted_labels = row_weights * sorted_rows.labels[None, :]
        positive_by_group = np.add.reduceat(weighted_labels, sorted_rows.starts, axis=1)
        total_by_group = np.add.reduceat(row_weights, sorted_rows.starts, axis=1)
        total_positive = positive_by_group.sum(axis=1)
        total_negative = total_by_group.sum(axis=1) - total_positive
        if np.any(total_positive <= 0) or np.any(total_negative <= 0):
            raise ValueError("A bootstrap replicate does not contain both labels")
        cumulative_positive = np.cumsum(positive_by_group, axis=1)
        cumulative_total = np.cumsum(total_by_group, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        result[start:stop, 0] = np.sum(precision * positive_by_group, axis=1) / total_positive
        negative_by_group = total_by_group - positive_by_group
        negative_after = total_negative[:, None] - np.cumsum(negative_by_group, axis=1)
        result[start:stop, 1] = np.sum(
            positive_by_group * (negative_after + 0.5 * negative_by_group), axis=1
        ) / (total_positive * total_negative)
    estimate = weighted_paired_metrics(label_array, score_array, unit_array)["auprc"]
    return {
        "auprc": result[:, 0],
        "auroc": result[:, 1],
        "estimate_auprc": float(estimate),
        "estimate_auroc": float(
            weighted_paired_metrics(label_array, score_array, unit_array)["auroc"]
        ),
    }


def summarize_bootstrap(values: Sequence[float]) -> dict[str, float]:
    """Return a deterministic percentile interval for a bootstrap vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    low, high = np.quantile(array, [0.025, 0.975])
    return {
        "lower_95": float(low),
        "upper_95": float(high),
        "median": float(np.median(array)),
    }


def make_stratified_bootstrap_weights(
    labels: Sequence[int], replicates: int, seed: int
) -> np.ndarray:
    """Sample positives and negatives separately for global negative panels.

    The returned row-weight matrix can be reused unchanged for every model in
    a comparison, which makes the evaluation resampling covariance explicit.
    It does not assume a positive-row/negative-row pairing.
    """

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or not len(values) or set(values.tolist()) != {0, 1}:
        raise ValueError("Stratified bootstrap requires non-empty binary labels")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    positive = np.flatnonzero(values == 1)
    negative = np.flatnonzero(values == 0)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(values)), dtype=np.int64)
    for replicate in range(replicates):
        positive_draw = rng.choice(positive, size=len(positive), replace=True)
        negative_draw = rng.choice(negative, size=len(negative), replace=True)
        weights[replicate] = np.bincount(
            np.concatenate((positive_draw, negative_draw)), minlength=len(values)
        )
    return weights


def stratified_bootstrap_metrics(
    labels: Sequence[int], scores: Sequence[float], row_weights: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Evaluate AUPRC/AUROC under shared stratified row resamples."""

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.int64)
    if (
        label_array.ndim != 1
        or score_array.shape != label_array.shape
        or weights.ndim != 2
        or weights.shape[1] != len(label_array)
    ):
        raise ValueError("Stratified bootstrap inputs are not aligned")
    if np.any(weights < 0) or np.any(~np.isfinite(score_array)):
        raise ValueError("Stratified bootstrap inputs contain invalid values")
    auprc = np.empty(len(weights), dtype=np.float64)
    auroc = np.empty(len(weights), dtype=np.float64)
    for index, replicate_weights in enumerate(weights):
        rows = np.repeat(np.arange(len(label_array)), replicate_weights)
        replicate_labels = label_array[rows]
        replicate_scores = score_array[rows]
        auprc[index] = average_precision_score(replicate_labels, replicate_scores)
        auroc[index] = roc_auc_score(replicate_labels, replicate_scores)
    return {
        "auprc": auprc,
        "auroc": auroc,
        "estimate_auprc": float(average_precision_score(label_array, score_array)),
        "estimate_auroc": float(roc_auc_score(label_array, score_array)),
    }


def seed_stability_summary(
    model_a: Mapping[int, float], model_b: Mapping[int, float]
) -> dict[str, Any]:
    """Summarize all seed-pair deltas for a fixed condition."""

    seeds_a = set(model_a)
    seeds_b = set(model_b)
    if seeds_a != seeds_b or not seeds_a:
        raise ValueError("Seed maps must have the same non-empty seed set")
    ordered = sorted(seeds_a)
    matrix = np.asarray(
        [[float(model_a[left]) - float(model_b[right]) for right in ordered] for left in ordered],
        dtype=np.float64,
    )
    mean_delta = float(np.mean(matrix))
    sign_fraction = float(np.mean(np.sign(matrix) == np.sign(mean_delta))) if mean_delta else 0.0
    return {
        "seeds": ordered,
        "matrix": matrix.tolist(),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(matrix)),
        "minimum_delta": float(np.min(matrix)),
        "maximum_delta": float(np.max(matrix)),
        "same_sign_fraction_as_mean": sign_fraction,
    }


def classify_reversal_categories(
    left_delta: float,
    right_delta: float,
    left_interval: Sequence[float],
    right_interval: Sequence[float],
    left_seed_summary: Mapping[str, Any],
    right_seed_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return descriptive, evaluation-supported, and seed-supported reversal flags."""

    descriptive = float(left_delta) * float(right_delta) < 0
    left_low, left_high = (float(value) for value in left_interval)
    right_low, right_high = (float(value) for value in right_interval)
    evaluation_supported = descriptive and (
        (left_high < 0 or left_low > 0) and (right_high < 0 or right_low > 0)
    )
    seed_supported = descriptive and (
        float(left_seed_summary["same_sign_fraction_as_mean"]) == 1.0
        and float(right_seed_summary["same_sign_fraction_as_mean"]) == 1.0
    )
    if evaluation_supported and seed_supported:
        category = "evaluation_and_seed_supported_reversal"
    elif seed_supported:
        category = "seed_stable_reversal"
    elif evaluation_supported:
        category = "evaluation_uncertainty_supported_reversal"
    elif descriptive:
        category = "descriptive_reversal"
    else:
        category = "no_reversal"
    return {
        "category": category,
        "descriptive_reversal": descriptive,
        "evaluation_uncertainty_supported_reversal": evaluation_supported,
        "seed_stable_reversal": seed_supported,
        "evaluation_and_seed_supported_reversal": evaluation_supported and seed_supported,
    }


def classify_reversal(
    left_delta: float,
    right_delta: float,
    left_interval: Sequence[float],
    right_interval: Sequence[float],
) -> str:
    """Classify a model-order sign flip under the predeclared reversal rule."""

    left_low, left_high = (float(value) for value in left_interval)
    right_low, right_high = (float(value) for value in right_interval)
    descriptive = float(left_delta) * float(right_delta) < 0
    if not descriptive:
        return "no_reversal"
    left_supported = left_high < 0 or left_low > 0
    right_supported = right_high < 0 or right_low > 0
    if left_supported and right_supported:
        return "uncertainty_supported_reversal"
    return "descriptive_only_reversal"
