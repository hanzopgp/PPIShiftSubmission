"""Deterministic metric and resampling helpers for E001 analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PAIR_CLASSIFICATION_METRICS = ("auprc", "auroc")
RANKING_METRICS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10")
CALIBRATION_METRICS = ("brier", "ece", "nll")


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], *, bins: int = 10
) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if len(labels_array) != len(scores_array) or not len(labels_array):
        raise ValueError("Calibration inputs must be non-empty and aligned")
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (scores_array >= lower) & (
            (scores_array < upper) if upper < 1.0 else (scores_array <= upper)
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(labels_array[selected].mean()) - float(scores_array[selected].mean())
            )
    return error


def select_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Select a validation-owned threshold by maximum MCC with deterministic ties."""

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if len(labels_array) != len(scores_array) or set(labels_array.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires aligned binary labels")
    candidates = sorted({0.0, 1.0, *[float(value) for value in scores_array]})
    evaluated = []
    for threshold in candidates:
        predicted = (scores_array >= threshold).astype(np.int64)
        evaluated.append(
            {
                "threshold": threshold,
                "mcc": float(matthews_corrcoef(labels_array, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels_array, predicted)),
            }
        )
    selected = max(
        evaluated,
        key=lambda row: (row["mcc"], row["balanced_accuracy"], -row["threshold"]),
    )
    return {"method": "validation_max_mcc", **selected}


def compute_binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    threshold_source: str = "fixed_0.5",
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if len(labels_array) != len(scores_array) or not len(labels_array):
        raise ValueError("Metric inputs must be non-empty and aligned")
    if set(labels_array.tolist()) != {0, 1}:
        raise ValueError("Binary metrics require both labels")
    predicted = (scores_array >= threshold).astype(np.int64)
    return {
        "rows": int(len(labels_array)),
        "positive_prevalence": float(labels_array.mean()),
        "auprc": float(average_precision_score(labels_array, scores_array)),
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "mcc": float(matthews_corrcoef(labels_array, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predicted)),
        "f1": float(f1_score(labels_array, predicted, zero_division=0)),
        "precision": float(precision_score(labels_array, predicted, zero_division=0)),
        "recall": float(recall_score(labels_array, predicted, zero_division=0)),
        "brier": float(brier_score_loss(labels_array, scores_array)),
        "ece": expected_calibration_error(labels_array, scores_array),
        "nll": float(log_loss(labels_array, scores_array, labels=[0, 1])),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
    }


def normalized_excess_average_precision(auprc: float, prevalence: float) -> float:
    """Return AP excess over the prevalence baseline, normalized to unit range.

    This is an evaluation diagnostic for natural-prevalence surfaces.  It is
    undefined when prevalence is one and invalid outside the open unit interval.
    """

    ap = float(auprc)
    baseline = float(prevalence)
    if not 0.0 <= ap <= 1.0:
        raise ValueError("AUPRC must lie in [0, 1]")
    if not 0.0 < baseline < 1.0:
        raise ValueError("Normalized excess AP requires prevalence in (0, 1)")
    return (ap - baseline) / (1.0 - baseline)


def external_prevalence_metrics(
    labels: Sequence[int], scores: Sequence[float], *, threshold: float = 0.5, threshold_source: str
) -> dict[str, Any]:
    """Compute binary metrics plus explicit natural-prevalence AP diagnostics."""

    result = compute_binary_metrics(
        labels,
        scores,
        threshold=threshold,
        threshold_source=threshold_source,
    )
    prevalence = float(result["positive_prevalence"])
    labels_array = np.asarray(labels, dtype=np.float64)
    # The prevalence-null predictor is the constant probability pi.  These
    # skills are explicitly relative to the evaluation surface and are not
    # used for checkpoint or threshold selection.
    null_brier = float(np.mean((labels_array - prevalence) ** 2))
    null_nll = float(
        -prevalence * np.log(prevalence)
        -(1.0 - prevalence) * np.log(1.0 - prevalence)
    )
    result.update(
        {
            "ap_prevalence_baseline": prevalence,
            "auprc_minus_prevalence": float(result["auprc"]) - prevalence,
            "normalized_excess_ap": normalized_excess_average_precision(
                float(result["auprc"]), prevalence
            ),
            "ap_lift": float(result["auprc"]) / prevalence if prevalence > 0 else None,
            "prevalence_null_probability": prevalence,
            "prevalence_null_brier": null_brier,
            "brier_skill_vs_prevalence_null": (
                1.0 - float(result["brier"]) / null_brier if null_brier > 0 else None
            ),
            "prevalence_null_nll": null_nll,
            "nll_skill_vs_prevalence_null": (
                1.0 - float(result["nll"]) / null_nll if null_nll > 0 else None
            ),
        }
    )
    return result


def kendall_tau(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Compute tie-aware-free Kendall concordance for distinct model scores."""

    first = np.asarray(values_a, dtype=np.float64)
    second = np.asarray(values_b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("Kendall inputs must be aligned one-dimensional arrays of length >= 2")
    concordant = discordant = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            product = (first[left] - first[right]) * (second[left] - second[right])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 0.0


def pairwise_model_order_reversals(
    model_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Compare model ordering across named environments."""

    environments = sorted(model_scores)
    models = sorted({model for values in model_scores.values() for model in values})
    orders = {
        environment: sorted(models, key=lambda model: (-model_scores[environment][model], model))
        for environment in environments
    }
    reversals: dict[str, int] = {}
    for left_index, left in enumerate(environments):
        for right in environments[left_index + 1 :]:
            count = 0
            for model_index, first in enumerate(models):
                for second in models[model_index + 1 :]:
                    left_difference = model_scores[left][first] - model_scores[left][second]
                    right_difference = model_scores[right][first] - model_scores[right][second]
                    if left_difference * right_difference < 0:
                        count += 1
            reversals[f"{left}__{right}"] = count
    return {"models": models, "orders": orders, "pairwise_reversals": reversals}


def anchor_ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute ranking metrics only when each group supplies exactly one positive."""

    if len(rows) != len(scores):
        raise ValueError("Ranking rows and scores must align")
    groups: dict[str, list[tuple[int, float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        group = str(row.get("anchor_group_id", row.get("positive_row_id", "")))
        groups.setdefault(group, []).append((int(row.get("interaction_label", 0)), float(score)))
    if not groups or any(sum(label for label, _score in values) != 1 for values in groups.values()):
        raise ValueError("Anchor ranking requires exactly one positive per anchor group")
    reciprocal_ranks = []
    hits = {int(k): 0 for k in k_values}
    for values in groups.values():
        ordered = sorted(values, key=lambda item: item[1], reverse=True)
        rank = next(index for index, (label, _score) in enumerate(ordered, start=1) if label == 1)
        reciprocal_ranks.append(1.0 / rank)
        for k in hits:
            hits[k] += int(rank <= k)
    count = len(groups)
    return {
        "groups": float(count),
        "mrr": float(np.mean(reciprocal_ranks)),
        **{f"hits_at_{k}": hits[k] / count for k in hits},
    }


def bootstrap_metric(
    labels: Sequence[int],
    scores: Sequence[float],
    metric: Callable[[Sequence[int], Sequence[float]], float],
    *,
    seed: int = 1729,
    replicates: int = 1000,
) -> dict[str, float | int]:
    """Deterministic IID bootstrap helper for pair metrics.

    Anchor-grouped bootstrap should be performed by the caller for ranking statistics.
    """

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    if labels_array.shape != scores_array.shape or labels_array.ndim != 1 or not len(labels_array):
        raise ValueError("Bootstrap inputs must be aligned one-dimensional arrays")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels_array), size=len(labels_array))
        values.append(float(metric(labels_array[indices], scores_array[indices])))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(metric(labels_array, scores_array)),
        "lower_95": float(low),
        "upper_95": float(high),
        "seed": seed,
        "replicates": replicates,
    }
