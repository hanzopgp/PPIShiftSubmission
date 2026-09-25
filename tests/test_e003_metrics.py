from __future__ import annotations

import numpy as np
import pytest

from ppi_shift.metrics import (
    external_prevalence_metrics,
    normalized_excess_average_precision,
)
from ppi_shift.uncertainty import (
    classify_reversal_categories,
    make_stratified_bootstrap_weights,
    seed_stability_summary,
    stratified_bootstrap_metrics,
)


def test_normalized_excess_ap_exposes_prevalence_baseline() -> None:
    assert normalized_excess_average_precision(0.5, 0.25) == pytest.approx(1 / 3)
    result = external_prevalence_metrics(
        [1, 0, 1, 0],
        [0.9, 0.2, 0.8, 0.1],
        threshold=0.5,
        threshold_source="gold_validation_threshold",
    )
    assert result["ap_prevalence_baseline"] == pytest.approx(0.5)
    assert result["normalized_excess_ap"] == pytest.approx(1.0)


def test_external_prevalence_metrics_reports_null_predictor_skills() -> None:
    result = external_prevalence_metrics(
        [1, 1, 0, 0],
        [0.9, 0.8, 0.2, 0.1],
        threshold=0.5,
        threshold_source="gold_validation_threshold",
    )

    assert result["prevalence_null_probability"] == pytest.approx(0.5)
    assert result["prevalence_null_brier"] == pytest.approx(0.25)
    assert result["brier_skill_vs_prevalence_null"] == pytest.approx(0.9)
    assert float(result["nll_skill_vs_prevalence_null"]) > 0.0


def test_stratified_bootstrap_is_shared_and_reproducible() -> None:
    labels = np.asarray([1, 1, 0, 0, 0])
    scores_a = np.asarray([0.9, 0.8, 0.4, 0.3, 0.2])
    scores_b = np.asarray([0.7, 0.6, 0.8, 0.4, 0.1])
    weights = make_stratified_bootstrap_weights(labels, replicates=8, seed=1729)
    assert np.array_equal(weights, make_stratified_bootstrap_weights(labels, 8, 1729))
    assert np.all(weights[:, labels == 1].sum(axis=1) == 2)
    assert np.all(weights[:, labels == 0].sum(axis=1) == 3)
    first = stratified_bootstrap_metrics(labels, scores_a, weights)
    second = stratified_bootstrap_metrics(labels, scores_b, weights)
    assert len(first["auprc"]) == len(second["auprc"]) == 8
    assert float(first["estimate_auprc"]) > float(second["estimate_auprc"])


def test_seed_stability_and_reversal_categories_use_all_seed_pairs() -> None:
    left = seed_stability_summary({11: 0.80, 22: 0.81, 33: 0.79}, {11: 0.70, 22: 0.71, 33: 0.69})
    right = seed_stability_summary({11: 0.60, 22: 0.61, 33: 0.59}, {11: 0.70, 22: 0.71, 33: 0.69})
    assert np.asarray(left["matrix"]).shape == (3, 3)
    assert left["same_sign_fraction_as_mean"] == 1.0
    result = classify_reversal_categories(
        left["mean_delta"],
        right["mean_delta"],
        (0.05, 0.15),
        (-0.15, -0.05),
        left,
        right,
    )
    assert result["category"] == "evaluation_and_seed_supported_reversal"
