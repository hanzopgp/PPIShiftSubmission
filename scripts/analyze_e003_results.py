#!/usr/bin/env python3
"""Analyze frozen E003 results and generate paper-ready, source-linked figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ppi_shift.artifacts import delimited_header, git_commit, iter_delimited, sha256_path
from ppi_shift.metrics import compute_binary_metrics, kendall_tau
from ppi_shift.uncertainty import (
    classify_reversal_categories,
    make_anchor_cluster_bootstrap_weights,
    make_stratified_bootstrap_weights,
    paired_bootstrap_metrics,
    seed_stability_summary,
    validate_paired_units,
)

MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2 pair MLP",
    "dscript_esm2": "D-SCRIPT-ESM2",
    "tuna_corrected": "corrected TUnA",
    "rapppid": "RAPPPID",
}
ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
ENVIRONMENT_LABELS = {
    "uniform_partner": "UNIF",
    "degree_partner_matched": "DEG",
    "sequence_nuisance_matched": "SEQ",
    "esm_partner_matched": "ESM",
    "bernett2026_ilp_exact": "BERN",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}
SUPPLEMENTARY_TEST_ENVIRONMENTS = {
    "degree_partner_matched",
    "tppni2025_scm_cl3_exact_balanced",
}
SEEDS = (11, 22, 33)


def _condition_key(train_environment: str, test_environment: str) -> str:
    return f"{train_environment}__{test_environment}"


def _environment_label(value: str) -> str:
    return ENVIRONMENT_LABELS.get(value, value)


def _model_label(value: str) -> str:
    return MODEL_LABELS.get(value, value)


def _tppni_pool_delimiter(path: Path) -> str | None:
    """Resolve the frozen TPPNI pool delimiter, including legacy suffix drift."""

    required = {"scm_probability", "l3_count"}
    suffix_header = set(delimited_header(path))
    if required.issubset(suffix_header):
        return None
    comma_header = set(delimited_header(path, delimiter=","))
    if required.issubset(comma_header):
        return ","
    raise SystemExit(
        "TPPNI pool lacks the required scm_probability/l3_count columns: "
        f"{path}"
    )


def _condition_label(value: str) -> str:
    train, separator, test = value.partition("__")
    if not separator:
        return value
    return f"{_environment_label(train)} x {_environment_label(test)}"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, with larger values ranked first."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(-array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < len(sorted_values):
        stop = start + 1
        while stop < len(sorted_values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 1.0 + 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(values_a: list[float], values_b: list[float]) -> float:
    first = _average_ranks(np.asarray(values_a, dtype=np.float64))
    second = _average_ranks(np.asarray(values_b, dtype=np.float64))
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _entropy(probabilities: list[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0]
    return float(-np.sum(values * np.log2(values))) if len(values) else 0.0


def _ranking_comparison_summary(
    values_by_condition: dict[str, np.ndarray], models: tuple[str, ...]
) -> dict[str, Any]:
    """Summarize ranking changes over all unordered condition pairs.

    The point estimand is the three-seed mean AUPRC for each model and
    condition. Ties use the declared model-roster order through the stable
    rank helper. Sign-change comparisons exclude a zero delta in either
    condition and report that exclusion explicitly.
    """

    if not values_by_condition:
        raise ValueError("At least one ranking condition is required")
    expected_size = len(models)
    if expected_size < 2:
        raise ValueError("At least two models are required")
    conditions = tuple(values_by_condition)
    values: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    winners: dict[str, int] = {}
    for condition in conditions:
        current = np.asarray(values_by_condition[condition], dtype=np.float64)
        if current.shape != (expected_size,) or np.any(~np.isfinite(current)):
            raise ValueError(f"Invalid ranking values for {condition!r}")
        values[condition] = current
        ranks[condition] = _average_ranks(current)
        winners[condition] = int(np.argmin(ranks[condition]))

    condition_pair_count = len(conditions) * (len(conditions) - 1) // 2
    top_model_changes = 0
    rank_displacement_sum = 0.0
    sign_change_count = 0
    sign_change_eligible = 0
    model_pair_count = expected_size * (expected_size - 1) // 2
    for left_index, left_condition in enumerate(conditions):
        for right_condition in conditions[left_index + 1 :]:
            if winners[left_condition] != winners[right_condition]:
                top_model_changes += 1
            rank_displacement_sum += float(
                np.abs(ranks[left_condition] - ranks[right_condition]).sum()
            )
            left_values = values[left_condition]
            right_values = values[right_condition]
            for model_index in range(expected_size):
                for other_index in range(model_index + 1, expected_size):
                    left_delta = left_values[model_index] - left_values[other_index]
                    right_delta = right_values[model_index] - right_values[other_index]
                    left_sign = np.sign(left_delta)
                    right_sign = np.sign(right_delta)
                    if left_sign == 0 or right_sign == 0:
                        continue
                    sign_change_eligible += 1
                    if left_sign != right_sign:
                        sign_change_count += 1

    rank_displacement_denominator = condition_pair_count * expected_size
    return {
        "scope": (
            "all unordered pairs of train-by-test conditions; three-seed mean "
            "AUPRC; ties use stable model-roster order"
        ),
        "condition_count": len(conditions),
        "condition_pair_count": condition_pair_count,
        "model_count": expected_size,
        "model_pair_count": model_pair_count,
        "top_model_change_count": top_model_changes,
        "top_model_change_fraction": (
            top_model_changes / condition_pair_count if condition_pair_count else 0.0
        ),
        "pairwise_sign_change_count": sign_change_count,
        "pairwise_sign_change_eligible": sign_change_eligible,
        "pairwise_sign_change_rate": (
            sign_change_count / sign_change_eligible if sign_change_eligible else 0.0
        ),
        "pairwise_sign_change_tie_exclusions": (
            condition_pair_count * model_pair_count - sign_change_eligible
        ),
        "absolute_rank_displacement_sum": rank_displacement_sum,
        "absolute_rank_displacement_denominator": rank_displacement_denominator,
        "average_absolute_rank_displacement": (
            rank_displacement_sum / rank_displacement_denominator
            if rank_displacement_denominator
            else 0.0
        ),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _panel_rows(manifest: dict[str, Any], environment: str) -> list[dict[str, str]]:
    spec = manifest["stages"]["test"]["samplers"][environment]
    path = Path(spec["panel_path"])
    if sha256_path(path) != spec["panel_sha256"]:
        raise ValueError(f"Test panel hash changed: {path}")
    rows = list(iter_delimited(path))
    if len(rows) != int(spec["panel_rows"]):
        raise ValueError(f"Test panel row count changed: {path}")
    return rows


def _cell_vectors(
    cell: dict[str, Any], panel_manifest: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    environment = str(cell["test_environment"])
    panel_rows = _panel_rows(panel_manifest, environment)
    prediction_path = Path(str(cell["prediction_path"]))
    if sha256_path(prediction_path) != cell["prediction_sha256"]:
        raise ValueError(f"Prediction hash changed: {prediction_path}")
    prediction_rows = list(iter_delimited(prediction_path))
    by_id: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        if row["row_id"] in by_id:
            raise ValueError(f"Duplicate prediction row id: {prediction_path}")
        by_id[row["row_id"]] = row
    labels: list[int] = []
    scores: list[float] = []
    positive_ids: list[str] = []
    anchor_ids: list[str] = []
    for panel_row in panel_rows:
        row_id = panel_row["row_id"]
        prediction = by_id.get(row_id) or by_id.get(f"{environment}::{row_id}")
        if prediction is None:
            raise ValueError(f"Prediction row missing for {environment}/{row_id}")
        label_key = "interaction_label" if "interaction_label" in prediction else "y_true"
        score_key = "score" if "score" in prediction else "y_score"
        label = int(prediction[label_key])
        expected_label = int(panel_row["interaction_label"])
        if label != expected_label:
            raise ValueError(f"Prediction label mismatch: {prediction_path}")
        labels.append(label)
        scores.append(float(prediction[score_key]))
        positive_ids.append(panel_row["positive_row_id"])
        anchor_ids.append(panel_row["anchor_id"])
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        positive_ids,
        anchor_ids,
    )


def _fast_stratified_bootstrap(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate shared positive/negative row weights without repeated sklearn sorting."""

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64, copy=False)
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    )
    auprc = np.empty(len(weights), dtype=np.float64)
    auroc = np.empty(len(weights), dtype=np.float64)
    for start in range(0, len(weights), 64):
        stop = min(start + 64, len(weights))
        row_weights = weights[start:stop][:, order].astype(np.float64, copy=False)
        positive_by_group = np.add.reduceat(row_weights * sorted_labels[None, :], starts, axis=1)
        total_by_group = np.add.reduceat(row_weights, starts, axis=1)
        total_positive = positive_by_group.sum(axis=1)
        total_negative = total_by_group.sum(axis=1) - total_positive
        if np.any(total_positive <= 0) or np.any(total_negative <= 0):
            raise ValueError("Bootstrap replicate lacks one binary class")
        cumulative_positive = np.cumsum(positive_by_group, axis=1)
        cumulative_total = np.cumsum(total_by_group, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        auprc[start:stop] = np.sum(precision * positive_by_group, axis=1) / total_positive
        negative_by_group = total_by_group - positive_by_group
        negative_after = total_negative[:, None] - np.cumsum(negative_by_group, axis=1)
        auroc[start:stop] = np.sum(
            positive_by_group * (negative_after + 0.5 * negative_by_group), axis=1
        ) / (total_positive * total_negative)
    return auprc, auroc


def _interval(values: np.ndarray, estimate: float, seed: int, replicates: int) -> dict[str, Any]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "seed": seed,
        "replicates": replicates,
        "scheme": "stratified_positive_negative_shared_rows",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: Any, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def _heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    base: Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(
        range(len(column_labels)),
        [_condition_label(value) for value in column_labels],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(row_labels)), [_condition_label(value) for value in row_labels])
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return _save_figure(figure, base)


def _read_external_rows(external: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in external["bundles"]:
        for surface in bundle["surfaces"].values():
            natural = surface["metrics"]["natural_prevalence"]
            balanced = surface["metrics"].get("balanced_posthoc_diagnostic", {}).get("metrics", {})
            prevalence_sensitivity = surface["metrics"].get("balanced_prevalence_sensitivity", {})
            rows.append(
                {
                    "model": bundle["model"],
                    "seed": int(bundle["seed"]),
                    "train_environment": bundle["train_environment"],
                    "source": surface["source"],
                    "surface": surface["surface"],
                    "auprc": float(natural["auprc"]),
                    "auroc": float(natural["auroc"]),
                    "prevalence": float(natural["positive_prevalence"]),
                    "auprc_minus_prevalence": float(natural["auprc_minus_prevalence"]),
                    "normalized_excess_ap": float(natural["normalized_excess_ap"]),
                    "ap_lift": float(natural["ap_lift"]),
                    "prevalence_null_probability": natural.get("prevalence_null_probability"),
                    "prevalence_null_brier": natural.get("prevalence_null_brier"),
                    "prevalence_null_nll": natural.get("prevalence_null_nll"),
                    "brier_skill_vs_prevalence_null": natural.get(
                        "brier_skill_vs_prevalence_null"
                    ),
                    "nll_skill_vs_prevalence_null": natural.get(
                        "nll_skill_vs_prevalence_null"
                    ),
                    "balanced_posthoc_auprc": balanced.get("auprc"),
                    "balanced_posthoc_auroc": balanced.get("auroc"),
                    "prevalence_sensitivity": json.dumps(prevalence_sensitivity, sort_keys=True),
                    "semantic_classes": ";".join(surface.get("semantic_classes", [])),
                }
            )
    return rows


REGISTRY_FIELDS = (
    "scope",
    "role",
    "model",
    "model_source_revision",
    "representation_source_revision",
    "training_environment",
    "validation_environment",
    "test_environment",
    "evaluation_surface",
    "seed",
    "rows",
    "panel_sha256",
    "checkpoint_sha256",
    "prediction_sha256",
    "checkpoint_selection_epoch",
    "threshold",
    "threshold_source",
    "auprc",
    "auroc",
    "mcc",
    "balanced_accuracy",
    "f1",
    "brier",
    "nll",
    "ece",
    "prevalence",
    "auprc_minus_prevalence",
    "normalized_excess_ap",
    "ap_lift",
    "prevalence_null_probability",
    "prevalence_null_brier",
    "prevalence_null_nll",
    "brier_skill_vs_prevalence_null",
    "nll_skill_vs_prevalence_null",
    "status",
)


def _source_revision_map(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    source_revisions = config.get("models", {}).get("source_revisions", {})
    if not isinstance(source_revisions, dict):
        raise SystemExit("E003 config lacks models.source_revisions")
    model_keys = {
        "dscript_esm2": "dscript_esm2",
        "tuna_corrected": "tuna_corrected",
        "rapppid": "rapppid",
    }
    model_sources = {
        model: str(source_revisions[key])
        for model, key in model_keys.items()
        if key in source_revisions
    }
    model_sources["symmetric_frozen_esm2_pair_mlp"] = (
        "E001v3_exact_checkpoint_reuse_or_local_frozen_pair_mlp"
    )
    representation_sources = {
        model: str(source_revisions["esm2_t30_150m"])
        for model in ("dscript_esm2", "tuna_corrected", "symmetric_frozen_esm2_pair_mlp")
        if "esm2_t30_150m" in source_revisions
    }
    if "rapppid" in source_revisions:
        representation_sources["rapppid"] = str(source_revisions["rapppid"])
    return model_sources, representation_sources


def _panel_hash(panel_manifest: dict[str, Any], environment: str) -> str:
    spec = panel_manifest["stages"]["test"]["samplers"][environment]
    return str(spec.get("panel_sha256", spec.get("sha256", "")))


def _registry_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": metrics.get("rows"),
        "auprc": metrics.get("auprc"),
        "auroc": metrics.get("auroc"),
        "mcc": metrics.get("mcc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1": metrics.get("f1"),
        "brier": metrics.get("brier"),
        "nll": metrics.get("nll"),
        "ece": metrics.get("ece"),
        "prevalence": metrics.get("positive_prevalence"),
        "auprc_minus_prevalence": metrics.get("auprc_minus_prevalence"),
        "normalized_excess_ap": metrics.get("normalized_excess_ap"),
        "ap_lift": metrics.get("ap_lift"),
        "prevalence_null_probability": metrics.get("prevalence_null_probability"),
        "prevalence_null_brier": metrics.get("prevalence_null_brier"),
        "prevalence_null_nll": metrics.get("prevalence_null_nll"),
        "brier_skill_vs_prevalence_null": metrics.get("brier_skill_vs_prevalence_null"),
        "nll_skill_vs_prevalence_null": metrics.get("nll_skill_vs_prevalence_null"),
    }


def _registry_row(
    *,
    scope: str,
    role: str,
    model: str,
    model_source_revision: str,
    representation_source_revision: str,
    training_environment: str,
    validation_environment: str,
    test_environment: str,
    evaluation_surface: str,
    seed: int,
    panel_sha256: str,
    checkpoint_sha256: str | None,
    prediction_sha256: str,
    checkpoint_selection_epoch: int | None,
    threshold: Any,
    threshold_source: Any,
    metrics: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    row = {
        "scope": scope,
        "role": role,
        "model": model,
        "model_source_revision": model_source_revision,
        "representation_source_revision": representation_source_revision,
        "training_environment": training_environment,
        "validation_environment": validation_environment,
        "test_environment": test_environment,
        "evaluation_surface": evaluation_surface,
        "seed": int(seed),
        "panel_sha256": panel_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": prediction_sha256,
        "checkpoint_selection_epoch": checkpoint_selection_epoch,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "status": status,
    }
    row.update(_registry_metrics(metrics))
    return row


def _build_result_registry(
    core: dict[str, Any],
    external: dict[str, Any],
    datasail: dict[str, Any] | None,
    panel_manifest: dict[str, Any],
    datasail_panel_sha256: str | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model_sources, representation_sources = _source_revision_map(config)
    rows: list[dict[str, Any]] = []
    for cell in core["cells"]:
        test_environment = str(cell["test_environment"])
        metrics = dict(cell["metrics"])
        threshold = cell.get("threshold", {})
        rows.append(
            _registry_row(
                scope="core",
                role=(
                    "core_supplementary_test_environment"
                    if test_environment in SUPPLEMENTARY_TEST_ENVIRONMENTS
                    else "core_primary"
                ),
                model=str(cell["model"]),
                model_source_revision=str(
                    cell.get("source_revision") or model_sources.get(str(cell["model"]), "")
                ),
                representation_source_revision=representation_sources.get(
                    str(cell["model"]), ""
                ),
                training_environment=str(cell["train_environment"]),
                validation_environment=str(cell["train_environment"]),
                test_environment=test_environment,
                evaluation_surface="gold_core",
                seed=int(cell["seed"]),
                panel_sha256=_panel_hash(panel_manifest, test_environment),
                checkpoint_sha256=cell.get("checkpoint_sha256"),
                prediction_sha256=str(cell["prediction_sha256"]),
                checkpoint_selection_epoch=cell.get("checkpoint_selection_epoch"),
                threshold=threshold.get("threshold"),
                threshold_source=metrics.get("threshold_source", threshold.get("method")),
                metrics=metrics,
                status=str(cell["status"]),
            )
        )

    for bundle in external["bundles"]:
        checkpoint = bundle.get("checkpoint", {})
        for surface_key, surface in bundle["surfaces"].items():
            natural = dict(surface["metrics"]["natural_prevalence"])
            rows.append(
                _registry_row(
                    scope="external",
                    role=(
                        "measured_transfer"
                        if surface["source"] in {"vidal", "intact"}
                        else "supplementary_external_source_surface"
                    ),
                    model=str(bundle["model"]),
                    model_source_revision=str(
                        checkpoint.get("source_revision")
                        or model_sources.get(str(bundle["model"]), "")
                    ),
                    representation_source_revision=representation_sources.get(
                        str(bundle["model"]), ""
                    ),
                    training_environment=str(bundle["train_environment"]),
                    validation_environment=str(bundle["train_environment"]),
                    test_environment=surface_key,
                    evaluation_surface=surface_key,
                    seed=int(bundle["seed"]),
                    panel_sha256=str(surface["panel"]["sha256"]),
                    checkpoint_sha256=str(
                        checkpoint.get("checkpoint_sha256", checkpoint.get("sha256", ""))
                    ),
                    prediction_sha256=str(surface["prediction"]["sha256"]),
                    checkpoint_selection_epoch=checkpoint.get("selected_epoch"),
                    threshold=natural.get("threshold"),
                    threshold_source=natural.get("threshold_source"),
                    metrics=natural,
                    status="complete_external_score_reused_core_checkpoint",
                )
            )

    if datasail is not None:
        if not datasail_panel_sha256:
            raise SystemExit("DataSAIL results require a DataSAIL panel manifest hash")
        for cell in datasail["cells"]:
            metrics = dict(cell["metrics"])
            threshold = cell.get("threshold", {})
            rows.append(
                _registry_row(
                    scope="datasail",
                    role="supplementary_official_datasail_sensitivity",
                    model=str(cell["model"]),
                    model_source_revision=str(
                        cell.get("source_revision")
                        or model_sources.get(str(cell["model"]), "")
                    ),
                    representation_source_revision=representation_sources.get(
                        str(cell["model"]), ""
                    ),
                    training_environment=str(cell["train_environment"]),
                    validation_environment=str(cell["train_environment"]),
                    test_environment=str(cell["test_environment"]),
                    evaluation_surface="official_datasail",
                    seed=int(cell["seed"]),
                    panel_sha256=datasail_panel_sha256,
                    checkpoint_sha256=cell.get("checkpoint_sha256"),
                    prediction_sha256=str(cell["prediction_sha256"]),
                    checkpoint_selection_epoch=cell.get("checkpoint_selection_epoch"),
                    threshold=threshold.get("threshold"),
                    threshold_source=metrics.get("threshold_source", threshold.get("method")),
                    metrics=metrics,
                    status=str(cell["status"]),
                )
            )

    rows.sort(
        key=lambda row: (
            row["scope"],
            row["model"],
            int(row["seed"]),
            row["training_environment"],
            row["test_environment"],
        )
    )
    return rows


def _write_result_registry(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    input_hashes: dict[str, str],
    datasail: dict[str, Any] | None,
) -> None:
    registry = {
        "schema_version": "ppi_shift.e003_result_registry.v1",
        "experiment_id": "E003",
        "rows": len(rows),
        "core_score_cells": sum(row["scope"] == "core" for row in rows),
        "external_score_cells": sum(row["scope"] == "external" for row in rows),
        "datasail_score_cells": sum(row["scope"] == "datasail" for row in rows),
        "input_hashes": input_hashes,
        "datasail_included": datasail is not None,
        "columns": list(REGISTRY_FIELDS),
        "rows_data": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e003_result_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_csv = output_dir / "e003_result_registry.csv"
    with registry_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGISTRY_FIELDS))
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in REGISTRY_FIELDS} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-results", type=Path, required=True)
    parser.add_argument("--external-results", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--sampler-trace-manifest", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--datasail-results", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    if args.bootstrap_replicates < 2000:
        raise SystemExit("E003 requires at least 2,000 bootstrap replicates")
    core = _json(args.core_results)
    external = _json(args.external_results)
    if core.get("status") != "complete_core_matrix" or len(core.get("cells", [])) != 432:
        raise SystemExit("E003 core matrix is incomplete")
    panel_manifest = yaml.safe_load(args.panel_manifest.read_text(encoding="utf-8"))
    if not isinstance(panel_manifest, dict):
        raise SystemExit("E003 panel manifest is not a YAML mapping")
    sampler_trace_manifest: dict[str, Any] | None = None
    if args.sampler_trace_manifest:
        if not args.sampler_trace_manifest.is_file():
            raise SystemExit(
                f"E003 sampler trace manifest is missing: {args.sampler_trace_manifest}"
            )
        sampler_trace_manifest = yaml.safe_load(
            args.sampler_trace_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(sampler_trace_manifest, dict):
            raise SystemExit("E003 sampler trace manifest is not a YAML mapping")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("E003 config is not a YAML mapping")
    if not args.protocol_freeze.is_file():
        raise SystemExit(f"E003 protocol freeze is missing: {args.protocol_freeze}")
    protocol_freeze = yaml.safe_load(args.protocol_freeze.read_text(encoding="utf-8"))
    if (
        not isinstance(protocol_freeze, dict)
        or protocol_freeze.get("freeze_status")
        != "protocol_frozen_before_any_corrected_test_score"
    ):
        raise SystemExit("E003 protocol freeze is not the required pre-score freeze")
    cells = core["cells"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.repo_root / "paper" / "figures" / "e003"
    figure_dir.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    vectors: dict[
        tuple[str, int, str, str], tuple[np.ndarray, np.ndarray, list[str], list[str]]
    ] = {}
    for cell in cells:
        key = (
            str(cell["model"]),
            int(cell["seed"]),
            str(cell["train_environment"]),
            str(cell["test_environment"]),
        )
        if key in by_key:
            raise SystemExit(f"Duplicate E003 cell: {key}")
        by_key[key] = cell
        labels, scores, positive_ids, anchors = _cell_vectors(cell, panel_manifest)
        if np.any(~np.isfinite(scores)):
            raise SystemExit(f"Non-finite prediction score in E003 cell: {key}")
        threshold_payload = cell.get("threshold", {})
        if not isinstance(threshold_payload, dict) or "threshold" not in threshold_payload:
            raise SystemExit(f"E003 cell lacks a validation-owned threshold: {key}")
        threshold = float(threshold_payload["threshold"])
        recorded_metrics = cell.get("metrics")
        if not isinstance(recorded_metrics, dict):
            raise SystemExit(f"E003 cell lacks metrics: {key}")
        recomputed_metrics = compute_binary_metrics(
            labels,
            scores,
            threshold=threshold,
            threshold_source=str(
                recorded_metrics.get("threshold_source", "matched_validation_max_mcc")
            ),
        )
        for metric_name in (
            "rows",
            "positive_prevalence",
            "auprc",
            "auroc",
            "mcc",
            "balanced_accuracy",
            "f1",
            "brier",
            "nll",
            "ece",
        ):
            if metric_name not in recorded_metrics:
                raise SystemExit(f"E003 cell metric is missing {metric_name}: {key}")
            if not np.isclose(
                float(recorded_metrics[metric_name]),
                float(recomputed_metrics[metric_name]),
                rtol=0.0,
                atol=5e-12,
            ):
                raise SystemExit(
                    f"E003 cell metric mismatch for {key}/{metric_name}: "
                    f"{recorded_metrics[metric_name]} != {recomputed_metrics[metric_name]}"
                )
        vectors[key] = (labels, scores, positive_ids, anchors)
    expected_keys = {
        (model, seed, train, test)
        for model in MODELS
        for seed in SEEDS
        for train in ENVIRONMENTS
        for test in ENVIRONMENTS
    }
    if set(by_key) != expected_keys:
        raise SystemExit(
            f"E003 core key grid mismatch: observed={len(by_key)} expected={len(expected_keys)}"
        )

    metric_means: dict[tuple[str, str, str, str], float] = {}
    metric_sds: dict[tuple[str, str, str, str], float] = {}
    metric_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for train in ENVIRONMENTS:
            for test in ENVIRONMENTS:
                for metric in ("auprc", "auroc"):
                    values = [
                        float(by_key[(model, seed, train, test)]["metrics"][metric])
                        for seed in SEEDS
                    ]
                    metric_means[(model, train, test, metric)] = float(np.mean(values))
                    metric_sds[(model, train, test, metric)] = float(np.std(values, ddof=1))
                    metric_rows.append(
                        {
                            "model": model,
                            "train_environment": train,
                            "test_environment": test,
                            "metric": metric,
                            "mean": metric_means[(model, train, test, metric)],
                            "sd_across_seeds": metric_sds[(model, train, test, metric)],
                            "n_seeds": len(values),
                        }
                    )
    _write_csv(args.output_dir / "core_metric_summary.csv", metric_rows)

    # Shared stratified bootstrap on every frozen test panel.  For a fixed test
    # environment the positive and negative row resamples are reused across all
    # models, train environments, and training seeds.  The bootstrap first
    # evaluates each seed and then averages those three values, matching the
    # primary point estimand without creating an artificial positive/negative
    # pairing for global negative sets.
    bootstrap_ap: dict[tuple[str, str, str], np.ndarray] = {}
    bootstrap_auroc: dict[tuple[str, str, str], np.ndarray] = {}
    bootstrap_summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for test_index, test_environment in enumerate(ENVIRONMENTS):
        labels = vectors[(MODELS[0], SEEDS[0], ENVIRONMENTS[0], test_environment)][0]
        weights = make_stratified_bootstrap_weights(
            labels, args.bootstrap_replicates, args.bootstrap_seed + test_index
        )
        for train_environment in ENVIRONMENTS:
            for model in MODELS:
                ap_values: list[np.ndarray] = []
                auroc_values: list[np.ndarray] = []
                for seed in SEEDS:
                    current_labels, scores, _positive_ids, _anchors = vectors[
                        (model, seed, train_environment, test_environment)
                    ]
                    if not np.array_equal(current_labels, labels):
                        raise SystemExit(
                            "Shared test labels changed: "
                            f"{model}/{train_environment}/{test_environment}/{seed}"
                        )
                    ap, auroc = _fast_stratified_bootstrap(current_labels, scores, weights)
                    ap_values.append(ap)
                    auroc_values.append(auroc)
                key = (model, train_environment, test_environment)
                bootstrap_ap[key] = np.mean(np.asarray(ap_values), axis=0)
                bootstrap_auroc[key] = np.mean(np.asarray(auroc_values), axis=0)
                bootstrap_summary[key] = {
                    "auprc": _interval(
                        bootstrap_ap[key],
                        metric_means[(model, train_environment, test_environment, "auprc")],
                        args.bootstrap_seed + test_index,
                        args.bootstrap_replicates,
                    ),
                    "auroc": _interval(
                        bootstrap_auroc[key],
                        metric_means[(model, train_environment, test_environment, "auroc")],
                        args.bootstrap_seed + test_index,
                        args.bootstrap_replicates,
                    ),
                }

    cluster_sensitivity: dict[str, Any] = {}
    for environment_index, environment in enumerate(ENVIRONMENTS):
        base_labels, _base_scores, base_positive_ids, base_anchors = vectors[
            (MODELS[0], SEEDS[0], environment, environment)
        ]
        try:
            paired = validate_paired_units(base_positive_ids, base_labels)
            cluster_weights = make_anchor_cluster_bootstrap_weights(
                np.asarray(base_anchors, dtype=object),
                paired,
                args.bootstrap_replicates,
                args.bootstrap_seed + 100_000 + environment_index,
            )
            summaries: dict[str, Any] = {}
            for model in MODELS:
                seed_results: list[dict[str, Any]] = []
                bootstrap_values: list[np.ndarray] = []
                for seed in SEEDS:
                    labels, scores, positive_ids, _anchors = vectors[
                        (model, seed, environment, environment)
                    ]
                    if positive_ids != base_positive_ids or not np.array_equal(
                        labels, base_labels
                    ):
                        raise ValueError(
                            "paired positive-row identities changed across models or seeds"
                        )
                    result = paired_bootstrap_metrics(
                        labels, scores, paired.row_to_unit, cluster_weights
                    )
                    bootstrap_values.append(np.asarray(result["auprc"], dtype=np.float64))
                    seed_results.append(
                        {
                            "seed": seed,
                            "estimate": float(result["estimate_auprc"]),
                        }
                    )
                mean_bootstrap = np.mean(np.asarray(bootstrap_values), axis=0)
                summaries[model] = _interval(
                    mean_bootstrap,
                    float(np.mean([row["estimate"] for row in seed_results])),
                    args.bootstrap_seed + 100_000 + environment_index,
                    args.bootstrap_replicates,
                )
                summaries[model]["seed_estimates"] = seed_results
            cluster_sensitivity[environment] = {
                "status": "complete",
                "scope": (
                    "anchor_endpoint_cluster_sensitivity; three-seed mean; "
                    "matched train=test"
                ),
                "unit": "positive_row_id clustered by anchor_id",
                "backbone_cluster_not_computed": True,
                "summaries": summaries,
            }
        except ValueError as error:
            cluster_sensitivity[environment] = {
                "status": "unsupported_panel_contract",
                "scope": "anchor_endpoint_cluster_sensitivity",
                "reason": str(error),
            }

    conditions = [(train, test) for train in ENVIRONMENTS for test in ENVIRONMENTS]
    seed_stability: list[dict[str, Any]] = []
    for model in MODELS:
        for train_environment, test_environment in conditions:
            values = {
                seed: float(
                    by_key[(model, seed, train_environment, test_environment)]["metrics"]["auprc"]
                )
                for seed in SEEDS
            }
            seed_stability.append(
                {
                    "model": model,
                    "train_environment": train_environment,
                    "test_environment": test_environment,
                    "mean_auprc": float(np.mean(list(values.values()))),
                    "sd_auprc": float(np.std(list(values.values()), ddof=1)),
                    "minimum_auprc": float(min(values.values())),
                    "maximum_auprc": float(max(values.values())),
                    "seed_values": values,
                }
            )
    _write_csv(
        args.output_dir / "seed_stability_summary.csv",
        [
            {
                "model": row["model"],
                "train_environment": row["train_environment"],
                "test_environment": row["test_environment"],
                "mean_auprc": row["mean_auprc"],
                "sd_auprc": row["sd_auprc"],
                "minimum_auprc": min(row["seed_values"].values()),
                "maximum_auprc": max(row["seed_values"].values()),
            }
            for row in seed_stability
        ],
    )

    # Pairwise architecture reversals across the complete 6x6 condition surface.
    # Point estimates average the three seeds; bootstrap intervals are shared
    # evaluation resamples, while seed stability is a separate flag.
    reversal_events: list[dict[str, Any]] = []
    for left_index, left_model in enumerate(MODELS):
        for right_model in MODELS[left_index + 1 :]:
            for first_index, first_condition in enumerate(conditions):
                first_train, first_test = first_condition
                left_seed_values = {
                    seed: float(
                        by_key[(left_model, seed, first_train, first_test)]["metrics"]["auprc"]
                    )
                    for seed in SEEDS
                }
                right_seed_values = {
                    seed: float(
                        by_key[(right_model, seed, first_train, first_test)]["metrics"]["auprc"]
                    )
                    for seed in SEEDS
                }
                first_delta = (
                    bootstrap_ap[(left_model, first_train, first_test)]
                    - bootstrap_ap[(right_model, first_train, first_test)]
                )
                first_summary = _interval(
                    first_delta,
                    metric_means[(left_model, first_train, first_test, "auprc")]
                    - metric_means[(right_model, first_train, first_test, "auprc")],
                    args.bootstrap_seed + ENVIRONMENTS.index(first_test),
                    args.bootstrap_replicates,
                )
                first_seed = seed_stability_summary(left_seed_values, right_seed_values)
                for second_condition in conditions[first_index + 1 :]:
                    second_train, second_test = second_condition
                    second_delta = (
                        bootstrap_ap[(left_model, second_train, second_test)]
                        - bootstrap_ap[(right_model, second_train, second_test)]
                    )
                    second_summary = _interval(
                        second_delta,
                        metric_means[(left_model, second_train, second_test, "auprc")]
                        - metric_means[(right_model, second_train, second_test, "auprc")],
                        args.bootstrap_seed + ENVIRONMENTS.index(second_test),
                        args.bootstrap_replicates,
                    )
                    second_seed = seed_stability_summary(
                        {
                            seed: float(
                                by_key[(left_model, seed, second_train, second_test)]["metrics"][
                                    "auprc"
                                ]
                            )
                            for seed in SEEDS
                        },
                        {
                            seed: float(
                                by_key[(right_model, seed, second_train, second_test)]["metrics"][
                                    "auprc"
                                ]
                            )
                            for seed in SEEDS
                        },
                    )
                    category = classify_reversal_categories(
                        first_summary["estimate"],
                        second_summary["estimate"],
                        (first_summary["lower_95"], first_summary["upper_95"]),
                        (second_summary["lower_95"], second_summary["upper_95"]),
                        first_seed,
                        second_seed,
                    )
                    reversal_events.append(
                        {
                            "left_model": left_model,
                            "right_model": right_model,
                            "condition_a": {
                                "train_environment": first_train,
                                "test_environment": first_test,
                            },
                            "condition_b": {
                                "train_environment": second_train,
                                "test_environment": second_test,
                            },
                            "delta_a": first_summary,
                            "delta_b": second_summary,
                            "seed_stability_a": first_seed,
                            "seed_stability_b": second_seed,
                            **category,
                        }
                    )

    categories = Counter(event["category"] for event in reversal_events)
    eligible_reversal_comparisons = len(reversal_events)
    reversal_summary = {
        "eligible_comparisons": eligible_reversal_comparisons,
        "category_counts": dict(sorted(categories.items())),
        "descriptive_rate": sum(event["descriptive_reversal"] for event in reversal_events)
        / eligible_reversal_comparisons,
        "evaluation_uncertainty_supported_rate": sum(
            event["evaluation_uncertainty_supported_reversal"] for event in reversal_events
        )
        / eligible_reversal_comparisons,
        "seed_stable_rate": sum(event["seed_stable_reversal"] for event in reversal_events)
        / eligible_reversal_comparisons,
        "evaluation_and_seed_supported_rate": sum(
            event["evaluation_and_seed_supported_reversal"] for event in reversal_events
        )
        / eligible_reversal_comparisons,
        "scope": "complete train-by-test condition surface; seed-averaged point estimates",
    }
    (args.output_dir / "reversal_events.json").write_text(
        json.dumps(reversal_events, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    core_matrices: dict[str, np.ndarray] = {}
    figure_manifest: list[dict[str, Any]] = []
    for model in MODELS:
        matrix = np.asarray(
            [
                [metric_means[(model, train, test, "auprc")] for test in ENVIRONMENTS]
                for train in ENVIRONMENTS
            ]
        )
        core_matrices[model] = matrix
        base = figure_dir / f"core_auprc_heatmap__{model}"
        paths = _heatmap(
            matrix,
            list(ENVIRONMENTS),
            list(ENVIRONMENTS),
            f"E003 mean AUPRC: {_model_label(model)}",
            base,
        )
        source = figure_dir / f"figure_data__core_auprc_heatmap__{model}.csv"
        _write_csv(
            source,
            [
                {
                    "model": model,
                    "train_environment": train,
                    "test_environment": test,
                    "mean_auprc": metric_means[(model, train, test, "auprc")],
                }
                for train in ENVIRONMENTS
                for test in ENVIRONMENTS
            ],
        )
        figure_manifest.append(
            {
                "figure": base.name,
                "files": [str(path) for path in paths],
                "source_data": str(source),
            }
        )

    # Main Figure 2: one common-scale, value-annotated panel per model.  The
    # marked test columns are retained in the six-environment roster but are
    # visually identified as supplementary method/sampler sensitivities.
    common_vmin = float(min(matrix.min() for matrix in core_matrices.values()))
    common_vmax = float(max(matrix.max() for matrix in core_matrices.values()))
    display_columns = [
        f"{_environment_label(environment)}"
        f"{'*' if environment in SUPPLEMENTARY_TEST_ENVIRONMENTS else ''}"
        for environment in ENVIRONMENTS
    ]
    figure, axes = plt.subplots(1, len(MODELS), figsize=(25, 6), constrained_layout=True)
    image = None
    for axis, model in zip(np.atleast_1d(axes), MODELS, strict=True):
        image = axis.imshow(
            core_matrices[model],
            aspect="auto",
            cmap="viridis",
            vmin=common_vmin,
            vmax=common_vmax,
        )
        axis.set_title(_model_label(model))
        axis.set_xticks(range(len(ENVIRONMENTS)), display_columns, rotation=45, ha="right")
        axis.set_yticks(
            range(len(ENVIRONMENTS)), [_environment_label(value) for value in ENVIRONMENTS]
        )
        axis.set_xlabel("test negative environment (* supplementary)")
        for row_index in range(len(ENVIRONMENTS)):
            for column_index in range(len(ENVIRONMENTS)):
                axis.text(
                    column_index,
                    row_index,
                    f"{core_matrices[model][row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color="white"
                    if core_matrices[model][row_index, column_index]
                    < (common_vmin + common_vmax) / 2
                    else "black",
                    fontsize=7,
                )
    if image is not None:
        figure.colorbar(image, ax=np.atleast_1d(axes).tolist(), shrink=0.75, label="mean AUPRC")
    figure.suptitle("E003 core train × test response surfaces (three-seed mean AUPRC)")
    figure2_base = figure_dir / "figure_2_core_response_surfaces"
    figure2_paths = _save_figure(figure, figure2_base)
    figure2_source = figure_dir / "figure_data__figure_2_core_response_surfaces.csv"
    _write_csv(
        figure2_source,
        [
            {
                "model": model,
                "train_environment": train,
                "test_environment": test,
                "test_role": (
                    "supplementary" if test in SUPPLEMENTARY_TEST_ENVIRONMENTS else "core"
                ),
                "mean_auprc": metric_means[(model, train, test, "auprc")],
            }
            for model in MODELS
            for train in ENVIRONMENTS
            for test in ENVIRONMENTS
        ],
    )
    figure_manifest.append(
        {
            "figure": figure2_base.name,
            "files": [str(path) for path in figure2_paths],
            "source_data": str(figure2_source),
            "common_color_scale": {"vmin": common_vmin, "vmax": common_vmax},
        }
    )
    model_concordance_tau = np.asarray(
        [
            [
                kendall_tau(
                    [metric_means[(model, train, test, "auprc")] for train, test in conditions],
                    [metric_means[(other, train, test, "auprc")] for train, test in conditions],
                )
                for other in MODELS
            ]
            for model in MODELS
        ]
    )
    condition_tau_matrix = np.asarray(
        [
            [
                kendall_tau(
                    [metric_means[(model, *first_condition, "auprc")] for model in MODELS],
                    [metric_means[(model, *second_condition, "auprc")] for model in MODELS],
                )
                for second_condition in conditions
            ]
            for first_condition in conditions
        ]
    )
    condition_spearman_matrix = np.asarray(
        [
            [
                _spearman(
                    [metric_means[(model, *first_condition, "auprc")] for model in MODELS],
                    [metric_means[(model, *second_condition, "auprc")] for model in MODELS],
                )
                for second_condition in conditions
            ]
            for first_condition in conditions
        ]
    )
    condition_labels = [_condition_key(train, test) for train, test in conditions]
    tau_base = figure_dir / "ranking_condition_kendall_tau"
    tau_paths = _heatmap(
        condition_tau_matrix,
        condition_labels,
        condition_labels,
        "E003 Kendall tau between condition-level model rankings",
        tau_base,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    tau_source = figure_dir / "figure_data__ranking_condition_kendall_tau.csv"
    _write_csv(
        tau_source,
        [
            {
                "condition_a": condition_labels[i],
                "condition_b": condition_labels[j],
                "kendall_tau": float(condition_tau_matrix[i, j]),
                "spearman_rho": float(condition_spearman_matrix[i, j]),
            }
            for i in range(len(conditions))
            for j in range(len(conditions))
        ],
    )
    figure_manifest.append(
        {
            "figure": tau_base.name,
            "files": [str(path) for path in tau_paths],
            "source_data": str(tau_source),
        }
    )

    # Complete condition-level ranking estimands.  Every condition uses the
    # same three-seed means as the reversal analysis; bootstrap rank
    # probabilities are based on the shared resamples for that test panel.
    winner_frequency: Counter[str] = Counter()
    condition_values = {
        _condition_key(train_environment, test_environment): np.asarray(
            [
                metric_means[(model, train_environment, test_environment, "auprc")]
                for model in MODELS
            ],
            dtype=np.float64,
        )
        for train_environment, test_environment in conditions
    }
    ranking_pairwise_summary = _ranking_comparison_summary(condition_values, MODELS)
    rank_rows: list[dict[str, Any]] = []
    rank_probability_by_condition: dict[str, dict[str, list[float]]] = {}
    margin_by_condition: dict[str, dict[str, Any]] = {}
    seed_rank_rows: list[dict[str, Any]] = []
    for train_environment, test_environment in conditions:
        condition = _condition_key(train_environment, test_environment)
        values = np.asarray(
            [
                metric_means[(model, train_environment, test_environment, "auprc")]
                for model in MODELS
            ],
            dtype=np.float64,
        )
        point_order = np.argsort(-values, kind="mergesort")
        winner = MODELS[int(point_order[0])]
        runner_up = MODELS[int(point_order[1])]
        winner_frequency[winner] += 1
        bootstrap_values = np.asarray(
            [bootstrap_ap[(model, train_environment, test_environment)] for model in MODELS]
        )
        bootstrap_order = np.argsort(-bootstrap_values, axis=0, kind="mergesort")
        winner_index = int(point_order[0])
        runner_index = int(point_order[1])
        winner_bootstrap = bootstrap_values[winner_index]
        runner_bootstrap = bootstrap_values[runner_index]
        top_two_delta = winner_bootstrap - runner_bootstrap
        top_two_interval = _interval(
            top_two_delta,
            float(values[winner_index] - values[runner_index]),
            args.bootstrap_seed + ENVIRONMENTS.index(test_environment),
            args.bootstrap_replicates,
        )
        rank_probabilities: dict[str, list[float]] = {}
        rank_array = np.empty_like(bootstrap_order, dtype=np.int64)
        for rank_index in range(len(MODELS)):
            rank_array[bootstrap_order[rank_index], np.arange(args.bootstrap_replicates)] = (
                rank_index + 1
            )
        for model_index, model in enumerate(MODELS):
            rank_probabilities[model] = [
                float(np.mean(rank_array[model_index] == rank_index + 1))
                for rank_index in range(len(MODELS))
            ]
        rank_probability_by_condition[condition] = rank_probabilities
        margin_by_condition[condition] = top_two_interval
        rank_order = [MODELS[int(index)] for index in point_order]
        rank_rows.append(
            {
                "condition": condition,
                "train_environment": train_environment,
                "test_environment": test_environment,
                "winner": winner,
                "runner_up": runner_up,
                "rank_order": ">".join(rank_order),
                "rank_positions": {
                    model: int(rank_order.index(model) + 1) for model in MODELS
                },
                "top1_top2_margin": float(values[winner_index] - values[runner_index]),
                "top1_top2_lower_95": top_two_interval["lower_95"],
                "top1_top2_upper_95": top_two_interval["upper_95"],
                "top1_top2_interval_contains_zero": top_two_interval["lower_95"]
                <= 0
                <= top_two_interval["upper_95"],
                "rank_entropy_bits": float(
                    np.mean([_entropy(values) for values in rank_probabilities.values()])
                ),
                "seed_winner_values": {},
            }
        )
        # Store seed-specific orders separately; this is a training-seed
        # sensitivity description and is never used as the primary ranking.
        seed_winners: dict[str, str] = {}
        for seed in SEEDS:
            seed_values = np.asarray(
                [
                    float(
                        by_key[(model, seed, train_environment, test_environment)]["metrics"][
                            "auprc"
                        ]
                    )
                    for model in MODELS
                ],
                dtype=np.float64,
            )
            seed_order = np.argsort(-seed_values, kind="mergesort")
            seed_winners[str(seed)] = MODELS[int(seed_order[0])]
            seed_rank_rows.append(
                {
                    "condition": condition,
                    "train_environment": train_environment,
                    "test_environment": test_environment,
                    "seed": seed,
                    "winner": MODELS[int(seed_order[0])],
                    "rank_order": ">".join(MODELS[int(index)] for index in seed_order),
                }
            )
        rank_rows[-1]["seed_winner_values"] = seed_winners

    # Store the mean per-model rank-position entropy. The distribution of
    # winners across all conditions is reported separately as winner entropy.
    for row in rank_rows:
        probabilities = rank_probability_by_condition[row["condition"]]
        row["rank_entropy_bits"] = float(
            np.mean([_entropy(values) for values in probabilities.values()])
        )
    _write_csv(
        args.output_dir / "ranking_condition_summary.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "seed_winner_values"},
                "seed_winners": json.dumps(row["seed_winner_values"], sort_keys=True),
            }
            for row in rank_rows
        ],
    )
    _write_csv(args.output_dir / "ranking_seed_sensitivity.csv", seed_rank_rows)
    winner_source = figure_dir / "figure_data__condition_rankings.json"
    winner_source.write_text(
        json.dumps(
            {
                "conditions": rank_rows,
                "rank_probability_by_condition": rank_probability_by_condition,
                "pairwise_comparison_summary": ranking_pairwise_summary,
                "winner_frequency": dict(sorted(winner_frequency.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Figure 3: complete winner map, a matched-condition bump chart, top-two
    # margin uncertainty, and reversal rates.  The map uses a categorical code
    # rather than a continuous color scale, so small performance differences
    # cannot look like large effects.
    model_codes = {model: index for index, model in enumerate(MODELS)}
    winner_matrix = np.asarray(
        [
            [model_codes[row["winner"]] for row in rank_rows if row["train_environment"] == train]
            for train in ENVIRONMENTS
        ]
    )
    margin_matrix = np.asarray(
        [
            [
                next(
                    row["top1_top2_margin"]
                    for row in rank_rows
                    if row["train_environment"] == train and row["test_environment"] == test
                )
                for test in ENVIRONMENTS
            ]
            for train in ENVIRONMENTS
        ]
    )
    uncertain_matrix = np.asarray(
        [
            [
                next(
                    bool(row["top1_top2_interval_contains_zero"])
                    for row in rank_rows
                    if row["train_environment"] == train
                    and row["test_environment"] == test
                )
                for test in ENVIRONMENTS
            ]
            for train in ENVIRONMENTS
        ],
        dtype=bool,
    )
    figure, axes = plt.subplots(2, 2, figsize=(19, 12), constrained_layout=True)
    axes = np.asarray(axes)
    axes[0, 0].imshow(
        winner_matrix, cmap="tab10", vmin=-0.5, vmax=len(MODELS) - 0.5
    )
    axes[0, 0].set_title("A. winner by condition")
    axes[0, 0].set_xticks(
        range(len(ENVIRONMENTS)),
        [_environment_label(value) for value in ENVIRONMENTS],
        rotation=45,
        ha="right",
    )
    axes[0, 0].set_yticks(
        range(len(ENVIRONMENTS)), [_environment_label(value) for value in ENVIRONMENTS]
    )
    for row_index in range(len(ENVIRONMENTS)):
        for column_index in range(len(ENVIRONMENTS)):
            axes[0, 0].text(
                column_index,
                row_index,
                str(int(winner_matrix[row_index, column_index]) + 1),
                ha="center",
                va="center",
                color="white",
            )
    axes[0, 0].set_xlabel("1–4 = model order in legend")
    winner_legend = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=plt.get_cmap("tab10")(index),
            markeredgecolor="none",
            label=f"{index + 1}: {_model_label(model)}",
        )
        for index, model in enumerate(MODELS)
    ]
    axes[0, 0].legend(handles=winner_legend, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

    # Matched-condition bump chart: each line is a model's mean rank across
    # the six test environments for a fixed train environment.
    train_positions = np.arange(len(ENVIRONMENTS))
    for model in MODELS:
        means = [
            float(
                np.mean(
                    [
                        row["rank_positions"][model]
                        for row in rank_rows
                        if row["train_environment"] == train
                    ]
                )
            )
            for train in ENVIRONMENTS
        ]
        axes[0, 1].plot(
            train_positions,
            means,
            marker="o",
            linewidth=2,
            label=_model_label(model),
        )
    axes[0, 1].set_title("B. matched-condition mean rank")
    axes[0, 1].set_xticks(
        train_positions,
        [_environment_label(value) for value in ENVIRONMENTS],
        rotation=45,
        ha="right",
    )
    axes[0, 1].set_yticks(range(1, len(MODELS) + 1))
    axes[0, 1].set_ylim(len(MODELS) + 0.25, 0.75)
    axes[0, 1].set_ylabel("rank (1 = highest AUPRC)")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].imshow(margin_matrix, cmap="magma", aspect="auto")
    axes[1, 0].set_title("C. top1 − top2 mean AUPRC")
    axes[1, 0].set_xticks(
        range(len(ENVIRONMENTS)),
        [_environment_label(value) for value in ENVIRONMENTS],
        rotation=45,
        ha="right",
    )
    axes[1, 0].set_yticks(
        range(len(ENVIRONMENTS)), [_environment_label(value) for value in ENVIRONMENTS]
    )
    for row_index in range(len(ENVIRONMENTS)):
        for column_index in range(len(ENVIRONMENTS)):
            if uncertain_matrix[row_index, column_index]:
                axes[1, 0].text(
                    column_index,
                    row_index,
                    "*",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=13,
                    fontweight="bold",
                )
    axes[1, 0].set_xlabel("* bootstrap interval contains zero")
    axes[1, 1].bar(
        [
            "descriptive",
            "eval-supported",
            "seed-stable",
            "eval + seed",
        ],
        [
            reversal_summary["descriptive_rate"],
            reversal_summary["evaluation_uncertainty_supported_rate"],
            reversal_summary["seed_stable_rate"],
            reversal_summary["evaluation_and_seed_supported_rate"],
        ],
        color=["#999999", "#377eb8", "#4daf4a", "#e41a1c"],
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("rate among eligible condition comparisons")
    axes[1, 1].set_title("D. pairwise reversal categories")
    axes[1, 1].tick_params(axis="x", rotation=45)
    figure.suptitle("E003 ranking stability: three-seed estimand")
    figure3_base = figure_dir / "figure_3_ranking_stability"
    figure3_paths = _save_figure(figure, figure3_base)
    figure3_source = figure_dir / "figure_data__figure_3_ranking_stability.json"
    figure3_source.write_text(
        json.dumps(
            {
                "models": list(MODELS),
                "environments": list(ENVIRONMENTS),
                "rank_rows": rank_rows,
                "reversal_summary": reversal_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_manifest.append(
        {
            "figure": figure3_base.name,
            "files": [str(path) for path in figure3_paths],
            "source_data": str(figure3_source),
        }
    )

    overall_mean = float(
        np.mean([value for key, value in metric_means.items() if key[-1] == "auprc"])
    )
    train_means = {
        train: float(
            np.mean(
                [
                    metric_means[(model, train, test, "auprc")]
                    for model in MODELS
                    for test in ENVIRONMENTS
                ]
            )
        )
        for train in ENVIRONMENTS
    }
    test_means = {
        test: float(
            np.mean(
                [
                    metric_means[(model, train, test, "auprc")]
                    for model in MODELS
                    for train in ENVIRONMENTS
                ]
            )
        )
        for test in ENVIRONMENTS
    }
    environment_effect_rows: list[dict[str, Any]] = []
    environment_effect_summary: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        train_deltas = []
        test_deltas = []
        train_auroc_deltas = []
        test_auroc_deltas = []
        for train in ENVIRONMENTS:
            for test in ENVIRONMENTS:
                train_delta = (
                    metric_means[(model, train, test, "auprc")]
                    - metric_means[(model, "uniform_partner", test, "auprc")]
                )
                test_delta = (
                    metric_means[(model, train, test, "auprc")]
                    - metric_means[(model, train, "uniform_partner", "auprc")]
                )
                train_auroc_delta = (
                    metric_means[(model, train, test, "auroc")]
                    - metric_means[(model, "uniform_partner", test, "auroc")]
                )
                test_auroc_delta = (
                    metric_means[(model, train, test, "auroc")]
                    - metric_means[(model, train, "uniform_partner", "auroc")]
                )
                train_deltas.append(train_delta)
                test_deltas.append(test_delta)
                train_auroc_deltas.append(train_auroc_delta)
                test_auroc_deltas.append(test_auroc_delta)
                environment_effect_rows.extend(
                    [
                        {
                            "model": model,
                            "axis": "train",
                            "changed_environment": train,
                            "held_environment": test,
                            "reference_environment": "uniform_partner",
                            "delta_auprc": float(train_delta),
                            "delta_auroc": float(train_auroc_delta),
                        },
                        {
                            "model": model,
                            "axis": "test",
                            "changed_environment": test,
                            "held_environment": train,
                            "reference_environment": "uniform_partner",
                            "delta_auprc": float(test_delta),
                            "delta_auroc": float(test_auroc_delta),
                        },
                    ]
                )
        environment_effect_summary[model] = {
            "train": {
                "minimum": float(min(train_deltas)),
                "maximum": float(max(train_deltas)),
                "maximum_absolute": float(max(abs(value) for value in train_deltas)),
                "auroc": {
                    "minimum": float(min(train_auroc_deltas)),
                    "maximum": float(max(train_auroc_deltas)),
                    "maximum_absolute": float(max(abs(value) for value in train_auroc_deltas)),
                },
            },
            "test": {
                "minimum": float(min(test_deltas)),
                "maximum": float(max(test_deltas)),
                "maximum_absolute": float(max(abs(value) for value in test_deltas)),
                "auroc": {
                    "minimum": float(min(test_auroc_deltas)),
                    "maximum": float(max(test_auroc_deltas)),
                    "maximum_absolute": float(max(abs(value) for value in test_auroc_deltas)),
                },
            },
        }
    _write_csv(args.output_dir / "environment_effects.csv", environment_effect_rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, effect_axis in zip(axes, ("train", "test"), strict=True):
        data = [
            [
                row["delta_auprc"]
                for row in environment_effect_rows
                if row["model"] == model and row["axis"] == effect_axis
            ]
            for model in MODELS
        ]
        axis.boxplot(data, labels=[_model_label(value) for value in MODELS], showfliers=False)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{effect_axis} environment vs uniform reference")
        axis.set_ylabel("AUPRC delta")
        axis.tick_params(axis="x", rotation=65)
    figure.suptitle("E003 environment effects relative to uniform_partner")
    effect_base = figure_dir / "figure_4_environment_effects"
    effect_paths = _save_figure(figure, effect_base)
    effect_source = figure_dir / "figure_data__figure_4_environment_effects.csv"
    _write_csv(effect_source, environment_effect_rows)
    figure_manifest.append(
        {
            "figure": effect_base.name,
            "files": [str(path) for path in effect_paths],
            "source_data": str(effect_source),
        }
    )
    interaction = np.asarray(
        [
            [
                float(np.mean([metric_means[(model, train, test, "auprc")] for model in MODELS]))
                - train_means[train]
                - test_means[test]
                + overall_mean
                for test in ENVIRONMENTS
            ]
            for train in ENVIRONMENTS
        ]
    )
    interaction_base = figure_dir / "train_test_interaction_residual"
    interaction_paths = _heatmap(
        interaction,
        list(ENVIRONMENTS),
        list(ENVIRONMENTS),
        "E003 train × test interaction residual",
        interaction_base,
        cmap="coolwarm",
    )
    interaction_source = figure_dir / "figure_data__train_test_interaction_residual.csv"
    _write_csv(
        interaction_source,
        [
            {
                "train_environment": train,
                "test_environment": test,
                "interaction_residual": float(interaction[i, j]),
            }
            for i, train in enumerate(ENVIRONMENTS)
            for j, test in enumerate(ENVIRONMENTS)
        ],
    )
    figure_manifest.append(
        {
            "figure": interaction_base.name,
            "files": [str(path) for path in interaction_paths],
            "source_data": str(interaction_source),
        }
    )

    external_rows = _read_external_rows(external)
    _write_csv(args.output_dir / "external_metric_summary.csv", external_rows)
    intact_sensitivity_rows: list[dict[str, Any]] = []
    for bundle in external["bundles"]:
        surface = bundle["surfaces"].get("intact__measured")
        if surface is None:
            continue
        payload = surface["metrics"].get("balanced_prevalence_sensitivity")
        if not payload:
            raise SystemExit("IntAct prevalence sensitivity is missing from an external bundle")
        for replicate in payload["replicates"]:
            intact_sensitivity_rows.append(
                {
                    "model": bundle["model"],
                    "seed": int(bundle["seed"]),
                    "train_environment": bundle["train_environment"],
                    "sensitivity_seed": int(replicate["seed"]),
                    "auprc": float(replicate["metrics"]["auprc"]),
                    "auroc": float(replicate["metrics"]["auroc"]),
                    "prevalence": float(replicate["metrics"]["positive_prevalence"]),
                }
            )
    if not intact_sensitivity_rows:
        raise SystemExit("No IntAct prevalence sensitivity rows found")
    _write_csv(
        args.output_dir / "intact_posthoc_prevalence_sensitivity.csv", intact_sensitivity_rows
    )
    external_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in external_rows:
        external_grouped[
            (row["model"], f"{row['source']}__{row['surface']}", "normalized_excess_ap")
        ].append(row["normalized_excess_ap"])
    external_summary = [
        {
            "model": model,
            "surface": surface,
            "metric": metric,
            "mean": float(np.mean(values)),
            "sd_across_units": float(np.std(values, ddof=1)),
            "n": len(values),
        }
        for (model, surface, metric), values in sorted(external_grouped.items())
    ]
    _write_csv(args.output_dir / "external_surface_summary.csv", external_summary)
    surface_names = sorted({row["surface"] for row in external_summary})
    transfer_matrix = np.asarray(
        [
            [
                next(
                    row["mean"]
                    for row in external_summary
                    if row["model"] == model and row["surface"] == surface
                )
                for surface in surface_names
            ]
            for model in MODELS
        ]
    )
    transfer_base = figure_dir / "external_normalized_excess_ap"
    transfer_paths = _heatmap(
        transfer_matrix,
        list(MODELS),
        surface_names,
        "E003 external normalized excess AP",
        transfer_base,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    transfer_source = figure_dir / "figure_data__external_normalized_excess_ap.csv"
    _write_csv(
        transfer_source,
        [
            {
                "model": model,
                "surface": surface,
                "normalized_excess_ap": float(transfer_matrix[i, j]),
            }
            for i, model in enumerate(MODELS)
            for j, surface in enumerate(surface_names)
        ],
    )
    figure_manifest.append(
        {
            "figure": transfer_base.name,
            "files": [str(path) for path in transfer_paths],
            "source_data": str(transfer_source),
        }
    )

    # Figure 5 keeps the measured surfaces visibly separate from synthetic
    # source-positive/synthetic-negative surfaces and makes the Vidal
    # train-environment transfer estimand explicit.
    external_all_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in external_rows:
        for metric in ("auprc", "auroc", "prevalence", "normalized_excess_ap"):
            external_all_grouped[
                (row["model"], f"{row['source']}__{row['surface']}", metric)
            ].append(float(row[metric]))
    external_all_summary = [
        {
            "model": model,
            "surface": surface,
            "metric": metric,
            "mean": float(np.mean(values)),
            "sd_across_units": float(np.std(values, ddof=1)),
            "n": len(values),
        }
        for (model, surface, metric), values in sorted(external_all_grouped.items())
    ]
    vidal_rows = [
        row
        for row in external_rows
        if row["source"] == "vidal" and row["surface"] == "measured"
    ]
    intact_rows = [
        row
        for row in external_rows
        if row["source"] == "intact" and row["surface"] == "measured"
    ]
    if not vidal_rows or not intact_rows:
        raise SystemExit("Measured Vidal and IntAct surfaces are required for Figure 5")

    # Figure 5 keeps raw AUPRC as the primary transfer metric.  Prevalence
    # baselines are shown as dotted reference lines; normalized and calibration
    # diagnostics remain available in the adjacent source files/supplements.
    figure, axes = plt.subplots(2, 3, figsize=(23, 12), constrained_layout=True)
    axes = np.asarray(axes)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(MODELS)))
    positions = np.arange(len(MODELS), dtype=np.float64)

    # A: the same final checkpoint is evaluated on Vidal while only its Gold
    # training-negative environment changes.  Solid/dashed encodings separate
    # AUPRC and AUROC without collapsing either metric.
    for model_index, model in enumerate(MODELS):
        auprc_values = [
            float(
                np.mean(
                    [
                        row["auprc"]
                        for row in vidal_rows
                        if row["model"] == model and row["train_environment"] == train
                    ]
                )
            )
            for train in ENVIRONMENTS
        ]
        axes[0, 0].plot(
            range(len(ENVIRONMENTS)),
            auprc_values,
            marker="o",
            color=colors[model_index],
            label=_model_label(model),
        )
    vidal_prevalence = float(np.mean([row["prevalence"] for row in vidal_rows]))
    axes[0, 0].axhline(
        vidal_prevalence,
        color="black",
        linestyle=":",
        linewidth=1,
        label=f"random AP = prevalence ({vidal_prevalence:.3f})",
    )
    axes[0, 0].set_xticks(
        range(len(ENVIRONMENTS)),
        [_environment_label(value) for value in ENVIRONMENTS],
        rotation=45,
        ha="right",
    )
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("A. Vidal transfer: raw AUPRC")
    axes[0, 0].set_ylabel("AUPRC")
    axes[0, 0].legend(fontsize=6, ncol=2)

    # B: measured transfer on the raw AUPRC scale; the dotted baselines in
    # panels A and E make prevalence effects immediately visible.
    measured_surfaces = ["vidal__measured", "intact__measured"]
    measured_matrix = np.asarray(
        [
            [
                next(
                    row["mean"]
                    for row in external_all_summary
                    if row["model"] == model
                    and row["surface"] == surface
                    and row["metric"] == "auprc"
                )
                for surface in measured_surfaces
            ]
            for model in MODELS
        ],
        dtype=np.float64,
    )
    image = axes[0, 1].imshow(measured_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0, 1].set_xticks(range(2), ("Vidal", "IntAct"))
    axes[0, 1].set_yticks(range(len(MODELS)), [_model_label(value) for value in MODELS])
    axes[0, 1].set_title("B. measured transfer: raw AUPRC")
    axes[0, 1].set_xlabel("raw AUPRC")
    for row_index in range(len(MODELS)):
        for column_index in range(2):
            axes[0, 1].text(
                column_index,
                row_index,
                f"{measured_matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white" if measured_matrix[row_index, column_index] < 0.55 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[0, 1], shrink=0.8, label="raw AUPRC")

    # C: the corrected BioGRID/STRING synthetic surfaces remain a separate
    # estimand from the measured assays, shown on the same raw-AUPRC scale.
    synthetic_surfaces = [
        surface for surface in surface_names if surface.startswith(("biogrid__", "string__"))
    ]
    synthetic_matrix = np.asarray(
        [
            [
                next(
                    row["mean"]
                    for row in external_all_summary
                    if row["model"] == model
                    and row["surface"] == surface
                    and row["metric"] == "auprc"
                )
                for surface in synthetic_surfaces
            ]
            for model in MODELS
        ],
        dtype=np.float64,
    )
    image = axes[0, 2].imshow(synthetic_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0, 2].set_xticks(
        range(len(synthetic_surfaces)),
        [surface.replace("__", "\\n") for surface in synthetic_surfaces],
        rotation=45,
        ha="right",
    )
    axes[0, 2].set_yticks(range(len(MODELS)), [_model_label(value) for value in MODELS])
    axes[0, 2].set_title("C. synthetic transfer: raw AUPRC")
    axes[0, 2].set_xlabel("raw AUPRC")
    figure.colorbar(image, ax=axes[0, 2], shrink=0.8, label="raw AUPRC")
    for row_index in range(len(MODELS)):
        for column_index in range(len(synthetic_surfaces)):
            axes[0, 2].text(
                column_index,
                row_index,
                f"{synthetic_matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white" if synthetic_matrix[row_index, column_index] < 0.55 else "black",
                fontsize=8,
            )

    # D: internal Gold performance at the uniform test surface versus Vidal
    # transfer, by Gold training environment.  This makes internal-to-assay
    # alignment visible instead of implying that a high Gold score transfers.
    for model_index, model in enumerate(MODELS):
        gold_values = [
            metric_means[(model, train, "uniform_partner", "auprc")]
            for train in ENVIRONMENTS
        ]
        vidal_values = [
            float(
                np.mean(
                    [
                        row["auprc"]
                        for row in vidal_rows
                        if row["model"] == model and row["train_environment"] == train
                    ]
                )
            )
            for train in ENVIRONMENTS
        ]
        axes[1, 0].plot(
            gold_values,
            vidal_values,
            marker="o",
            color=colors[model_index],
            label=_model_label(model),
        )
        for point_index, train in enumerate(ENVIRONMENTS):
            axes[1, 0].annotate(
                _environment_label(train),
                (gold_values[point_index], vidal_values[point_index]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axes[1, 0].plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.8)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Gold uniform-test AUPRC")
    axes[1, 0].set_ylabel("Vidal AUPRC")
    axes[1, 0].set_title("D. internal Gold versus Vidal transfer")
    axes[1, 0].legend(fontsize=7)

    # IntAct full-cohort diagnostic: raw AUPRC against the actual evaluation
    # prevalence, with the random constant-prevalence model as a dotted line.
    intact_auprc = [
        float(np.mean([row["auprc"] for row in intact_rows if row["model"] == model]))
        for model in MODELS
    ]
    intact_prevalence = float(np.mean([row["prevalence"] for row in intact_rows]))
    axes[1, 1].bar(positions, intact_auprc, width=0.55, label="raw AUPRC")
    axes[1, 1].axhline(
        intact_prevalence,
        color="black",
        linestyle=":",
        linewidth=1,
        label=f"random AP = prevalence ({intact_prevalence:.4f})",
    )
    axes[1, 1].set_xticks(
        positions, [_model_label(value) for value in MODELS], rotation=65, ha="right"
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("E. IntAct full cohort: raw AUPRC")
    axes[1, 1].legend(fontsize=7)

    measured_skill_rows = [row for row in external_rows if row["surface"] == "measured"]
    skill_values = np.asarray(
        [
            [
                [
                    float(
                        np.mean(
                            [
                                row[skill]
                                for row in measured_skill_rows
                                if row["model"] == model and row["source"] == source
                            ]
                        )
                    )
                    for skill in (
                        "brier_skill_vs_prevalence_null",
                        "nll_skill_vs_prevalence_null",
                    )
                ]
                for source in ("vidal", "intact")
            ]
            for model in MODELS
        ],
        dtype=np.float64,
    )
    skill_positions = np.arange(len(MODELS), dtype=np.float64)
    width = 0.18
    for source_index, source in enumerate(("vidal", "intact")):
        for skill_index, skill in enumerate(
            ("brier_skill_vs_prevalence_null", "nll_skill_vs_prevalence_null")
        ):
            offset = (source_index * 2 + skill_index - 1.5) * width
            axes[1, 2].bar(
                skill_positions + offset,
                skill_values[:, source_index, skill_index],
                width=width,
                label=f"{source} {skill.split('_')[0].upper()}",
            )
    axes[1, 2].axhline(0, color="black", linewidth=0.8)
    axes[1, 2].set_xticks(
        skill_positions, [_model_label(value) for value in MODELS], rotation=65, ha="right"
    )
    axes[1, 2].set_ylabel("skill versus constant pi predictor")
    axes[1, 2].set_title("F. supplementary prevalence-null skill")
    axes[1, 2].legend(fontsize=6, ncol=2)
    figure.suptitle("E003 external transfer: measured assays and synthetic surfaces")
    figure5_base = figure_dir / "figure_5_external_transfer"
    figure5_paths = _save_figure(figure, figure5_base)
    figure5_source = figure_dir / "figure_data__figure_5_external_transfer.csv"
    figure5_rows: list[dict[str, Any]] = []
    for row in external_rows:
        for metric in (
            "auprc",
            "auroc",
            "prevalence",
            "auprc_minus_prevalence",
            "normalized_excess_ap",
            "ap_lift",
            "prevalence_null_probability",
            "prevalence_null_brier",
            "prevalence_null_nll",
            "brier_skill_vs_prevalence_null",
            "nll_skill_vs_prevalence_null",
        ):
            figure5_rows.append(
                {
                    "scope": "external",
                    "model": row["model"],
                    "seed": row["seed"],
                    "train_environment": row["train_environment"],
                    "test_environment": "",
                    "source": row["source"],
                    "surface": row["surface"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    for model in MODELS:
        for seed in SEEDS:
            for train in ENVIRONMENTS:
                figure5_rows.append(
                    {
                        "scope": "gold_core",
                        "model": model,
                        "seed": seed,
                        "train_environment": train,
                        "test_environment": "uniform_partner",
                        "source": "gold",
                        "surface": "uniform_partner",
                        "metric": "auprc",
                        "value": by_key[(model, seed, train, "uniform_partner")]["metrics"][
                            "auprc"
                        ],
                    }
                )
    _write_csv(figure5_source, figure5_rows)
    figure_manifest.append(
        {
            "figure": figure5_base.name,
            "files": [str(path) for path in figure5_paths],
            "source_data": str(figure5_source),
        }
    )

    # Supplementary AUROC response surfaces use the same complete three-seed
    # estimand and common scale as the primary AUPRC surfaces.
    auroc_matrices = {
        model: np.asarray(
            [
                [metric_means[(model, train, test, "auroc")] for test in ENVIRONMENTS]
                for train in ENVIRONMENTS
            ],
            dtype=np.float64,
        )
        for model in MODELS
    }
    auroc_vmin = float(min(matrix.min() for matrix in auroc_matrices.values()))
    auroc_vmax = float(max(matrix.max() for matrix in auroc_matrices.values()))
    figure, axes = plt.subplots(1, len(MODELS), figsize=(25, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, MODELS, strict=True):
        image = axis.imshow(
            auroc_matrices[model],
            cmap="viridis",
            vmin=auroc_vmin,
            vmax=auroc_vmax,
            aspect="auto",
        )
        axis.set_title(_model_label(model))
        axis.set_xticks(
            range(len(ENVIRONMENTS)),
            [_environment_label(value) for value in ENVIRONMENTS],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(
            range(len(ENVIRONMENTS)), [_environment_label(value) for value in ENVIRONMENTS]
        )
        for row_index in range(len(ENVIRONMENTS)):
            for column_index in range(len(ENVIRONMENTS)):
                axis.text(
                    column_index,
                    row_index,
                    f"{auroc_matrices[model][row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if auroc_matrices[model][row_index, column_index]
                        < auroc_vmin + 0.55 * (auroc_vmax - auroc_vmin)
                        else "black"
                    ),
                    fontsize=7,
                )
    figure.colorbar(image, ax=axes.tolist(), shrink=0.8, label="mean AUROC")
    figure.suptitle("E003 supplementary AUROC train × test response surfaces")
    auroc_base = figure_dir / "supplement_auroc_heatmaps"
    auroc_paths = _save_figure(figure, auroc_base)
    auroc_source = figure_dir / "figure_data__supplement_auroc_heatmaps.csv"
    _write_csv(
        auroc_source,
        [
            {
                "model": model,
                "train_environment": train,
                "test_environment": test,
                "mean_auroc": float(metric_means[(model, train, test, "auroc")]),
            }
            for model in MODELS
            for train in ENVIRONMENTS
            for test in ENVIRONMENTS
        ],
    )
    figure_manifest.append(
        {
            "figure": auroc_base.name,
            "files": [str(path) for path in auroc_paths],
            "source_data": str(auroc_source),
        }
    )

    # Training-seed sensitivity and calibration are separate descriptive
    # estimands; neither is used to select a model or environment.
    seed_metric_rows = [
        {
            "model": str(cell["model"]),
            "seed": int(cell["seed"]),
            "train_environment": str(cell["train_environment"]),
            "test_environment": str(cell["test_environment"]),
            "auprc": float(cell["metrics"]["auprc"]),
            "auroc": float(cell["metrics"]["auroc"]),
            "ece": float(cell["metrics"]["ece"]),
            "brier": float(cell["metrics"]["brier"]),
            "nll": float(cell["metrics"]["nll"]),
        }
        for cell in cells
    ]
    seed_source = figure_dir / "figure_data__supplement_seed_and_calibration.csv"
    _write_csv(seed_source, seed_metric_rows)
    seed_means = [
        {
            "model": model,
            "seed": seed,
            "mean_auprc": float(
                np.mean(
                    [
                        row["auprc"]
                        for row in seed_metric_rows
                        if row["model"] == model and row["seed"] == seed
                    ]
                )
            ),
            "mean_ece": float(
                np.mean(
                    [
                        row["ece"]
                        for row in seed_metric_rows
                        if row["model"] == model and row["seed"] == seed
                    ]
                )
            ),
            "mean_brier": float(
                np.mean(
                    [
                        row["brier"]
                        for row in seed_metric_rows
                        if row["model"] == model and row["seed"] == seed
                    ]
                )
            ),
            "mean_nll": float(
                np.mean(
                    [
                        row["nll"]
                        for row in seed_metric_rows
                        if row["model"] == model and row["seed"] == seed
                    ]
                )
            ),
        }
        for model in MODELS
        for seed in SEEDS
    ]
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    for model in MODELS:
        values = [
            next(
                item["mean_auprc"]
                for item in seed_means
                if item["model"] == model and item["seed"] == seed
            )
            for seed in SEEDS
        ]
        axes[0].plot(SEEDS, values, marker="o", label=_model_label(model))
    axes[0].set_title("AUPRC by training seed")
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("mean across 36 train × test conditions")
    axes[0].legend(fontsize=7)
    metric_names = ("mean_ece", "mean_brier", "mean_nll")
    metric_labels = ("ECE", "Brier", "NLL")
    positions = np.arange(len(MODELS))
    width = 0.25
    for index, (metric_name, label) in enumerate(zip(metric_names, metric_labels, strict=True)):
        values = [
            float(np.mean([item[metric_name] for item in seed_means if item["model"] == model]))
            for model in MODELS
        ]
        axes[1].bar(positions + (index - 1) * width, values, width=width, label=label)
    axes[1].set_xticks(
        positions, [_model_label(value) for value in MODELS], rotation=65, ha="right"
    )
    axes[1].set_title("Calibration summaries")
    axes[1].set_ylabel("lower is better")
    axes[1].legend(fontsize=8)
    figure.suptitle("E003 supplementary seed sensitivity and calibration")
    seed_base = figure_dir / "supplement_seed_and_calibration"
    seed_paths = _save_figure(figure, seed_base)
    figure_manifest.append(
        {
            "figure": seed_base.name,
            "files": [str(path) for path in seed_paths],
            "source_data": str(seed_source),
        }
    )

    # IntAct post-hoc prevalence sensitivity is kept separate from the natural
    # prevalence headline and uses only the deterministic sensitivity rows.
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.boxplot(
        [
            [row["auprc"] for row in intact_sensitivity_rows if row["model"] == model]
            for model in MODELS
        ],
        labels=[_model_label(value) for value in MODELS],
        showfliers=False,
    )
    axis.set_title("IntAct post-hoc prevalence sensitivity")
    axis.set_ylabel("AUPRC after balanced prevalence diagnostic")
    axis.tick_params(axis="x", rotation=65)
    intact_base = figure_dir / "supplement_intact_prevalence_sensitivity"
    intact_paths = _save_figure(figure, intact_base)
    intact_source = figure_dir / "figure_data__supplement_intact_prevalence_sensitivity.csv"
    _write_csv(intact_source, intact_sensitivity_rows)
    figure_manifest.append(
        {
            "figure": intact_base.name,
            "files": [str(path) for path in intact_paths],
            "source_data": str(intact_source),
        }
    )

    # Sampler-fidelity diagnostics are derived from the frozen test panel and
    # source sampler artifacts, not from model predictions.
    test_samplers = panel_manifest["stages"]["test"]["samplers"]
    bernett_spec = test_samplers["bernett2026_ilp_exact"]
    bernett_audit = bernett_spec["pair_artifact"]["audit"]
    bernett_panel_rows = list(iter_delimited(Path(bernett_spec["panel_path"])))

    def _endpoint_degree_values(rows: list[dict[str, str]], label: str) -> np.ndarray:
        endpoints: Counter[str] = Counter()
        for row in rows:
            if row["interaction_label"] != label:
                continue
            endpoints[row["name1"]] += 1
            endpoints[row["name2"]] += 1
        return np.asarray(list(endpoints.values()), dtype=np.float64)

    bernett_positive_degrees = _endpoint_degree_values(bernett_panel_rows, "1")
    bernett_negative_degrees = _endpoint_degree_values(bernett_panel_rows, "0")
    tppni_manifest = sampler_trace_manifest or panel_manifest
    tppni_test_samplers = tppni_manifest["stages"]["test"]["samplers"]
    panel_tppni_pool = test_samplers["tppni2025_scm_cl3_exact_balanced"]["pool_artifact"]
    trace_tppni_pool = tppni_test_samplers["tppni2025_scm_cl3_exact_balanced"][
        "pool_artifact"
    ]
    if sampler_trace_manifest and (
        panel_tppni_pool.get("sha256") != trace_tppni_pool.get("sha256")
        or not Path(str(trace_tppni_pool.get("path", ""))).is_file()
        or sha256_path(Path(str(trace_tppni_pool["path"])))
        != str(trace_tppni_pool.get("sha256", ""))
    ):
        raise SystemExit(
            "E003 sampler trace manifest does not bind the frozen TPPNI pool"
        )
    tppni_spec = tppni_test_samplers["tppni2025_scm_cl3_exact_balanced"]
    tppni_pool_artifact = tppni_spec["pool_artifact"]
    tppni_diagnostics = tppni_pool_artifact["diagnostics"]
    tppni_pool_path = Path(tppni_pool_artifact["path"])
    tppni_pool_delimiter = _tppni_pool_delimiter(tppni_pool_path)
    tppni_probabilities: list[float] = []
    tppni_l3_counts: list[int] = []
    for row in iter_delimited(tppni_pool_path, delimiter=tppni_pool_delimiter):
        tppni_probabilities.append(float(row["scm_probability"]))
        tppni_l3_counts.append(int(row["l3_count"]))
    probability_array = np.asarray(tppni_probabilities, dtype=np.float64)
    l3_array = np.asarray(tppni_l3_counts, dtype=np.int64)
    tppni_scm = tppni_diagnostics["scm"]
    probability_order_tolerance = float(
        tppni_diagnostics.get("bottom_n_tie_tolerance", 0.0)
    )
    degree_trace = tppni_scm.get("degree_trace")
    if not isinstance(degree_trace, dict):
        raise SystemExit("TPPNI panel lacks the frozen SCM expected-vs-observed degree trace")
    observed_degrees = np.asarray(degree_trace.get("observed", []), dtype=np.float64)
    expected_degrees = np.asarray(degree_trace.get("expected", []), dtype=np.float64)
    degree_ids = degree_trace.get("protein_ids", [])
    if (
        not isinstance(degree_ids, list)
        or len(degree_ids) != int(tppni_scm["protein_count"])
        or observed_degrees.shape != expected_degrees.shape
        or len(observed_degrees) != len(degree_ids)
        or not np.all(np.isfinite(observed_degrees))
        or not np.all(np.isfinite(expected_degrees))
    ):
        raise SystemExit("TPPNI SCM degree trace is incomplete or non-finite")
    if len(probability_array) != int(tppni_diagnostics["effective_bottom_n"]):
        raise SystemExit("TPPNI pool row count does not match its declared effective bottom-N")
    if not len(probability_array) or not np.all(
        np.diff(probability_array) >= -probability_order_tolerance
    ):
        raise SystemExit("TPPNI pool is not in ascending exact-probability order")
    probability_hist, probability_edges = np.histogram(probability_array, bins=40)
    l3_hist, l3_edges = np.histogram(l3_array, bins=min(40, max(1, int(l3_array.max()) + 1)))
    bottom_n_cutoff_probability = float(probability_array[-1])
    sampler_source = figure_dir / "figure_data__figure_6_sampler_fidelity.json"
    sampler_source.write_text(
        json.dumps(
            {
                "panel_manifest_sha256": sha256_path(args.panel_manifest),
                "sampler_trace_manifest": (
                    {
                        "path": str(args.sampler_trace_manifest),
                        "sha256": sha256_path(args.sampler_trace_manifest),
                    }
                    if args.sampler_trace_manifest
                    else None
                ),
                "bernett": {
                    "selected_audit": bernett_audit,
                    "positive_endpoint_degree_count": int(len(bernett_positive_degrees)),
                    "negative_endpoint_degree_count": int(len(bernett_negative_degrees)),
                    "positive_endpoint_degree_values": bernett_positive_degrees.tolist(),
                    "negative_endpoint_degree_values": bernett_negative_degrees.tolist(),
                },
                "tppni": {
                    "diagnostics": tppni_diagnostics,
                    "pool_artifact": {
                        "path": str(tppni_pool_path),
                        "sha256": sha256_path(tppni_pool_path),
                        "delimiter": tppni_pool_delimiter or "suffix_resolved",
                    },
                    "scm_degree_trace": {
                        "protein_ids": degree_ids,
                        "observed": observed_degrees.tolist(),
                        "expected": expected_degrees.tolist(),
                    },
                    "probability_histogram": {
                        "counts": probability_hist.tolist(),
                        "edges": probability_edges.tolist(),
                    },
                    "l3_histogram": {
                        "counts": l3_hist.tolist(),
                        "edges": l3_edges.tolist(),
                    },
                    "pool_rows": int(len(probability_array)),
                    "bottom_n_cutoff_probability": bottom_n_cutoff_probability,
                    "probability_order_tolerance": probability_order_tolerance,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    axes[0, 0].hist(bernett_positive_degrees, bins=30, alpha=0.65, label="positive")
    axes[0, 0].hist(bernett_negative_degrees, bins=30, alpha=0.65, label="Bernett selected")
    axes[0, 0].set_title("Bernett endpoint-degree distributions")
    axes[0, 0].set_xlabel("endpoint degree")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(
        ["positive", "selected"],
        [
            bernett_audit["selected_audit"]["mean_positive_go_bp_jaccard"],
            bernett_audit["selected_audit"]["mean_selected_go_bp_jaccard"],
        ],
    )
    axes[0, 1].set_title("Bernett GO-BP Jaccard")
    axes[0, 1].set_ylabel("mean pairwise Jaccard")
    residual_labels = ("degree sum", "degree max", "taxon sum", "self loops")
    residual_values = [
        bernett_audit["selected_audit"]["degree_absolute_residual_sum"],
        bernett_audit["selected_audit"]["degree_max_absolute_residual"],
        bernett_audit["selected_audit"]["taxon_pair_absolute_residual_sum"],
        bernett_audit["selected_audit"]["selected_self_loop_count"],
    ]
    axes[0, 2].bar(residual_labels, residual_values)
    axes[0, 2].set_title("Bernett objective diagnostics")
    axes[0, 2].tick_params(axis="x", rotation=45)
    axes[1, 0].scatter(
        observed_degrees,
        expected_degrees,
        s=12,
        alpha=0.55,
        linewidths=0,
        label=(
            f"max |residual|={tppni_scm['max_degree_residual']:.2e}; "
            f"mean={tppni_scm['mean_abs_degree_residual']:.2e}"
        ),
    )
    degree_limits = [
        min(float(observed_degrees.min()), float(expected_degrees.min())),
        max(float(observed_degrees.max()), float(expected_degrees.max())),
    ]
    axes[1, 0].plot(degree_limits, degree_limits, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("TPPNI SCM expected vs observed degree")
    axes[1, 0].set_xlabel("observed target degree")
    axes[1, 0].set_ylabel("SCM expected degree")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].stairs(probability_hist, probability_edges, fill=True)
    axes[1, 1].axvline(
        bottom_n_cutoff_probability,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"bottom-N cutoff={bottom_n_cutoff_probability:.3g}",
    )
    axes[1, 1].set_title("TPPNI exact SCM probability distribution")
    axes[1, 1].set_xlabel("p_ij")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].stairs(l3_hist, l3_edges, fill=True)
    axes[1, 2].set_title(
        f"TPPNI L3 distribution; pool={tppni_diagnostics['effective_bottom_n']:,}/"
        f"{tppni_diagnostics['published_n']:,}, "
        f"L3=0={tppni_diagnostics['l3_zero_count']:,}"
    )
    axes[1, 2].set_xlabel("L3 count")
    figure.suptitle("E003 sampler fidelity diagnostics")
    sampler_base = figure_dir / "figure_6_sampler_fidelity"
    sampler_paths = _save_figure(figure, sampler_base)
    figure_manifest.append(
        {
            "figure": sampler_base.name,
            "files": [str(path) for path in sampler_paths],
            "source_data": str(sampler_source),
        }
    )

    datasail: dict[str, Any] | None = None
    datasail_summary: dict[str, Any] | None = None
    if args.datasail_results:
        datasail = _json(args.datasail_results)
        if datasail.get("status") != "complete_datasail_sensitivity":
            raise SystemExit("DataSAIL result matrix is not complete")
        datasail_summary = {
            "training_units": datasail["training_units"],
            "score_cells": datasail["score_cells"],
            "cells": datasail["cells"],
        }
        datasail_rows = [
            {
                "model": str(cell["model"]),
                "seed": int(cell["seed"]),
                "train_environment": str(cell["train_environment"]),
                "test_environment": str(cell["test_environment"]),
                "auprc": float(cell["metrics"]["auprc"]),
                "auroc": float(cell["metrics"]["auroc"]),
            }
            for cell in datasail["cells"]
        ]
        datasail_source = figure_dir / "figure_data__supplement_datasail.csv"
        _write_csv(datasail_source, datasail_rows)
        datasail_models = tuple(sorted({row["model"] for row in datasail_rows}))
        figure, axes = plt.subplots(
            1, len(datasail_models), figsize=(7 * len(datasail_models), 6), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        for axis, model in zip(axes, datasail_models, strict=True):
            matrix = np.asarray(
                [
                    [
                        float(
                            np.mean(
                                [
                                    row["auprc"]
                                    for row in datasail_rows
                                    if row["model"] == model
                                    and row["train_environment"] == train
                                    and row["test_environment"] == test
                                ]
                            )
                        )
                        for test in ("uniform_partner", "esm_partner_matched")
                    ]
                    for train in ("uniform_partner", "esm_partner_matched")
                ],
                dtype=np.float64,
            )
            image = axis.imshow(matrix, cmap="viridis", aspect="auto")
            axis.set_title(_model_label(model))
            axis.set_xticks(range(2), ("uniform", "ESM"))
            axis.set_yticks(range(2), ("uniform", "ESM"))
            for row_index in range(2):
                for column_index in range(2):
                    axis.text(
                        column_index,
                        row_index,
                        f"{matrix[row_index, column_index]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                    )
        figure.colorbar(image, ax=axes.tolist(), shrink=0.8, label="mean AUPRC")
        figure.suptitle("Official DataSAIL supplementary sensitivity")
        datasail_base = figure_dir / "supplement_datasail_sensitivity"
        datasail_paths = _save_figure(figure, datasail_base)
        figure_manifest.append(
            {
                "figure": datasail_base.name,
                "files": [str(path) for path in datasail_paths],
                "source_data": str(datasail_source),
            }
        )

    registry_rows = _build_result_registry(
        core,
        external,
        datasail,
        panel_manifest,
        (
            str(datasail.get("panel_manifest", {}).get("sha256", ""))
            if datasail is not None
            else None
        ),
        config,
    )
    _write_result_registry(
        args.output_dir,
        registry_rows,
        input_hashes={
            "core_results": sha256_path(args.core_results),
            "external_results": sha256_path(args.external_results),
            "panel_manifest": sha256_path(args.panel_manifest),
            **(
                {"sampler_trace_manifest": sha256_path(args.sampler_trace_manifest)}
                if args.sampler_trace_manifest
                else {}
            ),
            "config": sha256_path(args.config),
            "protocol_freeze": sha256_path(args.protocol_freeze),
            **(
                {"datasail_results": sha256_path(args.datasail_results)}
                if args.datasail_results
                else {}
            ),
        },
        datasail=datasail,
    )

    # A compact design figure is generated from the frozen roster itself.
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.axis("off")
    design_text = "\n".join(
        [
            "E003 method-fidelity design",
            (
                f"models: {len(MODELS)} | train environments: {len(ENVIRONMENTS)} | "
                f"test environments: {len(ENVIRONMENTS)} | seeds: {len(SEEDS)}"
            ),
            (
                f"core score cells: {len(cells)} | exact E001v3 reuse cells: "
                f"{core['reused_e001v3_cells']}"
            ),
            "selection: matched validation AUPRC; threshold: matched validation maximum MCC",
            "negative semantic class: synthetic_unobserved_pair; test adaptation: forbidden",
            "uncertainty: shared stratified positive/negative bootstrap; seed stability separate",
        ]
    )
    axis.text(0.02, 0.9, design_text, va="top", family="monospace", fontsize=13)
    design_base = figure_dir / "e003_design_schematic"
    design_paths = _save_figure(figure, design_base)
    design_source = figure_dir / "figure_data__e003_design.json"
    design_source.write_text(
        json.dumps(
            {
                "models": list(MODELS),
                "environments": list(ENVIRONMENTS),
                "seeds": list(SEEDS),
                "core_score_cells": len(cells),
                "reused_e001v3_cells": core["reused_e001v3_cells"],
                "selection": "matched validation AUPRC",
                "threshold": "matched validation maximum MCC",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_manifest.append(
        {
            "figure": design_base.name,
            "files": [str(path) for path in design_paths],
            "source_data": str(design_source),
        }
    )

    plotting_script = Path(__file__).resolve()
    plotting_script_git_sha = git_commit(args.repo_root)
    if plotting_script_git_sha == "unknown":
        raise SystemExit("Cannot resolve the plotting script Git SHA")
    figure_input_hashes = {
        "core_results": sha256_path(args.core_results),
        "external_results": sha256_path(args.external_results),
        "panel_manifest": sha256_path(args.panel_manifest),
        "config": sha256_path(args.config),
        "protocol_freeze": sha256_path(args.protocol_freeze),
        "result_registry": sha256_path(args.output_dir / "e003_result_registry.json"),
    }
    if args.sampler_trace_manifest:
        figure_input_hashes["sampler_trace_manifest"] = sha256_path(
            args.sampler_trace_manifest
        )
    if args.datasail_results:
        figure_input_hashes["datasail_results"] = sha256_path(args.datasail_results)
    for item in figure_manifest:
        item["source_data_sha256"] = sha256_path(Path(item["source_data"]))
        item["plotting_script"] = str(plotting_script)
        item["plotting_script_sha256"] = sha256_path(plotting_script)
        item["plotting_script_git_sha"] = plotting_script_git_sha
        item["input_hashes"] = dict(figure_input_hashes)

    analysis = {
        "schema_version": "ppi_shift.e003_analysis.v1",
        "experiment_id": "E003",
        "core_results": {"path": str(args.core_results), "sha256": sha256_path(args.core_results)},
        "external_results": {
            "path": str(args.external_results),
            "sha256": sha256_path(args.external_results),
        },
        "panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
        },
        "sampler_trace_manifest": (
            {
                "path": str(args.sampler_trace_manifest),
                "sha256": sha256_path(args.sampler_trace_manifest),
            }
            if args.sampler_trace_manifest
            else None
        ),
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "protocol_freeze": {
            "path": str(args.protocol_freeze),
            "sha256": sha256_path(args.protocol_freeze),
        },
        "result_registry": {
            "path": str(args.output_dir / "e003_result_registry.json"),
            "sha256": sha256_path(args.output_dir / "e003_result_registry.json"),
        },
        "plotting_script_git_sha": plotting_script_git_sha,
        "models": list(MODELS),
        "environments": list(ENVIRONMENTS),
        "seeds": list(SEEDS),
        "primary_metric": "auprc",
        "secondary_metrics": ["auroc", "mcc", "balanced_accuracy", "f1", "brier", "nll", "ece"],
        "metric_summary": metric_rows,
        "train_environment_mean_auprc": train_means,
        "test_environment_mean_auprc": test_means,
        "environment_effect_summary": environment_effect_summary,
        "train_test_interaction_residual": interaction.tolist(),
        "condition_labels": condition_labels,
        "condition_kendall_tau_matrix": condition_tau_matrix.tolist(),
        "condition_spearman_matrix": condition_spearman_matrix.tolist(),
        "model_concordance_kendall_tau": model_concordance_tau.tolist(),
        "ranking": {
            "conditions": rank_rows,
            "seed_sensitivity": seed_rank_rows,
            "winner_frequency": dict(sorted(winner_frequency.items())),
            "winner_entropy_bits": _entropy(
                [count / len(conditions) for count in (winner_frequency[model] for model in MODELS)]
            ),
            "pairwise_comparison_summary": ranking_pairwise_summary,
            "rank_probability_by_condition": rank_probability_by_condition,
            "top_two_margin_by_condition": margin_by_condition,
            "scope": "all 36 train-by-test conditions; three-seed means",
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "scope": (
                "all 36 train-by-test conditions; each seed evaluated separately, "
                "then averaged under shared stratified panel resamples"
            ),
            "summaries": {
                f"{model}|{train}|{test}": summary
                for (model, train, test), summary in bootstrap_summary.items()
            },
        },
        "seed_stability": seed_stability,
        "cluster_sensitivity": cluster_sensitivity,
        "reversal_summary": reversal_summary,
        "external_summary": external_summary,
        "external_all_summary": external_all_summary,
        "intact_posthoc_prevalence_sensitivity": intact_sensitivity_rows,
        "datasail_results": (
            {
                "path": str(args.datasail_results),
                "sha256": sha256_path(args.datasail_results),
            }
            if args.datasail_results
            else None
        ),
        "datasail": datasail_summary,
        "figure_manifest": figure_manifest,
        "interpretation_boundary": (
            "Synthetic negatives are unobserved-pair evaluation constructs, not "
            "biological non-interactions; "
            "external measured and synthetic surfaces remain separate estimands."
        ),
    }
    output = args.output_dir / "e003_analysis.json"
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_hashes = []
    for item in figure_manifest:
        figure_hashes.append(
            {
                "path": item["source_data"],
                "kind": "source_data",
                "sha256": item["source_data_sha256"],
            }
        )
        figure_hashes.extend(
            {
                "path": str(path),
                "kind": "rendered_figure",
                "sha256": sha256_path(path),
            }
            for path in [Path(value) for value in item["files"]]
        )
    (args.output_dir / "e003_figure_hashes.json").write_text(
        json.dumps(figure_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "analysis": str(output),
                "figures": len(figure_manifest),
                "reversal_events": reversal_summary["eligible_comparisons"],
                "status": "complete",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
