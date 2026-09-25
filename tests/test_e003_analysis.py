from __future__ import annotations

import gzip
import struct

import numpy as np

from ppi_shift.artifacts import sha256_path
from ppi_shift.uncertainty import make_stratified_bootstrap_weights, stratified_bootstrap_metrics
from scripts.analyze_e003_results import (
    _build_result_registry,
    _fast_stratified_bootstrap,
    _ranking_comparison_summary,
    _tppni_pool_delimiter,
)
from scripts.verify_e003_figures import _expected_input_hashes, _png_dpi


def test_figure_verifier_reads_png_dpi_metadata(tmp_path) -> None:
    png_path = tmp_path / "figure.png"
    payload = struct.pack(">IIB", 12000, 12000, 1)
    png_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I4s", len(payload), b"pHYs")
        + payload
        + b"\x00\x00\x00\x00"
        + struct.pack(">I4s", 0, b"IEND")
        + b"\x00\x00\x00\x00"
    )

    dpi = _png_dpi(png_path)

    assert dpi is not None
    assert dpi[0] == dpi[1] == 304.8


def test_tppni_pool_delimiter_accepts_comma_data_with_tsv_suffix(tmp_path) -> None:
    path = tmp_path / "pool.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("rank,protein1,protein2,scm_probability,l3_count\n")
        handle.write("1,A,B,0.1,2\n")

    assert _tppni_pool_delimiter(path) == ","


def test_figure_verifier_binds_analysis_input_hashes(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")

    expected = _expected_input_hashes(
        {"core_results": {"path": str(input_path), "sha256": sha256_path(input_path)}}
    )

    assert expected == {"core_results": sha256_path(input_path)}


def test_vectorized_stratified_bootstrap_matches_reference() -> None:
    labels = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.05], dtype=np.float64)
    weights = make_stratified_bootstrap_weights(labels, replicates=16, seed=1729)
    fast_auprc, fast_auroc = _fast_stratified_bootstrap(labels, scores, weights)
    reference = stratified_bootstrap_metrics(labels, scores, weights)
    assert np.allclose(fast_auprc, reference["auprc"])
    assert np.allclose(fast_auroc, reference["auroc"])


def test_result_registry_preserves_core_and_external_provenance() -> None:
    metrics = {
        "rows": 4,
        "positive_prevalence": 0.5,
        "auprc": 0.75,
        "auroc": 0.8,
        "mcc": 0.5,
        "balanced_accuracy": 0.75,
        "f1": 0.75,
        "brier": 0.2,
        "nll": 0.6,
        "ece": 0.1,
        "threshold": 0.4,
        "threshold_source": "matched_validation_max_mcc",
    }
    core = {
        "cells": [
            {
                "model": "dscript_esm2",
                "seed": 11,
                "train_environment": "uniform_partner",
                "test_environment": "uniform_partner",
                "status": "complete_new_training_unit",
                "metrics": metrics,
                "threshold": {"threshold": 0.4, "method": "validation_max_mcc"},
                "prediction_sha256": "prediction-core",
                "checkpoint_sha256": "checkpoint-core",
                "checkpoint_selection_epoch": 4,
                "source_revision": "model-revision",
            }
        ]
    }
    external = {
        "bundles": [
            {
                "model": "dscript_esm2",
                "seed": 11,
                "train_environment": "uniform_partner",
                "checkpoint": {
                    "source_revision": "model-revision",
                    "checkpoint_sha256": "checkpoint-core",
                    "selected_epoch": 4,
                },
                "surfaces": {
                    "vidal__measured": {
                        "source": "vidal",
                        "surface": "measured",
                        "panel": {"sha256": "panel-vidal"},
                        "prediction": {"sha256": "prediction-vidal"},
                        "metrics": {"natural_prevalence": metrics},
                    }
                },
            }
        ]
    }
    panel_manifest = {
        "stages": {
            "test": {"samplers": {"uniform_partner": {"panel_sha256": "panel-core"}}}
        }
    }
    config = {
        "models": {
            "source_revisions": {
                "dscript_esm2": "model-revision",
                "tuna_corrected": "tuna-revision",
                "senseppi_source": "sense-revision",
                "rapppid": "rapppid-revision",
                "esm2_t30_150m": "esm-revision",
                "esm2_t36_3b": "sense-representation",
            }
        }
    }

    rows = _build_result_registry(core, external, None, panel_manifest, None, config)

    assert len(rows) == 2
    assert rows[0]["scope"] == "core"
    assert rows[0]["panel_sha256"] == "panel-core"
    assert rows[0]["checkpoint_selection_epoch"] == 4
    assert rows[1]["scope"] == "external"
    assert rows[1]["normalized_excess_ap"] is None
    assert rows[1]["role"] == "measured_transfer"


def test_ranking_comparison_summary_uses_all_condition_pairs() -> None:
    summary = _ranking_comparison_summary(
        {
            "condition_a": np.asarray([0.90, 0.80, 0.70]),
            "condition_b": np.asarray([0.70, 0.80, 0.90]),
            "condition_c": np.asarray([0.85, 0.80, 0.90]),
        },
        ("model_a", "model_b", "model_c"),
    )

    assert summary["condition_pair_count"] == 3
    assert summary["top_model_change_count"] == 2
    assert summary["top_model_change_fraction"] == 2 / 3
    assert summary["pairwise_sign_change_eligible"] == 9
    assert summary["pairwise_sign_change_count"] == 6
    assert summary["absolute_rank_displacement_denominator"] == 9
    assert summary["average_absolute_rank_displacement"] > 0
