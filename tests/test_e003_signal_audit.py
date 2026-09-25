from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from ppi_shift.signal_audit import (
    EmbeddingStore,
    build_broken_pair_panel,
    build_positive_degree_table,
    embedding_contract_mismatches,
    fit_empirical_endpoint_audit,
    fit_pair_classifier_audit,
    fit_protein_sequence_audit,
    grouped_split_indices,
    pair_easy_features,
)
from scripts.analyze_e003_signal_audit import _validate_signal_csv
from scripts.build_e003_final_paper_figures import _build_transfer_sources, _expected_core_cells


def _rows() -> list[dict[str, str]]:
    rows = []
    for group_index in range(8):
        left = f"L{group_index}"
        positive_right = f"P{group_index}"
        negative_right = f"N{group_index}"
        left_sequence = "ACDEFGHIKLMNPQRSTVWY"[: 5 + group_index]
        positive_sequence = "YWVTSRQPNMLKIHGFEDCA"[: 6 + group_index]
        negative_sequence = "AAAAAAAAAAAAAAAAAAAA"[: 7 + group_index]
        for label, right, sequence in (
            ("1", positive_right, positive_sequence),
            ("0", negative_right, negative_sequence),
        ):
            rows.append(
                {
                    "row_id": f"row-{group_index}-{label}",
                    "name1": left,
                    "name2": right,
                    "seq1": left_sequence,
                    "seq2": sequence,
                    "interaction_label": label,
                    "positive_row_id": f"group-{group_index}",
                    "negative_environment": "uniform_partner",
                }
            )
    return rows


def test_group_split_is_deterministic_and_does_not_split_positive_groups() -> None:
    rows = _rows()
    first = grouped_split_indices(rows, 11)
    second = grouped_split_indices(rows, 11)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    fit_groups = {rows[index]["positive_row_id"] for index in first[0]}
    holdout_groups = {rows[index]["positive_row_id"] for index in first[1]}
    assert fit_groups.isdisjoint(holdout_groups)


def test_easy_feature_extraction_has_no_interaction_or_test_label_feature() -> None:
    rows = _rows()
    features, names = pair_easy_features(rows, degree_table=build_positive_degree_table(rows))
    assert features.shape[0] == len(rows)
    assert all("label" not in name and "score" not in name for name in names)
    assert "left_log_degree" in names


def test_pair_audit_fit_isolated_from_test_labels() -> None:
    rows = _rows()
    train = rows[:12]
    test = rows[12:]
    degree = build_positive_degree_table(train)

    def builder(values):
        return pair_easy_features(values, degree_table=degree)

    first = fit_pair_classifier_audit(
        train,
        test,
        feature_family="easy_sequence_degree",
        seed=11,
        feature_builder=builder,
    )
    altered_test = [
        dict(row, interaction_label="1" if row["interaction_label"] == "0" else "0") for row in test
    ]
    second = fit_pair_classifier_audit(
        train,
        altered_test,
        feature_family="easy_sequence_degree",
        seed=11,
        feature_builder=builder,
    )
    assert first["test_labels_used_for_fit"] is False
    assert first["coefficients"] == second["coefficients"]
    assert first["test_scores"] == second["test_scores"]


def test_protein_only_controls_do_not_expose_pair_features() -> None:
    rows = _rows()
    result = fit_protein_sequence_audit(rows[:12], rows[12:], seed=22)
    assert result["test_labels_used_for_fit"] is False
    assert all(name.startswith("single_protein_") for name in result["feature_names"])
    propensity = fit_empirical_endpoint_audit(rows[:12], rows[12:], seed=33)
    assert propensity["control"] == "empirical_endpoint_propensity"
    assert propensity["smoothing"] == {"alpha": 1.0, "beta": 1.0}


def test_broken_pair_panel_is_deterministic_and_preserves_marginals() -> None:
    rows = _rows()[:8]
    first, first_audit = build_broken_pair_panel(rows)
    second, second_audit = build_broken_pair_panel(rows)
    assert first == second
    assert first_audit == second_audit
    assert first_audit["label_marginals_preserved"] is True
    assert first_audit["endpoint_marginals_preserved"] is True
    assert Counter(row["interaction_label"] for row in first) == Counter(
        row["interaction_label"] for row in rows
    )
    assert first_audit["known_positive_collisions"] == 0


def test_embedding_gate_reports_exact_contract_mismatch() -> None:
    store = EmbeddingStore(
        vectors={},
        metadata={
            "model_name": "wrong",
            "model_revision": "wrong",
            "embedding_layer": 1,
            "pooling": "wrong",
            "embedding_dimension": 2,
        },
        source_paths={},
        source_hashes={},
    )
    mismatches = embedding_contract_mismatches(
        store,
        {
            "model": "facebook/esm2_t30_150M_UR50D",
            "revision": "a" * 40,
            "layer": 30,
            "pooling": "overlap_corrected_residue_mean",
            "dimension": 640,
        },
    )
    assert len(mismatches) == 5


def test_signal_config_is_frozen_with_declared_seed_roster() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/e003_signal_audit.yaml").read_text(encoding="utf-8")
    )
    assert config["status"] == "frozen_before_new_signal_output"
    assert config["seeds"] == [11, 22, 33]
    assert config["classifiers"]["logistic"]["threshold_selection"] == ("training_fit_maximum_mcc")


def test_signal_validator_reports_missing_seed_and_duplicate_cell(tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    fields = ["control", "environment", "seed", "status", "test_labels_used_for_fit", "test_auprc"]
    rows = [
        {
            "control": "easy_sequence_degree",
            "environment": environment,
            "seed": str(seed),
            "status": "PASS",
            "test_labels_used_for_fit": "False",
            "test_auprc": "0.5",
        }
        for environment in (
            "uniform_partner",
            "degree_partner_matched",
            "sequence_nuisance_matched",
            "esm_partner_matched",
            "bernett2026_ilp_exact",
            "tppni2025_scm_cl3_exact_balanced",
        )
        for seed in (11, 22, 33)
    ]
    rows.pop()
    rows[-1] = dict(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _observed, errors = _validate_signal_csv(
        path,
        expected_controls={"easy_sequence_degree"},
        expected_per_control=18,
        require_pass=True,
    )
    assert any(error.startswith("duplicate_cell:") for error in errors)
    assert any(error.startswith("row_count:") for error in errors)


def test_figure_source_csv_contains_only_machine_readable_values(tmp_path: Path) -> None:
    environments = (
        "uniform_partner",
        "degree_partner_matched",
        "sequence_nuisance_matched",
        "esm_partner_matched",
        "bernett2026_ilp_exact",
        "tppni2025_scm_cl3_exact_balanced",
    )
    core = [
        {
            "model": "test_model",
            "train_environment": train,
            "test_environment": test,
            "seed": seed,
            "auprc": str(0.4 + 0.01 * environments.index(train)),
        }
        for train in environments
        for test in environments
        for seed in (11, 22, 33)
    ]
    external = [
        {
            "model": "test_model",
            "train_environment": train,
            "evaluation_surface": surface,
            "seed": seed,
            "auroc": str(0.55 + 0.01 * environments.index(train)),
        }
        for train in environments
        for surface in ("vidal", "intact")
        for seed in (11, 22, 33)
    ]
    gold_rows, external_rows, _models, _surfaces = _build_transfer_sources(core, external)
    assert len(gold_rows) == len(environments) ** 2
    assert len(external_rows) == len(environments) * 2
    assert all(
        row["delta_metric"] == 0.0
        for row in gold_rows + external_rows
        if row["train_environment"] == "uniform_partner"
    )


def test_final_figure_gate_requires_the_complete_train_test_core_grid() -> None:
    assert _expected_core_cells(4) == 432
