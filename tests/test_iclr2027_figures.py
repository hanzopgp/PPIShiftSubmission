from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_e003_final_paper_figures import _require_committed_csv
from scripts.build_iclr2027_final_figures import (
    ENVIRONMENT_ORDER,
    MEASURED_SURFACES,
    MODEL_ORDER,
    _load_measured_source,
    aggregate_measured_rows,
    derive_pair_rows,
    validate_measured_summary,
)

ROOT = Path(__file__).parents[1]


def test_final_figure_build_rejects_changed_source_values() -> None:
    path = ROOT / "paper/figures/e003_final/figure_data__figure_02_negative_transfer.csv"
    before = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require_committed_csv(path, rows, ROOT)
    rows[0]["delta_metric"] = "999"
    with pytest.raises(ValueError, match="differs from the committed table"):
        _require_committed_csv(path, rows, ROOT)
    assert path.read_bytes() == before


def test_pair_heatmap_data_is_derived_from_the_frozen_pair_table() -> None:
    source = ROOT / "paper/figures/e003_final/figure_data__figure_03_pair_dependence.csv"
    rows = derive_pair_rows(source)

    assert len(rows) == len(MODEL_ORDER) * len(ENVIRONMENT_ORDER)
    for row in rows:
        delta = float(row["delta_auroc"])
        original = float(row["auroc_original"])
        shuffled = float(row["auroc_shuffled"])
        assert delta == pytest.approx(original - shuffled)
        assert row["sign_interpretation"] == (
            "original_pairing_higher"
            if delta > 0
            else "partners_shuffled_higher"
            if delta < 0
            else "no_difference"
        )
    assert any(row["sign_interpretation"] == "partners_shuffled_higher" for row in rows)
    assert any(row["sign_interpretation"] == "original_pairing_higher" for row in rows)


def test_measured_heatmap_data_has_three_seed_cells_and_matches_summary() -> None:
    source = ROOT / (
        "paper/figures/iclr2027_final/figure_data__figure_measured_transfer_heatmaps.csv"
    )
    summary = ROOT / "paper/figures/e003_final/figure_data__figure_05_measured_and_ood.csv"
    rows = _load_measured_source(source)
    aggregated = aggregate_measured_rows(rows)

    assert len(rows) == len(MEASURED_SURFACES) * len(MODEL_ORDER) * len(ENVIRONMENT_ORDER) * 3
    assert len(aggregated) == len(MEASURED_SURFACES) * len(MODEL_ORDER) * len(ENVIRONMENT_ORDER)
    assert {row["seed_count"] for row in aggregated} == {3}
    validate_measured_summary(aggregated, summary)
