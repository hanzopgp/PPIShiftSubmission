from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ppi_shift.e003_method_fidelity import (
    BernettConfig,
    generate_bernett_exact_pairs,
    generate_tppni_exact_pool,
    select_tppni_balanced_pairs,
    solve_tppni_scm_lambdas,
)
from ppi_shift.e003_panels import materialize_global_balanced_panel, validate_e003_panel
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    ExclusionRegistry,
    PositivePair,
    build_candidate_universe,
)


def _positive_rows() -> list[PositivePair]:
    sequences = {f"p{index}": ("ACDEFGHIKLMNPQRSTVWY"[index:] + "A" * index) for index in range(8)}
    pairs = [("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p4", "p5")]
    return [
        PositivePair(
            split="train",
            source_row_index=index,
            dataset_name="fixture",
            left_id=left,
            right_id=right,
            left_sequence=sequences[left],
            right_sequence=sequences[right],
            identity_schema_version=SCHEMA_VERSION_V3,
        )
        for index, (left, right) in enumerate(pairs)
    ]


def _universe(positives: list[PositivePair]):
    registry = ExclusionRegistry(
        name="fixture_registry",
        path=Path("fixture-registry.csv"),
        sha256="0" * 64,
        identifier_pairs=frozenset(),
        sequence_pairs=frozenset(),
    )
    return build_candidate_universe(
        positives,
        registry,
        stage="train",
        source_input_hashes={"fixture": "1" * 64},
        schema_version=SCHEMA_VERSION_V3,
    )


def _annotations(universe):
    taxonomy = {protein_id: "9606" for protein_id in universe.proteins}
    go_bp = {
        protein_id: frozenset({"GO:1", "GO:2"} if int(protein_id[1:]) % 2 else {"GO:1"})
        for protein_id in universe.proteins
    }
    return taxonomy, go_bp


def test_bernett_global_candidate_pool_and_solver_are_deterministic() -> None:
    pytest.importorskip("scipy")
    positives = _positive_rows()
    universe = _universe(positives)
    taxonomy, go_bp = _annotations(universe)
    config = BernettConfig(candidate_cap_multiplier=100, time_limit_seconds=30)
    first, first_audit = generate_bernett_exact_pairs(
        positives,
        universe,
        seed=11,
        taxon_by_id=taxonomy,
        go_bp_by_id=go_bp,
        config=config,
    )
    second, second_audit = generate_bernett_exact_pairs(
        positives,
        universe,
        seed=11,
        taxon_by_id=taxonomy,
        go_bp_by_id=go_bp,
        config=config,
    )
    assert first == second
    assert first_audit == second_audit
    assert len(first) == len(positives)
    assert first_audit["candidate_pool"]["candidate_pool_mode"] == "full_legal_upper_triangle"
    assert first_audit["candidate_pool"]["candidate_count"] > len(first)
    # The official candidate universe includes legal self-pairs; the active
    # self-loop objective can then choose zero of them without deleting them.
    assert first_audit["selected_audit"]["selected_self_loop_count"] == 0
    assert first_audit["solver_status"] == 0


def test_bernett_cap_records_official_subsampling_and_annotation_terms() -> None:
    pytest.importorskip("scipy")
    positives = _positive_rows()
    universe = _universe(positives)
    taxonomy, go_bp = _annotations(universe)
    pairs, audit = generate_bernett_exact_pairs(
        positives,
        universe,
        seed=22,
        taxon_by_id=taxonomy,
        go_bp_by_id=go_bp,
        config=BernettConfig(candidate_cap_multiplier=2, time_limit_seconds=30),
    )
    assert len(pairs) == len(positives)
    pool = audit["candidate_pool"]
    assert pool["candidate_cap_subsampled"] is True
    assert pool["candidate_count"] == 2 * len(positives)
    assert pool["candidate_subsampling"]["self_pairs_forced"] > 0
    assert audit["weights"] == {
        "degree": 1.0,
        "taxon_pair": 1.0,
        "self_loop": 1.0,
        "go_bp_jaccard": 1.0,
    }


def test_tppni_scm_residual_and_l3_filter_are_explicit() -> None:
    positives = _positive_rows()
    universe = _universe(positives)
    pool = generate_tppni_exact_pool(
        positives,
        universe,
        topology_positives=positives,
        published_n=10,
    )
    assert pool.diagnostics["effective_bottom_n"] == 10
    assert pool.diagnostics["scm"]["max_degree_residual"] <= 1e-8
    assert pool.diagnostics["scm"]["mean_abs_degree_residual"] <= 1e-8
    assert pool.diagnostics["scm"]["max_relative_degree_error"] <= 1e-8
    assert "pathological_degree_count" in pool.diagnostics["scm"]
    assert sum(l3 == 0 for l3 in pool.l3_counts) == len(pool.l3_zero_pairs)
    assert len(pool.l3_zero_pairs) < len(pool.bottom_n_pairs)
    assert pool.diagnostics["l3_definition"].startswith("sum_")


def test_tppni_lambdas_fit_a_small_degree_sequence() -> None:
    lambdas, diagnostics = solve_tppni_scm_lambdas(
        {"a": 1, "b": 2, "c": 2, "d": 1, "e": 2, "f": 2}, tolerance=1e-8
    )
    assert set(lambdas) == {"a", "b", "c", "d", "e", "f"}
    assert diagnostics["max_degree_residual"] <= 1e-8
    assert diagnostics["degree_trace"]["protein_ids"] == ["a", "b", "c", "d", "e", "f"]
    assert diagnostics["degree_trace"]["observed"] == [1.0, 2.0, 2.0, 1.0, 2.0, 2.0]
    assert np.allclose(
        diagnostics["degree_trace"]["expected"], diagnostics["degree_trace"]["observed"], atol=1e-8
    )


def test_tppni_balancing_is_reproducible_and_panel_contract_is_global() -> None:
    positives = _positive_rows()
    universe = _universe(positives)
    pool = generate_tppni_exact_pool(
        positives,
        universe,
        topology_positives=positives,
        published_n=20,
    )
    first, first_audit = select_tppni_balanced_pairs(pool, count=len(positives), seed=33)
    second, second_audit = select_tppni_balanced_pairs(pool, count=len(positives), seed=33)
    assert first == second
    assert first_audit == second_audit
    rows = materialize_global_balanced_panel(
        positives,
        universe,
        "tppni2025_scm_cl3_exact_balanced",
        seed=33,
        pairs=first,
        diagnostics=first_audit,
    )
    validate_e003_panel(
        rows,
        environment="tppni2025_scm_cl3_exact_balanced",
        universe=universe,
    )
    assert len(rows) == 2 * len(positives)
    assert all(
        "not_anchor_matched" in row["match_diagnostics_json"]
        for row in rows
        if str(row["interaction_label"]) == "0"
    )


def test_tppni_degree_trace_is_not_replicated_in_each_panel_row() -> None:
    positives = _positive_rows()
    universe = _universe(positives)
    pairs = [("p0", "p2"), ("p0", "p3"), ("p1", "p3"), ("p2", "p4")]
    rows = materialize_global_balanced_panel(
        positives,
        universe,
        "tppni2025_scm_cl3_exact_balanced",
        seed=33,
        pairs=pairs,
        diagnostics={
            "scm": {
                "protein_count": 8,
                "degree_trace": {
                    "protein_ids": [f"p{index}" for index in range(8)],
                    "observed": [1.0] * 8,
                    "expected": [1.0] * 8,
                },
            }
        },
    )
    negative = next(row for row in rows if str(row["interaction_label"]) == "0")
    diagnostics = json.loads(negative["match_diagnostics_json"])
    assert "degree_trace" not in diagnostics["scm"]
    assert diagnostics["scm_degree_trace_reference"] == (
        "pool_artifact.diagnostics.scm.degree_trace"
    )


def test_tppni_bottom_n_matches_brute_force_probability_order() -> None:
    positives = _positive_rows()
    universe = _universe(positives)
    pool = generate_tppni_exact_pool(
        positives,
        universe,
        topology_positives=positives,
        published_n=7,
    )
    lambdas, _diagnostics = solve_tppni_scm_lambdas(
        {
            protein_id: sum(
                protein_id in (positive.left_id, positive.right_id)
                and positive.left_id != positive.right_id
                for positive in positives
            )
            for protein_id in universe.proteins
        }
    )
    positive_pairs = {
        tuple(sorted((positive.left_id, positive.right_id))) for positive in positives
    }
    legal = []
    ids = sorted(universe.proteins)
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            pair = (left, right)
            if pair in positive_pairs:
                continue
            probability = 1.0 / (1.0 + math.exp(lambdas[left] + lambdas[right]))
            legal.append((probability, pair))
    ordered = sorted(legal, key=lambda item: (round(item[0], 12), item[1]))[:7]
    expected = [pair for _probability, pair in ordered]
    assert list(pool.bottom_n_pairs) == expected
    assert list(pool.probabilities) == pytest.approx(
        [probability for probability, _pair in ordered]
    )
