from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ppi_shift.e003_method_fidelity import BernettConfig, _candidate_pool_for_bernett
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    ExclusionRegistry,
    PositivePair,
    build_candidate_universe,
)


def _official_sampler_path() -> Path | None:
    candidates = []
    configured = os.environ.get("E003_BERNETT_SOURCE")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/tmp/e003-upstream-audit-20260917/bernett/bin/sample_negatives_ilp.py"),
            Path(
                r"\\wsl.localhost\Ubuntu\tmp\e003-upstream-audit-20260917\bernett\bin\sample_negatives_ilp.py"
            ),
        ]
    )
    return next((path for path in candidates if path.exists()), None)


def _fixture() -> tuple[list[PositivePair], object]:
    sequences = {f"p{i}": "ACDEFGHIKLMNPQRSTVWY"[: 8 + i] for i in range(6)}
    pairs = [("p0", "p1"), ("p1", "p2"), ("p2", "p3")]
    positives = [
        PositivePair(
            split="train",
            source_row_index=index,
            dataset_name="official-differential-fixture",
            left_id=left,
            right_id=right,
            left_sequence=sequences[left],
            right_sequence=sequences[right],
            identity_schema_version=SCHEMA_VERSION_V3,
        )
        for index, (left, right) in enumerate(pairs)
    ]
    registry = ExclusionRegistry(
        name="official-differential-fixture",
        path=Path("fixture-registry.csv"),
        sha256="0" * 64,
        identifier_pairs=frozenset(),
        sequence_pairs=frozenset(),
    )
    universe = build_candidate_universe(
        positives,
        registry,
        stage="train",
        source_input_hashes={"fixture": "1" * 64},
        schema_version=SCHEMA_VERSION_V3,
    )
    return positives, universe


def test_bernett_full_candidate_fixture_matches_official_script(monkeypatch) -> None:
    source = _official_sampler_path()
    if source is None:
        pytest.skip("Pinned Bernett source clone is not available in this environment")

    # The differential fixture exercises candidate construction only. The upstream
    # module imports cvxpy for later solver functions; a sentinel module keeps this
    # test independent of the optional solver package.
    monkeypatch.setitem(sys.modules, "cvxpy", types.ModuleType("cvxpy"))
    source_parent = str(source.parent)
    monkeypatch.syspath_prepend(source_parent)
    module_name = "e003_pinned_bernett_sampler"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    official = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = official
    try:
        spec.loader.exec_module(official)
        positives, universe = _fixture()
        ids, ours, audit = _candidate_pool_for_bernett(
            positives,
            universe,
            config=BernettConfig(candidate_cap_multiplier=100),
            seed=11,
            taxon_by_id={protein_id: "9606" for protein_id in universe.proteins},
            go_bp_by_id={protein_id: frozenset() for protein_id in universe.proteins},
        )
        index = {protein_id: position for position, protein_id in enumerate(ids)}
        pos_array = np.asarray(
            [
                [
                    min(index[row.left_id], index[row.right_id]),
                    max(index[row.left_id], index[row.right_id]),
                ]
                for row in positives
            ],
            dtype=np.int64,
        )
        expected = official.build_candidate_set(
            len(ids),
            pos_array,
            official.SamplingConfig(lambda_self_loop=1.0),
            max_candidates=10_000,
            seed=11,
        )
        assert audit["candidate_pool_mode"] == "full_legal_upper_triangle"
        assert set(ours) == {tuple(map(int, pair)) for pair in expected.tolist()}
    finally:
        sys.modules.pop(module_name, None)
