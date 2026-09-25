from __future__ import annotations

import pytest

from ppi_shift.metrics import compute_binary_metrics
from scripts.postprocess_e003_datasail import _assert_metrics_match


def test_datasail_postprocess_recomputes_frozen_metrics() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.2, 0.7, 0.1]
    recorded = compute_binary_metrics(
        labels,
        scores,
        threshold=0.5,
        threshold_source="matched_validation_max_mcc",
    )

    _assert_metrics_match(recorded, labels, scores, threshold=0.5, label="toy")

    recorded["auprc"] = 0.0
    with pytest.raises(SystemExit, match="DataSAIL metric mismatch"):
        _assert_metrics_match(recorded, labels, scores, threshold=0.5, label="toy")
