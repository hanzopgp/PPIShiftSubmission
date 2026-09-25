"""Model-native compatibility gates for common E001 evaluation cohorts.

This module performs only pre-sampler eligibility checks. It never consults model
scores, negative labels, or test diagnostics, and it never crops a sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ppi_shift.data import normalize_sequence
from ppi_shift.negatives import (
    SCHEMA_VERSION_V3,
    NegativeContractError,
    PositivePair,
)

COMPATIBILITY_SCHEMA_VERSION = "ppi_shift.e001_model_compatibility.v1"
COMMON_RAW_SEQUENCE_LENGTH = 512


@dataclass(frozen=True)
class LengthCompatibilityAudit:
    """Deterministic retention report for one Gold stage."""

    stage: str
    threshold: int
    input_positive_rows: int
    retained_positive_rows: int
    excluded_positive_rows: int
    maximum_input_length: int
    maximum_retained_length: int
    excluded_rows: tuple[dict[str, Any], ...]

    @property
    def retention_fraction(self) -> float:
        return self.retained_positive_rows / max(1, self.input_positive_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "stage": self.stage,
            "threshold": self.threshold,
            "input_positive_rows": self.input_positive_rows,
            "retained_positive_rows": self.retained_positive_rows,
            "excluded_positive_rows": self.excluded_positive_rows,
            "retention_fraction": self.retention_fraction,
            "maximum_input_length": self.maximum_input_length,
            "maximum_retained_length": self.maximum_retained_length,
            "excluded_rows": list(self.excluded_rows),
            "policy": (
                "retain only pairs whose two normalized sequences are at most threshold; never crop"
            ),
        }


def _positive_lengths(positive: PositivePair) -> tuple[int, int]:
    return len(normalize_sequence(positive.left_sequence)), len(
        normalize_sequence(positive.right_sequence)
    )


def filter_length_compatible_positive_rows(
    positives: Sequence[PositivePair],
    *,
    stage: str,
    threshold: int = COMMON_RAW_SEQUENCE_LENGTH,
) -> tuple[tuple[PositivePair, ...], LengthCompatibilityAudit]:
    """Retain only complete raw-sequence pairs within the audited threshold."""

    if threshold <= 0:
        raise NegativeContractError("Compatibility threshold must be positive")
    if not positives:
        raise NegativeContractError("Cannot audit compatibility without positive rows")
    if any(positive.identity_schema_version != SCHEMA_VERSION_V3 for positive in positives):
        raise NegativeContractError("v3 compatibility filtering requires v3 positive identities")

    retained: list[PositivePair] = []
    excluded: list[dict[str, Any]] = []
    maximum_input_length = 0
    maximum_retained_length = 0
    for positive in positives:
        left_length, right_length = _positive_lengths(positive)
        pair_maximum = max(left_length, right_length)
        maximum_input_length = max(maximum_input_length, pair_maximum)
        reasons = []
        if left_length > threshold:
            reasons.append("left_sequence_exceeds_threshold")
        if right_length > threshold:
            reasons.append("right_sequence_exceeds_threshold")
        if reasons:
            excluded.append(
                {
                    "positive_row_id": positive.positive_row_id,
                    "source_row_index": positive.source_row_index,
                    "left_id": positive.left_id,
                    "right_id": positive.right_id,
                    "left_length": left_length,
                    "right_length": right_length,
                    "reasons": reasons,
                }
            )
            continue
        retained.append(positive)
        maximum_retained_length = max(maximum_retained_length, pair_maximum)

    if not retained:
        raise NegativeContractError(
            f"No {stage} positive rows satisfy the raw-length threshold {threshold}"
        )
    audit = LengthCompatibilityAudit(
        stage=stage,
        threshold=threshold,
        input_positive_rows=len(positives),
        retained_positive_rows=len(retained),
        excluded_positive_rows=len(excluded),
        maximum_input_length=maximum_input_length,
        maximum_retained_length=maximum_retained_length,
        excluded_rows=tuple(excluded),
    )
    return tuple(retained), audit


def assert_pair_lengths_within_threshold(
    positives: Sequence[PositivePair], *, threshold: int = COMMON_RAW_SEQUENCE_LENGTH
) -> None:
    """Fail closed if any already-selected v3 pair is over the model contract."""

    for positive in positives:
        left_length, right_length = _positive_lengths(positive)
        if left_length > threshold or right_length > threshold:
            raise NegativeContractError(
                f"Over-threshold v3 positive row reached scoring surface: "
                f"{positive.positive_row_id} ({left_length}, {right_length}) > {threshold}"
            )
