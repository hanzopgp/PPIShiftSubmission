"""Leakage-control split contracts.

Implementation intentionally deferred until dataset and family-threshold audits are frozen.
"""

from __future__ import annotations

LEAKAGE_LEVELS = (
    "pair_disjoint",
    "protein_disjoint",
    "family_disjoint",
    "cross_source_family_disjoint",
)
