"""Canonical row and sequence utilities.

Keep this module deliberately small. All source-specific parsing should happen before
rows enter the canonical paper dataset contract.
"""

from __future__ import annotations

import hashlib


def normalize_sequence(sequence: str) -> str:
    """Canonicalize a protein sequence without silently changing residues."""
    return "".join(str(sequence).split()).upper()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence(sequence).encode("utf-8")).hexdigest()


def unordered_pair_key(a: str, b: str) -> tuple[str, str]:
    """Stable unordered pair key for non-directional PPI tasks."""
    left, right = str(a), str(b)
    return (left, right) if left <= right else (right, left)
