"""Small deterministic helpers for factual dataset audits.

This module deliberately does not assign biological meaning to labels.  It only
normalizes sequence text, constructs pair keys, counts row-level properties, and
computes exact set intersections.  Source-specific parsing and scientific
interpretation remain outside this package.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppi_shift.data import normalize_sequence, unordered_pair_key

LabelNormalizer = Callable[[str], str | None]


@dataclass(frozen=True)
class TableData:
    """A delimited or parquet table with stable string-valued row fields."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def ordered_pair_key(a: Any, b: Any) -> tuple[str, str]:
    """Return an orientation-preserving pair key."""

    return (str(a), str(b))


def canonical_pair_key(a: Any, b: Any, *, ordered: bool = False) -> tuple[str, str]:
    """Return either an ordered or non-directional pair key."""

    return ordered_pair_key(a, b) if ordered else unordered_pair_key(str(a), str(b))


def exact_sequence_pair_key(a: Any, b: Any, *, ordered: bool = False) -> tuple[str, str]:
    """Return an exact sequence-pair key after whitespace/case normalization."""

    return canonical_pair_key(
        normalize_sequence(str(a)), normalize_sequence(str(b)), ordered=ordered
    )


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_handle(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _delimiter_for(path: Path, sample: str, delimiter: str | None) -> str:
    if delimiter is not None:
        return delimiter
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".tsv" in suffixes or ".tab" in suffixes:
        return "\t"
    if ".csv" in suffixes:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def read_table(path: str | Path, *, delimiter: str | None = None) -> TableData:
    """Read CSV/TSV/TAB (optionally gzipped) or parquet rows.

    Delimited values are returned as strings, with missing cells represented by
    the empty string.  Parquet support is delegated to pandas because it is an
    optional engine concern; a clear error is raised if no parquet engine exists.
    """

    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            frame = pd.read_parquet(table_path)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("Parquet reading requires pandas and a parquet engine") from exc
        except (ValueError, OSError) as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(f"Could not read parquet table {table_path}") from exc
        columns = tuple(str(column) for column in frame.columns)
        rows = tuple(
            {column: "" if value is None else str(value) for column, value in row.items()}
            for row in frame.to_dict(orient="records")
        )
        return TableData(columns=columns, rows=rows)

    with _text_handle(table_path) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter_value = _delimiter_for(table_path, sample, delimiter)
        reader = csv.DictReader(handle, delimiter=delimiter_value)
        columns = tuple(reader.fieldnames or ())
        rows = tuple(
            {
                column: "" if row.get(column) is None else str(row.get(column, ""))
                for column in columns
            }
            for row in reader
        )
    return TableData(columns=columns, rows=rows)


def _row_value(row: Mapping[str, Any], column: str) -> str:
    value = row.get(column, "")
    return "" if value is None else str(value).strip()


def _key_sets(
    rows: Iterable[Mapping[str, Any]],
    *,
    pair_columns: tuple[str, str],
    sequence_columns: tuple[str, str],
) -> dict[str, set[tuple[str, str]]]:
    pair_keys: set[tuple[str, str]] = set()
    protein_keys: set[str] = set()
    sequence_keys: set[str] = set()
    sequence_pair_keys: set[tuple[str, str]] = set()
    for row in rows:
        left_id, right_id = (_row_value(row, column) for column in pair_columns)
        left_seq, right_seq = (
            normalize_sequence(_row_value(row, column)) for column in sequence_columns
        )
        if left_id and right_id:
            pair_keys.add(unordered_pair_key(left_id, right_id))
            protein_keys.update((left_id, right_id))
        if left_seq and right_seq:
            sequence_pair_keys.add(unordered_pair_key(left_seq, right_seq))
            sequence_keys.update((left_seq, right_seq))
    return {
        "pair": pair_keys,
        "protein": {(value, "") for value in protein_keys},
        "sequence": {(value, "") for value in sequence_keys},
        "sequence_pair": sequence_pair_keys,
    }


def overlap_rows(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
    *,
    pair_columns: tuple[str, str] = ("name1", "name2"),
    sequence_columns: tuple[str, str] = ("seq1", "seq2"),
) -> dict[str, int]:
    """Compute exact unordered pair/protein/sequence intersections."""

    left = _key_sets(left_rows, pair_columns=pair_columns, sequence_columns=sequence_columns)
    right = _key_sets(right_rows, pair_columns=pair_columns, sequence_columns=sequence_columns)
    return {
        "pair_overlap": len(left["pair"] & right["pair"]),
        "protein_overlap": len(left["protein"] & right["protein"]),
        "exact_sequence_overlap": len(left["sequence"] & right["sequence"]),
        "exact_sequence_pair_overlap": len(left["sequence_pair"] & right["sequence_pair"]),
    }


def audit_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    pair_columns: tuple[str, str] = ("name1", "name2"),
    sequence_columns: tuple[str, str] = ("seq1", "seq2"),
    label_column: str | None = None,
    label_normalizer: LabelNormalizer | None = None,
) -> dict[str, Any]:
    """Return factual row, key, duplicate, conflict, and sequence statistics.

    Duplicate counts are ``complete rows - unique keys``.  Conflict counts are
    distinct complete pair keys having more than one raw label, or more than one
    normalized label when ``label_normalizer`` is supplied.  No label conversion
    is performed implicitly.
    """

    materialized = [dict(row) for row in rows]
    pair_values: list[tuple[str, str]] = []
    ordered_values: list[tuple[str, str]] = []
    sequence_pair_values: list[tuple[str, str]] = []
    protein_to_sequences: dict[str, set[str]] = defaultdict(set)
    sequence_values: set[str] = set()
    protein_values: set[str] = set()
    lengths: list[int] = []
    missing_identifier_rows = 0
    missing_sequence_rows = 0
    empty_sequence_cells = 0
    self_pair_rows = 0
    identical_sequence_pair_rows = 0

    raw_labels: Counter[str] = Counter()
    normalized_labels: Counter[str] = Counter()
    labels_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    normalized_labels_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in materialized:
        left_id, right_id = (_row_value(row, column) for column in pair_columns)
        left_seq, right_seq = (
            normalize_sequence(_row_value(row, column)) for column in sequence_columns
        )
        complete_pair = bool(left_id and right_id)
        complete_sequence_pair = bool(left_seq and right_seq)
        if not complete_pair:
            missing_identifier_rows += 1
        else:
            pair_key = unordered_pair_key(left_id, right_id)
            pair_values.append(pair_key)
            ordered_values.append(ordered_pair_key(left_id, right_id))
            protein_values.update((left_id, right_id))
            self_pair_rows += int(left_id == right_id)
        if not complete_sequence_pair:
            missing_sequence_rows += 1
        else:
            sequence_pair_values.append(unordered_pair_key(left_seq, right_seq))
            sequence_values.update((left_seq, right_seq))
            lengths.extend((len(left_seq), len(right_seq)))
            identical_sequence_pair_rows += int(left_seq == right_seq)
        empty_sequence_cells += int(not left_seq) + int(not right_seq)
        if complete_pair and complete_sequence_pair:
            protein_to_sequences[left_id].add(left_seq)
            protein_to_sequences[right_id].add(right_seq)

        if label_column is not None:
            raw_label = _row_value(row, label_column)
            raw_labels[raw_label] += 1
            if complete_pair:
                labels_by_pair[unordered_pair_key(left_id, right_id)].add(raw_label)
            if label_normalizer is not None:
                normalized = label_normalizer(raw_label)
                if normalized is not None:
                    normalized_value = str(normalized)
                    normalized_labels[normalized_value] += 1
                    if complete_pair:
                        normalized_labels_by_pair[unordered_pair_key(left_id, right_id)].add(
                            normalized_value
                        )

    duplicate_sequence_groups = sum(
        len(sequence_ids) > 1 for sequence_ids in protein_to_sequences.values()
    )
    duplicate_sequence_mappings = sum(
        max(0, len(sequence_ids) - 1) for sequence_ids in protein_to_sequences.values()
    )
    conflicting_raw_label_pairs = sum(len(values) > 1 for values in labels_by_pair.values())
    conflicting_normalized_label_pairs = sum(
        len(values) > 1 for values in normalized_labels_by_pair.values()
    )
    conflicting_label_pairs = (
        conflicting_raw_label_pairs
        if label_normalizer is None
        else conflicting_normalized_label_pairs
    )

    result: dict[str, Any] = {
        "rows": len(materialized),
        "unique_unordered_pairs": len(set(pair_values)),
        "unique_ordered_pairs": len(set(ordered_values)),
        "unique_proteins": len(protein_values),
        "unique_sequences": len(sequence_values),
        "unique_sequence_pairs": len(set(sequence_pair_values)),
        "duplicate_pair_rows": len(pair_values) - len(set(pair_values)),
        "duplicate_sequence_pair_rows": len(sequence_pair_values) - len(set(sequence_pair_values)),
        "conflicting_label_pairs": conflicting_label_pairs,
        "conflicting_raw_label_pairs": conflicting_raw_label_pairs,
        "conflicting_normalized_label_pairs": conflicting_normalized_label_pairs,
        "missing_identifier_rows": missing_identifier_rows,
        "missing_sequence_rows": missing_sequence_rows,
        "empty_sequence_cells": empty_sequence_cells,
        "self_pair_rows": self_pair_rows,
        "identical_sequence_pair_rows": identical_sequence_pair_rows,
        "duplicate_exact_sequence_groups_across_identifiers": duplicate_sequence_groups,
        "duplicate_exact_sequence_mappings_across_identifiers": duplicate_sequence_mappings,
        "sequence_length_min": min(lengths) if lengths else None,
        "sequence_length_max": max(lengths) if lengths else None,
        "sequence_length_mean": (sum(lengths) / len(lengths)) if lengths else None,
        "raw_label_counts": dict(sorted(raw_labels.items())),
    }
    if label_normalizer is not None:
        result["normalized_label_counts"] = dict(sorted(normalized_labels.items()))
    return result


def audit_table(
    path: str | Path,
    *,
    pair_columns: tuple[str, str] = ("name1", "name2"),
    sequence_columns: tuple[str, str] = ("seq1", "seq2"),
    label_column: str | None = None,
    label_normalizer: LabelNormalizer | None = None,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Read and audit one supported table, including its file hash and columns."""

    table = read_table(path, delimiter=delimiter)
    result = audit_rows(
        table.rows,
        pair_columns=pair_columns,
        sequence_columns=sequence_columns,
        label_column=label_column,
        label_normalizer=label_normalizer,
    )
    result.update(
        {
            "path": str(Path(path)),
            "columns": list(table.columns),
            "sha256": file_sha256(path),
        }
    )
    return result


def binary_label(value: str) -> str | None:
    """Normalize common binary spellings without changing unknown values."""

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "positive", "pos", "yes"}:
        return "1"
    if normalized in {"0", "false", "negative", "neg", "no"}:
        return "0"
    return None


def finite_number(value: float | int | None) -> bool:
    """Small public predicate useful to report JSON-safe optional statistics."""

    return value is not None and math.isfinite(float(value))
