"""Small deterministic helpers shared by the pre-freeze data layer.

The large scientific tables are deliberately streamed and written outside Git.  This
module keeps hashing, delimited input, FASTA input, and manifest bookkeeping in one
place so that each D-package uses the same low-level conventions.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def sha256_path(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any], *, omit_key: str | None = None) -> str:
    """Hash a JSON mapping using the canonical request-manifest encoding."""

    payload = dict(value)
    if omit_key is not None:
        payload.pop(omit_key, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def open_text(path: str | Path) -> TextIO:
    """Open a UTF-8 text or gzip-compressed text file."""

    file_path = Path(path)
    if file_path.name.lower().endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", newline="")
    return file_path.open("r", encoding="utf-8", newline="")


def delimiter_for(path: str | Path, *, sample: str = "", delimiter: str | None = None) -> str:
    """Resolve a delimiter from an explicit value, suffix, or a small sample."""

    if delimiter is not None:
        return delimiter
    file_path = Path(path)
    suffixes = {suffix.lower() for suffix in file_path.suffixes}
    if ".tsv" in suffixes or ".tab" in suffixes:
        return "\t"
    if ".csv" in suffixes:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def iter_delimited(
    path: str | Path,
    *,
    delimiter: str | None = None,
) -> Iterator[dict[str, str]]:
    """Stream a CSV/TSV table as string-valued dictionaries."""

    file_path = Path(path)
    with open_text(file_path) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.DictReader(
            handle, delimiter=delimiter_for(file_path, sample=sample, delimiter=delimiter)
        )
        for row in reader:
            yield {
                column: "" if row.get(column) is None else str(row.get(column, ""))
                for column in (reader.fieldnames or ())
            }


def delimited_header(path: str | Path, *, delimiter: str | None = None) -> tuple[str, ...]:
    """Read only the header of a delimited table."""

    file_path = Path(path)
    with open_text(file_path) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.reader(
            handle, delimiter=delimiter_for(file_path, sample=sample, delimiter=delimiter)
        )
        return tuple(next(reader, ()))


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Stream FASTA records, preserving the first whitespace-delimited ID."""

    identifier: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(chunks)
                identifier = line[1:].split(None, 1)[0]
                chunks = []
            elif identifier is not None:
                chunks.append(line)
    if identifier is not None:
        yield identifier, "".join(chunks)


def write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    delimiter: str = ",",
) -> int:
    """Write deterministic UTF-8 delimited rows and return the data-row count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field, "")
                    for field in fieldnames
                }
            )
            count += 1
    return count


def write_fasta(path: str | Path, records: Iterable[tuple[str, str]]) -> int:
    """Write deterministic two-line FASTA records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")
            count += 1
    return count


def git_commit(repo: str | Path) -> str:
    """Return the checked-out commit, or an explicit unknown marker."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a small YAML manifest with stable key order."""

    import yaml

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
