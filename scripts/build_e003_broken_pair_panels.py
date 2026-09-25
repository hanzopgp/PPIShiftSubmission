#!/usr/bin/env python3
"""Build deterministic label-stratified broken-pair E003 diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from e003_signal_common import (
    ENVIRONMENTS,
    base_metadata,
    load_signal_panel,
    parse_csv_values,
    write_json,
)

from ppi_shift.signal_audit import (
    _pair_sequence_key,
    build_broken_pair_panel,
    sha256_path,
    write_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("audit/E003_SIGNAL_AUDIT_PROTOCOL.md")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--environments")
    args = parser.parse_args()
    environments = parse_csv_values(args.environments, ENVIRONMENTS)
    panel_rows: dict[str, tuple[list[dict[str, str]], Path]] = {}
    known_positive_pairs: set[str] = set()
    for environment in environments:
        rows, path = load_signal_panel(args.panel_root, "test", environment)
        panel_rows[environment] = (rows, path)
        known_positive_pairs.update(
            _pair_sequence_key(str(row.get("seq1", "")), str(row.get("seq2", "")))
            for row in rows
            if str(row.get("interaction_label", "")) == "1"
        )
    results = []
    for environment in environments:
        rows, input_path = panel_rows[environment]
        broken, audit = build_broken_pair_panel(rows, known_positive_pairs=known_positive_pairs)
        output_path = args.output_dir / f"test__{environment}__broken.csv.gz"
        write_rows(output_path, broken)
        audit.update(
            {
                "environment": environment,
                "input": {"path": str(input_path), "sha256": sha256_path(input_path)},
                "output": {"path": str(output_path), "sha256": sha256_path(output_path)},
            }
        )
        write_json(args.output_dir / f"test__{environment}__broken.audit.json", audit)
        results.append(audit)
    metadata = base_metadata(args.repo_root, args.protocol)
    metadata.update(
        {
            "control": "broken_pair_panel",
            "environments": list(environments),
            "known_positive_pair_count": len(known_positive_pairs),
            "results": results,
        }
    )
    write_json(args.output_dir / "broken_pair_manifest.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
