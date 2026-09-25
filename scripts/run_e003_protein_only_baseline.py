#!/usr/bin/env python3
"""Run the two fixed protein-only E003 signal controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from e003_signal_common import (
    ENVIRONMENTS,
    SEEDS,
    base_metadata,
    load_signal_panel,
    metric_row,
    parse_csv_values,
    write_csv,
    write_json,
)

from ppi_shift.signal_audit import fit_empirical_endpoint_audit, fit_protein_sequence_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("audit/E003_SIGNAL_AUDIT_PROTOCOL.md")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--environments")
    parser.add_argument("--seeds")
    args = parser.parse_args()
    environments = parse_csv_values(args.environments, ENVIRONMENTS)
    seeds = tuple(
        int(value) for value in parse_csv_values(args.seeds, tuple(str(seed) for seed in SEEDS))
    )
    results = []
    input_panels = {}
    for environment in environments:
        train_rows, train_path = load_signal_panel(args.panel_root, "train", environment)
        test_rows, test_path = load_signal_panel(args.panel_root, "test", environment)
        input_panels[environment] = {
            "train": {"path": str(train_path)},
            "test": {"path": str(test_path)},
        }
        for seed in seeds:
            empirical = fit_empirical_endpoint_audit(train_rows, test_rows, seed=seed)
            empirical["environment"] = environment
            results.append(empirical)
            sequence = fit_protein_sequence_audit(train_rows, test_rows, seed=seed)
            sequence["environment"] = environment
            results.append(sequence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [metric_row(result, environment=result["environment"]) for result in results]
    write_csv(args.output_dir / "protein_only_baseline.csv", rows)
    write_csv(args.output_dir / "e003_protein_only_results.csv", rows)
    metadata = base_metadata(args.repo_root, args.protocol)
    metadata.update(
        {
            "controls": ["empirical_endpoint_propensity", "single_protein_sequence_logistic"],
            "environments": list(environments),
            "seeds": list(seeds),
            "input_panels": input_panels,
            "results": results,
        }
    )
    write_json(args.output_dir / "protein_only_baseline.json", metadata)
    write_json(args.output_dir / "e003_protein_only_results.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
