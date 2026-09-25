#!/usr/bin/env python3
"""Run the frozen easy-feature positive-vs-negative signal audit."""

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

from ppi_shift.signal_audit import (
    build_positive_degree_table,
    fit_pair_classifier_audit,
    pair_easy_features,
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
        degree_table = build_positive_degree_table(train_rows)

        def split_degree_features(rows, *, reference_rows):
            return pair_easy_features(
                rows,
                degree_table=build_positive_degree_table(reference_rows),
            )

        for seed in seeds:
            result = fit_pair_classifier_audit(
                train_rows,
                test_rows,
                feature_family="easy_sequence_degree",
                seed=seed,
                feature_builder=lambda rows, table=degree_table: pair_easy_features(
                    rows, degree_table=table
                ),
                split_feature_builder=split_degree_features,
            )
            result["environment"] = environment
            results.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [metric_row(result, environment=result["environment"]) for result in results]
    write_csv(args.output_dir / "easy_feature_audit.csv", rows)
    write_csv(args.output_dir / "e003_easy_feature_shortcuts.csv", rows)
    metadata = base_metadata(args.repo_root, args.protocol)
    metadata.update(
        {
            "control": "easy_sequence_degree",
            "environments": list(environments),
            "seeds": list(seeds),
            "input_panels": input_panels,
            "results": results,
        }
    )
    write_json(args.output_dir / "easy_feature_audit.json", metadata)
    write_json(args.output_dir / "e003_easy_feature_shortcuts.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
