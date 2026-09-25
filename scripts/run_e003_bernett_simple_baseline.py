#!/usr/bin/env python3
"""Run the exact-representation Bernett-style low-capacity diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
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
    EmbeddingStore,
    _fit_classifier,
    embedding_contract_mismatches,
    fit_pair_classifier_audit,
    pair_embedding_features,
    row_label,
)


def _blocked(environment: str, seed: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "ppi_shift.e003_signal_audit.v1",
        "status": "BLOCKED_MISSING_FEATURE",
        "control": "bernett_simple",
        "environment": environment,
        "seed": seed,
        "reason": reason,
        "test_labels_used_for_fit": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("audit/E003_SIGNAL_AUDIT_PROTOCOL.md")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e003_signal_audit.yaml")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--environments")
    parser.add_argument("--seeds")
    args = parser.parse_args()
    environments = parse_csv_values(args.environments, ENVIRONMENTS)
    seeds = tuple(
        int(value) for value in parse_csv_values(args.seeds, tuple(str(seed) for seed in SEEDS))
    )
    expected = yaml.safe_load(args.config.read_text(encoding="utf-8"))["inputs"]["representation"]
    results = []
    input_panels = {}
    store = None
    store_error = ""
    feature_errors: dict[str, str] = {}
    if not args.embeddings or not args.records or not args.summary:
        store_error = "No complete exact ESM2 embedding store arguments were supplied"
    else:
        try:
            store = EmbeddingStore.from_files(args.embeddings, args.records, args.summary)
            mismatches = embedding_contract_mismatches(store, expected)
            if mismatches:
                store_error = "Representation contract mismatch: " + "; ".join(mismatches)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            store_error = f"Could not load exact ESM2 embedding store: {error}"
    for environment in environments:
        train_rows, train_path = load_signal_panel(args.panel_root, "train", environment)
        test_rows, test_path = load_signal_panel(args.panel_root, "test", environment)
        input_panels[environment] = {
            "train": {"path": str(train_path)},
            "test": {"path": str(test_path)},
        }
        feature_error = store_error
        cached_train = cached_test = None
        cached_names = None
        full_fit = None
        if not feature_error:
            try:
                cached_train, cached_names = pair_embedding_features(train_rows, store)
                cached_test, test_names = pair_embedding_features(test_rows, store)
                if cached_names != test_names:
                    raise ValueError("Cached train and test feature names differ")
                full_fit = _fit_classifier(
                    cached_train,
                    np.asarray([row_label(row) for row in train_rows], dtype=np.int64),
                    seeds[0],
                )
            except (OSError, ValueError, KeyError) as error:
                feature_error = str(error)
        if feature_error:
            feature_errors[environment] = feature_error
        for seed in seeds:
            if feature_error:
                results.append(_blocked(environment, seed, feature_error))
                continue
            try:

                def cached_builder(rows, train=train_rows, test=test_rows):
                    if rows is train:
                        return cached_train, cached_names
                    if rows is test:
                        return cached_test, cached_names
                    raise ValueError("Unexpected row table passed to cached feature builder")

                result = fit_pair_classifier_audit(
                    train_rows,
                    test_rows,
                    feature_family="bernett_simple",
                    seed=seed,
                    feature_builder=cached_builder,
                    full_fit=full_fit,
                )
            except (OSError, ValueError, KeyError) as error:
                result = _blocked(environment, seed, str(error))
            result["environment"] = environment
            results.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [metric_row(result, environment=result["environment"]) for result in results]
    write_csv(args.output_dir / "bernett_simple_baseline.csv", rows)
    write_csv(args.output_dir / "e003_bernett_simple_baseline.csv", rows)
    metadata = base_metadata(args.repo_root, args.protocol)
    metadata.update(
        {
            "control": "bernett_simple",
            "environments": list(environments),
            "seeds": list(seeds),
            "representation": expected,
            "representation_error": store_error,
            "feature_errors": feature_errors,
            "input_panels": input_panels,
            "results": results,
        }
    )
    write_json(args.output_dir / "bernett_simple_baseline.json", metadata)
    write_json(args.output_dir / "e003_bernett_simple_baseline.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
