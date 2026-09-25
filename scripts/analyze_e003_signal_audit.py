#!/usr/bin/env python3
"""Validate and summarize the machine-readable E003 signal-audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from e003_signal_common import ENVIRONMENTS, SEEDS, base_metadata, write_csv, write_json
except ModuleNotFoundError:  # Imported as ``scripts.analyze_e003_signal_audit`` in tests.
    from scripts.e003_signal_common import ENVIRONMENTS, SEEDS, base_metadata, write_csv, write_json

from ppi_shift.signal_audit import sha256_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def _broken_metric_rows(audit_dir: Path) -> tuple[list[dict[str, str]], list[Path]]:
    paths = sorted(audit_dir.rglob("broken_metrics__*.csv"))
    rows = [row for path in paths for row in _read_csv(path)]
    return rows, paths


def _status_is_false(value: Any) -> bool:
    return value is False or str(value).lower() == "false"


def _validate_signal_csv(
    path: Path,
    *,
    expected_controls: set[str],
    expected_per_control: int,
    require_pass: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"missing:{path}"]
    rows = _read_csv(path)
    errors: list[str] = []
    keys: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        control = str(row.get("control", ""))
        environment = str(row.get("environment", ""))
        seed = str(row.get("seed", ""))
        keys[(control, environment, seed)] += 1
        if control not in expected_controls:
            errors.append(f"unexpected_control:{control}")
        if environment not in ENVIRONMENTS:
            errors.append(f"unexpected_environment:{environment}")
        if seed not in {str(seed_value) for seed_value in SEEDS}:
            errors.append(f"unexpected_seed:{seed}")
        status = str(row.get("status", ""))
        if require_pass and status != "PASS":
            errors.append(f"non_pass:{control}:{environment}:{seed}:{status}")
        if status == "PASS" and not _status_is_false(row.get("test_labels_used_for_fit")):
            errors.append(f"test_label_fit_flag:{control}:{environment}:{seed}")
        if status == "PASS" and not row.get("test_auprc"):
            errors.append(f"missing_metric:{control}:{environment}:{seed}")
    errors.extend(
        f"duplicate_cell:{key[0]}:{key[1]}:{key[2]}" for key, count in keys.items() if count > 1
    )
    expected_count = len(expected_controls) * expected_per_control
    if len(rows) != expected_count:
        errors.append(f"row_count:{len(rows)}:expected:{expected_count}")
    return rows, sorted(set(errors))


def _validate_core_registry(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "BLOCKED_MISSING_CORE_REGISTRY",
            "rows": 0,
            "errors": ["argument_missing"],
        }
    if not path.is_file():
        return {"status": "BLOCKED_MISSING_CORE_REGISTRY", "rows": 0, "errors": [str(path)]}
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows_data", payload.get("cells", []))
    else:
        rows = _read_csv(path)
    if not isinstance(rows, list):
        return {"status": "FAIL", "rows": 0, "errors": ["registry_rows_not_list"]}
    keys: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        scope = str(row.get("scope", "core"))
        if scope != "core":
            continue
        keys[
            (
                str(row.get("model", "")),
                str(row.get("training_environment", row.get("train_environment", ""))),
                str(row.get("test_environment", "")),
                str(row.get("seed", "")),
            )
        ] += 1
    errors = [f"duplicate_core_cell:{key}" for key, count in keys.items() if count > 1]
    if len(keys) != 432:
        errors.append(f"core_cell_count:{len(keys)}:expected:432")
    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(keys),
        "errors": errors,
        "path": str(path),
        "sha256": sha256_path(path),
    }


def _validate_broken_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "BLOCKED_MISSING_BROKEN_PANELS", "errors": [str(path)]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    errors: list[str] = []
    observed = {str(row.get("environment", "")) for row in rows}
    if observed != set(ENVIRONMENTS):
        errors.append(f"environment_set:{sorted(observed)}")
    for row in rows:
        if row.get("status") != "PASS":
            errors.append(f"non_pass:{row.get('environment')}:{row.get('status')}")
        if not row.get("endpoint_marginals_preserved"):
            errors.append(f"endpoint_marginal_failure:{row.get('environment')}")
        if not row.get("label_marginals_preserved"):
            errors.append(f"label_marginal_failure:{row.get('environment')}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(rows),
        "errors": sorted(set(errors)),
        "path": str(path),
        "sha256": sha256_path(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol", type=Path, default=Path("audit/E003_SIGNAL_AUDIT_PROTOCOL.md")
    )
    parser.add_argument("--core-registry", type=Path)
    parser.add_argument("--broken-dir", type=Path)
    parser.add_argument("--allow-blocked-optional", action="store_true")
    args = parser.parse_args()
    easy_path = _first_existing(
        args.audit_dir / "easy_feature_audit.csv",
        args.audit_dir / "e003_easy_feature_shortcuts.csv",
    )
    protein_path = _first_existing(
        args.audit_dir / "protein_only_baseline.csv",
        args.audit_dir / "e003_protein_only_results.csv",
    )
    easy_rows, easy_errors = _validate_signal_csv(
        easy_path,
        expected_controls={"easy_sequence_degree"},
        expected_per_control=len(ENVIRONMENTS) * len(SEEDS),
        require_pass=True,
    )
    protein_rows, protein_errors = _validate_signal_csv(
        protein_path,
        expected_controls={"empirical_endpoint_propensity", "single_protein_sequence_logistic"},
        expected_per_control=len(ENVIRONMENTS) * len(SEEDS),
        require_pass=True,
    )
    bernett_path = _first_existing(
        args.audit_dir / "bernett_simple_baseline.csv",
        args.audit_dir / "e003_bernett_simple_baseline.csv",
    )
    bernett_rows = _read_csv(bernett_path) if bernett_path.is_file() else []
    bernett_errors = []
    if bernett_rows:
        _rows, bernett_errors = _validate_signal_csv(
            bernett_path,
            expected_controls={"bernett_simple"},
            expected_per_control=len(ENVIRONMENTS) * len(SEEDS),
            require_pass=False,
        )
    broken_manifest = args.broken_dir / "broken_pair_manifest.json" if args.broken_dir else None
    if broken_manifest is None or not broken_manifest.is_file():
        broken_manifest = args.audit_dir / "broken_pair_manifest.json"
    if not broken_manifest.is_file():
        nested = args.audit_dir / "broken_pairs" / "broken_pair_manifest.json"
        if nested.is_file():
            broken_manifest = nested
    broken = _validate_broken_manifest(broken_manifest)
    broken_rows, broken_paths = _broken_metric_rows(args.audit_dir)
    core = _validate_core_registry(args.core_registry)
    statuses = {
        "easy_feature": "PASS" if not easy_errors else "FAIL",
        "protein_only": "PASS" if not protein_errors else "FAIL",
        "bernett_simple": "PASS" if bernett_rows and not bernett_errors else "BLOCKED_OPTIONAL",
        "broken_pairs": broken["status"],
        "core_registry": core["status"],
    }
    required_failures = [
        name
        for name, status in statuses.items()
        if status in {"FAIL", "BLOCKED_MISSING_CORE_REGISTRY", "BLOCKED_MISSING_BROKEN_PANELS"}
    ]
    if not args.allow_blocked_optional and statuses["bernett_simple"] == "BLOCKED_OPTIONAL":
        required_failures.append("bernett_simple_optional")
    summary_rows = [
        {
            "surface": "easy_feature",
            "status": statuses["easy_feature"],
            "rows": len(easy_rows),
            "errors": " | ".join(easy_errors),
        },
        {
            "surface": "protein_only",
            "status": statuses["protein_only"],
            "rows": len(protein_rows),
            "errors": " | ".join(protein_errors),
        },
        {
            "surface": "bernett_simple",
            "status": statuses["bernett_simple"],
            "rows": len(bernett_rows),
            "errors": " | ".join(bernett_errors),
        },
        {
            "surface": "broken_pairs",
            "status": broken["status"],
            "rows": broken.get("rows", 0),
            "errors": " | ".join(broken.get("errors", [])),
        },
        {
            "surface": "core_registry",
            "status": core["status"],
            "rows": core.get("rows", 0),
            "errors": " | ".join(core.get("errors", [])),
        },
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "e003_signal_audit_summary.csv", summary_rows)
    metadata = base_metadata(args.repo_root, args.protocol)
    metadata.update(
        {
            "status": "PASS" if not required_failures else "BLOCKED_OR_FAIL",
            "required_failures": sorted(set(required_failures)),
            "surfaces": statuses,
            "summary_rows": summary_rows,
            "source_files": {
                name: {"path": str(path), "sha256": sha256_path(path)}
                for name, path in {
                    "easy": easy_path,
                    "protein": protein_path,
                    "bernett": bernett_path,
                    "broken": broken_manifest,
                    "broken_metrics": broken_paths[0] if broken_paths else None,
                }.items()
                if path is not None and path.is_file()
            },
        }
    )
    write_json(args.output_dir / "e003_signal_audit_summary.json", metadata)
    if broken_rows:
        write_csv(args.output_dir / "e003_broken_pair_results.csv", broken_rows)
        write_json(
            args.output_dir / "e003_broken_pair_results.json",
            {
                **metadata,
                "control": "broken_pair_score",
                "rows": len(broken_rows),
                "metric_files": [
                    {"path": str(path), "sha256": sha256_path(path)}
                    for path in broken_paths
                ],
                "results": broken_rows,
            },
        )
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
