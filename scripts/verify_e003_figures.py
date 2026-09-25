#!/usr/bin/env python3
"""Verify E003 figure files, source data, hashes, and plotted core values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import git_commit, sha256_path

EXPECTED_RENDERED_SUFFIXES = {".pdf", ".svg", ".png"}
CORE_MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)
CORE_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
CORE_SEEDS = (11, 22, 33)
# PNG pHYs stores integer pixels per metre; Matplotlib's exact 300-dpi output
# is therefore decoded as 299.9994 dpi.
MIN_PNG_DPI = 299.9


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON mapping: {path}")
    return value


def _close(actual: Any, expected: float, *, label: str, tolerance: float = 5e-12) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value, float(expected), rel_tol=0.0, abs_tol=tolerance
    ):
        raise SystemExit(f"Figure source value mismatch for {label}: {value} != {expected}")


def _png_dpi(path: Path) -> tuple[float, float] | None:
    """Read PNG pHYs resolution without requiring Pillow."""

    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"Not a PNG file: {path}")
        while True:
            header = handle.read(8)
            if len(header) != 8:
                return None
            length, chunk_type = struct.unpack(">I4s", header)
            payload = handle.read(length)
            handle.read(4)  # CRC
            if chunk_type == b"pHYs" and len(payload) == 9:
                x_ppm, y_ppm, unit = struct.unpack(">IIB", payload)
                if unit != 1:
                    return None
                return (x_ppm * 0.0254, y_ppm * 0.0254)
            if chunk_type == b"IEND":
                return None


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _core_means(
    registry_rows: list[dict[str, Any]], metric: str = "auprc"
) -> dict[tuple[str, str, str, str], float]:
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in registry_rows:
        if row.get("scope") != "core":
            continue
        key = (
            str(row["model"]),
            str(row["training_environment"]),
            str(row["test_environment"]),
            metric,
        )
        values[key].append(float(row[metric]))
    expected: dict[tuple[str, str, str, str], float] = {}
    for key, items in values.items():
        expected[key] = sum(items) / len(items)
    return expected


def _core_metric_values(
    registry_rows: list[dict[str, Any]], metric: str
) -> dict[tuple[str, int, str, str], float]:
    result: dict[tuple[str, int, str, str], float] = {}
    for row in registry_rows:
        if row.get("scope") != "core":
            continue
        key = (
            str(row["model"]),
            int(row["seed"]),
            str(row["training_environment"]),
            str(row["test_environment"]),
        )
        if key in result:
            raise SystemExit(f"Duplicate core registry key: {key}")
        result[key] = float(row[metric])
    return result


def _check_core_grid(registry_rows: list[dict[str, Any]]) -> None:
    expected_keys = {
        (model, seed, train, test)
        for model in CORE_MODELS
        for seed in CORE_SEEDS
        for train in CORE_ENVIRONMENTS
        for test in CORE_ENVIRONMENTS
    }
    observed = {
        (
            str(row["model"]),
            int(row["seed"]),
            str(row["training_environment"]),
            str(row["test_environment"]),
        )
        for row in registry_rows
        if row.get("scope") == "core"
    }
    if observed != expected_keys:
        raise SystemExit(
            f"Core registry grid mismatch: observed={len(observed)} expected={len(expected_keys)}"
        )


def _check_core_source(
    path: Path,
    core_values: dict[tuple[str, str, str, str], float],
    metric: str,
) -> int:
    rows = _csv_rows(path)
    value_field = f"mean_{metric}"
    if not rows:
        raise SystemExit(f"Empty core figure source: {path}")
    for row in rows:
        key = (row["model"], row["train_environment"], row["test_environment"], metric)
        if key not in core_values:
            raise SystemExit(f"Unknown core figure-source key: {key}")
        _close(row[value_field], core_values[key], label=f"{path.name}:{key}")
    return len(rows)


def _check_seed_source(path: Path, registry_rows: list[dict[str, Any]]) -> int:
    rows = _csv_rows(path)
    expected = {
        (
            str(row["model"]),
            int(row["seed"]),
            str(row["training_environment"]),
            str(row["test_environment"]),
        ): row
        for row in registry_rows
        if row.get("scope") == "core"
    }
    for row in rows:
        key = (row["model"], int(row["seed"]), row["train_environment"], row["test_environment"])
        source = expected.get(key)
        if source is None:
            raise SystemExit(f"Unknown seed/calibration source key: {key}")
        for field in ("auprc", "auroc", "ece", "brier", "nll"):
            _close(row[field], float(source[field]), label=f"{path.name}:{key}:{field}")
    return len(rows)


def _check_external_source(
    path: Path,
    registry_rows: list[dict[str, Any]],
    core_values: dict[tuple[str, int, str, str], float],
) -> int:
    rows = _csv_rows(path)
    expected = {
        (
            str(row["model"]),
            int(row["seed"]),
            str(row["training_environment"]),
            str(row["test_environment"]),
        ): row
        for row in registry_rows
        if row.get("scope") == "external"
    }
    fields = (
        "auprc",
        "auroc",
        "prevalence",
        "auprc_minus_prevalence",
        "normalized_excess_ap",
        "ap_lift",
        "prevalence_null_probability",
        "prevalence_null_brier",
        "prevalence_null_nll",
        "brier_skill_vs_prevalence_null",
        "nll_skill_vs_prevalence_null",
    )
    checked = 0
    for row in rows:
        key = (
            row["model"],
            int(row["seed"]),
            row["train_environment"],
            f"{row['source']}__{row['surface']}",
        )
        source = expected.get(key)
        if source is None:
            # Gold rows are included in this source file for the internal-vs-
            # Vidal panel and are checked against the core registry below.
            if row["scope"] != "gold_core":
                raise SystemExit(f"Unknown external figure-source key: {key}")
            continue
        metric = row["metric"]
        if metric not in fields:
            raise SystemExit(f"Unknown external plotted metric: {metric}")
        _close(row["value"], float(source[metric]), label=f"{path.name}:{key}:{metric}")
        checked += 1
    for row in rows:
        if row.get("scope") != "gold_core":
            continue
        key = (
            row["model"],
            int(row["seed"]),
            row["train_environment"],
            row["test_environment"],
        )
        if row["metric"] != "auprc" or key not in core_values:
            raise SystemExit(f"Invalid Gold figure-source row: {key}")
        _close(row["value"], core_values[key], label=f"{path.name}:gold:{key}")
        checked += 1
    return checked


def _check_sampler_source(path: Path) -> int:
    payload = _json(path)
    bernett = payload.get("bernett")
    tppni = payload.get("tppni")
    if not isinstance(bernett, dict) or not isinstance(tppni, dict):
        raise SystemExit(f"Sampler figure source lacks Bernett/TPPNI sections: {path}")
    selected_audit = bernett.get("selected_audit")
    if not isinstance(selected_audit, dict):
        raise SystemExit(f"Sampler figure source lacks Bernett selected audit: {path}")
    # The source preserves the complete frozen Bernett audit under this key;
    # the numerical diagnostics used for verification are its nested audit.
    if isinstance(selected_audit.get("selected_audit"), dict):
        selected_audit = selected_audit["selected_audit"]
    expected_fields = (
        "degree_absolute_residual_sum",
        "degree_max_absolute_residual",
        "taxon_pair_absolute_residual_sum",
        "selected_self_loop_count",
        "mean_positive_go_bp_jaccard",
        "mean_selected_go_bp_jaccard",
    )
    missing_fields = [field for field in expected_fields if field not in selected_audit]
    if missing_fields:
        raise SystemExit(
            "Bernett selected audit is missing fields "
            f"{missing_fields}; observed={sorted(selected_audit)}: {path}"
        )
    for field in expected_fields:
        value = float(selected_audit[field])
        if not math.isfinite(value) or value < 0.0:
            raise SystemExit(f"Invalid Bernett sampler diagnostic {field}: {path}")

    diagnostics = tppni.get("diagnostics")
    trace = tppni.get("scm_degree_trace")
    if not isinstance(diagnostics, dict) or not isinstance(trace, dict):
        raise SystemExit(f"Sampler figure source lacks TPPNI SCM diagnostics: {path}")
    scm = diagnostics.get("scm")
    if not isinstance(scm, dict):
        raise SystemExit(f"Sampler figure source lacks nested TPPNI SCM diagnostics: {path}")
    protein_ids = trace.get("protein_ids")
    observed = trace.get("observed")
    expected = trace.get("expected")
    if (
        not isinstance(protein_ids, list)
        or not isinstance(observed, list)
        or not isinstance(expected, list)
        or len(protein_ids) != int(scm["protein_count"])
        or len(observed) != len(expected)
        or len(observed) != len(protein_ids)
        or not all(math.isfinite(float(value)) for value in observed)
        or not all(math.isfinite(float(value)) for value in expected)
    ):
        raise SystemExit(f"Invalid TPPNI SCM degree trace: {path}")
    absolute_residuals = [
        abs(float(actual) - float(target))
        for actual, target in zip(expected, observed, strict=True)
    ]
    _close(
        max(absolute_residuals, default=0.0),
        float(scm["max_degree_residual"]),
        label=f"{path.name}:TPPNI max degree residual",
        tolerance=1e-10,
    )
    _close(
        math.fsum(absolute_residuals) / len(absolute_residuals),
        float(scm["mean_abs_degree_residual"]),
        label=f"{path.name}:TPPNI mean degree residual",
        tolerance=1e-10,
    )
    pool_rows = int(tppni["pool_rows"])
    diagnostics_pool_rows = int(diagnostics["effective_bottom_n"])
    if pool_rows < 1 or pool_rows != diagnostics_pool_rows:
        raise SystemExit(f"TPPNI figure pool size mismatch: {path}")
    published_n = int(diagnostics["published_n"])
    if diagnostics_pool_rows > published_n:
        raise SystemExit(f"TPPNI effective pool exceeds published bottom-N: {path}")
    cutoff = float(tppni["bottom_n_cutoff_probability"])
    if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise SystemExit(f"Invalid TPPNI bottom-N cutoff probability: {path}")
    for histogram_name in ("probability_histogram", "l3_histogram"):
        histogram = tppni.get(histogram_name)
        if not isinstance(histogram, dict):
            raise SystemExit(f"Missing TPPNI histogram {histogram_name}: {path}")
        counts = histogram.get("counts")
        edges = histogram.get("edges")
        if (
            not isinstance(counts, list)
            or not isinstance(edges, list)
            or len(edges) != len(counts) + 1
            or not all(int(value) >= 0 for value in counts)
            or not all(math.isfinite(float(value)) for value in edges)
        ):
            raise SystemExit(f"Invalid TPPNI histogram {histogram_name}: {path}")
    return len(protein_ids) + pool_rows


def _expected_input_hashes(analysis: dict[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for name in (
        "core_results",
        "external_results",
        "panel_manifest",
        "sampler_trace_manifest",
        "config",
        "protocol_freeze",
        "result_registry",
        "datasail_results",
    ):
        reference = analysis.get(name)
        if reference is None:
            continue
        if not isinstance(reference, dict):
            raise SystemExit(f"Invalid analysis input reference: {name}")
        path = Path(str(reference.get("path", "")))
        digest = str(reference.get("sha256", ""))
        if not path.is_file() or sha256_path(path) != digest:
            raise SystemExit(f"Analysis input hash mismatch: {name} ({path})")
        expected[name] = digest
    return expected


def _check_figure_manifest(
    analysis: dict[str, Any], hashes: list[dict[str, Any]], repo_root: Path
) -> tuple[int, int]:
    hash_map = {str(item["path"]): str(item["sha256"]) for item in hashes}
    figures = analysis.get("figure_manifest")
    if not isinstance(figures, list) or not figures:
        raise SystemExit("Analysis has no figure manifest")
    rendered = 0
    sources = 0
    expected_input_hashes = _expected_input_hashes(analysis)
    for item in figures:
        files = [Path(str(value)) for value in item.get("files", [])]
        if {path.suffix.lower() for path in files} != EXPECTED_RENDERED_SUFFIXES:
            raise SystemExit(f"Figure does not contain PDF/SVG/PNG: {item.get('figure')}")
        source = Path(str(item.get("source_data", "")))
        if not source.is_file() or any(
            path.parent.resolve() != source.parent.resolve() for path in files
        ):
            raise SystemExit(f"Figure source is not adjacent to rendered files: {source}")
        if not source.exists():
            raise SystemExit(f"Missing figure source: {source}")
        if hash_map.get(str(source)) != sha256_path(source):
            raise SystemExit(f"Figure source hash mismatch: {source}")
        sources += 1
        script = Path(str(item.get("plotting_script", "")))
        if not script.is_file() or not item.get("plotting_script_sha256"):
            raise SystemExit(f"Figure lacks plotting-script provenance: {item.get('figure')}")
        if sha256_path(script) != str(item["plotting_script_sha256"]):
            raise SystemExit(f"Plotting-script hash mismatch: {script}")
        if item.get("input_hashes") != expected_input_hashes:
            raise SystemExit(f"Figure input-hash set mismatch: {item.get('figure')}")
        script_sha = str(item.get("plotting_script_git_sha", ""))
        if len(script_sha) != 40 or any(
            char not in "0123456789abcdef" for char in script_sha.lower()
        ):
            raise SystemExit(f"Figure lacks a valid plotting-script Git SHA: {item.get('figure')}")
        for path in files:
            if not path.is_file() or hash_map.get(str(path)) != sha256_path(path):
                raise SystemExit(f"Figure hash mismatch or missing file: {path}")
            if path.suffix.lower() == ".png":
                dpi = _png_dpi(path)
                if dpi is None or min(dpi) < MIN_PNG_DPI:
                    raise SystemExit(f"PNG is below 300 dpi or lacks pHYs metadata: {path}")
            elif path.suffix.lower() == ".pdf" and path.read_bytes()[:5] != b"%PDF-":
                raise SystemExit(f"Invalid PDF header: {path}")
            elif path.suffix.lower() == ".svg" and "<svg" not in path.read_text(
                encoding="utf-8"
            )[:2000]:
                raise SystemExit(f"Invalid SVG content: {path}")
            rendered += 1
    if repo_root.is_dir() and git_commit(repo_root) == "unknown":
        raise SystemExit(f"Cannot resolve repository Git SHA: {repo_root}")
    return rendered, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--figure-hashes", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    analysis = _json(args.analysis)
    registry = _json(args.registry)
    hashes_value = json.loads(args.figure_hashes.read_text(encoding="utf-8"))
    if not isinstance(hashes_value, list):
        raise SystemExit("Figure hash manifest is not a list")
    registry_rows = registry.get("rows_data", [])
    if not isinstance(registry_rows, list):
        raise SystemExit("Result registry has no rows_data list")
    _check_core_grid(registry_rows)
    core_auprc = _core_means(registry_rows)
    core_auroc = _core_means(registry_rows, "auroc")
    checked_sources = 0
    sampler_source_values_checked = 0
    figure_sources = {
        Path(str(item["source_data"])).name: Path(str(item["source_data"]))
        for item in analysis["figure_manifest"]
    }
    for name, path in figure_sources.items():
        if name.startswith("figure_data__core_auprc_heatmap__") or name == (
            "figure_data__figure_2_core_response_surfaces.csv"
        ):
            checked_sources += _check_core_source(path, core_auprc, "auprc")
        elif name == "figure_data__supplement_auroc_heatmaps.csv":
            checked_sources += _check_core_source(path, core_auroc, "auroc")
        elif name == "figure_data__supplement_seed_and_calibration.csv":
            checked_sources += _check_seed_source(path, registry_rows)
        elif name == "figure_data__figure_5_external_transfer.csv":
            checked_sources += _check_external_source(
                path, registry_rows, _core_metric_values(registry_rows, "auprc")
            )
        elif name == "figure_data__figure_6_sampler_fidelity.json":
            sampler_source_values_checked += _check_sampler_source(path)

    rendered, sources = _check_figure_manifest(analysis, hashes_value, args.repo_root)
    result = {
        "schema_version": "ppi_shift.e003_figure_integrity_audit.v1",
        "experiment_id": "E003",
        "status": "complete",
        "analysis": {"path": str(args.analysis), "sha256": sha256_path(args.analysis)},
        "registry": {"path": str(args.registry), "sha256": sha256_path(args.registry)},
        "figure_hashes": {
            "path": str(args.figure_hashes),
            "sha256": sha256_path(args.figure_hashes),
        },
        "core_registry_rows": len([row for row in registry_rows if row.get("scope") == "core"]),
        "figures": len(analysis["figure_manifest"]),
        "rendered_files": rendered,
        "adjacent_sources": sources,
        "source_values_checked": checked_sources,
        "sampler_source_values_checked": sampler_source_values_checked,
        "checks": {
            "core_grid": True,
            "input_hashes": True,
            "source_hashes": True,
            "adjacent_source_data": True,
            "pdf_svg_png": True,
            "png_dpi_at_least_300": True,
            "plotting_script_hash_and_git_sha": True,
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
