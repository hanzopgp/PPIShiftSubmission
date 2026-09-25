#!/usr/bin/env python3
"""Build the compact, source-linked E003 figures used by the ICLR paper.

The script is intentionally fail-closed.  It derives every plotted value from
the committed pair-dependence table or from the canonical E003 registry, and
records the source hashes, aggregation rules, and current revision beside the
rendered figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

try:
    from paper_figure_style import (
        AXIS_SIZE,
        PAPER_WIDTH,
        SAMPLER_LABELS,
        apply_style,
        heatmap,
        panel_header,
    )
    from paper_figure_style import (
        MODEL_LABELS as PAPER_MODEL_LABELS,
    )
except ModuleNotFoundError:
    from scripts.paper_figure_style import (
        AXIS_SIZE,
        PAPER_WIDTH,
        SAMPLER_LABELS,
        apply_style,
        heatmap,
        panel_header,
    )
    from scripts.paper_figure_style import (
        MODEL_LABELS as PAPER_MODEL_LABELS,
    )

MODEL_ORDER = (
    "symmetric_frozen_esm2_pair_mlp",
    "tuna_corrected",
    "dscript_esm2",
    "rapppid",
)
MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2 pair MLP",
    "tuna_corrected": "Corrected TUnA",
    "dscript_esm2": "D-SCRIPT-ESM2",
    "rapppid": "RAPPPID",
}
MODEL_SHORT_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}
ENVIRONMENT_ORDER = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
ENVIRONMENT_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Degree\nmatched",
    "sequence_nuisance_matched": "Sequence\nmatched",
    "esm_partner_matched": "ESM2\nmatched",
    "bernett2026_ilp_exact": "Bernett\n2026",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI\nbalanced",
}
ENVIRONMENT_SHORT_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Degree",
    "sequence_nuisance_matched": "Sequence",
    "esm_partner_matched": "ESM2",
    "bernett2026_ilp_exact": "Bernett",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}
MEASURED_SURFACES = ("vidal__measured", "intact__measured")
MEASURED_SURFACE_LABELS = {
    "vidal__measured": "Vidal/Y2H",
    "intact__measured": "IntAct",
}
SEEDS = (11, 22, 33)

apply_style()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def derive_pair_rows(source_path: Path) -> list[dict[str, Any]]:
    """Compute original-minus-shuffled AUROC from the frozen pair table."""

    rows = [row for row in _read_csv(source_path) if row.get("panel") == "B"]
    expected = {
        (model, environment, series)
        for model in MODEL_ORDER
        for environment in ENVIRONMENT_ORDER
        for series in ("Original pairs", "Partners shuffled")
    }
    observed: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row.get("model", ""), row.get("environment", ""), row.get("series", ""))
        if key in observed:
            raise ValueError(f"Duplicate pair-dependence row: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected pair-dependence row: {key}")
        value = float(row["auroc"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite pair-dependence AUROC: {key}")
        observed[key] = value
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"Pair table does not have the frozen 4x6x2 layout: {missing} {extra}")

    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for environment in ENVIRONMENT_ORDER:
            original = observed[(model, environment, "Original pairs")]
            shuffled = observed[(model, environment, "Partners shuffled")]
            delta = original - shuffled
            output.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "test_environment": environment,
                    "test_environment_label": ENVIRONMENT_LABELS[environment].replace("\n", " "),
                    "auroc_original": f"{original:.17g}",
                    "auroc_shuffled": f"{shuffled:.17g}",
                    "delta_auroc": f"{delta:.17g}",
                    "sign_interpretation": (
                        "original_pairing_higher"
                        if delta > 0
                        else "partners_shuffled_higher"
                        if delta < 0
                        else "no_difference"
                    ),
                    "source_seed_aggregation": "mean over frozen seeds 11,22,33",
                }
            )
    return output


def _registry_measured_rows(registry_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = payload.get("rows_data")
    if not isinstance(rows, list):
        raise ValueError("E003 result registry lacks rows_data")
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"E003 registry row {index} is not an object")
        if row.get("scope") == "external" and row.get("evaluation_surface") in MEASURED_SURFACES:
            selected.append(
                {
                    "surface": str(row["evaluation_surface"]),
                    "model": str(row["model"]),
                    "train_environment": str(row["training_environment"]),
                    "seed": int(row["seed"]),
                    "auroc": float(row["auroc"]),
                    "registry_row_index": index,
                }
            )
    expected_count = len(MEASURED_SURFACES) * len(MODEL_ORDER) * len(ENVIRONMENT_ORDER) * len(SEEDS)
    if len(selected) != expected_count:
        raise ValueError(
            f"Measured registry selection has {len(selected)} rows; expected {expected_count}"
        )
    for row in selected:
        if not math.isfinite(row["auroc"]):
            raise ValueError(f"Non-finite measured AUROC: {row}")
    return sorted(
        selected,
        key=lambda row: (
            MEASURED_SURFACES.index(row["surface"]),
            MODEL_ORDER.index(row["model"]),
            ENVIRONMENT_ORDER.index(row["train_environment"]),
            SEEDS.index(row["seed"]),
        ),
    )


def _source_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "surface": row["surface"],
            "model": row["model"],
            "train_environment": row["train_environment"],
            "seed": row["seed"],
            "auroc": f"{row['auroc']:.17g}",
            "registry_row_index": row["registry_row_index"],
        }
        for row in rows
    ]


def _load_measured_source(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    required = {
        "surface",
        "model",
        "train_environment",
        "seed",
        "auroc",
        "registry_row_index",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Measured source CSV lacks required columns: {path}")
    parsed = [
        {
            "surface": row["surface"],
            "model": row["model"],
            "train_environment": row["train_environment"],
            "seed": int(row["seed"]),
            "auroc": float(row["auroc"]),
            "registry_row_index": int(row["registry_row_index"]),
        }
        for row in rows
    ]
    expected = {
        (surface, model, environment, seed)
        for surface in MEASURED_SURFACES
        for model in MODEL_ORDER
        for environment in ENVIRONMENT_ORDER
        for seed in SEEDS
    }
    observed = {
        (row["surface"], row["model"], row["train_environment"], row["seed"]) for row in parsed
    }
    if observed != expected or len(parsed) != len(expected):
        raise ValueError("Measured source CSV does not have the frozen 2x4x6x3 layout")
    return sorted(
        parsed,
        key=lambda row: (
            MEASURED_SURFACES.index(row["surface"]),
            MODEL_ORDER.index(row["model"]),
            ENVIRONMENT_ORDER.index(row["train_environment"]),
            SEEDS.index(row["seed"]),
        ),
    )


def aggregate_measured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["surface"], row["model"], row["train_environment"])].append(
            float(row["auroc"])
        )
    expected_keys = {
        (surface, model, environment)
        for surface in MEASURED_SURFACES
        for model in MODEL_ORDER
        for environment in ENVIRONMENT_ORDER
    }
    if set(grouped) != expected_keys or any(
        sorted(SEEDS)
        != sorted(
            row["seed"]
            for row in rows
            if (row["surface"], row["model"], row["train_environment"]) == key
        )
        for key in expected_keys
    ):
        raise ValueError("Measured source rows do not contain exactly one row per frozen seed")
    output = []
    for surface in MEASURED_SURFACES:
        for model in MODEL_ORDER:
            for environment in ENVIRONMENT_ORDER:
                values = grouped[(surface, model, environment)]
                output.append(
                    {
                        "surface": surface,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "train_environment": environment,
                        "train_environment_label": ENVIRONMENT_LABELS[environment].replace(
                            "\n", " "
                        ),
                        "auroc_mean": float(np.mean(values)),
                        "seed_count": len(values),
                    }
                )
    return output


def validate_measured_summary(
    aggregated: list[dict[str, Any]], summary_path: Path, tolerance: float = 1e-12
) -> None:
    summary = {
        (row["model"], row["surface"]): (float(row["auroc"]), int(row["row_count"]))
        for row in _read_csv(summary_path)
        if row.get("block") == "measured"
    }
    for model in MODEL_ORDER:
        for surface in MEASURED_SURFACES:
            cells = [
                row["auroc_mean"]
                for row in aggregated
                if row["model"] == model and row["surface"] == surface
            ]
            if len(cells) != len(ENVIRONMENT_ORDER):
                raise ValueError(f"Missing measured aggregate cells for {model}/{surface}")
            expected, row_count = summary.get((model, surface), (None, None))
            observed = float(np.mean(cells))
            if row_count != len(ENVIRONMENT_ORDER) * len(SEEDS) or expected is None:
                raise ValueError(f"Missing measured summary row for {model}/{surface}")
            if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
                raise ValueError(
                    f"Measured AUROC discrepancy for {model}/{surface}: "
                    f"registry-derived {observed:.17g} vs summary {expected:.17g}"
                )


def _output_record(path: Path, repo_root: Path, kind: str) -> dict[str, str]:
    return {
        "path": _relative_path(path, repo_root),
        "kind": kind,
        "sha256": sha256_path(path),
    }


def _metadata_input(path: Path, repo_root: Path) -> dict[str, str]:
    return {"path": _relative_path(path, repo_root), "sha256": sha256_path(path)}


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, tight: bool = True) -> list[Path]:
    outputs = []
    for extension in ("pdf", "png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight" if tight else None,
            facecolor="white",
            metadata={"Creator": "PPIShift ICLR 2027 figure build"} if extension != "png" else None,
        )
        if extension == "svg":
            normalized = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            path.write_text(normalized, encoding="utf-8", newline="\n")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_pair_heatmap(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    values = np.asarray([float(row["delta_auroc"]) for row in rows], dtype=np.float64).reshape(
        len(MODEL_ORDER), len(ENVIRONMENT_ORDER)
    )
    limit = max(float(np.max(np.abs(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(PAPER_WIDTH, 2.55))
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1, 0.045), left=0.18, right=0.87, bottom=0.27, top=0.88, wspace=0.18
    )
    axis = fig.add_subplot(grid[0, 0])
    image = heatmap(axis, values, cmap=cmap, norm=norm, annotations="{:+.3f}", annotation_size=9.0)
    axis.set_xticks(range(len(ENVIRONMENT_ORDER)))
    axis.set_xticklabels(
        [SAMPLER_LABELS[key] for key in ENVIRONMENT_ORDER], rotation=38, ha="right"
    )
    axis.set_yticks(range(len(MODEL_ORDER)))
    axis.set_yticklabels([PAPER_MODEL_LABELS[key] for key in MODEL_ORDER])
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 1]))
    colorbar.ax.set_title("ΔAUROC", size=AXIS_SIZE, pad=5)
    colorbar.set_ticks([-limit, 0.0, limit])
    colorbar.set_ticklabels([f"{value:.3f}" for value in (-limit, 0.0, limit)])
    colorbar.ax.tick_params(labelsize=8.8)
    return _save_figure(fig, output_dir, "figure_pair_dependence_heatmap", tight=False)


def plot_measured_heatmaps(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    values = np.asarray([row["auroc_mean"] for row in rows], dtype=np.float64)
    vmin = math.floor(float(np.min(values)) * 20) / 20
    vmax = math.ceil(float(np.max(values)) * 20) / 20
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(PAPER_WIDTH, 4.15))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1, 0.045),
        left=0.18,
        right=0.87,
        bottom=0.18,
        top=0.93,
        hspace=0.34,
        wspace=0.18,
    )
    image = None
    for panel_index, surface in enumerate(MEASURED_SURFACES):
        axis = fig.add_subplot(grid[panel_index, 0])
        surface_rows = [row for row in rows if row["surface"] == surface]
        matrix = np.asarray([row["auroc_mean"] for row in surface_rows], dtype=np.float64).reshape(
            len(MODEL_ORDER), len(ENVIRONMENT_ORDER)
        )
        image = heatmap(
            axis, matrix, cmap=cmap, norm=norm, annotations="{:.3f}", annotation_size=9.0
        )
        panel_label = "a" if panel_index == 0 else "b"
        panel_header(axis, f"{panel_label}  {MEASURED_SURFACE_LABELS[surface]}")
        axis.set_xticks(range(len(ENVIRONMENT_ORDER)))
        if panel_index == 1:
            axis.set_xticklabels(
                [SAMPLER_LABELS[key] for key in ENVIRONMENT_ORDER], rotation=38, ha="right"
            )
        else:
            axis.tick_params(axis="x", labelbottom=False)
        axis.set_yticks(range(len(MODEL_ORDER)))
        axis.set_yticklabels([PAPER_MODEL_LABELS[key] for key in MODEL_ORDER])
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, 1]))
    colorbar.ax.set_title("AUROC", size=AXIS_SIZE, pad=5)
    ticks = sorted({vmin, 0.5, vmax})
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.2f}" for value in ticks])
    colorbar.ax.tick_params(labelsize=8.8)
    colorbar.ax.axhline(0.5, color="#203040", linewidth=0.8)
    return _save_figure(fig, output_dir, "figure_measured_transfer_heatmaps", tight=False)


def _pair_metadata(
    repo_root: Path,
    script_path: Path,
    source_path: Path,
    output_csv: Path,
    outputs: list[Path],
) -> dict[str, Any]:
    return {
        "schema_version": "ppi_shift.iclr2027_figure_provenance.v1",
        "figure": "pair_dependence_heatmap",
        "git_revision": _git_revision(repo_root),
        "script_sha256": sha256_path(script_path),
        "style_sha256": sha256_path(script_path.with_name("paper_figure_style.py")),
        "inputs": [_metadata_input(source_path, repo_root)],
        "source_data_csv": _metadata_input(output_csv, repo_root),
        "formula": "delta_auroc = auroc_original_pairs - auroc_partners_shuffled",
        "seed_aggregation": "input AUROC values are the frozen mean over seeds 11, 22, 33",
        "layout": {
            "models": list(MODEL_ORDER),
            "test_environments": list(ENVIRONMENT_ORDER),
            "display_models": [MODEL_SHORT_LABELS[key] for key in MODEL_ORDER],
            "display_test_environments": [
                ENVIRONMENT_SHORT_LABELS[key] for key in ENVIRONMENT_ORDER
            ],
        },
        "color_scale": "symmetric about zero",
        "interpretation_boundary": (
            "This is a necessity diagnostic on the evaluated test constructions; "
            "it is not proof that a model never uses pair-specific information."
        ),
        "outputs": [_output_record(path, repo_root, "rendered_figure") for path in outputs],
    }


def _measured_metadata(
    repo_root: Path,
    script_path: Path,
    source_csv: Path,
    summary_path: Path,
    registry_path: Path | None,
    outputs: list[Path],
) -> dict[str, Any]:
    inputs = [_metadata_input(source_csv, repo_root), _metadata_input(summary_path, repo_root)]
    if registry_path is not None:
        inputs.append(_metadata_input(registry_path, repo_root))
    return {
        "schema_version": "ppi_shift.iclr2027_figure_provenance.v1",
        "figure": "measured_transfer_heatmaps",
        "git_revision": _git_revision(repo_root),
        "script_sha256": sha256_path(script_path),
        "style_sha256": sha256_path(script_path.with_name("paper_figure_style.py")),
        "inputs": inputs,
        "source_data_csv": _metadata_input(source_csv, repo_root),
        "formula": (
            "cell = mean AUROC over seeds 11, 22, 33 for one model/train-environment/surface"
        ),
        "seed_aggregation": "arithmetic mean over exactly three frozen seeds: 11, 22, 33",
        "layout": {
            "models": list(MODEL_ORDER),
            "training_environments": list(ENVIRONMENT_ORDER),
            "surfaces": list(MEASURED_SURFACES),
            "display_models": [MODEL_SHORT_LABELS[key] for key in MODEL_ORDER],
            "display_training_environments": [
                ENVIRONMENT_SHORT_LABELS[key] for key in ENVIRONMENT_ORDER
            ],
        },
        "color_scale": {"metric": "AUROC", "chance_value": 0.5, "shared_across_panels": True},
        "scope_boundary": (
            "Only Vidal/Y2H and IntAct measured-assay panels are shown. "
            "DataSAIL remains a separate supplementary analysis and RAPPPID is not imputed there."
        ),
        "outputs": [_output_record(path, repo_root, "rendered_figure") for path in outputs],
    }


def build_figures(
    repo_root: Path,
    pair_source: Path,
    registry_path: Path | None,
    measured_source: Path,
    measured_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = derive_pair_rows(pair_source)
    pair_csv = output_dir / "figure_data__figure_pair_dependence_heatmap.csv"
    _write_csv(
        pair_csv,
        [
            "model",
            "model_label",
            "test_environment",
            "test_environment_label",
            "auroc_original",
            "auroc_shuffled",
            "delta_auroc",
            "sign_interpretation",
            "source_seed_aggregation",
        ],
        pair_rows,
    )
    pair_outputs = plot_pair_heatmap(pair_rows, output_dir)
    pair_metadata = output_dir / "figure_pair_dependence_heatmap.metadata.json"
    _write_json(
        pair_metadata,
        _pair_metadata(repo_root, Path(__file__).resolve(), pair_source, pair_csv, pair_outputs),
    )

    if measured_source.is_file():
        measured_rows = _load_measured_source(measured_source)
    elif registry_path is not None and registry_path.is_file():
        measured_rows = _registry_measured_rows(registry_path)
        _write_csv(
            measured_source,
            [
                "surface",
                "model",
                "train_environment",
                "seed",
                "auroc",
                "registry_row_index",
            ],
            _source_csv_rows(measured_rows),
        )
    else:
        raise FileNotFoundError(
            "Measured source CSV is absent and the canonical E003 registry is unavailable"
        )
    if registry_path is not None and registry_path.is_file():
        registry_rows = _registry_measured_rows(registry_path)
        if _source_csv_rows(registry_rows) != _source_csv_rows(measured_rows):
            raise ValueError(
                "Committed measured source CSV differs from the canonical E003 registry"
            )
    measured_aggregated = aggregate_measured_rows(measured_rows)
    validate_measured_summary(measured_aggregated, measured_summary)
    measured_outputs = plot_measured_heatmaps(measured_aggregated, output_dir)
    measured_metadata = output_dir / "figure_measured_transfer_heatmaps.metadata.json"
    _write_json(
        measured_metadata,
        _measured_metadata(
            repo_root,
            Path(__file__).resolve(),
            measured_source,
            measured_summary,
            registry_path if registry_path is not None and registry_path.is_file() else None,
            measured_outputs,
        ),
    )
    manifest = {
        "schema_version": "ppi_shift.iclr2027_final_figure_manifest.v1",
        "git_revision": _git_revision(repo_root),
        "script_sha256": sha256_path(Path(__file__).resolve()),
        "style_sha256": sha256_path(Path(__file__).with_name("paper_figure_style.py")),
        "figures": [
            _output_record(pair_metadata, repo_root, "metadata"),
            _output_record(measured_metadata, repo_root, "metadata"),
            *[_output_record(path, repo_root, "rendered_figure") for path in pair_outputs],
            *[_output_record(path, repo_root, "rendered_figure") for path in measured_outputs],
        ],
    }
    _write_json(output_dir / "figure_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--pair-source",
        type=Path,
        default=Path("paper/figures/e003_final/figure_data__figure_03_pair_dependence.csv"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("results/analysis/e003_result_registry.json"),
    )
    parser.add_argument(
        "--measured-source",
        type=Path,
        default=Path(
            "paper/figures/iclr2027_final/figure_data__figure_measured_transfer_heatmaps.csv"
        ),
    )
    parser.add_argument(
        "--measured-summary",
        type=Path,
        default=Path("paper/figures/e003_final/figure_data__figure_05_measured_and_ood.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/iclr2027_final"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    build_figures(
        repo_root=repo_root,
        pair_source=resolve(args.pair_source),
        registry_path=resolve(args.registry),
        measured_source=resolve(args.measured_source),
        measured_summary=resolve(args.measured_summary),
        output_dir=resolve(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
