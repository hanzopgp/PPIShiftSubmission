#!/usr/bin/env python3
"""Render the compact appendix characterization from frozen E003 source tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

try:
    from paper_figure_style import (
        AXIS_SIZE,
        PAPER_WIDTH,
        apply_style,
        clean_axis,
        panel_header,
        text_on_color,
    )
except ModuleNotFoundError:
    from scripts.paper_figure_style import (
        AXIS_SIZE,
        PAPER_WIDTH,
        apply_style,
        clean_axis,
        panel_header,
        text_on_color,
    )

SAMPLERS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
ANCHOR_FIXED_SAMPLERS = SAMPLERS[:4]
SAMPLER_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Degree",
    "sequence_nuisance_matched": "Sequence",
    "esm_partner_matched": "ESM2",
    "bernett2026_ilp_exact": "Bernett",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}
GROUP_LABELS = {"observed_positives": "Observed positives", **SAMPLER_LABELS}
GROUP_COLORS = {
    "observed_positives": "#30343b",
    "uniform_partner": "#276fbf",
    "degree_partner_matched": "#2a9d8f",
    "sequence_nuisance_matched": "#d79b27",
    "esm_partner_matched": "#e47c32",
    "bernett2026_ilp_exact": "#8064a2",
    "tppni2025_scm_cl3_exact_balanced": "#c54b5c",
}
SOURCE_HASHES = {
    "pair_overlap.csv": "8dcdc9aa405f3b234fc2e77a34f415a9d75afcc3e56379b6a3be6a400aaa74de",
    "endpoint_set_overlap.csv": "a5ecf2a080ade43def3408c40b86a42c9f91fe82b3a4d3bab873b4a04e832fd4",
    "pair_esm_similarity.csv": "0124abfed25f10ca212f83c89a3ceefd01a9e58d0e85b3b3ae5ec79c68da1f93",
    "anchor_candidate_esm_distance.csv": (
        "909e629ca76f36a01006b3bd8f768300551371f5ce83c6b152cd2d9dbaa4152e"
    ),
}

apply_style()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _save(figure: plt.Figure, base: Path) -> list[Path]:
    outputs = []
    for extension in ("pdf", "png", "svg"):
        path = base.with_suffix(f".{extension}")
        figure.savefig(
            path,
            dpi=300,
            facecolor="white",
            metadata={"Creator": "PPIShift negative-construction figure build"}
            if extension != "png"
            else None,
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        outputs.append(path)
    plt.close(figure)
    return outputs


def _write_metadata(
    *,
    repo_root: Path,
    output_dir: Path,
    name: str,
    input_paths: dict[str, Path],
    outputs: list[Path],
    formula: str,
    interpretation_boundary: dict[str, bool],
) -> Path:
    output_path = output_dir / f"{name}.json"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "ppi_shift.negative_construction_figure.v1",
                "figure": name,
                "git_revision": _git_revision(repo_root),
                "script_sha256": _sha256(Path(__file__).resolve()),
                "style_sha256": _sha256(Path(__file__).with_name("paper_figure_style.py")),
                "input_files": {
                    f"results/analysis/e003_negative_representation/{path.name}": _sha256(path)
                    for path in input_paths.values()
                },
                "formula": formula,
                "interpretation_boundary": interpretation_boundary,
                "outputs": {path.suffix.removeprefix("."): _sha256(path) for path in outputs},
                "source_hashes": {
                    f"base/{path.name}": _sha256(path) for path in input_paths.values()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def _overlap_matrix(
    rows: list[dict[str, str]], field: str, *, endpoint_scale: bool
) -> tuple[np.ndarray, str, Normalize]:
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["sampler_a"], row["sampler_b"])
        if key in values:
            raise ValueError(f"Duplicate overlap cell: {key}")
        value = float(row[field])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid Jaccard value for {key}: {value}")
        values[key] = value
    expected = {(a, b) for a in SAMPLERS for b in SAMPLERS}
    if set(values) != expected:
        raise ValueError("Overlap table does not contain the complete 6x6 sampler matrix")
    matrix = np.asarray([[values[(a, b)] for b in SAMPLERS] for a in SAMPLERS])

    off_diagonal = matrix[~np.eye(len(SAMPLERS), dtype=bool)]
    if endpoint_scale:
        lower = max(0.0, np.floor(float(off_diagonal.min()) * 100) / 100)
        upper = 1.0
        cmap = "Blues"
    else:
        lower = 0.0
        upper = max(0.01, np.ceil(float(off_diagonal.max()) * 100) / 100)
        cmap = "YlGnBu"
    return matrix, cmap, Normalize(vmin=lower, vmax=upper)


def _draw_overlap_axis(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap_name: str,
    norm: Normalize,
    title: str,
    show_y_labels: bool = True,
    show_x_labels: bool = True,
) -> None:
    masked = np.ma.array(matrix, mask=np.eye(len(SAMPLERS), dtype=bool))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#edf0f3")
    axis.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    for edge in np.arange(-0.5, len(SAMPLERS), 1):
        axis.axvline(edge, color="white", linewidth=0.8)
        axis.axhline(edge, color="white", linewidth=0.8)
    axis.set_xticks(range(len(SAMPLERS)))
    if show_x_labels:
        axis.set_xticklabels(
            [SAMPLER_LABELS[key] for key in SAMPLERS], rotation=48, ha="right", fontsize=8.5
        )
    else:
        axis.tick_params(axis="x", labelbottom=False)
    axis.set_yticks(range(len(SAMPLERS)))
    if show_y_labels:
        axis.set_yticklabels([SAMPLER_LABELS[key] for key in SAMPLERS], fontsize=8.5)
    else:
        axis.tick_params(axis="y", labelleft=False)
    panel_header(axis, title)
    axis.tick_params(length=0, pad=3)
    for (row, column), value in np.ndenumerate(matrix):
        if row != column:
            text = f"{100 * value:.1f}%"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9.0,
                color=text_on_color(cmap(norm(float(value)))),
            )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _build_overlap(*, repo_root: Path, output_dir: Path, inputs: dict[str, Path]) -> Path:
    pair_matrix, pair_cmap, pair_norm = _overlap_matrix(
        _read_csv(inputs["pair_overlap.csv"]), "jaccard", endpoint_scale=False
    )
    endpoint_matrix, endpoint_cmap, endpoint_norm = _overlap_matrix(
        _read_csv(inputs["endpoint_set_overlap.csv"]), "jaccard", endpoint_scale=True
    )
    figure = plt.figure(figsize=(PAPER_WIDTH, 5.45))
    grid = figure.add_gridspec(
        2,
        1,
        left=0.18,
        right=0.98,
        bottom=0.16,
        top=0.95,
        hspace=0.32,
    )
    top_axis = figure.add_subplot(grid[0, 0])
    bottom_axis = figure.add_subplot(grid[1, 0])
    _draw_overlap_axis(
        top_axis,
        pair_matrix,
        cmap_name=pair_cmap,
        norm=pair_norm,
        title="a  Exact-pair Jaccard (%)",
        show_x_labels=False,
    )
    _draw_overlap_axis(
        bottom_axis,
        endpoint_matrix,
        cmap_name=endpoint_cmap,
        norm=endpoint_norm,
        title="b  Protein-set Jaccard (%)",
    )
    outputs = _save(figure, output_dir / "figure_negative_construction_characterization_main")
    return _write_metadata(
        repo_root=repo_root,
        output_dir=output_dir,
        name="figure_negative_construction_characterization_main",
        input_paths={
            "pair_overlap.csv": inputs["pair_overlap.csv"],
            "endpoint_set_overlap.csv": inputs["endpoint_set_overlap.csv"],
        },
        outputs=outputs,
        formula="cell = Jaccard(set_a, set_b); diagonal omitted; displayed as percent",
        interpretation_boundary={
            "overlap_is_descriptive": True,
            "low_exact_pair_overlap_does_not_establish_distribution_separation": True,
            "protein_set_overlap_is_distinct_from_pair_overlap": True,
        },
    )


def _pair_groups(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    groups: dict[str, list[float]] = {key: [] for key in GROUP_LABELS}
    for row in rows:
        group = "observed_positives" if row["label_type"] == "observed_positive" else row["sampler"]
        if group not in groups:
            raise ValueError(f"Unexpected pair-similarity group: {group}")
        value = float(row["cosine_similarity"])
        if not np.isfinite(value):
            raise ValueError("Pair-similarity table contains a non-finite value")
        groups[group].append(value)
    if any(not values for values in groups.values()):
        raise ValueError("Pair-similarity table is missing one or more frozen groups")
    return {key: np.sort(np.asarray(values, dtype=np.float64)) for key, values in groups.items()}


def _distance_groups(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    groups = {key: [] for key in ANCHOR_FIXED_SAMPLERS}
    for row in rows:
        sampler = row["sampler"]
        if sampler not in groups:
            raise ValueError(f"Unexpected selected-partner distance group: {sampler}")
        value = float(row["cosine_distance"])
        if not np.isfinite(value):
            raise ValueError("Selected-partner distance table contains a non-finite value")
        groups[sampler].append(value)
    if any(not values for values in groups.values()):
        raise ValueError("Selected-partner distance table is missing an anchor-fixed construction")
    return {key: np.sort(np.asarray(values, dtype=np.float64)) for key, values in groups.items()}


def _draw_ecdf(axis: plt.Axes, groups: dict[str, np.ndarray], keys: tuple[str, ...]) -> None:
    for key in keys:
        values = groups[key]
        y_values = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
        axis.plot(
            values,
            y_values,
            color=GROUP_COLORS[key],
            linewidth=2.0 if key == "observed_positives" else 1.4,
            label=GROUP_LABELS[key],
        )
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", color="#d9dee7", linewidth=0.55, alpha=0.7)
    clean_axis(axis)


def _build_ecdf(*, repo_root: Path, output_dir: Path, inputs: dict[str, Path]) -> Path:
    pair_groups = _pair_groups(_read_csv(inputs["pair_esm_similarity.csv"]))
    distance_groups = _distance_groups(_read_csv(inputs["anchor_candidate_esm_distance.csv"]))
    figure = plt.figure(figsize=(PAPER_WIDTH, 3.75))
    grid = figure.add_gridspec(1, 2, left=0.105, right=0.92, bottom=0.33, top=0.91, wspace=0.20)
    pair_axis = figure.add_subplot(grid[0, 0])
    distance_axis = figure.add_subplot(grid[0, 1], sharey=pair_axis)
    _draw_ecdf(pair_axis, pair_groups, tuple(GROUP_LABELS))
    _draw_ecdf(distance_axis, distance_groups, ANCHOR_FIXED_SAMPLERS)
    panel_header(pair_axis, "a  Pair similarity")
    panel_header(distance_axis, "b  Partner distance")
    pair_axis.set_xlabel("ESM2 cosine similarity", fontsize=AXIS_SIZE)
    distance_axis.set_xlabel("ESM2 cosine distance", fontsize=AXIS_SIZE)
    pair_axis.set_ylabel("ECDF", fontsize=AXIS_SIZE)
    distance_axis.tick_params(axis="y", labelleft=False)
    handles = [
        Line2D([], [], color=GROUP_COLORS[key], linewidth=1.7, label=GROUP_LABELS[key])
        for key in GROUP_LABELS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.035),
        ncol=4,
        frameon=False,
        fontsize=8.7,
        handlelength=1.6,
        columnspacing=1.0,
    )
    outputs = _save(figure, output_dir / "figure_negative_construction_characterization_ecdf")
    return _write_metadata(
        repo_root=repo_root,
        output_dir=output_dir,
        name="figure_negative_construction_characterization_ecdf",
        input_paths={
            "pair_esm_similarity.csv": inputs["pair_esm_similarity.csv"],
            "anchor_candidate_esm_distance.csv": inputs["anchor_candidate_esm_distance.csv"],
        },
        outputs=outputs,
        formula="ECDF over every row in each source table; no filtering or reweighting",
        interpretation_boundary={
            "frozen_esm2_similarity_is_not_a_ppi_probability": True,
            "esm2_matching_effect_is_expected_by_design": True,
            "selected_partner_distance_defined_only_for_anchor_fixed_constructions": True,
        },
    )


def build_figures(repo_root: Path, source_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {name: source_dir / name for name in SOURCE_HASHES}
    observed_hashes = {name: _sha256(path) for name, path in inputs.items()}
    if observed_hashes != SOURCE_HASHES:
        differences = {
            name: {"expected": SOURCE_HASHES[name], "observed": observed_hashes.get(name)}
            for name in SOURCE_HASHES
            if observed_hashes.get(name) != SOURCE_HASHES[name]
        }
        raise ValueError(f"Frozen characterization source hashes differ: {differences}")
    overlap_metadata = _build_overlap(repo_root=repo_root, output_dir=output_dir, inputs=inputs)
    ecdf_metadata = _build_ecdf(repo_root=repo_root, output_dir=output_dir, inputs=inputs)
    return [overlap_metadata, ecdf_metadata]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/analysis/e003_negative_representation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/e003_negative_representation"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    source_dir = args.source_dir if args.source_dir.is_absolute() else repo_root / args.source_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    build_figures(repo_root, source_dir.resolve(), output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
