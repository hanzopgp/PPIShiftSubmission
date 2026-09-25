#!/usr/bin/env python3
"""Build the machine-readable E003 final paper figure pack.

The script is deliberately fail-closed: it accepts only the canonical result
registry and the frozen signal-audit outputs, derives every plotted value from
those inputs, and records the exact source hashes beside every figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch

try:
    from paper_figure_style import (
        AXIS_SIZE,
        BLUE,
        INK,
        MUTED,
        ORANGE,
        PAPER_WIDTH,
        SAMPLER_LABELS,
        apply_style,
        clean_axis,
        heatmap,
        panel_header,
    )
    from paper_figure_style import (
        MODEL_LABELS as PAPER_MODEL_LABELS,
    )
except ModuleNotFoundError:
    from scripts.paper_figure_style import (
        AXIS_SIZE,
        BLUE,
        INK,
        MUTED,
        ORANGE,
        PAPER_WIDTH,
        SAMPLER_LABELS,
        apply_style,
        clean_axis,
        heatmap,
        panel_header,
    )
    from scripts.paper_figure_style import (
        MODEL_LABELS as PAPER_MODEL_LABELS,
    )

try:
    from e003_signal_common import ENVIRONMENTS, SEEDS, base_metadata, write_csv, write_json
except ModuleNotFoundError:  # Imported as a package in tests/tools.
    from scripts.e003_signal_common import ENVIRONMENTS, SEEDS, base_metadata, write_csv, write_json

from ppi_shift.signal_audit import sha256_path

MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "PPIShift Frozen-ESM2 baseline",
    "dscript_esm2": "D-SCRIPT-ESM2 (Reim et al.)",
    "tuna_corrected": "TUnA-R (corrected)",
    "rapppid": "RAPPPID",
}
ENVIRONMENT_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Matched by degree",
    "sequence_nuisance_matched": "Matched by sequence",
    "esm_partner_matched": "Matched in ESM2 space",
    "bernett2026_ilp_exact": "Bernett 2026",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}
CONTROL_LABELS = {
    "empirical_endpoint_propensity": "Empirical protein propensity",
    "single_protein_sequence_logistic": "Single-protein sequence propensity",
    "easy_sequence_degree": "Easy properties",
    "bernett_simple": "Simple pooled-ESM classifier",
}
SHORT_MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}
SHORT_ENVIRONMENT_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Degree",
    "sequence_nuisance_matched": "Sequence",
    "esm_partner_matched": "ESM2",
    "bernett2026_ilp_exact": "Bernett",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}
FIGURE_MODEL_ORDER = (
    "symmetric_frozen_esm2_pair_mlp",
    "tuna_corrected",
    "dscript_esm2",
    "rapppid",
)

apply_style()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_committed_csv(path: Path, rows: list[dict[str, Any]], repo_root: Path) -> None:
    """Fail closed if rederived data differs from the committed figure table."""
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    def normalized(value: bytes) -> bytes:
        return value.replace(b"\r\n", b"\n")

    committed = normalized(result.stdout)
    if not path.is_file() or normalized(path.read_bytes()) != committed:
        raise ValueError(f"Committed paper figure data is missing or locally modified: {relative}")
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        candidate = Path(temporary) / path.name
        write_csv(candidate, rows)
        if normalized(candidate.read_bytes()) != committed:
            raise ValueError(f"Rederived figure data differs from the committed table: {relative}")


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows_data", payload.get("cells", []))
    else:
        rows = _read_csv(path)
    if not isinstance(rows, list):
        raise SystemExit(f"Registry rows are not a list: {path}")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        # The canonical result registry uses the explicit
        # ``training_environment`` field; the figure-pack source schema uses
        # the shorter ``train_environment`` label.  Keep the registry
        # immutable and add only a derived alias for plotting.
        if "training_environment" in current:
            current.setdefault("train_environment", current["training_environment"])
        metrics = current.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                current.setdefault(key, value)
        normalized.append(current)
    return normalized


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Missing numeric {key!r} in row: {row}") from error
    if not np.isfinite(result):
        raise SystemExit(f"Non-finite {key!r} in row: {row}")
    return result


def _mean_exact(rows: Iterable[dict[str, Any]], key: str, *, context: str) -> float:
    selected = list(rows)
    observed = {int(row["seed"]) for row in selected}
    if len(selected) != len(SEEDS) or observed != set(SEEDS):
        raise SystemExit(
            f"Expected exactly seeds {SEEDS} for {context}; observed {sorted(observed)}"
        )
    return float(np.mean([_float(row, key) for row in selected]))


def _save(figure: Any, base: Path, *, tight: bool = True) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
        if path.suffix == ".svg":
            normalized = (
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n"
            )
            path.write_text(normalized, encoding="utf-8", newline="\n")
    plt.close(figure)
    return [str(path) for path in paths]


def _human_environment(value: str) -> str:
    return ENVIRONMENT_LABELS.get(value, value.replace("_", " ").title())


def _human_control(value: str) -> str:
    return CONTROL_LABELS.get(value, value.replace("_", " ").title())


def _human_surface(value: str) -> str:
    paper_labels = {
        "vidal__measured": "Vidal/Y2H",
        "intact__measured": "IntAct",
        "biogrid__uniform_partner": "BioGRID + Uniform",
        "biogrid__bernett2026_ilp_exact": "BioGRID + Bernett 2026",
        "biogrid__tppni2025_scm_cl3_exact_balanced": "BioGRID + TPPNI",
        "string__uniform_partner": "STRING + Uniform",
        "string__bernett2026_ilp_exact": "STRING + Bernett 2026",
        "string__tppni2025_scm_cl3_exact_balanced": "STRING + TPPNI",
    }
    if value in paper_labels:
        return paper_labels[value]

    replacements = {
        "vidal": "Vidal",
        "intact": "IntAct",
        "biogrid": "BioGRID",
        "string": "STRING",
        "datasail": "DataSAIL",
        "ood": "strict OOD",
    }
    label = value.replace("_", " ")
    for source, target in replacements.items():
        label = label.replace(source, target)
    return (
        label.title()
        .replace("Intact", "IntAct")
        .replace("Biogrid", "BioGRID")
        .replace("Datasail", "DataSAIL")
        .replace("Ood", "OOD")
        .replace("Esm", "ESM")
    )


def _surface_sort(value: str) -> tuple[int, str]:
    lowered = value.lower()
    if "vidal" in lowered:
        return (0, value)
    if "intact" in lowered:
        return (1, value)
    if "biogrid" in lowered:
        return (2, value)
    if "string" in lowered:
        return (3, value)
    if "ood" in lowered or "unseen" in lowered or "family" in lowered:
        return (4, value)
    if "datasail" in lowered:
        return (5, value)
    return (6, value)


def _expected_core_cells(model_count: int) -> int:
    return model_count * len(ENVIRONMENTS) * len(ENVIRONMENTS) * len(SEEDS)


def _is_measured(surface: str) -> bool:
    lowered = surface.lower()
    return "vidal" in lowered or "intact" in lowered


def _is_strict_ood(surface: str) -> bool:
    lowered = surface.lower()
    return any(token in lowered for token in ("ood", "unseen", "family", "novel"))


def _metric_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    metric: str,
    context_prefix: str,
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)
    return {
        key: _mean_exact(value, metric, context=f"{context_prefix}:{key}")
        for key, value in grouped.items()
    }


def _metric_rows_seed_covered(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    metric: str,
    context_prefix: str,
) -> dict[tuple[str, ...], float]:
    """Aggregate rows with any number of train environments, but all seeds."""

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)
    values: dict[tuple[str, ...], float] = {}
    for key, selected in grouped.items():
        observed = {int(row["seed"]) for row in selected}
        if observed != set(SEEDS):
            raise SystemExit(
                f"Expected every seed {SEEDS} for {context_prefix}:{key}; "
                f"observed {sorted(observed)}"
            )
        numbers = [_float(row, metric) for row in selected]
        values[key] = float(np.mean(numbers))
    return values


def _source_hashes(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": sha256_path(path)} for path in paths if path.is_file()]


def _figure_metadata(
    *,
    args: argparse.Namespace,
    name: str,
    source_csv: Path,
    input_paths: list[Path],
    formula: str,
    filters: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    panel_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    representation_revisions: set[str] = set()
    for path in input_paths:
        if path != args.core_registry or not path.is_file():
            continue
        for row in _load_registry(path):
            for field, target in (
                ("panel_sha256", panel_hashes),
                ("checkpoint_sha256", checkpoint_hashes),
                ("representation_source_revision", representation_revisions),
            ):
                value = str(row.get(field, ""))
                if value:
                    target.add(value)
    payload = base_metadata(args.repo_root, args.protocol)
    payload.update(
        {
            "schema_version": "ppi_shift.e003_figure_provenance.v1",
            "figure": name,
            "style_sha256": sha256_path(Path(__file__).with_name("paper_figure_style.py")),
            "source_csv": {"path": str(source_csv), "sha256": sha256_path(source_csv)},
            "input_files": _source_hashes(input_paths),
            "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
            "seeds": list(SEEDS),
            "filters": filters,
            "formula": formula,
            "panel_hashes": sorted(panel_hashes),
            "checkpoint_hashes": sorted(checkpoint_hashes),
            "representation_source_revisions": sorted(representation_revisions),
        }
    )
    if extra:
        payload.update(extra)
    path = args.output_dir / f"{name}.metadata.json"
    write_json(path, payload)
    return path


def _write_source_and_metadata(
    *,
    args: argparse.Namespace,
    name: str,
    source_rows: list[dict[str, Any]],
    input_paths: list[Path],
    formula: str,
    filters: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    source = args.output_dir / f"figure_data__{name}.csv"
    _require_committed_csv(source, source_rows, args.repo_root)
    return _figure_metadata(
        args=args,
        name=name,
        source_csv=source,
        input_paths=input_paths,
        formula=formula,
        filters=filters,
        extra=extra,
    )


def _figure_one(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    name = "figure_01_negative_construction_explainer"
    matrix = config["matrix"]
    construction_rows = [
        {
            "construction": environment,
            "label": _human_environment(environment),
            "description": description,
        }
        for environment, description in (
            ("uniform_partner", "random legal partner"),
            ("degree_partner_matched", "similar network degree"),
            ("sequence_nuisance_matched", "similar length/composition"),
            ("esm_partner_matched", "similar frozen ESM2 representation"),
            ("bernett2026_ilp_exact", "global bias-reduced construction"),
            ("tppni2025_scm_cl3_exact_balanced", "topology-based construction"),
        )
    ]
    source_rows = [{"section": "construction", **row} for row in construction_rows] + [
        {
            "section": "evaluation",
            "construction": "same_source",
            "label": "Same dataset",
            "description": "Gold -> Gold",
        },
        {
            "section": "evaluation",
            "construction": "cross_source",
            "label": "External datasets",
            "description": "Gold -> Vidal / IntAct / BioGRID / STRING",
        },
        {
            "section": "evaluation",
            "construction": "strict_ood",
            "label": "Strict OOD",
            "description": "Gold -> protein/family-unseen external subset",
        },
    ]
    figure = _draw_construction_schematic()
    files = _save(figure, args.output_dir / name, tight=False)
    metadata = _write_source_and_metadata(
        args=args,
        name=name,
        source_rows=source_rows,
        input_paths=[args.config, args.protocol],
        formula="schematic labels are taken from the frozen E003 construction contract",
        filters={"construction_count": len(construction_rows), "seed_values": matrix["seeds"]},
    )
    return {
        "figure": name,
        "files": files,
        "source_csv": str(args.output_dir / f"figure_data__{name}.csv"),
        "metadata": str(metadata),
    }


def _draw_construction_schematic() -> Any:
    """Draw the experiment as a compact sequence, without interpretation text."""
    figure = plt.figure(figsize=(PAPER_WIDTH, 2.45))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set(xlim=(0, PAPER_WIDTH), ylim=(0, 2.45))
    axis.axis("off")

    def pair(cx: float, cy: float, partner: str, color: str) -> None:
        axis.plot([cx - 0.17, cx + 0.17], [cy, cy], color=MUTED, lw=1.6, zorder=1)
        for x, label, fill in ((cx - 0.21, "A", BLUE), (cx + 0.21, partner, color)):
            axis.add_patch(Circle((x, cy), 0.17, facecolor=fill, edgecolor="white", lw=1))
            axis.text(
                x, cy, label, ha="center", va="center", color="white", size=9.2, weight="bold"
            )

    def arrow(x0: float, x1: float, y: float = 1.34) -> None:
        axis.add_patch(
            FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, lw=1, color=MUTED)
        )

    pair(0.43, 1.34, "B", "#4b927b")
    axis.text(0.43, 0.88, "Observed", ha="center", size=9.2)
    arrow(0.78, 0.96)

    labels = ("Uniform", "Degree", "Sequence", "ESM2", "Bernett", "TPPNI")
    for index, label in enumerate(labels):
        column, row = divmod(index, 3)
        x = 1.02 + column * 0.63
        y = 1.77 - row * 0.43
        axis.add_patch(
            plt.Rectangle(
                (x, y - 0.14), 0.57, 0.29, facecolor="#edf3f7", edgecolor="#c9d7e1", lw=0.6
            )
        )
        axis.text(x + 0.285, y, label, ha="center", va="center", size=8.7)
    arrow(2.27, 2.43)

    pair(2.78, 1.34, "C", ORANGE)
    axis.text(2.78, 0.88, "Constructed", ha="center", size=9.2)
    arrow(3.13, 3.32)

    x0, y0, step = 3.43, 0.95, 0.115
    for row in range(6):
        for column in range(6):
            axis.add_patch(
                plt.Rectangle(
                    (x0 + column * step, y0 + (5 - row) * step),
                    step - 0.012,
                    step - 0.012,
                    facecolor=BLUE if row == column else "#cadbe5",
                    edgecolor="white",
                    lw=0.3,
                )
            )
    axis.text(3.76, 1.86, "6 train × 6 test", ha="center", size=9.1, weight="bold")
    axis.text(3.76, 0.72, "4 models × 3 seeds", ha="center", size=8.8)
    arrow(4.18, 4.43)

    axis.text(4.93, 1.91, "Evaluation", ha="center", size=9.2, weight="bold")
    for row, label in enumerate(("Measured", "External", "Controls")):
        axis.text(
            4.93,
            1.57 - row * 0.43,
            label,
            ha="center",
            va="center",
            size=9.0,
            color=INK if label != "Controls" else BLUE,
        )
    return figure


def _build_transfer_sources(
    core: list[dict[str, Any]], external: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    core_values = _metric_rows(
        core,
        group_keys=("model", "train_environment", "test_environment"),
        metric="auprc",
        context_prefix="core-auprc",
    )
    surfaces = sorted({str(row["evaluation_surface"]) for row in external}, key=_surface_sort)
    external_values = _metric_rows(
        external,
        group_keys=("model", "train_environment", "evaluation_surface"),
        metric="auroc",
        context_prefix="external-auroc",
    )
    models = sorted({str(row["model"]) for row in core})
    core_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    for model in models:
        for test in ENVIRONMENTS:
            reference = core_values[(model, "uniform_partner", test)]
            for train in ENVIRONMENTS:
                absolute = core_values[(model, train, test)]
                core_rows.append(
                    {
                        "kind": "gold",
                        "model": model,
                        "train_environment": train,
                        "test_environment": test,
                        "absolute_metric": absolute,
                        "reference_uniform_train": reference,
                        "delta_metric": absolute - reference,
                        "metric": "AUPRC",
                    }
                )
        for surface in surfaces:
            reference = external_values[(model, "uniform_partner", surface)]
            for train in ENVIRONMENTS:
                absolute = external_values[(model, train, surface)]
                external_rows.append(
                    {
                        "kind": "external",
                        "model": model,
                        "train_environment": train,
                        "evaluation_surface": surface,
                        "absolute_metric": absolute,
                        "reference_uniform_train": reference,
                        "delta_metric": absolute - reference,
                        "metric": "AUROC",
                    }
                )
    return core_rows, external_rows, models, surfaces


def _short_surface(value: str) -> str:
    measured = {"vidal__measured": "Vidal/\nY2H", "intact__measured": "IntAct"}
    if value in measured:
        return measured[value]
    source, environment = value.split("__", maxsplit=1)
    source_label = "BioGRID" if source == "biogrid" else "STRING"
    return f"{source_label}\n{SHORT_ENVIRONMENT_LABELS.get(environment, environment)}"


def _transfer_matrix(
    rows: list[dict[str, Any]], model: str, column: str, column_values: Iterable[str]
) -> np.ndarray:
    row_values = list(ENVIRONMENTS)
    lookup = {
        (str(row["train_environment"]), str(row[column])): float(row["delta_metric"])
        for row in rows
        if row["model"] == model
    }
    expected = {(train, str(value)) for train in row_values for value in column_values}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete {column} matrix for {model}")
    return np.asarray(
        [[lookup[(train, str(value))] for value in column_values] for train in row_values]
    )


def _save_external_transfer_supplement(
    delta_external: list[dict[str, Any]],
    models: list[str],
    surfaces: list[str],
    args: argparse.Namespace,
    source_csv: Path,
) -> tuple[list[str], Path]:
    values = np.asarray([float(row["delta_metric"]) for row in delta_external])
    limit = max(float(np.max(np.abs(values))), 1e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    # Eight heterogeneous external surfaces remain separate rows; no trajectory is implied.
    surface_labels = {
        "vidal__measured": "Vidal/Y2H",
        "intact__measured": "IntAct",
        "biogrid__uniform_partner": "BioGRID Uniform",
        "biogrid__bernett2026_ilp_exact": "BioGRID Bernett",
        "biogrid__tppni2025_scm_cl3_exact_balanced": "BioGRID TPPNI",
        "string__uniform_partner": "STRING Uniform",
        "string__bernett2026_ilp_exact": "STRING Bernett",
        "string__tppni2025_scm_cl3_exact_balanced": "STRING TPPNI",
    }
    figure = plt.figure(figsize=(PAPER_WIDTH, 8.45))
    grid = figure.add_gridspec(
        len(models),
        2,
        width_ratios=(1, 0.05),
        left=0.27,
        right=0.87,
        bottom=0.105,
        top=0.94,
        hspace=0.34,
        wspace=0.17,
    )
    image = None
    for index, model in enumerate(models):
        axis = figure.add_subplot(grid[index, 0])
        matrix = _transfer_matrix(delta_external, model, "evaluation_surface", surfaces).T
        image = heatmap(axis, matrix, cmap=cmap, norm=norm)
        panel_header(axis, PAPER_MODEL_LABELS[model])
        axis.set_yticks(range(len(surfaces)))
        axis.set_yticklabels([surface_labels[value] for value in surfaces], fontsize=8.8)
        axis.set_xticks(range(len(ENVIRONMENTS)))
        if index == len(models) - 1:
            axis.set_xticklabels(
                [SAMPLER_LABELS[value] for value in ENVIRONMENTS], rotation=38, ha="right"
            )
        else:
            axis.tick_params(axis="x", labelbottom=False)
    colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[:, 1]))
    colorbar.ax.set_title("ΔAUROC", size=AXIS_SIZE, pad=5)
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.set_ticklabels([f"{value:.3f}" for value in (-limit, 0, limit)])
    colorbar.ax.tick_params(labelsize=8.8)
    files = _save(figure, args.output_dir / "figure_02_external_transfer_supplement", tight=False)
    metadata = _figure_metadata(
        args=args,
        name="figure_02_external_transfer_supplement",
        source_csv=source_csv,
        input_paths=[args.core_registry, args.protocol, args.config],
        formula=(
            "mean external AUROC(train,surface) - "
            "mean external AUROC(Uniform train,surface), seeds {11,22,33}"
        ),
        filters={"kind": "external", "models": models, "surfaces": surfaces},
        extra={
            "uniform_row_is_zero": True,
            "cell_annotations": False,
            "exact_values_in_source_csv": True,
        },
    )
    return files, metadata


def _gold_grid_figure(
    rows: list[dict[str, Any]],
    models: list[str],
    *,
    field: str,
    cmap: Any,
    norm: Any,
    bar_label: str,
) -> Any:
    figure = plt.figure(figsize=(PAPER_WIDTH, 4.55))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1, 1, 0.075),
        left=0.17,
        right=0.87,
        bottom=0.19,
        top=0.92,
        wspace=0.24,
        hspace=0.40,
    )
    bar_axis = figure.add_subplot(grid[:, 2])
    image = None
    labels = [SAMPLER_LABELS[key] for key in ENVIRONMENTS]
    for index, model in enumerate(models):
        row_index, column_index = divmod(index, 2)
        axis = figure.add_subplot(grid[row_index, column_index])
        lookup = {
            (str(row["train_environment"]), str(row["test_environment"])): float(row[field])
            for row in rows
            if row["model"] == model and row["kind"] == "gold"
        }
        if set(lookup) != {(train, test) for train in ENVIRONMENTS for test in ENVIRONMENTS}:
            raise ValueError(f"Incomplete Gold matrix for {model}")
        matrix = np.asarray(
            [[lookup[(train, test)] for test in ENVIRONMENTS] for train in ENVIRONMENTS]
        )
        image = heatmap(axis, matrix, cmap=cmap, norm=norm)
        panel_header(axis, PAPER_MODEL_LABELS[model])
        axis.set_xticks(range(6))
        if row_index == 1:
            axis.set_xticklabels(labels, rotation=45, ha="right")
        else:
            axis.tick_params(axis="x", labelbottom=False)
        axis.set_yticks(range(6))
        if column_index == 0:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(axis="y", labelleft=False)
    colorbar = figure.colorbar(image, cax=bar_axis)
    colorbar.ax.set_title(bar_label, size=AXIS_SIZE, pad=5)
    colorbar.ax.tick_params(labelsize=8.8)
    if field == "delta_metric":
        ticks = [norm.vmin, 0, norm.vmax]
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{value:.3f}" for value in ticks])
    return figure


def _figure_two(
    core: list[dict[str, Any]], external: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    delta_core, delta_external, models, surfaces = _build_transfer_sources(core, external)
    core_source = args.output_dir / "figure_data__figure_02_negative_transfer.csv"
    absolute_source = args.output_dir / "figure_data__figure_02_absolute_reference.csv"
    source_rows = [*delta_core, *delta_external]
    _require_committed_csv(core_source, source_rows, args.repo_root)
    _require_committed_csv(
        absolute_source, [dict(row, delta_metric="") for row in source_rows], args.repo_root
    )
    limit = max(max(abs(float(row["delta_metric"])) for row in delta_core), 1e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    names: list[dict[str, Any]] = []
    plot_models = [model for model in FIGURE_MODEL_ORDER if model in models]
    if (
        plot_models != list(FIGURE_MODEL_ORDER)
        or len(delta_core) != len(plot_models) * len(ENVIRONMENTS) ** 2
    ):
        raise ValueError("Gold transfer data does not match the frozen four-model 6x6 layout")
    cmap = plt.get_cmap("RdBu_r")
    figure = _gold_grid_figure(
        delta_core,
        plot_models,
        field="delta_metric",
        cmap=cmap,
        norm=norm,
        bar_label="ΔAUPRC",
    )
    files = _save(figure, args.output_dir / "figure_02_negative_transfer", tight=False)
    metadata = _figure_metadata(
        args=args,
        name="figure_02_negative_transfer",
        source_csv=core_source,
        input_paths=[args.core_registry, args.protocol, args.config],
        formula=(
            "mean metric(train,test) - mean metric(uniform train,test), "
            "with exact seed set {11,22,33}"
        ),
        filters={"kind": "gold", "models": plot_models, "gold_tests": list(ENVIRONMENTS)},
        extra={
            "companion_source_csv": str(absolute_source),
            "uniform_row_is_zero": True,
            "external_supplement": "figure_02_external_transfer_supplement",
            "cell_annotations": False,
            "exact_values_in_source_csv": True,
        },
    )
    names.append(
        {
            "figure": "figure_02_negative_transfer",
            "files": files,
            "source_csv": str(core_source),
            "metadata": str(metadata),
        }
    )
    gold_absolute = [row for row in source_rows if row["kind"] == "gold"]
    absolute_values = [float(row["absolute_metric"]) for row in gold_absolute]
    absolute_norm = Normalize(vmin=min(absolute_values), vmax=max(absolute_values))
    absolute_cmap = plt.get_cmap("viridis")
    figure = _gold_grid_figure(
        gold_absolute,
        plot_models,
        field="absolute_metric",
        cmap=absolute_cmap,
        norm=absolute_norm,
        bar_label="AUPRC",
    )
    files = _save(figure, args.output_dir / "figure_02_absolute_reference", tight=False)
    metadata = _figure_metadata(
        args=args,
        name="figure_02_absolute_reference",
        source_csv=absolute_source,
        input_paths=[args.core_registry, args.protocol, args.config],
        formula="mean metric(train,test) over exact seed set {11,22,33}",
        filters={"kind": "gold", "models": plot_models, "gold_tests": list(ENVIRONMENTS)},
        extra={
            "absolute_range": [min(absolute_values), max(absolute_values)],
            "cell_annotations": False,
            "exact_values_in_source_csv": True,
        },
    )
    names.append(
        {
            "figure": "figure_02_absolute_reference",
            "files": files,
            "source_csv": str(absolute_source),
            "metadata": str(metadata),
        }
    )
    external_files, external_metadata = _save_external_transfer_supplement(
        delta_external, plot_models, surfaces, args, core_source
    )
    names.append(
        {
            "figure": "figure_02_external_transfer_supplement",
            "files": external_files,
            "source_csv": str(core_source),
            "metadata": str(external_metadata),
        }
    )
    return names


def _figure_three(
    easy: list[dict[str, str]],
    protein: list[dict[str, str]],
    bernett: list[dict[str, str]],
    core: list[dict[str, Any]],
    broken: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = "figure_03_pair_dependence"
    easy_values = _metric_rows(
        easy, group_keys=("environment",), metric="test_auroc", context_prefix="easy-auroc"
    )
    protein_controls = sorted({str(row["control"]) for row in protein})
    protein_values = _metric_rows(
        protein,
        group_keys=("control", "environment"),
        metric="test_auroc",
        context_prefix="protein-auroc",
    )
    bernett_values = _metric_rows(
        bernett, group_keys=("environment",), metric="test_auroc", context_prefix="bernett-auroc"
    )
    original_values = _metric_rows_seed_covered(
        core,
        group_keys=("model", "test_environment"),
        metric="auroc",
        context_prefix="original-auroc",
    )
    broken_values = _metric_rows_seed_covered(
        broken, group_keys=("model", "environment"), metric="auroc", context_prefix="broken-auroc"
    )
    source_rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        source_rows.append(
            {
                "panel": "A",
                "series": "Easy properties",
                "environment": environment,
                "auroc": easy_values[(environment,)],
            }
        )
        for control in protein_controls:
            source_rows.append(
                {
                    "panel": "A",
                    "series": f"Protein-only: {_human_control(control)}",
                    "environment": environment,
                    "auroc": protein_values[(control, environment)],
                }
            )
        source_rows.append(
            {
                "panel": "C",
                "series": "Simple pooled-ESM classifier",
                "environment": environment,
                "auroc": bernett_values[(environment,)],
            }
        )
    for model in sorted({str(row["model"]) for row in core}):
        for environment in ENVIRONMENTS:
            source_rows.extend(
                [
                    {
                        "panel": "B",
                        "series": "Original pairs",
                        "model": model,
                        "environment": environment,
                        "auroc": original_values[(model, environment)],
                    },
                    {
                        "panel": "B",
                        "series": "Partners shuffled",
                        "model": model,
                        "environment": environment,
                        "auroc": broken_values[(model, environment)],
                    },
                ]
            )
    source = args.output_dir / f"figure_data__{name}.csv"
    _require_committed_csv(source, source_rows, args.repo_root)
    figure = plt.figure(figsize=(PAPER_WIDTH, 7.55))
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1, 0.055),
        height_ratios=(1, 1.3, 1),
        left=0.19,
        right=0.87,
        bottom=0.145,
        top=0.96,
        hspace=0.70,
        wspace=0.18,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    positions = np.arange(len(ENVIRONMENTS))
    labels = [SAMPLER_LABELS[key] for key in ENVIRONMENTS]
    handles = []
    (handle,) = axis_a.plot(
        positions,
        [easy_values[(env,)] for env in ENVIRONMENTS],
        "o-",
        color=BLUE,
        markersize=4.5,
        label="Easy properties",
    )
    handles.append(handle)
    control_colors = ("#60768e", "#4b927b")
    for control, color in zip(protein_controls, control_colors, strict=True):
        (handle,) = axis_a.plot(
            positions,
            [protein_values[(control, env)] for env in ENVIRONMENTS],
            "o-",
            color=color,
            markersize=4.5,
            label="Endpoint propensity" if "endpoint" in control else "Single-protein sequence",
        )
        handles.append(handle)
    panel_header(axis_a, "a  Single-protein controls")
    axis_a.set_ylabel("AUROC")
    axis_a.set_xticks(positions, labels, rotation=35, ha="right")
    clean_axis(axis_a, grid=True)

    models = list(FIGURE_MODEL_ORDER)
    matrix = np.asarray(
        [
            [original_values[(model, env)] - broken_values[(model, env)] for env in ENVIRONMENTS]
            for model in models
        ]
    )
    limit = max(float(np.max(np.abs(matrix))), 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    axis_b = figure.add_subplot(grid[1, 0])
    image = heatmap(
        axis_b, matrix, cmap=cmap, norm=norm, annotations="{:+.3f}", annotation_size=8.8
    )
    panel_header(axis_b, "b  Pair dependence")
    axis_b.set_yticks(range(len(models)))
    axis_b.set_yticklabels([PAPER_MODEL_LABELS[model] for model in models])
    axis_b.set_xticks(positions, labels, rotation=35, ha="right")
    colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[1, 1]))
    colorbar.ax.set_title("ΔAUROC", size=AXIS_SIZE, pad=5)
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.set_ticklabels([f"{value:.3f}" for value in (-limit, 0, limit)])
    colorbar.ax.tick_params(labelsize=8.5)

    axis_c = figure.add_subplot(grid[2, 0])
    axis_c.plot(
        positions,
        [bernett_values[(env,)] for env in ENVIRONMENTS],
        "o-",
        color=ORANGE,
        markersize=4.5,
    )
    axis_c.axhline(0.5, color=MUTED, linestyle=":", linewidth=0.8)
    panel_header(axis_c, "c  Pooled-ESM2 classifier")
    axis_c.set_ylabel("AUROC")
    axis_c.set_xticks(positions, labels, rotation=35, ha="right")
    clean_axis(axis_c, grid=True)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )
    files = _save(figure, args.output_dir / name, tight=False)
    metadata = _figure_metadata(
        args=args,
        name=name,
        source_csv=source,
        input_paths=[*getattr(args, "signal_inputs", []), args.core_registry, args.protocol],
        formula=(
            "mean AUROC over exact seeds; original and shuffled panels are "
            "evaluated without retraining"
        ),
        filters={
            "controls": ["easy properties", *protein_controls, "simple pooled-ESM classifier"],
            "deep_models": sorted({str(row["model"]) for row in core}),
        },
    )
    return {"figure": name, "files": files, "source_csv": str(source), "metadata": str(metadata)}


def _figure_four(
    easy: list[dict[str, str]],
    core: list[dict[str, Any]],
    external: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = "figure_04_control_vs_transfer"
    easy_values = _metric_rows(
        easy, group_keys=("environment",), metric="test_auroc", context_prefix="easy-auroc"
    )
    same_values = _metric_rows_seed_covered(
        [row for row in core if str(row["train_environment"]) == str(row["test_environment"])],
        group_keys=("test_environment",),
        metric="auroc",
        context_prefix="same-source-auroc",
    )
    cross_values = _metric_rows_seed_covered(
        [row for row in core if str(row["train_environment"]) != str(row["test_environment"])],
        group_keys=("test_environment",),
        metric="auroc",
        context_prefix="cross-gold-auroc",
    )
    measured = [row for row in external if _is_measured(str(row["evaluation_surface"]))]
    measured_values = _metric_rows_seed_covered(
        measured,
        group_keys=("train_environment", "evaluation_surface"),
        metric="auroc",
        context_prefix="measured-auroc",
    )
    measured_surfaces = sorted(
        {str(row["evaluation_surface"]) for row in measured}, key=_surface_sort
    )
    source_rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        source_rows.extend(
            [
                {
                    "panel": "A",
                    "environment": environment,
                    "x_easy_auroc": easy_values[(environment,)],
                    "y_metric": same_values[(environment,)],
                    "y_definition": "deep-model same-source AUROC",
                },
                {
                    "panel": "B",
                    "environment": environment,
                    "x_easy_auroc": easy_values[(environment,)],
                    "y_metric": cross_values[(environment,)],
                    "y_definition": "mean AUROC across other Gold test constructions",
                },
            ]
        )
        for surface in measured_surfaces:
            source_rows.append(
                {
                    "panel": "C",
                    "environment": environment,
                    "surface": surface,
                    "x_easy_auroc": easy_values[(environment,)],
                    "y_metric": measured_values[(environment, surface)],
                    "y_definition": f"{surface} transfer AUROC",
                }
            )
    source = args.output_dir / f"figure_data__{name}.csv"
    _require_committed_csv(source, source_rows, args.repo_root)
    column_labels = ["Easy", "Same Gold", "Other Gold", "Vidal/Y2H", "IntAct"]
    surface_lookup = {"Vidal/Y2H": "vidal__measured", "IntAct": "intact__measured"}
    matrix = np.asarray(
        [
            [easy_values[(env,)], same_values[(env,)], cross_values[(env,)]]
            + [measured_values[(env, surface_lookup[name])] for name in column_labels[3:]]
            for env in ENVIRONMENTS
        ]
    )
    norm = Normalize(
        vmin=np.floor(float(matrix.min()) * 20) / 20, vmax=np.ceil(float(matrix.max()) * 20) / 20
    )
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(PAPER_WIDTH, 3.6))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1, 0.055), left=0.17, right=0.9, bottom=0.20, top=0.93, wspace=0.16
    )
    axis = figure.add_subplot(grid[0, 0])
    image = heatmap(axis, matrix, cmap=cmap, norm=norm, annotations="{:.3f}", annotation_size=8.8)
    axis.set_yticks(range(len(ENVIRONMENTS)))
    axis.set_yticklabels([SAMPLER_LABELS[key] for key in ENVIRONMENTS])
    axis.set_xticks(range(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=22, ha="right")
    colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[0, 1]))
    colorbar.ax.set_title("AUROC", size=AXIS_SIZE, pad=5)
    colorbar.ax.tick_params(labelsize=8.8)
    files = _save(figure, args.output_dir / name, tight=False)
    metadata = _figure_metadata(
        args=args,
        name=name,
        source_csv=source,
        input_paths=[*getattr(args, "signal_inputs", []), args.core_registry, args.protocol],
        formula=(
            "descriptive AUROC matrix from easy-property, same-Gold, "
            "other-Gold, and measured-transfer source values; no correlation test is used"
        ),
        filters={"measured_surfaces": measured_surfaces, "construction_count": len(ENVIRONMENTS)},
    )
    return {"figure": name, "files": files, "source_csv": str(source), "metadata": str(metadata)}


def _figure_five(
    external: list[dict[str, Any]], datasail: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    name = "figure_05_measured_and_ood"
    all_rows = [*external, *datasail]
    surface_key = (
        "evaluation_surface"
        if all_rows and "evaluation_surface" in all_rows[0]
        else "test_environment"
    )
    measured = [row for row in all_rows if _is_measured(str(row.get(surface_key, "")))]
    strict = [
        row
        for row in all_rows
        if _is_strict_ood(str(row.get(surface_key, ""))) or str(row.get("scope", "")) == "datasail"
    ]
    if not measured:
        raise SystemExit("Measured external rows are required for Figure 5")
    grouped = _metric_rows_seed_covered(
        measured,
        group_keys=("model", surface_key),
        metric="auroc",
        context_prefix="measured-figure5",
    )
    strict_grouped = (
        _metric_rows_seed_covered(
            strict,
            group_keys=("model", surface_key),
            metric="auroc",
            context_prefix="strict-figure5",
        )
        if strict
        else {}
    )
    source_rows: list[dict[str, Any]] = []
    for (model, surface), value in sorted(
        grouped.items(), key=lambda item: (item[0][0], _surface_sort(item[0][1]))
    ):
        source_rows.append(
            {
                "block": "measured",
                "model": model,
                "surface": surface,
                "auroc": value,
                "row_count": len(
                    [
                        row
                        for row in measured
                        if row["model"] == model and row[surface_key] == surface
                    ]
                ),
            }
        )
    for (model, surface), value in sorted(
        strict_grouped.items(), key=lambda item: (item[0][0], _surface_sort(item[0][1]))
    ):
        source_rows.append(
            {
                "block": "strict_ood",
                "model": model,
                "surface": surface,
                "auroc": value,
                "row_count": len(
                    [row for row in strict if row["model"] == model and row[surface_key] == surface]
                ),
            }
        )
    source = args.output_dir / f"figure_data__{name}.csv"
    _require_committed_csv(source, source_rows, args.repo_root)
    models = sorted({str(row["model"]) for row in source_rows})
    surfaces = sorted({str(row["surface"]) for row in source_rows}, key=_surface_sort)
    models = [model for model in FIGURE_MODEL_ORDER if model in models]
    surfaces = [
        surface
        for surface in ("vidal__measured", "intact__measured", "official_datasail")
        if surface in surfaces
    ]
    lookup = {(str(row["model"]), str(row["surface"])): float(row["auroc"]) for row in source_rows}
    matrix = np.ma.masked_invalid(
        np.asarray(
            [[lookup.get((model, surface), np.nan) for surface in surfaces] for model in models]
        )
    )
    valid = matrix.compressed()
    norm = Normalize(
        vmin=np.floor(float(valid.min()) * 20) / 20, vmax=np.ceil(float(valid.max()) * 20) / 20
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#edf0f3")
    figure = plt.figure(figsize=(PAPER_WIDTH, 2.75))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1, 0.05), left=0.2, right=0.90, bottom=0.16, top=0.9, wspace=0.17
    )
    axis = figure.add_subplot(grid[0, 0])
    image = heatmap(axis, matrix, cmap=cmap, norm=norm, annotations="{:.3f}", annotation_size=9.0)
    for row_index, column_index in np.argwhere(np.ma.getmaskarray(matrix)):
        axis.text(column_index, row_index, "—", ha="center", va="center", fontsize=9, color=MUTED)
    axis.set_yticks(range(len(models)))
    axis.set_yticklabels([PAPER_MODEL_LABELS[model] for model in models])
    axis.set_xticks(range(len(surfaces)))
    axis.set_xticklabels(
        [
            {
                "vidal__measured": "Vidal/Y2H",
                "intact__measured": "IntAct",
                "official_datasail": "DataSAIL",
            }[surface]
            for surface in surfaces
        ]
    )
    colorbar = figure.colorbar(image, cax=figure.add_subplot(grid[0, 1]))
    colorbar.ax.set_title("AUROC", size=AXIS_SIZE, pad=5)
    colorbar.ax.tick_params(labelsize=8.8)
    files = _save(figure, args.output_dir / name, tight=False)
    metadata = _figure_metadata(
        args=args,
        name=name,
        source_csv=source,
        input_paths=[*getattr(args, "signal_inputs", []), args.core_registry, args.protocol],
        formula=(
            "mean AUROC over exact seeds {11,22,33}; measured and strict-OOD "
            "estimands remain separate"
        ),
        filters={
            "measured_surfaces": sorted(
                {str(row[surface_key]) for row in measured}, key=_surface_sort
            ),
            "strict_ood_rows": len(strict),
            "strict_ood_included": bool(strict),
        },
        extra={
            "strict_ood_status": "included" if strict else "omitted_insufficient_exact_novelty_rows"
        },
    )
    return {"figure": name, "files": files, "source_csv": str(source), "metadata": str(metadata)}


def _require_sources(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    list[Path],
]:
    def first(*paths: Path) -> Path:
        return next((path for path in paths if path.is_file()), paths[0])

    easy_path = first(
        args.audit_dir / "easy_feature_audit.csv",
        args.audit_dir / "e003_easy_feature_shortcuts.csv",
    )
    protein_path = first(
        args.audit_dir / "protein_only_baseline.csv",
        args.audit_dir / "e003_protein_only_results.csv",
    )
    bernett_path = first(
        args.audit_dir / "bernett_simple_baseline.csv",
        args.audit_dir / "e003_bernett_simple_baseline.csv",
    )
    if not easy_path.is_file() or not protein_path.is_file() or not bernett_path.is_file():
        raise SystemExit("Signal-audit CSVs are incomplete; final figures remain closed")
    broken_paths = sorted(args.audit_dir.rglob("broken_metrics__*.csv"))
    if not broken_paths:
        committed_broken = args.audit_dir / "e003_broken_pair_results.csv"
        if committed_broken.is_file():
            broken_paths = [committed_broken]
    if not broken_paths:
        raise SystemExit("Broken-pair score CSVs are missing; final figures remain closed")
    registry = _load_registry(args.core_registry)
    core = [row for row in registry if str(row.get("scope", "core")) == "core"]
    external = [row for row in registry if str(row.get("scope", "")) == "external"]
    datasail = [row for row in registry if str(row.get("scope", "")) == "datasail"]
    models = sorted({str(row["model"]) for row in core})
    expected_core = _expected_core_cells(len(models))
    if len(core) != expected_core:
        raise SystemExit(f"Core registry is incomplete: {len(core)} rows, expected {expected_core}")
    if not external:
        raise SystemExit("External registry rows are missing; final figures remain closed")
    easy = _read_csv(easy_path)
    protein = _read_csv(protein_path)
    bernett = _read_csv(bernett_path)
    broken = [row for path in broken_paths for row in _read_csv(path)]
    sources = [
        args.config,
        args.protocol,
        args.core_registry,
        easy_path,
        protein_path,
        bernett_path,
        *broken_paths,
    ]
    return easy, protein, bernett, core, external, datasail, broken, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--core-registry", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/e003_method_fidelity_final.yaml")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("audit/E003_SIGNAL_AUDIT_PROTOCOL.md")
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/e003_final"))
    parser.add_argument(
        "--figures",
        type=int,
        choices=(1, 2, 3, 4, 5),
        nargs="+",
        help="Build selected figures; by default build the complete E003 pack.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    easy, protein, bernett, core, external, datasail, broken, sources = _require_sources(args)
    args.signal_inputs = sources
    selected_figures = set(args.figures or (1, 2, 3, 4, 5))
    figures = []
    if 1 in selected_figures:
        figures.append(_figure_one(config, args))
    if 2 in selected_figures:
        figures.extend(_figure_two(core, external, args))
    if 3 in selected_figures:
        figures.append(_figure_three(easy, protein, bernett, core, broken, args))
    if 4 in selected_figures:
        figures.append(_figure_four(easy, core, external, args))
    if 5 in selected_figures:
        figures.append(_figure_five(external, datasail, args))

    manifest_path = args.output_dir / "figure_manifest.json"
    if args.figures and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacements = {figure["figure"]: figure for figure in figures}
        merged = []
        for old_figure in existing.get("figures", []):
            merged.append(replacements.pop(old_figure["figure"], old_figure))
        figures = [*merged, *replacements.values()]

    payload = base_metadata(args.repo_root, args.protocol)
    payload.update(
        {
            "schema_version": "ppi_shift.e003_final_figure_manifest.v2",
            "style_sha256": sha256_path(Path(__file__).with_name("paper_figure_style.py")),
            "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
            "source_files": _source_hashes(sources),
            "figures": figures,
            "dpi": 300,
            "scientific_values_source": (
                "canonical_machine_readable_registry_and_signal_audit_outputs"
            ),
        }
    )
    write_json(manifest_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
