"""E002 negative-environment contracts and published-method adapters.

This module deliberately sits beside the frozen E001 implementation.  E001 panels are
referenced by hash and are never rewritten here.  The two new samplers consume the
same E001-compatible positive rows, candidate universes, and exclusion registries.
Their output keeps the single authoritative panel columns used by the E001 runners,
but uses a distinct schema and explicit diagnostics for the published-method
adaptations required by the anchor-fixed factorial.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import sha256_path, write_csv
from ppi_shift.negatives import (
    ANCHOR_ORIENTATION_RULE,
    PANEL_COLUMNS,
    POSITIVE_SEMANTIC_CLASS,
    CandidateUniverse,
    NegativeContractError,
    PositivePair,
    ProteinRecord,
    anchor_orientation,
    canonical_json,
    contract_hash,
)

E001_V3_SCHEMA_VERSION = "ppi_shift.e001_negative_environment.v3"
E002_SCHEMA_VERSION = "ppi_shift.e002_negative_environment.v1"
E002_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett_bias_aware_ilp",
    "tppni_contrastive_l3",
)
E002_OLD_ENVIRONMENTS = E002_ENVIRONMENTS[:4]
E002_NEW_ENVIRONMENTS = E002_ENVIRONMENTS[4:]
NEGATIVE_SEMANTIC_CLASS = "synthetic_unobserved_pair"
MAX_RAW_SEQUENCE_LENGTH = 512


def expected_e002_panel_schema(
    manifest: Mapping[str, Any], split: str, environment: str
) -> str:
    """Return the immutable schema for an E001 reference or a new E002 panel."""
    if environment in manifest.get("e001_references", {}).get(split, {}):
        return E001_V3_SCHEMA_VERSION
    return E002_SCHEMA_VERSION


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _hash_rank(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest(),
        byteorder="big",
        signed=False,
    )


def _row_id(environment: str, seed: int, positive_id: str, candidate_id: str) -> str:
    digest = contract_hash(
        {
            "schema_version": E002_SCHEMA_VERSION,
            "environment": environment,
            "seed": seed,
            "positive_row_id": positive_id,
            "candidate_id": candidate_id,
        }
    )
    return f"negative:{digest[:32]}"


def _positive_row(
    positive: PositivePair,
    universe: CandidateUniverse,
    environment: str,
    seed: int,
) -> dict[str, Any]:
    anchor, true_partner = anchor_orientation(positive)
    return {
        "row_id": positive.positive_row_id,
        "split": positive.split,
        "dataset_name": positive.dataset_name,
        "name1": anchor.protein_id,
        "name2": true_partner.protein_id,
        "seq1": anchor.sequence,
        "seq2": true_partner.sequence,
        "interaction_label": 1,
        "positive_row_id": positive.positive_row_id,
        "positive_pair_key": positive.positive_pair_key,
        "anchor_id": anchor.protein_id,
        "true_partner_id": true_partner.protein_id,
        "candidate_partner_id": "",
        "anchor_sequence_sha256": anchor.sequence_sha256,
        "true_partner_sequence_sha256": true_partner.sequence_sha256,
        "candidate_sequence_sha256": "",
        "negative_environment": environment,
        "negative_semantic_class": POSITIVE_SEMANTIC_CLASS,
        "sampler_version": E002_SCHEMA_VERSION,
        "generation_seed": seed,
        "anchor_orientation_rule": ANCHOR_ORIENTATION_RULE,
        "candidate_universe_id": universe.universe_id,
        "candidate_universe_sha256": universe.universe_sha256,
        "exclusion_registry_name": universe.exclusion_registry.name,
        "exclusion_registry_sha256": universe.exclusion_registry.sha256,
        "source_input_hashes_json": canonical_json(universe.source_input_hashes),
        "eligible_candidate_count": "",
        "candidate_rank": "",
        "match_distance": "",
        "match_diagnostics_json": "{}",
        "fallback_status": "not_applicable",
        "rejection_reason": "",
    }


def negative_row(
    positive: PositivePair,
    universe: CandidateUniverse,
    environment: str,
    seed: int,
    candidate: ProteinRecord,
    *,
    rank: int,
    diagnostics: Mapping[str, Any],
    fallback_status: str = "none",
    match_distance: float | str = "",
) -> dict[str, Any]:
    anchor, true_partner = anchor_orientation(positive)
    return {
        "row_id": _row_id(environment, seed, positive.positive_row_id, candidate.protein_id),
        "split": positive.split,
        "dataset_name": f"{positive.dataset_name}__{environment}",
        "name1": anchor.protein_id,
        "name2": candidate.protein_id,
        "seq1": anchor.sequence,
        "seq2": candidate.sequence,
        "interaction_label": 0,
        "positive_row_id": positive.positive_row_id,
        "positive_pair_key": positive.positive_pair_key,
        "anchor_id": anchor.protein_id,
        "true_partner_id": true_partner.protein_id,
        "candidate_partner_id": candidate.protein_id,
        "anchor_sequence_sha256": anchor.sequence_sha256,
        "true_partner_sequence_sha256": true_partner.sequence_sha256,
        "candidate_sequence_sha256": candidate.sequence_sha256,
        "negative_environment": environment,
        "negative_semantic_class": NEGATIVE_SEMANTIC_CLASS,
        "sampler_version": f"{E002_SCHEMA_VERSION}:{environment}",
        "generation_seed": seed,
        "anchor_orientation_rule": ANCHOR_ORIENTATION_RULE,
        "candidate_universe_id": universe.universe_id,
        "candidate_universe_sha256": universe.universe_sha256,
        "exclusion_registry_name": universe.exclusion_registry.name,
        "exclusion_registry_sha256": universe.exclusion_registry.sha256,
        "source_input_hashes_json": canonical_json(universe.source_input_hashes),
        "eligible_candidate_count": universe.candidate_count(positive),
        "candidate_rank": rank,
        "match_distance": match_distance,
        "match_diagnostics_json": canonical_json(dict(diagnostics)),
        "fallback_status": fallback_status,
        "rejection_reason": "",
    }


def validate_e002_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    environment: str | None = None,
    universe: CandidateUniverse | None = None,
) -> None:
    """Validate the balanced E002 row contract without inspecting model outputs."""

    if environment not in E002_ENVIRONMENTS and environment is not None:
        raise NegativeContractError(f"Unknown E002 environment: {environment}")
    seen: set[str] = set()
    positive_ids: set[str] = set()
    negative_ids: set[str] = set()
    observed_environments: set[str] = set()
    observed_universes: set[str] = set()
    positives_by_id = (
        {positive.positive_row_id: positive for positive in universe.positives} if universe else {}
    )
    for row in rows:
        missing = [field for field in PANEL_COLUMNS if field not in row]
        if missing:
            raise NegativeContractError(f"E002 row lacks fields: {missing}")
        row_id = str(row["row_id"])
        if row_id in seen:
            raise NegativeContractError(f"Duplicate E002 row ID: {row_id}")
        seen.add(row_id)
        current_environment = str(row["negative_environment"])
        observed_environments.add(current_environment)
        observed_universes.add(str(row["candidate_universe_sha256"]))
        if current_environment not in E002_ENVIRONMENTS:
            raise NegativeContractError(f"Unknown E002 environment: {current_environment}")
        label = str(row["interaction_label"])
        if label == "1":
            positive_ids.add(str(row["positive_row_id"]))
            if row["negative_semantic_class"] != POSITIVE_SEMANTIC_CLASS:
                raise NegativeContractError(f"Positive row has synthetic semantics: {row_id}")
            if not str(row["candidate_partner_id"]) == "":
                raise NegativeContractError(f"Positive row has a candidate: {row_id}")
        elif label == "0":
            negative_ids.add(str(row["positive_row_id"]))
            if row["negative_semantic_class"] != NEGATIVE_SEMANTIC_CLASS:
                raise NegativeContractError(f"Synthetic row has non-synthetic semantics: {row_id}")
            if row["anchor_id"] == row["candidate_partner_id"]:
                raise NegativeContractError(f"Self-pair negative: {row_id}")
            if row["true_partner_id"] == row["candidate_partner_id"]:
                raise NegativeContractError(f"Positive partner reused: {row_id}")
            if not str(row["sampler_version"]).startswith(f"{E002_SCHEMA_VERSION}:"):
                raise NegativeContractError(f"Unsupported E002 sampler schema: {row_id}")
        else:
            raise NegativeContractError(f"E002 panel label is not binary: {row_id}")
        for field in (
            "anchor_sequence_sha256",
            "true_partner_sequence_sha256",
            "candidate_universe_sha256",
            "exclusion_registry_sha256",
        ):
            if label == "1" and field == "candidate_universe_sha256":
                value = str(row[field])
            else:
                value = str(row[field])
            if not _sha256(value):
                raise NegativeContractError(f"Invalid SHA-256 in {field}: {row_id}")
        if label == "0" and not _sha256(str(row["candidate_sequence_sha256"])):
            raise NegativeContractError(f"Invalid candidate sequence hash: {row_id}")
        try:
            json.loads(str(row["match_diagnostics_json"]))
            json.loads(str(row["source_input_hashes_json"]))
        except json.JSONDecodeError as error:
            raise NegativeContractError(f"Invalid E002 provenance JSON: {row_id}") from error
        if (
            len(str(row["seq1"])) > MAX_RAW_SEQUENCE_LENGTH
            or len(str(row["seq2"])) > MAX_RAW_SEQUENCE_LENGTH
        ):
            raise NegativeContractError(f"Overlength sequence reached E002 panel: {row_id}")
        if universe is not None and label == "0":
            if str(row["candidate_universe_id"]) != universe.universe_id:
                raise NegativeContractError(f"Candidate universe ID mismatch: {row_id}")
            positive = positives_by_id.get(str(row["positive_row_id"]))
            if positive is None or not universe.is_eligible(
                positive, str(row["candidate_partner_id"])
            ):
                raise NegativeContractError(f"Known-positive or forbidden candidate: {row_id}")
    if environment is not None and observed_environments != {environment}:
        raise NegativeContractError(
            f"E002 panel environment mismatch: {sorted(observed_environments)} != {[environment]}"
        )
    if len(observed_environments) != 1:
        raise NegativeContractError(
            f"E002 panel mixes environments: {sorted(observed_environments)}"
        )
    if len(observed_universes) != 1:
        raise NegativeContractError("E002 panel mixes candidate universes")
    if not positive_ids or positive_ids != negative_ids:
        raise NegativeContractError("E002 panel is not one-to-one by positive identity")
    if len(rows) != 2 * len(positive_ids):
        raise NegativeContractError("E002 panel has duplicate positive identities")


def positive_from_panel_row(row: Mapping[str, Any]) -> PositivePair:
    """Recover a positive identity for universe validation from a panel row."""

    return PositivePair(
        split=str(row["split"]),
        source_row_index=0,
        dataset_name=str(row.get("dataset_name", "")),
        left_id=str(row["anchor_id"]),
        right_id=str(row["true_partner_id"]),
        left_sequence=str(row["seq1"]),
        right_sequence=str(row["seq2"]),
        identity_schema_version="ppi_shift.e001_negative_environment.v3",
    )


def write_e002_panel(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic gzip CSV bytes for a new E002 panel."""

    validate_e002_panel(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name.endswith(".gz"):
        with output.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=list(PANEL_COLUMNS),
                        extrasaction="ignore",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(
                        {field: row.get(field, "") for field in PANEL_COLUMNS} for row in rows
                    )
    else:
        write_csv(output, PANEL_COLUMNS, rows)


def read_e002_panel(path: str | Path, *, environment: str | None = None) -> list[dict[str, str]]:
    path_value = Path(path)
    opener = gzip.open if path_value.name.endswith(".gz") else open
    with opener(path_value, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PANEL_COLUMNS:
            raise NegativeContractError(f"Unexpected E002 panel columns in {path}")
        rows = list(reader)
    validate_e002_panel(rows, environment=environment)
    return rows


def _candidate_records(
    positive: PositivePair, universe: CandidateUniverse
) -> tuple[ProteinRecord, ...]:
    candidates = universe.candidate_records(positive)
    if not candidates:
        raise NegativeContractError(f"No legal E002 candidates for {positive.positive_row_id}")
    return candidates


def bernett_bias_costs(
    positive_degrees: Mapping[str, int],
    candidate_pairs: Sequence[tuple[str, str]],
    *,
    negative_ratio: float = 1.0,
    degree_weight: float = 0.6,
) -> dict[str, float]:
    """Return the official Bernett degree-bias weights for a fixture or audit.

    The upstream implementation weights per-protein absolute residuals by
    ``1/log1p(d_plus)`` and targets ``negative_ratio*d_plus``.  This helper exposes
    that formula independently of the solver so the integration test can compare
    adapter arithmetic against the pinned upstream source.
    """

    degree = Counter()
    for left, right in candidate_pairs:
        degree[left] += 1
        degree[right] += 1
    residual = 0.0
    for protein, observed_degree in positive_degrees.items():
        target = negative_ratio * float(observed_degree)
        residual += abs(float(degree[protein]) - target) / max(1.0, math.log1p(observed_degree))
    return {
        "degree_bias_weight": float(degree_weight),
        "negative_ratio": float(negative_ratio),
        "unweighted_absolute_residual": float(residual),
        "weighted_objective": float(degree_weight * residual),
    }


def _select_hash_capped_candidates(
    positive: PositivePair,
    universe: CandidateUniverse,
    candidates: Sequence[ProteinRecord],
    *,
    seed: int,
    cap: int,
    degree_table: Mapping[str, int],
) -> list[ProteinRecord]:
    """Build a deterministic legal candidate network for the open solver.

    The cap is a declared feasibility control, not a hidden sampler fallback.  It is
    recorded per row and in the manifest.  Candidate-network support is an official
    Bernett input mode; the anchor-fixed cap preserves the E002 one-negative-per-row
    contract while keeping HiGHS tractable.
    """

    if cap < 1:
        raise NegativeContractError("Bernett candidate cap must be positive")
    _anchor, true_partner = anchor_orientation(positive)
    target_degree = int(degree_table.get(true_partner.protein_id, 0))
    ordered = sorted(
        candidates,
        key=lambda record: (
            abs(int(degree_table.get(record.protein_id, 0)) - target_degree),
            _hash_rank(
                E002_SCHEMA_VERSION,
                "bernett-candidate-network",
                seed,
                positive.positive_row_id,
                record.protein_id,
            ),
            record.protein_id,
        ),
    )
    return ordered[: min(cap, len(ordered))]


def generate_bernett_negatives(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    *,
    seed: int,
    degree_table: Mapping[str, int],
    candidate_cap_per_anchor: int = 8,
    time_limit_seconds: float = 1800.0,
    mip_rel_gap: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve the anchor-fixed Bernett objective with open-source HiGHS.

    Active terms retain the upstream degree-bias formula.  Taxon, self-loop, and GO
    terms are recorded as inactive when the E001 Gold contract supplies no such
    annotations; no synthetic metadata is invented.  Failure is explicit and stops
    materialization rather than silently switching to a different sampler.
    """

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:  # pragma: no cover - exercised on unsupported hosts
        raise NegativeContractError("Bernett requires scipy.optimize.milp/HiGHS") from error

    if not positives:
        raise NegativeContractError("Bernett requires positive rows")
    candidate_network: list[list[ProteinRecord]] = []
    variable_pairs: list[tuple[int, ProteinRecord]] = []
    for anchor_index, positive in enumerate(positives):
        capped = _select_hash_capped_candidates(
            positive,
            universe,
            _candidate_records(positive, universe),
            seed=seed,
            cap=candidate_cap_per_anchor,
            degree_table=degree_table,
        )
        candidate_network.append(capped)
        variable_pairs.extend((anchor_index, candidate) for candidate in capped)
    if not variable_pairs:
        raise NegativeContractError("Bernett candidate network is empty")

    protein_ids = tuple(sorted(universe.proteins))
    protein_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
    zero_degree_ids = {
        protein_id for protein_id in protein_ids if int(degree_table.get(protein_id, 0)) == 0
    }
    zero_degree_cap = (
        max(1, math.ceil(len(positives) / max(1, len(zero_degree_ids)))) if zero_degree_ids else 0
    )
    variable_count = len(variable_pairs)
    deviation_offset = variable_count
    variable_total = variable_count + len(protein_ids)
    rows_count = len(positives)
    # one anchor equality, two absolute-residual inequalities, one hard cap per protein
    constraint_count = rows_count + 3 * len(protein_ids)
    matrix = lil_matrix((constraint_count, variable_total), dtype=np.float64)
    lower = np.full(constraint_count, -np.inf, dtype=np.float64)
    upper = np.full(constraint_count, np.inf, dtype=np.float64)
    for variable_index, (anchor_index, candidate) in enumerate(variable_pairs):
        matrix[anchor_index, variable_index] = 1.0
        candidate_index = protein_index[candidate.protein_id]
        matrix[rows_count + candidate_index, variable_index] += 1.0
        matrix[rows_count + len(protein_ids) + candidate_index, variable_index] += -1.0
        matrix[rows_count + 2 * len(protein_ids) + candidate_index, variable_index] += 1.0
    lower[:rows_count] = 1.0
    upper[:rows_count] = 1.0
    max_degree_slack = 5
    for index, protein_id in enumerate(protein_ids):
        deviation_variable = deviation_offset + index
        positive_degree = float(degree_table.get(protein_id, 0))
        target = positive_degree
        matrix[rows_count + index, deviation_variable] = -1.0
        upper[rows_count + index] = target
        matrix[rows_count + len(protein_ids) + index, deviation_variable] = -1.0
        upper[rows_count + len(protein_ids) + index] = -target
        official_cap = (1.0 + max_degree_slack) * positive_degree
        upper[rows_count + 2 * len(protein_ids) + index] = float(
            max(official_cap, zero_degree_cap) if positive_degree == 0.0 else official_cap
        )
        if positive_degree == 0.0:
            # No positive degree means there is no meaningful upstream target; keep
            # the hard cap but do not reward arbitrary residual movement.  Remove
            # both residual rows entirely; otherwise the zero target would
            # accidentally forbid every candidate incident on this protein.
            residual_row = rows_count + index
            reverse_residual_row = rows_count + len(protein_ids) + index
            matrix[residual_row, :] = 0.0
            matrix[reverse_residual_row, :] = 0.0
            upper[residual_row] = np.inf
            upper[reverse_residual_row] = np.inf
        else:
            # Store the coefficient in the objective below through the same weight.
            pass
    objective = np.zeros(variable_total, dtype=np.float64)
    for index, protein_id in enumerate(protein_ids):
        positive_degree = float(degree_table.get(protein_id, 0))
        objective[deviation_offset + index] = 0.6 / max(1.0, math.log1p(positive_degree))
    integrality = np.zeros(variable_total, dtype=np.int8)
    integrality[:variable_count] = 1
    variable_lower = np.zeros(variable_total, dtype=np.float64)
    variable_upper = np.ones(variable_total, dtype=np.float64)
    variable_upper[deviation_offset:] = np.inf
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": True,
        },
    )
    status = str(getattr(result, "message", "unknown"))
    if not bool(getattr(result, "success", False)) or result.x is None:
        raise NegativeContractError(f"Bernett HiGHS solve failed: status={status}")
    selected: dict[int, ProteinRecord] = {}
    for variable_index, (anchor_index, candidate) in enumerate(variable_pairs):
        if float(result.x[variable_index]) >= 0.5:
            selected[anchor_index] = candidate
    if set(selected) != set(range(len(positives))):
        raise NegativeContractError(
            "Bernett solver did not select exactly one candidate per anchor"
        )
    negatives = []
    selected_pairs = []
    for index, positive in enumerate(positives):
        candidate = selected[index]
        anchor, _true_partner = anchor_orientation(positive)
        selected_pairs.append((anchor.protein_id, candidate.protein_id))
        anchor_degree = float(degree_table.get(anchor.protein_id, 0))
        negatives.append(
            negative_row(
                positive,
                universe,
                "bernett_bias_aware_ilp",
                seed,
                candidate,
                rank=next(
                    rank
                    for rank, record in enumerate(candidate_network[index], start=1)
                    if record.protein_id == candidate.protein_id
                ),
                diagnostics={
                    "method": "bionetslab/ppi-splitting-pipeline",
                    "objective": "official_degree_bias_absolute_residual",
                    "lambda_degree": 0.6,
                    "lambda_taxon": 0.0,
                    "lambda_self": 0.1,
                    "lambda_jaccard": 0.3,
                    "candidate_network_mode": "anchor_fixed_legal_hash_degree_cap",
                    "candidate_network_cap_per_anchor": candidate_cap_per_anchor,
                    "candidate_network_count": len(candidate_network[index]),
                    "degree_target_ratio": 1.0,
                    "max_degree_slack": max_degree_slack,
                    "zero_degree_cap_adjustment": zero_degree_cap
                    if anchor_degree == 0.0
                    else None,
                    "solver": "scipy.optimize.milp/HiGHS",
                    "solver_status": status,
                    "solver_objective": float(result.fun),
                    "solver_mip_gap": getattr(result, "mip_gap", None),
                    "seed": seed,
                },
            )
        )
    objective_audit = bernett_bias_costs(degree_table, selected_pairs)
    diagnostics = {
        "status": "complete",
        "solver": "HiGHS",
        "solver_status": status,
        "solver_objective": float(result.fun),
        "solver_mip_gap": getattr(result, "mip_gap", None),
        "candidate_network": {
            "legal_candidates": int(sum(universe.candidate_count(p) for p in positives)),
            "selected_candidates": int(variable_count),
            "cap_per_anchor": int(candidate_cap_per_anchor),
            "cap_rule": "closest official degree residual, then SHA256 tie order",
        },
        "zero_degree_cap_adjustment": {
            "affected_proteins": len(zero_degree_ids),
            "cap": zero_degree_cap,
            "rule": (
                "ceil(number_of_anchors / zero_degree_proteins), minimum 1, "
                "only when the official degree cap is zero"
            ),
            "reason": (
                "solver feasibility for protein-disjoint or degree-zero candidate populations"
            ),
        },
        "selected_residual_audit": objective_audit,
        "terms": {
            "degree": "active",
            "taxon_pair": (
                "inactive; all E001 Gold proteins are human and no taxon labels are introduced"
            ),
            "self_loop": "inactive; self-pairs are prohibited by the E002 candidate contract",
            "go_jaccard": "inactive; no GO annotation source is added to E001",
        },
        "selected_candidate_counts": dict(
            sorted(Counter(candidate.protein_id for candidate in selected.values()).items())
        ),
    }
    return negatives, diagnostics


def _l3_count(
    left: str,
    right: str,
    adjacency: Mapping[str, set[str]],
) -> int:
    neighbors_left = adjacency.get(left, set())
    neighbors_right = adjacency.get(right, set())
    return sum(len(adjacency.get(middle, set()) & neighbors_right) for middle in neighbors_left)


def _config_probability(
    left: str,
    right: str,
    degree_table: Mapping[str, int],
    total_degree: int,
) -> float:
    if total_degree <= 0:
        return 0.0
    return float(degree_table.get(left, 0) * degree_table.get(right, 0) / total_degree)


def _config_product_cutoff(
    candidate_pool: Sequence[ProteinRecord],
    degree_table: Mapping[str, int],
    published_n: int,
    total_degree: int,
) -> tuple[float, dict[str, int]]:
    """Find the inclusive bottom-N probability cutoff from degree histograms.

    The published N is much smaller than the Cartesian candidate universe for some
    external sources.  Enumerating every pair just to find the cutoff is needlessly
    quadratic in memory, so the integer degree products are counted by histogram
    bins.  The resulting inclusive cutoff intentionally retains all ties at the
    boundary; the exact retained count is recorded for audit.
    """

    if published_n < 1:
        raise NegativeContractError("TPPNI published N must be positive")
    degree_counts = Counter(
        int(degree_table.get(record.protein_id, 0)) for record in candidate_pool
    )
    product_counts: Counter[int] = Counter()
    degrees = sorted(degree_counts)
    for left_index, left_degree in enumerate(degrees):
        left_count = degree_counts[left_degree]
        for right_degree in degrees[left_index:]:
            pair_count = left_count * degree_counts[right_degree]
            if left_degree == right_degree:
                pair_count = left_count * (left_count - 1) // 2
            if pair_count:
                product_counts[left_degree * right_degree] += pair_count
    total_pairs = sum(product_counts.values())
    if not total_pairs:
        raise NegativeContractError("TPPNI candidate pool has no unique pairs")
    target = min(int(published_n), total_pairs)
    cumulative = 0
    cutoff_product = 0
    for product, count in sorted(product_counts.items()):
        cumulative += count
        cutoff_product = product
        if cumulative >= target:
            break
    inclusive_count = sum(
        count for product, count in product_counts.items() if product <= cutoff_product
    )
    return cutoff_product / max(1, total_degree), {
        "unique_candidate_pairs": int(total_pairs),
        "requested_bottom_n": int(published_n),
        "effective_bottom_n": int(target),
        "cutoff_degree_product": int(cutoff_product),
        "inclusive_cutoff_pair_count": int(inclusive_count),
    }


def generate_tppni_negatives(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    *,
    topology_positives: Sequence[PositivePair],
    seed: int,
    published_n: int = 10_000_000,
    concentration_max_fraction: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct TPPNI's config-model plus Contrastive-L3 surface legally.

    The public UPNA-PPI repository publishes the method/data contract but not a
    reusable constructor.  This implementation computes the unipartite configuration
    probability, takes the legal bottom-N pool, and then applies the L3==0 filter.
    Anchor fixing is explicit because the E002 factorial requires one negative per
    fixed positive; no external pair list is substituted.
    """

    adjacency: dict[str, set[str]] = defaultdict(set)
    for positive in topology_positives:
        adjacency[positive.left_id].add(positive.right_id)
        adjacency[positive.right_id].add(positive.left_id)
    degree_table = {protein_id: len(neighbors) for protein_id, neighbors in adjacency.items()}
    for protein_id in universe.proteins:
        degree_table.setdefault(protein_id, 0)
    total_degree = sum(degree_table.values())
    # UPNA-PPI's published N is larger than the legal unique-pair universe for the
    # E001 cohorts.  Detect that case without materializing the same candidate pair
    # once per positive anchor (which would multiply memory by the number of rows).
    unique_pair_upper_bound = len(universe.candidate_pool) * (len(universe.candidate_pool) - 1) // 2
    complete_pool = published_n >= unique_pair_upper_bound
    global_probability_cutoff: float | None = None
    cutoff_audit: dict[str, int] = {}
    if not complete_pool:
        global_probability_cutoff, cutoff_audit = _config_product_cutoff(
            universe.candidate_pool,
            degree_table,
            published_n,
            total_degree,
        )
    l3_forbidden_by_anchor: dict[str, set[str]] = {}
    for anchor in adjacency:
        forbidden: set[str] = set()
        for middle in adjacency[anchor]:
            forbidden.update(adjacency.get(middle, set()))
        l3_forbidden_by_anchor[anchor] = forbidden
    l3_survivors = 0
    legal_count = 0
    negatives: list[dict[str, Any]] = []
    fallback_count = 0
    selected_candidates: Counter[str] = Counter()
    selected_l3: Counter[int] = Counter()
    for positive in positives:
        anchor, _true = anchor_orientation(positive)
        legal_forbidden = universe.forbidden_by_positive[positive.positive_row_id]
        l3_forbidden = l3_forbidden_by_anchor.get(anchor.protein_id, set())
        current: list[tuple[float, int, ProteinRecord, int]] = []
        all_legal: list[tuple[float, int, ProteinRecord, int]] = []
        for candidate in universe.candidate_pool:
            if candidate.protein_id in legal_forbidden:
                continue
            legal_count += 1
            probability = _config_probability(
                anchor.protein_id, candidate.protein_id, degree_table, total_degree
            )
            rank_hash = _hash_rank(
                E002_SCHEMA_VERSION,
                "tppni-config-model",
                seed,
                positive.positive_row_id,
                candidate.protein_id,
            )
            l3 = 1 if candidate.protein_id in l3_forbidden else 0
            item = (probability, rank_hash, candidate, l3)
            all_legal.append(item)
            in_low_probability_pool = complete_pool or probability <= float(
                global_probability_cutoff
            )
            if in_low_probability_pool:
                current.append(item)
                if l3 == 0:
                    l3_survivors += 1
        fallback_status = "none"
        if not current:
            current = all_legal
            fallback_status = "no_legal_candidate_in_bottom_n_pool"
            fallback_count += 1
        current.sort(key=lambda item: (item[0], item[1], item[2].protein_id))
        in_pool = [
            item for item in current if complete_pool or item[0] <= float(global_probability_cutoff)
        ]
        eligible = [item for item in in_pool if item[3] == 0]
        if not eligible:
            eligible = in_pool
            if not eligible:
                eligible = current
            fallback_status = (
                "no_contrastive_l3_zero_candidate_in_legal_bottom_n"
                if fallback_status == "none"
                else f"{fallback_status};no_contrastive_l3_zero_candidate"
            )
            fallback_count += 1
        if not eligible:
            raise NegativeContractError(
                f"TPPNI legal bottom-N pool has no candidate for {positive.positive_row_id}"
            )
        probability, _hash, candidate, l3 = eligible[0]
        rank = next(
            index
            for index, item in enumerate(current, start=1)
            if item[2].protein_id == candidate.protein_id
        )
        selected_candidates[candidate.protein_id] += 1
        selected_l3[l3] += 1
        negatives.append(
            negative_row(
                positive,
                universe,
                "tppni_contrastive_l3",
                seed,
                candidate,
                rank=rank,
                diagnostics={
                    "method": "alxndgb/UPNA-PPI",
                    "config_model": "unipartite_configuration_model",
                    "config_probability": probability,
                    "published_n": published_n,
                    "legal_bottom_n_pool": len(in_pool),
                    "contrastive_l3_count": l3,
                    "contrastive_l3_filter": "l3_count == 0",
                    "topology_positive_rows": len(topology_positives),
                    "topology_protein_count": len(adjacency),
                    "topology_only_train_visible": True,
                    "seed": seed,
                },
                fallback_status=fallback_status,
                match_distance=probability,
            )
        )
    max_reuse = max(selected_candidates.values(), default=0)
    max_fraction = max_reuse / max(1, len(positives))
    topology_proteins = set(adjacency)
    candidate_proteins = set(universe.proteins)
    overlap = topology_proteins & candidate_proteins
    meaningful_support = l3_survivors > 0 and legal_count > 0
    role_reasons: list[str] = []
    if fallback_count:
        role_reasons.append("anchor_fallbacks_present")
    if not meaningful_support:
        role_reasons.append("no_meaningful_contrastive_l3_support")
    if max_fraction > concentration_max_fraction:
        role_reasons.append("selected_candidate_concentration_above_frozen_limit")
    if not overlap:
        role_reasons.append("test_candidate_proteins_disjoint_from_topology_graph")
    role = "primary_eligible" if not role_reasons else "supplementary_due_to_topology_contract"
    diagnostics = {
        "status": "complete",
        "published_repo": "alxndgb/UPNA-PPI",
        "published_n": published_n,
        "legal_candidate_count": legal_count,
        "legal_bottom_n_pool_count": (
            legal_count if complete_pool else "probability-cutoff-per-anchor-inclusive-ties"
        ),
        "bottom_n_cutoff": cutoff_audit,
        "contrastive_l3_zero_legal_candidates": l3_survivors,
        "fallback_anchor_count": fallback_count,
        "selected_l3_counts": dict(sorted((str(key), value) for key, value in selected_l3.items())),
        "selected_candidate_unique_count": len(selected_candidates),
        "selected_candidate_max_reuse": max_reuse,
        "selected_candidate_max_reuse_fraction": max_fraction,
        "concentration_max_fraction": concentration_max_fraction,
        "topology_protein_count": len(topology_proteins),
        "candidate_universe_protein_count": len(candidate_proteins),
        "topology_candidate_protein_overlap_count": len(overlap),
        "role": role,
        "role_reasons": role_reasons,
        "anchor_fixed_adaptation": (
            "one legal negative per fixed E001 positive; no published pair-list substitution"
        ),
    }
    return negatives, diagnostics


def panel_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact pre-score counts for a materialized E002 panel."""

    validate_e002_panel(rows)
    negatives = [row for row in rows if str(row["interaction_label"]) == "0"]
    fallbacks = Counter(str(row["fallback_status"]) for row in negatives)
    candidates = Counter(str(row["candidate_partner_id"]) for row in negatives)
    return {
        "rows": len(rows),
        "positive_rows": len(
            {str(row["positive_row_id"]) for row in rows if str(row["interaction_label"]) == "1"}
        ),
        "negative_rows": len(negatives),
        "label_counts": {"0": len(negatives), "1": len(rows) - len(negatives)},
        "fallback_status_counts": dict(sorted(fallbacks.items())),
        "unique_candidates": len(candidates),
        "maximum_candidate_reuse": max(candidates.values(), default=0),
        "semantic_class": NEGATIVE_SEMANTIC_CLASS,
        "panel_sha256_if_written": None,
    }


def panel_hash(path: str | Path) -> str:
    return sha256_path(path)
