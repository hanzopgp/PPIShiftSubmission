"""Method-faithful E003 negative-environment constructors.

E001 and E002 sampler modules are historical contracts and are deliberately not
modified here.  This module implements the two corrected published-method
surfaces required by E003:

* Bernett et al.'s global bias-aware ILP candidate construction and objective;
* TPPNI's logistic unipartite configuration model, exact bottom-N ranking, and
  Contrastive-L3 filter.

The functions return global pair identities first.  Mapping those pairs into a
balanced model panel is a separate, deterministic adapter step so that the
paper-level negative set cannot accidentally become an anchor-local sampler.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ppi_shift.data import unordered_pair_key
from ppi_shift.negatives import (
    CandidateUniverse,
    NegativeContractError,
    PositivePair,
    ProteinRecord,
    contract_hash,
)

E003_SCHEMA_VERSION = "ppi_shift.e003_method_fidelity.v1"
BERNETT_ENVIRONMENT = "bernett2026_ilp_exact"
TPPNI_ENVIRONMENT = "tppni2025_scm_cl3_exact"
TPPNI_BALANCED_ENVIRONMENT = "tppni2025_scm_cl3_exact_balanced"
TPPNI_LEAKAGE_SAFE_ENVIRONMENT = "tppni2025_scm_cl3_leakage_safe"
E003_NEGATIVE_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    BERNETT_ENVIRONMENT,
    TPPNI_BALANCED_ENVIRONMENT,
)
# Public compatibility name used by the E003 runners and audit scripts.  Keep
# one canonical tuple so the panel, matrix, and postprocess contracts cannot
# drift to different environment rosters.
E003_ENVIRONMENTS = E003_NEGATIVE_ENVIRONMENTS
MAX_DEGREE_SLACK = 5.0


@dataclass(frozen=True)
class BernettConfig:
    """The official Nextflow ILP defaults used by the E003 adapter."""

    lambda_degree: float = 1.0
    lambda_taxon_pair: float = 1.0
    lambda_self_loop: float = 1.0
    lambda_jaccard: float = 1.0
    negative_ratio: float = 1.0
    candidate_cap_multiplier: int = 4
    solver: str = "scipy.optimize.milp/HiGHS"
    time_limit_seconds: float = 7200.0
    mip_rel_gap: float = 0.01

    def __post_init__(self) -> None:
        if self.negative_ratio <= 0:
            raise ValueError("Bernett negative_ratio must be positive")
        if self.candidate_cap_multiplier < 1:
            raise ValueError("Bernett candidate_cap_multiplier must be positive")
        if self.time_limit_seconds <= 0:
            raise ValueError("Bernett time limit must be positive")
        if not 0 <= self.mip_rel_gap:
            raise ValueError("Bernett mip_rel_gap must be nonnegative")
        for name in (
            "lambda_degree",
            "lambda_taxon_pair",
            "lambda_self_loop",
            "lambda_jaccard",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"Bernett {name} must be nonnegative")


@dataclass(frozen=True)
class TPPNIExactPool:
    """The exact TPPNI ranking surface and its L3-zero subset."""

    bottom_n_pairs: tuple[tuple[str, str], ...]
    l3_zero_pairs: tuple[tuple[str, str], ...]
    probabilities: tuple[float, ...]
    l3_counts: tuple[int, ...]
    diagnostics: dict[str, Any]


def _pair_id(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _record_map(universe: CandidateUniverse) -> dict[str, ProteinRecord]:
    return dict(universe.proteins)


def _positive_id_pairs(positives: Sequence[PositivePair]) -> set[tuple[str, str]]:
    return {_pair_id(positive.left_id, positive.right_id) for positive in positives}


def _positive_sequence_pairs(positives: Sequence[PositivePair]) -> set[tuple[str, str]]:
    return {unordered_pair_key(positive.left_hash, positive.right_hash) for positive in positives}


def _is_legal_global_pair(
    left: ProteinRecord,
    right: ProteinRecord,
    *,
    positive_id_pairs: set[tuple[str, str]],
    positive_sequence_pairs: set[tuple[str, str]],
    universe: CandidateUniverse,
    allow_self: bool,
) -> bool:
    if not allow_self and left.protein_id == right.protein_id:
        return False
    if _pair_id(left.protein_id, right.protein_id) in positive_id_pairs:
        return False
    if unordered_pair_key(left.sequence_sha256, right.sequence_sha256) in positive_sequence_pairs:
        return False
    return not universe.exclusion_registry.excludes(
        left.protein_id,
        right.protein_id,
        left.sequence_sha256,
        right.sequence_sha256,
    )


def _positive_degree_array(positives: Sequence[PositivePair], ids: Sequence[str]) -> np.ndarray:
    index = {protein_id: position for position, protein_id in enumerate(ids)}
    degree = np.zeros(len(ids), dtype=np.float64)
    for positive in positives:
        left = index[positive.left_id]
        right = index[positive.right_id]
        degree[left] += 1.0
        if left != right:
            degree[right] += 1.0
    return degree


def _pair_incidence(
    pairs: Sequence[tuple[int, int]], n_proteins: int
) -> tuple[np.ndarray, np.ndarray]:
    incidence = np.zeros((n_proteins, len(pairs)), dtype=np.float64)
    for column, (left, right) in enumerate(pairs):
        incidence[left, column] = 1.0
        if left != right:
            incidence[right, column] = 1.0
    return incidence, incidence.sum(axis=1)


def _jaccard(left: Collection[str], right: Collection[str]) -> float:
    union = len(set(left) | set(right))
    return float(len(set(left) & set(right)) / union) if union else 0.0


def _pair_terms(
    pairs: Sequence[tuple[int, int]],
    ids: Sequence[str],
    taxon_by_id: Mapping[str, str],
    go_bp_by_id: Mapping[str, Collection[str]],
) -> tuple[list[tuple[str, str]], np.ndarray]:
    taxon_pairs = [
        tuple(sorted((str(taxon_by_id[ids[left]]), str(taxon_by_id[ids[right]]))))
        for left, right in pairs
    ]
    jaccards = np.asarray(
        [
            _jaccard(
                go_bp_by_id.get(ids[left], ()),
                go_bp_by_id.get(ids[right], ()),
            )
            for left, right in pairs
        ],
        dtype=np.float64,
    )
    return taxon_pairs, jaccards


def _official_pair_sampling(
    *,
    legal_pairs: Sequence[tuple[int, int]] | None,
    excluded_pairs: set[tuple[int, int]],
    positive_pairs: Sequence[tuple[int, int]],
    ids: Sequence[str],
    degree: np.ndarray,
    taxon_by_id: Mapping[str, str],
    go_bp_by_id: Mapping[str, Collection[str]],
    target: int,
    seed: int,
    lambda_self_loop: float,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Reproduce the official cap policy without materializing a huge pool.

    The pinned Bernett implementation force-keeps legal self-pairs when the
    self-loop term is active, stratifies the remaining draw to the positive
    same-taxonomy ratio, prioritizes nonzero GO-BP Jaccard, and samples protein
    endpoints proportional to positive degree.  For pools that fit the cap the
    caller bypasses this routine and keeps the exact full complement.
    """

    if target < 1:
        raise NegativeContractError("Bernett candidate target must be positive")
    legal_count = (
        len(legal_pairs)
        if legal_pairs is not None
        else len(ids) * (len(ids) + 1) // 2 - len(excluded_pairs)
    )
    if legal_count < target:
        raise NegativeContractError(
            f"Bernett legal candidate pool has {legal_count} pairs; {target} required"
        )
    self_pairs = [
        (index, index) for index in range(len(ids)) if (index, index) not in excluded_pairs
    ]
    if lambda_self_loop > 0 and len(self_pairs) > target:
        raise NegativeContractError(
            "Bernett official self-pair force-keep policy exceeds candidate cap"
        )
    selected: set[tuple[int, int]] = set(self_pairs if lambda_self_loop > 0 else ())
    rng = np.random.default_rng(seed)
    remaining = target - len(selected)
    if remaining <= 0:
        return sorted(selected), {
            "self_pairs_forced": len(selected),
            "same_taxon_target": 0,
            "cross_taxon_target": 0,
            "sampling_rounds": 0,
        }

    weights = np.asarray(degree, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    positive_taxon_same = 0
    if positive_pairs:
        positive_taxon_same = sum(
            taxon_by_id[ids[left]] == taxon_by_id[ids[right]] for left, right in positive_pairs
        )
    same_ratio = positive_taxon_same / max(1, len(positive_pairs))
    same_target = int(round(same_ratio * remaining))
    cross_target = remaining - same_target
    rounds = 0

    def collect(quota: int, want_same: bool | None) -> int:
        nonlocal rounds
        if quota <= 0:
            return 0
        priority: set[tuple[int, int]] = set()
        filler: set[tuple[int, int]] = set()
        for _ in range(200):
            rounds += 1
            if len(priority | filler) >= quota:
                break
            batch_size = max(10_000, (quota - len(priority | filler)) * 5)
            first = rng.choice(len(ids), size=batch_size, p=weights)
            second = rng.choice(len(ids), size=batch_size, p=weights)
            for left_raw, right_raw in zip(first.tolist(), second.tolist(), strict=True):
                left, right = sorted((int(left_raw), int(right_raw)))
                pair = (left, right)
                if pair in excluded_pairs or pair in selected:
                    continue
                if want_same is not None:
                    is_same = taxon_by_id[ids[left]] == taxon_by_id[ids[right]]
                    if is_same != want_same:
                        continue
                jac = _jaccard(
                    go_bp_by_id.get(ids[left], ()),
                    go_bp_by_id.get(ids[right], ()),
                )
                (priority if jac > 0 else filler).add(pair)
            if len(priority | filler) >= quota:
                break
        ordered = sorted(priority) + sorted(filler)
        for pair in ordered[:quota]:
            selected.add(pair)
        return min(quota, len(ordered))

    same_found = collect(same_target, True)
    cross_found = collect(cross_target, False)
    # The official sampler uses an unstratified shortfall fill when a stratum is
    # smaller than its quota.
    shortfall = target - len(selected)
    unstratified_found = collect(shortfall, None)
    if len(selected) < target:
        # For a small known pool, complete deterministically. For a large implicit
        # pool, do not materialize the complement just to hide a sampling failure.
        if legal_pairs is not None:
            for pair in sorted(set(legal_pairs) - selected):
                selected.add(pair)
                if len(selected) == target:
                    break
    if len(selected) != target:
        raise NegativeContractError("Bernett official cap sampler could not fill its target")
    return sorted(selected), {
        "self_pairs_forced": len(self_pairs) if lambda_self_loop > 0 else 0,
        "same_taxon_target": same_target,
        "cross_taxon_target": cross_target,
        "same_taxon_found": same_found,
        "cross_taxon_found": cross_found,
        "unstratified_found": unstratified_found,
        "sampling_rounds": rounds,
        "shortfall_completion": len(selected) == target and unstratified_found > 0,
    }


def _candidate_pool_for_bernett(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    *,
    config: BernettConfig,
    seed: int,
    taxon_by_id: Mapping[str, str],
    go_bp_by_id: Mapping[str, Collection[str]],
) -> tuple[list[str], list[tuple[int, int]], dict[str, Any]]:
    ids = sorted(universe.proteins)
    records = _record_map(universe)
    positive_id_pairs = _positive_id_pairs(positives)
    positive_sequence_pairs = _positive_sequence_pairs(positives)
    index = {protein_id: position for position, protein_id in enumerate(ids)}
    ids_by_hash: dict[str, list[int]] = defaultdict(list)
    for protein_id, position in index.items():
        ids_by_hash[records[protein_id].sequence_sha256].append(position)
    excluded_pairs: set[tuple[int, int]] = set()

    def exclude_pair(left: int, right: int) -> None:
        excluded_pairs.add((min(left, right), max(left, right)))

    for left_id, right_id in positive_id_pairs:
        if left_id in index and right_id in index:
            exclude_pair(index[left_id], index[right_id])
    for left_hash, right_hash in positive_sequence_pairs:
        for left in ids_by_hash.get(left_hash, []):
            for right in ids_by_hash.get(right_hash, []):
                exclude_pair(left, right)
    for left_id, right_id in universe.exclusion_registry.identifier_pairs:
        if left_id in index and right_id in index:
            exclude_pair(index[left_id], index[right_id])
    for left_hash, right_hash in universe.exclusion_registry.sequence_pairs:
        for left in ids_by_hash.get(left_hash, []):
            for right in ids_by_hash.get(right_hash, []):
                exclude_pair(left, right)

    full_pair_count = len(ids) * (len(ids) + 1) // 2
    full_legal_count = full_pair_count - len(excluded_pairs)
    target = config.candidate_cap_multiplier * len(positives)
    if full_legal_count <= target:
        legal = [
            (left, right)
            for left in range(len(ids))
            for right in range(left, len(ids))
            if (left, right) not in excluded_pairs
        ]
        return (
            ids,
            legal,
            {
                "candidate_pool_mode": "full_legal_upper_triangle",
                "full_legal_candidate_count": full_legal_count,
                "candidate_cap": target,
                "candidate_count": len(legal),
                "candidate_cap_subsampled": False,
                "candidate_subsampling": {},
            },
        )
    positive_pairs = [_pair_id(positive.left_id, positive.right_id) for positive in positives]
    positive_index_pairs = [
        (min(index[left], index[right]), max(index[left], index[right]))
        for left, right in positive_pairs
    ]
    degree = _positive_degree_array(positives, ids)
    selected, sampling = _official_pair_sampling(
        legal_pairs=None,
        excluded_pairs=excluded_pairs,
        positive_pairs=positive_index_pairs,
        ids=ids,
        degree=degree,
        taxon_by_id=taxon_by_id,
        go_bp_by_id=go_bp_by_id,
        target=target,
        seed=seed,
        lambda_self_loop=config.lambda_self_loop,
    )
    return (
        ids,
        selected,
        {
            "candidate_pool_mode": "official_weighted_stratified_subsample",
            "full_legal_candidate_count": full_legal_count,
            "candidate_cap": target,
            "candidate_count": len(selected),
            "candidate_cap_subsampled": True,
            "candidate_subsampling": sampling,
        },
    )


def _build_bernett_milp(
    positives: Sequence[PositivePair],
    ids: Sequence[str],
    candidate_pairs: Sequence[tuple[int, int]],
    *,
    config: BernettConfig,
    taxon_by_id: Mapping[str, str],
    go_bp_by_id: Mapping[str, Collection[str]],
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:  # pragma: no cover
        raise NegativeContractError("E003 Bernett requires scipy.optimize.milp") from error

    index = {protein_id: position for position, protein_id in enumerate(ids)}
    positive_index_pairs = [
        (index[positive.left_id], index[positive.right_id]) for positive in positives
    ]
    n_proteins = len(ids)
    n_candidates = len(candidate_pairs)
    n_negative = int(round(config.negative_ratio * len(positives)))
    if n_candidates < n_negative:
        raise NegativeContractError("Bernett candidate cap is smaller than negative target")
    incidence, candidate_degree = _pair_incidence(candidate_pairs, n_proteins)
    positive_degree = _positive_degree_array(positives, ids)

    taxon_pairs, candidate_jaccard = _pair_terms(candidate_pairs, ids, taxon_by_id, go_bp_by_id)
    positive_taxon_pairs = [
        tuple(sorted((taxon_by_id[positive.left_id], taxon_by_id[positive.right_id])))
        for positive in positives
    ]
    taxon_groups = sorted(set(taxon_pairs) | set(positive_taxon_pairs))
    taxon_index = {group: position for position, group in enumerate(taxon_groups)}
    candidate_taxon_group = np.asarray(
        [taxon_index[group] for group in taxon_pairs], dtype=np.int64
    )
    positive_taxon_counts = np.zeros(len(taxon_groups), dtype=np.float64)
    for group in positive_taxon_pairs:
        positive_taxon_counts[taxon_index[group]] += 1.0
    candidate_taxon_counts = np.bincount(candidate_taxon_group, minlength=len(taxon_groups)).astype(
        np.float64
    )
    taxon_gamma = np.full(
        len(taxon_groups),
        1.0 / math.log(2.0),
        dtype=np.float64,
    )
    positive_taxon_mask = positive_taxon_counts > 0
    taxon_gamma[positive_taxon_mask] = 1.0 / np.log1p(positive_taxon_counts[positive_taxon_mask])

    self_mask = np.asarray([left == right for left, right in candidate_pairs])
    self_candidate_count = int(self_mask.sum())
    positive_self_count = sum(left == right for left, right in positive_index_pairs)
    jaccard_positive = np.asarray(
        [
            _jaccard(
                go_bp_by_id.get(positive.left_id, ()),
                go_bp_by_id.get(positive.right_id, ()),
            )
            for positive in positives
        ],
        dtype=np.float64,
    )
    jaccard_target = float(jaccard_positive.mean()) if len(jaccard_positive) else 0.0

    active_degree = np.flatnonzero((positive_degree > 0) | (candidate_degree > 0))
    degree_coef = 1.0 / np.log1p(positive_degree[active_degree])
    degree_target = config.negative_ratio * positive_degree[active_degree]
    degree_u = float(
        np.sum(
            degree_coef * np.maximum(degree_target, candidate_degree[active_degree] - degree_target)
        )
    )
    taxon_u = float(
        np.sum(
            taxon_gamma
            * np.maximum(
                config.negative_ratio * positive_taxon_counts,
                candidate_taxon_counts - config.negative_ratio * positive_taxon_counts,
            )
        )
    )
    self_target = config.negative_ratio * positive_self_count
    self_u = float(max(self_target, self_candidate_count - self_target))
    jaccard_u = float(max(jaccard_target, 1.0 - jaccard_target))

    offsets: dict[str, int] = {"x": 0}
    cursor = n_candidates
    offsets["degree"] = cursor
    cursor += len(active_degree)
    offsets["taxon"] = cursor
    cursor += len(taxon_groups)
    offsets["self"] = cursor
    cursor += 1
    offsets["jaccard"] = cursor
    cursor += 1
    variable_count = cursor

    rows: list[tuple[str, float, float]] = []
    # Rows are assembled below as (kind, lower, upper); the sparse coefficients
    # are kept explicit so the source objective remains inspectable.
    rows.append(("count", float(n_negative), float(n_negative)))
    for cap in positive_degree:
        rows.append(("cap", -np.inf, config.negative_ratio * (1.0 + MAX_DEGREE_SLACK) * cap))
    for _ in active_degree:
        rows.extend([("degree_plus", -np.inf, 0.0), ("degree_minus", -np.inf, 0.0)])
    for _ in taxon_groups:
        rows.extend([("taxon_plus", -np.inf, 0.0), ("taxon_minus", -np.inf, 0.0)])
    rows.extend([("self_plus", -np.inf, 0.0), ("self_minus", -np.inf, 0.0)])
    rows.extend([("jaccard_plus", -np.inf, 0.0), ("jaccard_minus", -np.inf, 0.0)])
    matrix = lil_matrix((len(rows), variable_count), dtype=np.float64)
    lower = np.asarray([row[1] for row in rows], dtype=np.float64)
    upper = np.asarray([row[2] for row in rows], dtype=np.float64)
    row_index = 0
    matrix[row_index, :n_candidates] = 1.0
    row_index += 1
    for protein_index, cap in enumerate(positive_degree):
        matrix[row_index, :n_candidates] = incidence[protein_index]
        upper[row_index] = config.negative_ratio * (1.0 + MAX_DEGREE_SLACK) * cap
        row_index += 1
    for local, protein_index in enumerate(active_degree):
        residual = offsets["degree"] + local
        coefficients = incidence[protein_index]
        matrix[row_index, :n_candidates] = coefficients
        matrix[row_index, residual] = -1.0
        upper[row_index] = degree_target[local]
        row_index += 1
        matrix[row_index, :n_candidates] = -coefficients
        matrix[row_index, residual] = -1.0
        upper[row_index] = -degree_target[local]
        row_index += 1
    for group_index, group in enumerate(taxon_groups):
        residual = offsets["taxon"] + group_index
        mask = candidate_taxon_group == group_index
        matrix[row_index, :n_candidates] = mask.astype(np.float64)
        matrix[row_index, residual] = -1.0
        target = config.negative_ratio * positive_taxon_counts[group_index]
        upper[row_index] = target
        row_index += 1
        matrix[row_index, :n_candidates] = -mask.astype(np.float64)
        matrix[row_index, residual] = -1.0
        upper[row_index] = -target
        row_index += 1
    self_variable = offsets["self"]
    matrix[row_index, :n_candidates] = self_mask.astype(np.float64)
    matrix[row_index, self_variable] = -1.0
    upper[row_index] = self_target
    row_index += 1
    matrix[row_index, :n_candidates] = -self_mask.astype(np.float64)
    matrix[row_index, self_variable] = -1.0
    upper[row_index] = -self_target
    row_index += 1
    jac_variable = offsets["jaccard"]
    jac_coefficients = candidate_jaccard / n_negative
    matrix[row_index, :n_candidates] = jac_coefficients
    matrix[row_index, jac_variable] = -1.0
    upper[row_index] = jaccard_target
    row_index += 1
    matrix[row_index, :n_candidates] = -jac_coefficients
    matrix[row_index, jac_variable] = -1.0
    upper[row_index] = -jaccard_target

    objective = np.zeros(variable_count, dtype=np.float64)
    if degree_u > 0:
        objective[offsets["degree"] : offsets["taxon"]] = (
            config.lambda_degree * degree_coef / degree_u
        )
    if taxon_u > 0:
        objective[offsets["taxon"] : offsets["self"]] = (
            config.lambda_taxon_pair * taxon_gamma / taxon_u
        )
    if self_u > 0:
        objective[self_variable] = config.lambda_self_loop / self_u
    if jaccard_u > 0:
        objective[jac_variable] = config.lambda_jaccard / jaccard_u
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:n_candidates] = 1
    variable_lower = np.zeros(variable_count, dtype=np.float64)
    variable_upper = np.full(variable_count, np.inf, dtype=np.float64)
    variable_upper[:n_candidates] = 1.0
    metadata = {
        "n_negative": n_negative,
        "n_proteins": n_proteins,
        "n_candidates": n_candidates,
        "positive_degree": positive_degree,
        "candidate_degree": candidate_degree,
        "candidate_jaccard": candidate_jaccard,
        "candidate_taxon_group": candidate_taxon_group,
        "positive_taxon_counts": positive_taxon_counts,
        "taxon_gamma": taxon_gamma,
        "self_mask": self_mask,
        "self_target": self_target,
        "jaccard_target": jaccard_target,
        "degree_u": degree_u,
        "taxon_u": taxon_u,
        "self_u": self_u,
        "jaccard_u": jaccard_u,
    }
    return (
        (
            milp,
            Bounds(variable_lower, variable_upper),
            LinearConstraint(matrix.tocsr(), lower, upper),
        ),
        objective,
        {"integrality": integrality, "metadata": metadata},
    )


def _bernett_selected_diagnostics(
    selected: Sequence[tuple[int, int]], metadata: Mapping[str, Any], *, config: BernettConfig
) -> dict[str, Any]:
    selected_array = np.zeros(metadata["n_candidates"], dtype=np.float64)
    # Caller supplies the selected candidate indices through this temporary field.
    selected_array[np.asarray(metadata["selected_indices"], dtype=np.int64)] = 1.0
    degree_minus = metadata["candidate_degree_vector"] @ selected_array
    positive_degree = metadata["positive_degree"]
    degree_residual = np.abs(degree_minus - config.negative_ratio * positive_degree)
    taxon_counts = np.bincount(
        metadata["candidate_taxon_group"][metadata["selected_indices"]],
        minlength=len(metadata["positive_taxon_counts"]),
    )
    taxon_residual = np.abs(
        taxon_counts - config.negative_ratio * metadata["positive_taxon_counts"]
    )
    selected_jaccard = float(metadata["candidate_jaccard"][metadata["selected_indices"]].mean())
    self_count = int(metadata["self_mask"][metadata["selected_indices"]].sum())
    return {
        "selected_pair_count": len(selected),
        "selected_self_loop_count": self_count,
        "degree_absolute_residual_sum": float(degree_residual.sum()),
        "degree_max_absolute_residual": float(degree_residual.max(initial=0.0)),
        "taxon_pair_absolute_residual_sum": float(taxon_residual.sum()),
        "mean_positive_go_bp_jaccard": float(metadata["jaccard_target"]),
        "mean_selected_go_bp_jaccard": selected_jaccard,
    }


def generate_bernett_exact_pairs(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    *,
    seed: int,
    taxon_by_id: Mapping[str, str] | None = None,
    go_bp_by_id: Mapping[str, Collection[str]] | None = None,
    config: BernettConfig | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Generate one global Bernett negative set with the official ILP objective."""

    if not positives:
        raise NegativeContractError("Bernett requires positive rows")
    resolved = config or BernettConfig()
    taxon = dict(taxon_by_id or {})
    go_bp = {key: frozenset(value) for key, value in (go_bp_by_id or {}).items()}
    missing_taxon = sorted(set(universe.proteins) - set(taxon))
    if missing_taxon and resolved.lambda_taxon_pair > 0:
        raise NegativeContractError(
            f"Bernett taxonomy annotation missing for {len(missing_taxon)} proteins"
        )
    if missing_taxon:
        taxon.update({protein_id: "" for protein_id in missing_taxon})
    missing_go = sorted(set(universe.proteins) - set(go_bp))
    if missing_go and resolved.lambda_jaccard > 0:
        raise NegativeContractError(
            f"Bernett GO-BP annotation missing for {len(missing_go)} proteins"
        )
    go_bp.update({protein_id: frozenset() for protein_id in missing_go})
    ids, candidate_pairs, pool_diagnostics = _candidate_pool_for_bernett(
        positives,
        universe,
        config=resolved,
        seed=seed,
        taxon_by_id=taxon,
        go_bp_by_id=go_bp,
    )
    if len(candidate_pairs) < int(round(resolved.negative_ratio * len(positives))):
        raise NegativeContractError("Bernett candidate pool cannot support the negative target")
    problem, objective, payload = _build_bernett_milp(
        positives,
        ids,
        candidate_pairs,
        config=resolved,
        taxon_by_id=taxon,
        go_bp_by_id=go_bp,
    )
    milp, bounds, constraints = problem
    try:
        result = milp(
            c=objective,
            integrality=payload["integrality"],
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": resolved.time_limit_seconds,
                "mip_rel_gap": resolved.mip_rel_gap,
                "presolve": True,
            },
        )
    except Exception as error:  # pragma: no cover - solver/environment specific
        raise NegativeContractError(f"Bernett HiGHS invocation failed: {error}") from error
    if result.x is None:
        raise NegativeContractError(f"Bernett solver returned no solution: {result.message}")
    selected_indices = np.flatnonzero(np.asarray(result.x[: len(candidate_pairs)]) >= 0.5)
    expected = int(round(resolved.negative_ratio * len(positives)))
    if len(selected_indices) != expected:
        raise NegativeContractError(
            f"Bernett solver selected {len(selected_indices)} pairs; expected {expected}; "
            f"status={result.status} message={result.message}"
        )
    if int(getattr(result, "status", 1)) != 0:
        raise NegativeContractError(
            "Bernett solver did not reach an optimal exact solution: "
            f"status={result.status} message={result.message}"
        )
    selected_pairs = [candidate_pairs[index] for index in selected_indices.tolist()]
    selected_id_pairs = [_pair_id(ids[left], ids[right]) for left, right in selected_pairs]
    payload["metadata"]["n_candidates"] = len(candidate_pairs)
    payload["metadata"]["selected_indices"] = selected_indices
    payload["metadata"]["candidate_degree_vector"] = _pair_incidence(candidate_pairs, len(ids))[0]
    selected_audit = _bernett_selected_diagnostics(
        selected_pairs, payload["metadata"], config=resolved
    )
    diagnostics = {
        "status": "optimal",
        "method": "Bernett2026_bias_aware_negative_ilp",
        "upstream_repository": "https://github.com/bionetslab/ppi-splitting-pipeline",
        "upstream_revision": "8bf9adb183b702c88db62ff878de482e2f20e4b2",
        "candidate_pairs_sha256": contract_hash(
            [[ids[left], ids[right]] for left, right in candidate_pairs]
        ),
        "selected_pairs_sha256": contract_hash([list(pair) for pair in selected_id_pairs]),
        "solver": resolved.solver,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_objective": float(result.fun),
        "solver_mip_gap": resolved.mip_rel_gap,
        "solver_time_limit_seconds": resolved.time_limit_seconds,
        "seed": seed,
        "weights": {
            "degree": resolved.lambda_degree,
            "taxon_pair": resolved.lambda_taxon_pair,
            "self_loop": resolved.lambda_self_loop,
            "go_bp_jaccard": resolved.lambda_jaccard,
        },
        "candidate_pool": pool_diagnostics,
        "positive_rows": len(positives),
        "negative_rows": len(selected_id_pairs),
        "annotation_coverage": {
            "taxonomy_proteins": len(taxon),
            "go_bp_proteins": len(go_bp),
            "go_bp_nonempty_proteins": sum(bool(terms) for terms in go_bp.values()),
        },
        "selected_audit": selected_audit,
        "semantic_class": "synthetic_unobserved_pair",
    }
    return selected_id_pairs, diagnostics


def _scm_expected_degrees(theta: np.ndarray, *, block_size: int = 256) -> np.ndarray:
    expected = np.zeros(len(theta), dtype=np.float64)
    try:
        from scipy.special import expit
    except ImportError as error:  # pragma: no cover
        raise NegativeContractError("TPPNI requires scipy.special.expit") from error
    for start in range(0, len(theta), block_size):
        stop = min(len(theta), start + block_size)
        probabilities = expit(theta[:, None] + theta[start:stop][None, :])
        rows = np.arange(start, stop)
        probabilities[rows, rows - start] = 0.0
        expected += probabilities.sum(axis=1)
    return expected


def solve_tppni_scm_lambdas(
    degree_by_id: Mapping[str, int],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Solve the exact logistic degree equations numerically.

    ``theta_i = -lambda_i`` is fitted so that
    ``sum_j expit(theta_i + theta_j) = d_i`` for every distinct pair.  A damped
    iterative-proportional-fitting update is used for large graphs to avoid a
    dense ``n x n`` Hessian; it still solves the exact equations and refuses to
    return a model when the final degree residual is above the declared tolerance.
    """

    ids = sorted(degree_by_id)
    if len(ids) < 2:
        raise NegativeContractError("TPPNI SCM requires at least two proteins")
    n = len(ids)
    degrees = np.asarray([float(degree_by_id[protein_id]) for protein_id in ids])
    if np.any(degrees < 0) or np.any(degrees > n - 1):
        raise NegativeContractError("TPPNI degrees must lie in [0, n-1]")
    theta = np.log((degrees + 0.5) / (n - 0.5 - degrees))
    theta = np.clip(theta, -50.0, 50.0)
    theta[degrees == 0] = -50.0
    theta[degrees >= n - 1] = 50.0
    converged = False
    residual_max = math.inf
    iteration = 0
    if n <= 512:
        try:
            from scipy.optimize import least_squares
            from scipy.special import expit
        except ImportError as error:  # pragma: no cover
            raise NegativeContractError("TPPNI requires scipy.optimize") from error

        def residual_function(values: np.ndarray) -> np.ndarray:
            probabilities = expit(values[:, None] + values[None, :])
            np.fill_diagonal(probabilities, 0.0)
            return probabilities.sum(axis=1) - degrees

        def jacobian_function(values: np.ndarray) -> np.ndarray:
            probabilities = expit(values[:, None] + values[None, :])
            np.fill_diagonal(probabilities, 0.0)
            weights = probabilities * (1.0 - probabilities)
            np.fill_diagonal(weights, 0.0)
            jacobian = weights.copy()
            np.fill_diagonal(jacobian, weights.sum(axis=1))
            return jacobian

        result = least_squares(
            residual_function,
            theta,
            jac=jacobian_function,
            bounds=(-50.0, 50.0),
            xtol=min(tolerance, 1e-12),
            ftol=min(tolerance, 1e-12),
            gtol=min(tolerance, 1e-12),
            max_nfev=max_iterations,
        )
        theta = np.asarray(result.x, dtype=np.float64)
        residual_vector = residual_function(theta)
        residual_max = float(np.max(np.abs(residual_vector)))
        iteration = int(result.nfev)
        converged = bool(result.success) and residual_max <= tolerance
    else:
        residual_vector = np.zeros_like(degrees)
        for iteration in range(1, max_iterations + 1):
            expected = _scm_expected_degrees(theta)
            residual = expected - degrees
            residual_vector = residual
            residual_max = float(np.max(np.abs(residual)))
            if residual_max <= tolerance:
                converged = True
                break
            update = np.zeros_like(theta)
            active = degrees > 0
            update[active] = np.log(
                np.maximum(degrees[active], 1e-12) / np.maximum(expected[active], 1e-12)
            )
            update[~active] = -50.0 - theta[~active]
            # Halving the step when a sparse degree sequence overshoots is stable
            # for the large frozen backbone.
            damping = 0.5 if iteration < 8 else 0.8
            theta = np.clip(theta + damping * update, -50.0, 50.0)
            theta[degrees == 0] = -50.0
    if not converged:
        raise NegativeContractError(
            "TPPNI SCM did not converge to the declared degree tolerance: "
            f"max_residual={residual_max} iterations={max_iterations}"
        )
    absolute_residual = np.abs(residual_vector)
    expected_degrees = degrees + residual_vector
    relative_denominator = np.maximum(degrees, 1.0)
    relative_residual = absolute_residual / relative_denominator
    pathological = np.flatnonzero((degrees <= 0.0) | (degrees >= float(n - 1)) | (degrees < 1.0))
    return {protein_id: float(-theta[position]) for position, protein_id in enumerate(ids)}, {
        "solver": "exact_logistic_degree_equations_damped_ipf",
        "iterations": iteration,
        "max_degree_residual": residual_max,
        "mean_abs_degree_residual": float(np.mean(absolute_residual)),
        "max_relative_degree_error": float(np.max(relative_residual)),
        "mean_relative_degree_error": float(np.mean(relative_residual)),
        "tolerance": tolerance,
        "boundary_lambda_count": int(np.sum(np.abs(theta) >= 49.999)),
        "pathological_degree_count": int(len(pathological)),
        "pathological_degree_proteins": [ids[int(index)] for index in pathological[:100]],
        "protein_count": n,
        "degree_trace": {
            "protein_ids": ids,
            "observed": degrees.tolist(),
            "expected": expected_degrees.tolist(),
        },
    }


def _tppni_probability(lambda_left: float, lambda_right: float) -> float:
    value = -(lambda_left + lambda_right)
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _l3_count(left: str, right: str, adjacency: Mapping[str, set[str]]) -> int:
    neighbors_left = adjacency.get(left, set())
    neighbors_right = adjacency.get(right, set())
    return sum(len(adjacency.get(middle, set()) & neighbors_right) for middle in neighbors_left)


def generate_tppni_exact_pool(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    *,
    topology_positives: Sequence[PositivePair],
    published_n: int = 10_000_000,
    scm_tolerance: float = 1e-8,
) -> TPPNIExactPool:
    """Construct the exact paper-faithful bottom-N and L3-zero TPPNI pool."""

    if published_n < 1:
        raise ValueError("TPPNI published_n must be positive")
    ids = sorted(universe.proteins)
    positive_id_pairs = _positive_id_pairs(positives)
    positive_sequence_pairs = _positive_sequence_pairs(positives)
    records = _record_map(universe)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for positive in topology_positives:
        adjacency[positive.left_id].add(positive.right_id)
        adjacency[positive.right_id].add(positive.left_id)
    degree = {protein_id: len(adjacency.get(protein_id, set())) for protein_id in ids}
    lambdas, scm_diagnostics = solve_tppni_scm_lambdas(degree, tolerance=scm_tolerance)
    # The TPPNI protein-pair graph is simple and non-self-interacting: enumerate
    # i<j only. A min-heap merges rows of the pairwise lambda-sum ordering, so the
    # exact bottom-N surface is obtained without materializing every pair.
    # lambda is lower for high degree and p increases as -(lambda_i+lambda_j);
    # therefore sort by probability directly with the corresponding theta order.
    theta = {protein_id: -lambdas[protein_id] for protein_id in ids}
    # The upstream method ranks floating-point probabilities. Numerical SCM
    # solves can differ by a few ulps for mathematically tied pairs, so E003
    # freezes a small comparison tolerance and resolves those ties by IDs.
    tie_tolerance = 1e-12
    order = sorted(
        ids,
        key=lambda protein_id: (
            round(theta[protein_id] / tie_tolerance),
            protein_id,
        ),
    )
    heap: list[tuple[float, str, str, float, int, int]] = []
    for left_index in range(len(order) - 1):
        right_index = left_index + 1
        left = order[left_index]
        right = order[right_index]
        heapq.heappush(
            heap,
            (
                round(_tppni_probability(lambdas[left], lambdas[right]) / tie_tolerance)
                * tie_tolerance,
                left,
                right,
                _tppni_probability(lambdas[left], lambdas[right]),
                left_index,
                right_index,
            ),
        )
    bottom: list[tuple[tuple[str, str], float, int]] = []
    target = published_n
    while heap and len(bottom) < target:
        _sort_probability, left, right, probability, left_index, right_index = heapq.heappop(heap)
        if _is_legal_global_pair(
            records[left],
            records[right],
            positive_id_pairs=positive_id_pairs,
            positive_sequence_pairs=positive_sequence_pairs,
            universe=universe,
            allow_self=False,
        ):
            pair = _pair_id(left, right)
            bottom.append((pair, probability, _l3_count(left, right, adjacency)))
        next_right_index = right_index + 1
        if next_right_index < len(order):
            next_right = order[next_right_index]
            heapq.heappush(
                heap,
                (
                    round(_tppni_probability(lambdas[left], lambdas[next_right]) / tie_tolerance)
                    * tie_tolerance,
                    left,
                    next_right,
                    _tppni_probability(lambdas[left], lambdas[next_right]),
                    left_index,
                    next_right_index,
                ),
            )
    bottom_pairs = tuple(item[0] for item in bottom)
    bottom_probabilities = tuple(float(item[1]) for item in bottom)
    bottom_l3 = tuple(int(item[2]) for item in bottom)
    l3_zero = tuple(pair for pair, _probability, l3 in bottom if l3 == 0)
    diagnostics = {
        "status": "complete" if len(bottom) < published_n else "complete_requested_bottom_n",
        "method": "TPPNI_exact_logistic_scm_contrastive_l3",
        "upstream_repository": "https://github.com/jbusto/UPNA-PPI",
        "upstream_revision": "20d1d8225e9d377434669973b7391697be6b4aa2",
        "published_n": published_n,
        "effective_bottom_n": len(bottom),
        "l3_zero_count": len(l3_zero),
        "l3_nonzero_count": len(bottom) - len(l3_zero),
        "topology_positive_rows": len(topology_positives),
        "topology_protein_count": len(adjacency),
        "candidate_protein_count": len(ids),
        "positive_exclusion_count": len(positive_id_pairs),
        "scm": scm_diagnostics,
        "bottom_n_pair_order": "ascending exact SCM probability, then pair IDs",
        "bottom_n_tie_tolerance": tie_tolerance,
        "l3_definition": "sum_{u in N(i)} |N(u) intersection N(j)|",
        "candidate_pair_semantic_class": "synthetic_unobserved_pair",
        "leakage_safe": False,
        "pair_probabilities_sha256": contract_hash(
            [[list(pair), probability] for pair, probability, _l3 in bottom]
        ),
    }
    return TPPNIExactPool(
        bottom_n_pairs=bottom_pairs,
        l3_zero_pairs=l3_zero,
        probabilities=bottom_probabilities,
        l3_counts=bottom_l3,
        diagnostics=diagnostics,
    )


def select_tppni_balanced_pairs(
    pool: TPPNIExactPool,
    *,
    count: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Select a deterministic balanced sensitivity subset from the exact L3 pool."""

    if count < 1:
        raise ValueError("Balanced TPPNI count must be positive")
    if len(pool.l3_zero_pairs) < count:
        raise NegativeContractError(
            f"TPPNI L3-zero pool has {len(pool.l3_zero_pairs)} pairs; {count} required"
        )
    rng = np.random.default_rng(seed)
    chosen_indices = np.sort(rng.choice(len(pool.l3_zero_pairs), size=count, replace=False))
    selected = [pool.l3_zero_pairs[int(index)] for index in chosen_indices]
    diagnostics = {
        "status": "complete",
        "source_pool": "tppni2025_scm_cl3_exact",
        "source_l3_zero_count": len(pool.l3_zero_pairs),
        "selected_count": len(selected),
        "seed": seed,
        "selection": "numpy.default_rng_without_replacement_over_sorted_l3_zero_pool",
        "selected_pairs_sha256": contract_hash([list(pair) for pair in selected]),
        "negative_semantic_class": "synthetic_unobserved_pair",
    }
    return selected, diagnostics
