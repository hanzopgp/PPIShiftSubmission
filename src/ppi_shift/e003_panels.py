"""E003 panel contracts for global published-method negative pools."""

from __future__ import annotations

import csv
import gzip
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ppi_shift.e003_method_fidelity import (
    E003_NEGATIVE_ENVIRONMENTS,
    E003_SCHEMA_VERSION,
    _is_legal_global_pair,
    _positive_id_pairs,
    _positive_sequence_pairs,
)
from ppi_shift.negatives import (
    ANCHOR_ORIENTATION_RULE,
    PANEL_COLUMNS,
    POSITIVE_SEMANTIC_CLASS,
    CandidateUniverse,
    NegativeContractError,
    PositivePair,
    anchor_orientation,
    canonical_json,
    contract_hash,
)

NEGATIVE_SEMANTIC_CLASS = "synthetic_unobserved_pair"
MAX_RAW_SEQUENCE_LENGTH = 512


def _compact_row_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep row provenance compact while retaining the full sampler audit once.

    TPPNI's expected-vs-observed SCM degree trace is a pool-level diagnostic and
    is retained in the immutable pool artifact referenced by the panel manifest.
    Replicating that trace in every balanced negative row makes panel files
    needlessly large without adding row-level provenance.
    """

    compact = dict(diagnostics)
    scm = compact.get("scm")
    if isinstance(scm, Mapping) and "degree_trace" in scm:
        compact_scm = dict(scm)
        compact_scm.pop("degree_trace", None)
        compact["scm"] = compact_scm
        compact["scm_degree_trace_reference"] = "pool_artifact.diagnostics.scm.degree_trace"
    return compact


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
        "sampler_version": E003_SCHEMA_VERSION,
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


def global_negative_row(
    positive: PositivePair,
    universe: CandidateUniverse,
    environment: str,
    seed: int,
    pair: tuple[str, str],
    *,
    rank: int,
    diagnostics: Mapping[str, Any],
    match_distance: float | str = "",
) -> dict[str, Any]:
    """Represent a global negative while retaining a fixed-panel row identity.

    The pair is selected by the global published-method sampler. The assigned
    ``positive_row_id`` is only a deterministic panel container for a balanced
    model input; ``match_diagnostics_json`` makes that non-anchor relationship
    explicit so it cannot be mistaken for E001/E002 anchor matching.
    """

    records = universe.proteins
    try:
        left = records[pair[0]]
        right = records[pair[1]]
    except KeyError as error:
        raise NegativeContractError(
            f"E003 pair is outside the candidate universe: {pair}"
        ) from error
    pair_key = "||".join(pair)
    row_hash = contract_hash(
        {
            "schema_version": E003_SCHEMA_VERSION,
            "environment": environment,
            "seed": seed,
            "positive_row_id": positive.positive_row_id,
            "pair": list(pair),
        }
    )
    assigned_anchor, assigned_true_partner = anchor_orientation(positive)
    merged_diagnostics = {
        **_compact_row_diagnostics(diagnostics),
        "panel_assignment": "deterministic_container_only_global_pair_not_anchor_matched",
        "assigned_positive_row_id": positive.positive_row_id,
        "assigned_positive_anchor_id": assigned_anchor.protein_id,
        "assigned_positive_true_partner_id": assigned_true_partner.protein_id,
    }
    return {
        "row_id": f"negative:{row_hash[:32]}",
        "split": positive.split,
        "dataset_name": f"{positive.dataset_name}__{environment}",
        "name1": left.protein_id,
        "name2": right.protein_id,
        "seq1": left.sequence,
        "seq2": right.sequence,
        "interaction_label": 0,
        "positive_row_id": positive.positive_row_id,
        "positive_pair_key": positive.positive_pair_key,
        "anchor_id": left.protein_id,
        "true_partner_id": assigned_true_partner.protein_id,
        "candidate_partner_id": right.protein_id,
        "anchor_sequence_sha256": left.sequence_sha256,
        "true_partner_sequence_sha256": assigned_true_partner.sequence_sha256,
        "candidate_sequence_sha256": right.sequence_sha256,
        "negative_environment": environment,
        "negative_semantic_class": NEGATIVE_SEMANTIC_CLASS,
        "sampler_version": f"{E003_SCHEMA_VERSION}:{environment}",
        "generation_seed": seed,
        "anchor_orientation_rule": ANCHOR_ORIENTATION_RULE,
        "candidate_universe_id": universe.universe_id,
        "candidate_universe_sha256": universe.universe_sha256,
        "exclusion_registry_name": universe.exclusion_registry.name,
        "exclusion_registry_sha256": universe.exclusion_registry.sha256,
        "source_input_hashes_json": canonical_json(universe.source_input_hashes),
        "eligible_candidate_count": len(universe.proteins),
        "candidate_rank": rank,
        "match_distance": match_distance,
        "match_diagnostics_json": canonical_json(
            {**merged_diagnostics, "global_pair_key": pair_key}
        ),
        "fallback_status": "none",
        "rejection_reason": "",
    }


def materialize_global_balanced_panel(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    environment: str,
    *,
    seed: int,
    pairs: Sequence[tuple[str, str]],
    diagnostics: Mapping[str, Any],
    match_distances: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create a balanced E003 panel from a global pair set."""

    if environment not in E003_NEGATIVE_ENVIRONMENTS:
        raise NegativeContractError(f"Unknown E003 environment: {environment}")
    if len(pairs) != len(positives):
        raise NegativeContractError(
            f"Balanced E003 panel needs {len(positives)} pairs; received {len(pairs)}"
        )
    if len(set(pairs)) != len(pairs):
        raise NegativeContractError("Balanced E003 panel contains duplicate global pairs")
    positive_id_pairs = _positive_id_pairs(positives)
    positive_sequence_pairs = _positive_sequence_pairs(positives)
    records = universe.proteins
    for pair in pairs:
        if pair[0] not in records or pair[1] not in records:
            raise NegativeContractError(f"Global E003 pair is outside the universe: {pair}")
        if not _is_legal_global_pair(
            records[pair[0]],
            records[pair[1]],
            positive_id_pairs=positive_id_pairs,
            positive_sequence_pairs=positive_sequence_pairs,
            universe=universe,
            allow_self=True,
        ):
            raise NegativeContractError(f"Global E003 pair is not legal: {pair}")
    ordered_positives = sorted(positives, key=lambda positive: positive.positive_row_id)
    rows: list[dict[str, Any]] = []
    distances = list(match_distances or [""] * len(pairs))
    if len(distances) != len(pairs):
        raise NegativeContractError("E003 match distance count does not match pair count")
    for rank, (positive, pair, distance) in enumerate(
        zip(ordered_positives, pairs, distances, strict=True), start=1
    ):
        rows.append(_positive_row(positive, universe, environment, seed))
        rows.append(
            global_negative_row(
                positive,
                universe,
                environment,
                seed,
                pair,
                rank=rank,
                diagnostics=diagnostics,
                match_distance=distance,
            )
        )
    validate_e003_panel(rows, environment=environment, universe=universe)
    return rows


def adapt_anchor_matched_rows_to_e003(
    rows: Sequence[Mapping[str, Any]], *, environment: str
) -> list[dict[str, Any]]:
    """Retain E001-compatible samplers as explicit E003 historical references."""

    if environment not in E003_NEGATIVE_ENVIRONMENTS[:4]:
        raise NegativeContractError(f"Not an anchor-matched E003 environment: {environment}")
    adapted = []
    for row in rows:
        copy = dict(row)
        copy["negative_environment"] = environment
        copy["sampler_version"] = (
            E003_SCHEMA_VERSION
            if str(copy["interaction_label"]) == "1"
            else f"{E003_SCHEMA_VERSION}:{environment}"
        )
        adapted.append(copy)
    return adapted


def validate_e003_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    environment: str | None = None,
    universe: CandidateUniverse | None = None,
) -> None:
    """Validate the E003 balanced panel without model-output inspection."""

    if environment is not None and environment not in E003_NEGATIVE_ENVIRONMENTS:
        raise NegativeContractError(f"Unknown E003 environment: {environment}")
    seen: set[str] = set()
    positive_ids: set[str] = set()
    negative_ids: set[str] = set()
    environments: set[str] = set()
    universes: set[str] = set()
    for row in rows:
        missing = [field for field in PANEL_COLUMNS if field not in row]
        if missing:
            raise NegativeContractError(f"E003 row lacks fields: {missing}")
        row_id = str(row["row_id"])
        if row_id in seen:
            raise NegativeContractError(f"Duplicate E003 row ID: {row_id}")
        seen.add(row_id)
        current_environment = str(row["negative_environment"])
        environments.add(current_environment)
        universes.add(str(row["candidate_universe_sha256"]))
        if current_environment not in E003_NEGATIVE_ENVIRONMENTS:
            raise NegativeContractError(f"Unknown E003 environment: {current_environment}")
        label = str(row["interaction_label"])
        if label == "1":
            positive_ids.add(str(row["positive_row_id"]))
            if row["negative_semantic_class"] != POSITIVE_SEMANTIC_CLASS:
                raise NegativeContractError(f"Positive row has synthetic semantics: {row_id}")
            if str(row["candidate_partner_id"]):
                raise NegativeContractError(f"Positive row has a candidate: {row_id}")
        elif label == "0":
            negative_ids.add(str(row["positive_row_id"]))
            if row["negative_semantic_class"] != NEGATIVE_SEMANTIC_CLASS:
                raise NegativeContractError(f"Synthetic row has non-synthetic semantics: {row_id}")
            if not str(row["candidate_partner_id"]):
                raise NegativeContractError(f"Global negative lacks pair-right identity: {row_id}")
            if not str(row["sampler_version"]).startswith(f"{E003_SCHEMA_VERSION}:"):
                raise NegativeContractError(f"Unsupported E003 sampler schema: {row_id}")
        else:
            raise NegativeContractError(f"E003 panel label is not binary: {row_id}")
        for field in (
            "anchor_sequence_sha256",
            "true_partner_sequence_sha256",
            "candidate_universe_sha256",
            "exclusion_registry_sha256",
        ):
            value = str(row[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise NegativeContractError(f"Invalid SHA-256 in {field}: {row_id}")
        if label == "0":
            value = str(row["candidate_sequence_sha256"])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise NegativeContractError(f"Invalid candidate sequence hash: {row_id}")
        if (
            len(str(row["seq1"])) > MAX_RAW_SEQUENCE_LENGTH
            or len(str(row["seq2"])) > MAX_RAW_SEQUENCE_LENGTH
        ):
            raise NegativeContractError(f"Overlength sequence reached E003 panel: {row_id}")
        try:
            import json

            json.loads(str(row["match_diagnostics_json"]))
            json.loads(str(row["source_input_hashes_json"]))
        except (ValueError, TypeError) as error:
            raise NegativeContractError(f"Invalid E003 provenance JSON: {row_id}") from error
        if universe is not None and label == "0":
            left = universe.proteins.get(str(row["anchor_id"]))
            right = universe.proteins.get(str(row["candidate_partner_id"]))
            if left is None or right is None:
                raise NegativeContractError(f"E003 pair is not in the candidate universe: {row_id}")
            if not _is_legal_global_pair(
                left,
                right,
                positive_id_pairs=_positive_id_pairs(universe.positives),
                positive_sequence_pairs=_positive_sequence_pairs(universe.positives),
                universe=universe,
                allow_self=True,
            ):
                raise NegativeContractError(f"Known positive or excluded global pair: {row_id}")
    if environment is not None and environments != {environment}:
        raise NegativeContractError(f"E003 panel environment mismatch: {sorted(environments)}")
    if len(environments) != 1:
        raise NegativeContractError(f"E003 panel mixes environments: {sorted(environments)}")
    if len(universes) != 1:
        raise NegativeContractError("E003 panel mixes candidate universes")
    if not positive_ids or positive_ids != negative_ids:
        raise NegativeContractError("E003 panel is not one-to-one by positive identity")
    if len(rows) != 2 * len(positive_ids):
        raise NegativeContractError("E003 panel has duplicate positive identities")


def write_e003_panel(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    validate_e003_panel(rows)
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
        with output.open("w", encoding="utf-8", newline="") as handle:
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


def read_e003_panel(path: str | Path, *, environment: str | None = None) -> list[dict[str, str]]:
    input_path = Path(path)
    opener = gzip.open if input_path.name.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PANEL_COLUMNS:
            raise NegativeContractError(f"Unexpected E003 panel columns in {path}")
        rows = list(reader)
    validate_e003_panel(rows, environment=environment)
    return rows


def summarize_e003_panel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_e003_panel(rows)
    negatives = [row for row in rows if str(row["interaction_label"]) == "0"]
    candidates = Counter(str(row["candidate_partner_id"]) for row in negatives)
    return {
        "rows": len(rows),
        "positive_rows": len(
            {
                str(row["positive_row_id"])
                for row in rows
                if str(row["interaction_label"]) == "1"
            }
        ),
        "negative_rows": len(negatives),
        "unique_global_pairs": len(
            {
                (str(row["anchor_id"]), str(row["candidate_partner_id"]))
                for row in negatives
            }
        ),
        "maximum_candidate_endpoint_reuse": max(candidates.values(), default=0),
        "negative_environment": sorted({str(row["negative_environment"]) for row in rows}),
        "semantic_class": NEGATIVE_SEMANTIC_CLASS,
    }
