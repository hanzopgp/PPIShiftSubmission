"""Frozen E001 negative-environment contracts and deterministic samplers.

The samplers operate on positive rows plus a stage-specific positive exclusion registry.
They never inspect negative labels and never consult a future/OOD source while building a
training or validation candidate universe. ``synthetic_unobserved_pair`` therefore means
"not present in the legal frozen registry", not experimentally verified non-interaction.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import iter_delimited, sha256_path, write_csv
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key

SCHEMA_VERSION = "ppi_shift.e001_negative_environment.v1"
SCHEMA_VERSION_V2 = "ppi_shift.e001_negative_environment.v2"
SCHEMA_VERSION_V3 = "ppi_shift.e001_negative_environment.v3"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V2, SCHEMA_VERSION_V3)
ANCHOR_ORIENTATION_RULE = "sha256_parity:ppi_shift_anchor_v1"
NEGATIVE_SEMANTIC_CLASS = "synthetic_unobserved_pair"
POSITIVE_SEMANTIC_CLASS = "curated_positive_reference"
SAMPLER_NAMES = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
)
NEGATIVE_SEMANTICS = (
    NEGATIVE_SEMANTIC_CLASS,
    "explicit_assay_negative",
    "assay_non_detection",
    "curated_noninteraction",
    "apms_non_call",
    "database_absence_unknown",
)
PANEL_COLUMNS = (
    "row_id",
    "split",
    "dataset_name",
    "name1",
    "name2",
    "seq1",
    "seq2",
    "interaction_label",
    "positive_row_id",
    "positive_pair_key",
    "anchor_id",
    "true_partner_id",
    "candidate_partner_id",
    "anchor_sequence_sha256",
    "true_partner_sequence_sha256",
    "candidate_sequence_sha256",
    "negative_environment",
    "negative_semantic_class",
    "sampler_version",
    "generation_seed",
    "anchor_orientation_rule",
    "candidate_universe_id",
    "candidate_universe_sha256",
    "exclusion_registry_name",
    "exclusion_registry_sha256",
    "source_input_hashes_json",
    "eligible_candidate_count",
    "candidate_rank",
    "match_distance",
    "match_diagnostics_json",
    "fallback_status",
    "rejection_reason",
)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class NegativeContractError(ValueError):
    """Base error for a violated frozen negative-environment contract."""


class CandidateUniverseError(NegativeContractError):
    """Raised when a legal common candidate universe cannot be built."""


class SamplerInsufficiencyError(NegativeContractError):
    """Raised when the requested sampler cannot produce one row per positive."""


class EmbeddingCompatibilityError(NegativeContractError):
    """Raised when a frozen single-protein embedding asset is incompatible."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _label_is_positive(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "positive"}


@dataclass(frozen=True)
class PositivePair:
    """One positive row with a stable positive-order identity."""

    split: str
    source_row_index: int
    dataset_name: str
    left_id: str
    right_id: str
    left_sequence: str
    right_sequence: str
    identity_schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.left_id or not self.right_id:
            raise NegativeContractError("Positive rows require two non-empty protein IDs")
        if not self.left_sequence or not self.right_sequence:
            raise NegativeContractError("Positive rows require two non-empty sequences")
        if self.left_id == self.right_id:
            raise NegativeContractError(f"Positive self-pair is prohibited: {self.left_id!r}")
        _validate_schema_version(self.identity_schema_version)

    @property
    def left_hash(self) -> str:
        return sequence_sha256(self.left_sequence)

    @property
    def right_hash(self) -> str:
        return sequence_sha256(self.right_sequence)

    @property
    def positive_pair_key(self) -> str:
        return "||".join(unordered_pair_key(self.left_hash, self.right_hash))

    @property
    def positive_row_id(self) -> str:
        payload = {
            "schema_version": self.identity_schema_version,
            "split": self.split,
            "source_row_index": self.source_row_index,
            "dataset_name": self.dataset_name,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_sequence_sha256": self.left_hash,
            "right_sequence_sha256": self.right_hash,
        }
        return f"positive:{contract_hash(payload)[:32]}"


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    sequence_sha256: str

    @classmethod
    def from_pair_endpoint(cls, protein_id: str, sequence: str) -> ProteinRecord:
        normalized = normalize_sequence(sequence)
        return cls(protein_id, normalized, sequence_sha256(normalized))


@dataclass(frozen=True)
class ExclusionRegistry:
    name: str
    path: Path
    sha256: str
    identifier_pairs: frozenset[tuple[str, str]]
    sequence_pairs: frozenset[tuple[str, str]]

    def excludes(self, left_id: str, right_id: str, left_hash: str, right_hash: str) -> bool:
        return (
            unordered_pair_key(left_id, right_id) in self.identifier_pairs
            or unordered_pair_key(left_hash, right_hash) in self.sequence_pairs
        )


def load_exclusion_registry(path: str | Path, *, name: str | None = None) -> ExclusionRegistry:
    """Load a registry and retain both identifier and exact-sequence namespaces."""

    identifier_pairs: set[tuple[str, str]] = set()
    sequence_pairs: set[tuple[str, str]] = set()
    for row in iter_delimited(path):
        identifier = row.get("identifier_pair_key", "").strip()
        sequence = row.get("sequence_pair_key", "").strip()
        if identifier and "||" in identifier:
            left, right = identifier.split("||", 1)
            identifier_pairs.add(unordered_pair_key(left, right))
        if sequence and "||" in sequence:
            left, right = sequence.split("||", 1)
            sequence_pairs.add(unordered_pair_key(left, right))
    registry_path = Path(path)
    return ExclusionRegistry(
        name=name or registry_path.stem,
        path=registry_path,
        sha256=sha256_path(registry_path),
        identifier_pairs=frozenset(identifier_pairs),
        sequence_pairs=frozenset(sequence_pairs),
    )


def load_positive_rows(
    path: str | Path,
    split: str,
    *,
    identity_schema_version: str = SCHEMA_VERSION,
) -> list[PositivePair]:
    """Read only positive rows; negative labels cannot influence sampler construction."""

    _validate_schema_version(identity_schema_version)
    positives: list[PositivePair] = []
    for _source_row_index, row in enumerate(iter_delimited(path)):
        if not _label_is_positive(row.get("interaction_label", "")):
            continue
        positives.append(
            PositivePair(
                split=split,
                # Count only positives. Reordering or relabeling rows outside the
                # positive backbone must not alter sampler identities or choices.
                source_row_index=len(positives),
                dataset_name=row.get("dataset_name", "").strip() or split,
                left_id=row.get("name1", "").strip(),
                right_id=row.get("name2", "").strip(),
                left_sequence=row.get("seq1", ""),
                right_sequence=row.get("seq2", ""),
                identity_schema_version=identity_schema_version,
            )
        )
    if not positives:
        raise NegativeContractError(f"No positive rows found in {path}")
    return positives


def anchor_orientation(positive: PositivePair) -> tuple[ProteinRecord, ProteinRecord]:
    """Return a label-independent, input-order-independent anchor/true-partner pair."""

    left = ProteinRecord.from_pair_endpoint(positive.left_id, positive.left_sequence)
    right = ProteinRecord.from_pair_endpoint(positive.right_id, positive.right_sequence)
    ordered = sorted((left, right), key=lambda item: (item.sequence_sha256, item.protein_id))
    digest = hashlib.sha256(
        (
            f"{ANCHOR_ORIENTATION_RULE}|{ordered[0].protein_id}|{ordered[0].sequence_sha256}|"
            f"{ordered[1].protein_id}|{ordered[1].sequence_sha256}"
        ).encode()
    ).digest()
    return (ordered[0], ordered[1]) if digest[0] % 2 == 0 else (ordered[1], ordered[0])


@dataclass
class CandidateUniverse:
    stage: str
    positives: tuple[PositivePair, ...]
    proteins: dict[str, ProteinRecord]
    candidate_pool: tuple[ProteinRecord, ...]
    candidate_index_by_id: dict[str, int]
    forbidden_by_positive: dict[str, frozenset[str]]
    candidate_counts_by_positive: dict[str, int]
    exclusion_registry: ExclusionRegistry
    source_input_hashes: dict[str, str]
    universe_id: str
    universe_sha256: str
    schema_version: str = SCHEMA_VERSION

    @property
    def protein_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.proteins))

    def candidate_records(self, positive: PositivePair) -> tuple[ProteinRecord, ...]:
        try:
            forbidden = self.forbidden_by_positive[positive.positive_row_id]
        except KeyError as error:
            raise CandidateUniverseError(
                f"Positive row is not part of the frozen {self.stage} universe: "
                f"{positive.positive_row_id}"
            ) from error
        return tuple(record for record in self.candidate_pool if record.protein_id not in forbidden)

    def is_eligible(self, positive: PositivePair, protein_id: str) -> bool:
        try:
            forbidden = self.forbidden_by_positive[positive.positive_row_id]
        except KeyError as error:
            raise CandidateUniverseError(
                f"Positive row is not part of the frozen {self.stage} universe: "
                f"{positive.positive_row_id}"
            ) from error
        return protein_id not in forbidden

    def candidate_at(self, positive: PositivePair, index: int) -> ProteinRecord:
        """Return the index-th legal candidate without scanning the pool."""

        if index < 0 or index >= self.candidate_count(positive):
            raise CandidateUniverseError(f"Candidate index is out of range: {index}")
        try:
            forbidden = self.forbidden_by_positive[positive.positive_row_id]
        except KeyError as error:
            raise CandidateUniverseError(
                f"Positive row is not part of the frozen {self.stage} universe: "
                f"{positive.positive_row_id}"
            ) from error
        forbidden_indices = sorted(
            self.candidate_index_by_id[protein_id]
            for protein_id in forbidden
            if protein_id in self.candidate_index_by_id
        )
        low = 0
        high = len(self.candidate_pool) - 1
        while low < high:
            middle = (low + high) // 2
            legal_through_middle = middle + 1 - bisect_right(forbidden_indices, middle)
            if legal_through_middle > index:
                high = middle
            else:
                low = middle + 1
        if low in forbidden_indices:
            raise CandidateUniverseError(
                "Indexed candidate selection landed on a forbidden protein"
            )
        return self.candidate_pool[low]

    def candidate_count(self, positive: PositivePair) -> int:
        try:
            self.forbidden_by_positive[positive.positive_row_id]
        except KeyError as error:
            raise CandidateUniverseError(
                f"Positive row is not part of the frozen {self.stage} universe: "
                f"{positive.positive_row_id}"
            ) from error
        return self.candidate_counts_by_positive[positive.positive_row_id]

    def manifest_payload(self) -> dict[str, Any]:
        counts = [self.candidate_count(positive) for positive in self.positives]
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "universe_id": self.universe_id,
            "universe_sha256": self.universe_sha256,
            "protein_count": len(self.proteins),
            "positive_rows": len(self.positives),
            "candidate_count_min": min(counts, default=0),
            "candidate_count_max": max(counts, default=0),
            "candidate_count_mean": sum(counts) / max(1, len(counts)),
            "exclusion_registry": {
                "name": self.exclusion_registry.name,
                "path": str(self.exclusion_registry.path),
                "sha256": self.exclusion_registry.sha256,
                "identifier_pairs": len(self.exclusion_registry.identifier_pairs),
                "sequence_pairs": len(self.exclusion_registry.sequence_pairs),
            },
            "source_input_hashes": dict(sorted(self.source_input_hashes.items())),
            "candidate_rule": (
                "all proteins in the fixed positive stage population, excluding the anchor, "
                "the observed partner, and any pair present in the stage-specific registry"
            ),
        }


def build_candidate_universe(
    positives: Sequence[PositivePair],
    exclusion_registry: ExclusionRegistry,
    *,
    stage: str,
    source_input_hashes: Mapping[str, str] | None = None,
    universe_id: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> CandidateUniverse:
    """Build one common eligible universe shared by all four samplers."""

    if not positives:
        raise CandidateUniverseError("Cannot build a candidate universe without positives")
    _validate_schema_version(schema_version)
    mismatched_identities = [
        positive.positive_row_id
        for positive in positives
        if positive.identity_schema_version != schema_version
    ]
    if mismatched_identities:
        raise CandidateUniverseError(
            "Positive identities and candidate-universe schema do not match: "
            f"schema={schema_version}, first={mismatched_identities[0]}"
        )
    proteins: dict[str, ProteinRecord] = {}
    for positive in positives:
        for record in (
            ProteinRecord.from_pair_endpoint(positive.left_id, positive.left_sequence),
            ProteinRecord.from_pair_endpoint(positive.right_id, positive.right_sequence),
        ):
            previous = proteins.get(record.protein_id)
            if previous is not None and previous.sequence_sha256 != record.sequence_sha256:
                raise CandidateUniverseError(
                    f"Protein ID maps to incompatible sequences: {record.protein_id}"
                )
            proteins[record.protein_id] = record

    candidate_pool = tuple(
        sorted(proteins.values(), key=lambda item: (item.sequence_sha256, item.protein_id))
    )
    identifier_neighbors: dict[str, set[str]] = defaultdict(set)
    sequence_neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in exclusion_registry.identifier_pairs:
        identifier_neighbors[left].add(right)
        identifier_neighbors[right].add(left)
    for left, right in exclusion_registry.sequence_pairs:
        sequence_neighbors[left].add(right)
        sequence_neighbors[right].add(left)
    ids_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in candidate_pool:
        ids_by_hash[record.sequence_sha256].add(record.protein_id)
    forbidden_by_positive: dict[str, frozenset[str]] = {}
    candidate_counts_by_positive: dict[str, int] = {}
    for positive in positives:
        anchor, true_partner = anchor_orientation(positive)
        forbidden = {
            anchor.protein_id,
            true_partner.protein_id,
            *identifier_neighbors.get(anchor.protein_id, set()),
        }
        for sequence_hash in sequence_neighbors.get(anchor.sequence_sha256, set()):
            forbidden.update(ids_by_hash.get(sequence_hash, set()))
        forbidden_by_positive[positive.positive_row_id] = frozenset(forbidden)
        candidate_counts_by_positive[positive.positive_row_id] = len(proteins) - sum(
            identifier in proteins for identifier in forbidden
        )

    input_hashes = dict(sorted((source_input_hashes or {}).items()))
    identity_payload = {
        "schema_version": schema_version,
        "stage": stage,
        "proteins": [
            {
                "protein_id": record.protein_id,
                "sequence_sha256": record.sequence_sha256,
                "length": len(record.sequence),
            }
            for record in candidate_pool
        ],
        "exclusion_registry_sha256": exclusion_registry.sha256,
        "source_input_hashes": input_hashes,
        "anchor_orientation_rule": ANCHOR_ORIENTATION_RULE,
    }
    universe_sha = contract_hash(identity_payload)
    resolved_id = universe_id or f"e001_{stage}_candidate_universe_{universe_sha[:16]}"
    return CandidateUniverse(
        stage=stage,
        positives=tuple(positives),
        proteins=proteins,
        candidate_pool=candidate_pool,
        candidate_index_by_id={
            record.protein_id: index for index, record in enumerate(candidate_pool)
        },
        forbidden_by_positive=forbidden_by_positive,
        candidate_counts_by_positive=candidate_counts_by_positive,
        exclusion_registry=exclusion_registry,
        source_input_hashes=input_hashes,
        universe_id=resolved_id,
        universe_sha256=universe_sha,
        schema_version=schema_version,
    )


def write_candidate_universe(path: str | Path, universe: CandidateUniverse) -> None:
    """Write a compact, reproducible candidate-universe manifest outside Git."""

    payload = universe.manifest_payload()
    payload["proteins"] = [
        {
            "protein_id": record.protein_id,
            "sequence_sha256": record.sequence_sha256,
            "length": len(record.sequence),
        }
        for record in sorted(universe.proteins.values(), key=lambda item: item.protein_id)
    ]
    payload["positive_candidate_counts"] = [
        {
            "positive_row_id": positive.positive_row_id,
            "eligible_candidate_count": universe.candidate_count(positive),
        }
        for positive in sorted(universe.positives, key=lambda item: item.positive_row_id)
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sequence_nuisance_features(sequence: str) -> np.ndarray:
    normalized = normalize_sequence(sequence)
    composition = np.asarray(
        [normalized.count(amino) / max(1, len(normalized)) for amino in AMINO_ACIDS],
        dtype=np.float64,
    )
    return np.concatenate(([math.log1p(len(normalized))], composition))


@dataclass(frozen=True)
class NuisanceScaler:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    source: str
    scaler_sha256: str

    def transform(self, sequence: str) -> np.ndarray:
        values = _sequence_nuisance_features(sequence)
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)


def build_nuisance_scaler(universe: CandidateUniverse) -> NuisanceScaler:
    values = np.asarray(
        [_sequence_nuisance_features(record.sequence) for record in universe.proteins.values()],
        dtype=np.float64,
    )
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    names = ("log_length",) + tuple(f"composition_{amino}" for amino in AMINO_ACIDS)
    payload = {
        "schema_version": universe.schema_version,
        "feature_names": names,
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "source": "candidate_universe_proteins_only",
        "candidate_universe_sha256": universe.universe_sha256,
    }
    return NuisanceScaler(
        feature_names=names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        source="candidate_universe_proteins_only",
        scaler_sha256=contract_hash(payload),
    )


def build_degree_table(positives: Sequence[PositivePair]) -> dict[str, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for positive in positives:
        adjacency[positive.left_id].add(positive.right_id)
        adjacency[positive.right_id].add(positive.left_id)
    return {protein_id: len(neighbors) for protein_id, neighbors in adjacency.items()}


@dataclass(frozen=True)
class EmbeddingStore:
    vectors: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    source_paths: Mapping[str, str]
    source_hashes: Mapping[str, str]

    @classmethod
    def from_files(
        cls,
        embeddings_path: str | Path,
        records_path: str | Path,
        summary_path: str | Path,
    ) -> EmbeddingStore:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        vectors = np.load(embeddings_path, mmap_mode="r")
        records = list(iter_delimited(records_path))
        if vectors.ndim != 2 or vectors.shape[0] != len(records):
            raise EmbeddingCompatibilityError(
                f"Embedding rows do not align with records: shape={vectors.shape}, "
                f"records={len(records)}"
            )
        expected_dimension = summary.get("embedding_dimension")
        if expected_dimension is not None and int(expected_dimension) != vectors.shape[1]:
            raise EmbeddingCompatibilityError("Embedding dimension disagrees with summary")
        model_name = str(summary.get("model_name", ""))
        revision = str(summary.get("model_revision", ""))
        layer = summary.get("embedding_layer")
        pooling = str(summary.get("pooling", ""))
        if not model_name or not revision or layer is None or not pooling:
            raise EmbeddingCompatibilityError(
                "Embedding summary lacks frozen model contract fields"
            )
        mapping: dict[str, np.ndarray] = {}
        for index, row in enumerate(records):
            digest = row.get("sequence_sha256", "").strip()
            if not _sha256(digest):
                raise EmbeddingCompatibilityError(f"Invalid embedding sequence hash at row {index}")
            previous = mapping.get(digest)
            if previous is not None and not np.array_equal(previous, vectors[index]):
                raise EmbeddingCompatibilityError(f"Conflicting vectors for sequence {digest}")
            mapping[digest] = vectors[index]
        return cls(
            vectors=mapping,
            metadata={
                "model_name": model_name,
                "model_revision": revision,
                "embedding_layer": int(layer),
                "pooling": pooling,
                "embedding_dimension": int(vectors.shape[1]),
                "dtype": str(vectors.dtype),
                "embedding_contract_sha256": summary.get("embedding_contract_sha256", ""),
                "summary_schema_version": summary.get("schema_version", ""),
            },
            source_paths={
                "embeddings": str(embeddings_path),
                "records": str(records_path),
                "summary": str(summary_path),
            },
            source_hashes={
                "embeddings": sha256_path(embeddings_path),
                "records": sha256_path(records_path),
                "summary": sha256_path(summary_path),
            },
        )

    def missing(self, sequence_hashes: Iterable[str]) -> list[str]:
        return sorted({value for value in sequence_hashes if value not in self.vectors})


def _validate_schema_version(schema_version: str) -> None:
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise NegativeContractError(f"Unsupported negative-environment schema: {schema_version!r}")


def _hash_rank(
    seed: int,
    sampler: str,
    positive_id: str,
    candidate_id: str,
    schema_version: str = SCHEMA_VERSION,
) -> int:
    _validate_schema_version(schema_version)
    value = hashlib.sha256(
        f"{schema_version}|{seed}|{sampler}|{positive_id}|{candidate_id}".encode()
    ).digest()
    return int.from_bytes(value, byteorder="big", signed=False)


def _ranked_by_hash(
    candidates: Sequence[ProteinRecord],
    seed: int,
    sampler: str,
    positive_id: str,
    schema_version: str = SCHEMA_VERSION,
) -> list[ProteinRecord]:
    return sorted(
        candidates,
        key=lambda record: _hash_rank(
            seed, sampler, positive_id, record.protein_id, schema_version
        ),
    )


def _precompute_distance_selections(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    sampler: str,
    *,
    seed: int,
    nuisance_scaler: NuisanceScaler,
    embeddings: EmbeddingStore | None,
) -> dict[str, tuple[ProteinRecord, int, float, dict[str, Any]]]:
    """Compute exact distance orders in batches and retain one compact choice per row.

    A target may occur in thousands of positive assertions. The distance calculation is
    therefore grouped by target sequence and is never repeated per assertion. Only the
    resulting candidate choice is retained, rather than a Python list of every candidate
    for every target.
    """

    pool = tuple(
        sorted(universe.proteins.values(), key=lambda item: (item.sequence_sha256, item.protein_id))
    )
    target_groups: dict[str, list[PositivePair]] = defaultdict(list)
    target_sequences: dict[str, str] = {}
    for positive in positives:
        _anchor, target = anchor_orientation(positive)
        target_groups[target.sequence_sha256].append(positive)
        target_sequences[target.sequence_sha256] = target.sequence
    if sampler == "sequence_nuisance_matched":
        candidate_matrix = np.asarray(
            [nuisance_scaler.transform(record.sequence) for record in pool], dtype=np.float64
        )
    elif sampler == "esm_partner_matched":
        if embeddings is None:
            raise EmbeddingCompatibilityError(
                "esm_partner_matched requires a frozen embedding store"
            )
        missing = embeddings.missing(
            [record.sequence_sha256 for record in pool] + list(target_sequences)
        )
        if missing:
            raise EmbeddingCompatibilityError(
                f"Embedding store lacks {len(missing)} candidate-universe hashes; "
                f"first={missing[0]}"
            )
        candidate_matrix = np.asarray(
            [embeddings.vectors[record.sequence_sha256] for record in pool], dtype=np.float32
        )
    else:
        raise NegativeContractError(f"Distance precomputation does not support {sampler!r}")

    result: dict[str, tuple[ProteinRecord, int, float, dict[str, Any]]] = {}
    target_hashes = sorted(target_groups)
    batch_size = 64 if sampler == "esm_partner_matched" else 512
    for start in range(0, len(target_hashes), batch_size):
        batch_hashes = target_hashes[start : start + batch_size]
        targets = np.asarray(
            [
                (
                    nuisance_scaler.transform(target_sequences[target_hash])
                    if sampler == "sequence_nuisance_matched"
                    else embeddings.vectors[target_hash]
                )
                for target_hash in batch_hashes
            ],
            dtype=candidate_matrix.dtype,
        )
        if sampler == "sequence_nuisance_matched":
            distances = np.linalg.norm(targets[:, None, :] - candidate_matrix[None, :, :], axis=2)
        else:
            candidate_norms = np.sum(candidate_matrix * candidate_matrix, axis=1)
            distances = (
                np.sum(targets * targets, axis=1)[:, None]
                + candidate_norms[None, :]
                - 2.0 * (targets @ candidate_matrix.T)
            )
            distances = np.sqrt(np.maximum(distances, 0.0))
        for target_index, target_hash in enumerate(batch_hashes):
            order = np.argsort(distances[target_index], kind="stable")
            for positive in target_groups[target_hash]:
                candidate = None
                rank = 0
                for candidate_rank, pool_index in enumerate(order, start=1):
                    candidate_record = pool[int(pool_index)]
                    if universe.is_eligible(positive, candidate_record.protein_id):
                        candidate = candidate_record
                        rank = candidate_rank
                        break
                if candidate is None:
                    raise SamplerInsufficiencyError(
                        f"No ranked legal candidate for {positive.positive_row_id}"
                    )
                diagnostics: dict[str, Any] = {
                    "rule": (
                        "standardized_log_length_amino_acid_composition_l2"
                        if sampler == "sequence_nuisance_matched"
                        else "frozen_single_protein_embedding_l2"
                    ),
                    "tie_break": "candidate_sequence_hash_then_id_order",
                }
                if sampler == "sequence_nuisance_matched":
                    diagnostics.update(
                        {
                            "feature_names": list(nuisance_scaler.feature_names),
                            "scaler_source": nuisance_scaler.source,
                            "scaler_sha256": nuisance_scaler.scaler_sha256,
                        }
                    )
                else:
                    diagnostics.update(
                        {
                            "model_name": embeddings.metadata["model_name"],
                            "model_revision": embeddings.metadata["model_revision"],
                            "embedding_layer": embeddings.metadata["embedding_layer"],
                            "pooling": embeddings.metadata["pooling"],
                            "embedding_dimension": embeddings.metadata["embedding_dimension"],
                            "embedding_contract_sha256": embeddings.metadata.get(
                                "embedding_contract_sha256", ""
                            ),
                        }
                    )
                result[positive.positive_row_id] = (
                    candidate,
                    rank,
                    float(distances[target_index, order[rank - 1]]),
                    diagnostics,
                )
    return result


def _first_eligible(
    ranked: Sequence[ProteinRecord], eligible_ids: set[str]
) -> tuple[ProteinRecord, int]:
    for index, record in enumerate(ranked, start=1):
        if record.protein_id in eligible_ids:
            return record, index
    raise SamplerInsufficiencyError("Ranked candidate cache contains no legal candidate")


def _select_indexed_legal_candidate(
    group: Sequence[ProteinRecord],
    group_index_by_id: Mapping[str, int],
    forbidden: frozenset[str],
    index: int,
) -> ProteinRecord:
    forbidden_indices = sorted(
        group_index_by_id[protein_id] for protein_id in forbidden if protein_id in group_index_by_id
    )
    low = 0
    high = len(group) - 1
    while low < high:
        middle = (low + high) // 2
        legal_through_middle = middle + 1 - bisect_right(forbidden_indices, middle)
        if legal_through_middle > index:
            high = middle
        else:
            low = middle + 1
    if low in forbidden_indices:
        raise CandidateUniverseError("Indexed degree selection landed on a forbidden protein")
    return group[low]


def _select_candidate(
    positive: PositivePair,
    universe: CandidateUniverse,
    sampler: str,
    seed: int,
    *,
    degree_table: Mapping[str, int],
    nuisance_scaler: NuisanceScaler,
    embeddings: EmbeddingStore | None,
    rank_cache: Mapping[str, tuple[ProteinRecord, ...]] | None = None,
    degree_groups: Mapping[int, tuple[ProteinRecord, ...]] | None = None,
    degree_group_indices: Mapping[int, Mapping[str, int]] | None = None,
    degree_order_cache: Mapping[int, tuple[int, ...]] | None = None,
    schema_version: str = SCHEMA_VERSION,
    max_distance: float | None = None,
) -> tuple[ProteinRecord, int, float | str, dict[str, Any], str]:
    _validate_schema_version(schema_version)
    candidate_count = universe.candidate_count(positive)
    if not candidate_count:
        raise SamplerInsufficiencyError(
            f"No eligible candidates for {positive.positive_row_id} in {universe.stage}"
        )
    _anchor, true_partner = anchor_orientation(positive)
    positive_id = positive.positive_row_id
    fallback_status = "none"
    if sampler == "uniform_partner":
        # Hash-to-index is a deterministic uniform draw over the same sorted legal
        # candidate set for every sampler. It avoids an O(n log n) sort per positive.
        # CandidateUniverse stores each eligible tuple in the global deterministic
        # sequence-hash/ID order, so no per-row sort is required here.
        index = (
            _hash_rank(seed, sampler, positive_id, "candidate-index", schema_version)
            % candidate_count
        )
        candidate = universe.candidate_at(positive, index)
        rank = index + 1
        distance: float | str = ""
        diagnostics: dict[str, Any] = {
            "rule": "uniform_hash_index",
            "hash_rank": _hash_rank(seed, sampler, positive_id, "candidate-index", schema_version),
        }
    elif sampler == "degree_partner_matched":
        target_degree = int(degree_table.get(true_partner.protein_id, 0))
        if schema_version == SCHEMA_VERSION_V2:
            if degree_groups is None:
                raise NegativeContractError(
                    "E001v2 degree selection requires indexed degree groups"
                )
            if degree_group_indices is None:
                raise NegativeContractError("E001v2 degree selection requires degree-group indices")
            ordered_degrees = (
                degree_order_cache.get(target_degree) if degree_order_cache is not None else None
            )
            if ordered_degrees is None:
                ordered_degrees = tuple(
                    sorted(
                        degree_groups,
                        key=lambda value: (abs(int(value) - target_degree), int(value)),
                    )
                )
            minimum_distance: int | None = None
            tied_groups: list[tuple[tuple[ProteinRecord, ...], Mapping[str, int], int]] = []
            forbidden = universe.forbidden_by_positive[positive_id]
            for candidate_degree in ordered_degrees:
                distance_value = abs(int(candidate_degree) - target_degree)
                if minimum_distance is not None and distance_value > minimum_distance:
                    break
                group = degree_groups[candidate_degree]
                group_index_by_id = degree_group_indices[candidate_degree]
                legal_count = len(group) - sum(
                    protein_id in group_index_by_id for protein_id in forbidden
                )
                if not legal_count:
                    continue
                if minimum_distance is None:
                    minimum_distance = distance_value
                if distance_value == minimum_distance:
                    tied_groups.append((group, group_index_by_id, legal_count))
                else:
                    break
            if not tied_groups:
                raise SamplerInsufficiencyError(
                    f"No legal degree candidate for {positive.positive_row_id}"
                )
            minimum_distance_candidate_count = sum(item[2] for item in tied_groups)
            tie_index = (
                _hash_rank(seed, sampler, positive_id, "candidate-index", schema_version)
                % minimum_distance_candidate_count
            )
            candidate = None
            for group, group_index_by_id, legal_count in tied_groups:
                if tie_index >= legal_count:
                    tie_index -= legal_count
                    continue
                candidate = _select_indexed_legal_candidate(
                    group, group_index_by_id, forbidden, tie_index
                )
                break
            if candidate is None:
                raise NegativeContractError("Indexed degree tie selection failed")
            rank = 1
            distance = float(minimum_distance)
            diagnostics = {
                "rule": "absolute_degree_difference",
                "degree_source": "provided_legal_positive_graph",
                "true_partner_degree": target_degree,
                "candidate_degree": int(degree_table.get(candidate.protein_id, 0)),
                "tie_break": "positive_row_id_specific_hash_to_index",
                "minimum_distance_candidate_count": minimum_distance_candidate_count,
                "sampler_schema_version": schema_version,
            }
        else:
            if rank_cache is None or str(target_degree) not in rank_cache:
                candidates = list(universe.candidate_records(positive))
                ranked = sorted(
                    candidates,
                    key=lambda record: (
                        abs(int(degree_table.get(record.protein_id, 0)) - target_degree),
                        _hash_rank(seed, sampler, positive_id, record.protein_id),
                    ),
                )
                rank = 1
                candidate = ranked[0]
            else:
                candidate = None
                rank = 0
                for rank, record in enumerate(rank_cache[str(target_degree)], start=1):
                    if universe.is_eligible(positive, record.protein_id):
                        candidate = record
                        break
                if candidate is None:
                    raise SamplerInsufficiencyError(
                        f"No legal degree candidate for {positive.positive_row_id}"
                    )
            distance = float(abs(int(degree_table.get(candidate.protein_id, 0)) - target_degree))
            diagnostics = {
                "rule": "absolute_degree_difference",
                "degree_source": "provided_legal_positive_graph",
                "true_partner_degree": target_degree,
                "candidate_degree": int(degree_table.get(candidate.protein_id, 0)),
            }
    elif sampler == "sequence_nuisance_matched":
        candidates = list(universe.candidate_records(positive))
        target = nuisance_scaler.transform(true_partner.sequence)
        eligible_ids = {record.protein_id for record in candidates}
        if rank_cache is not None and true_partner.sequence_sha256 in rank_cache:
            candidate, rank = _first_eligible(
                rank_cache[true_partner.sequence_sha256], eligible_ids
            )
            distance = float(np.linalg.norm(nuisance_scaler.transform(candidate.sequence) - target))
        else:
            scored = [
                (
                    float(np.linalg.norm(nuisance_scaler.transform(record.sequence) - target)),
                    _hash_rank(seed, sampler, positive_id, record.protein_id, schema_version),
                    record,
                )
                for record in candidates
            ]
            scored.sort(key=lambda item: (item[0], item[1]))
            distance, _tie, candidate = scored[0]
        rank = 1
        diagnostics = {
            "rule": "standardized_log_length_amino_acid_composition_l2",
            "feature_names": list(nuisance_scaler.feature_names),
            "scaler_source": nuisance_scaler.source,
            "scaler_sha256": nuisance_scaler.scaler_sha256,
        }
    elif sampler == "esm_partner_matched":
        if embeddings is None:
            raise EmbeddingCompatibilityError(
                "esm_partner_matched requires a frozen embedding store"
            )
        candidates = list(universe.candidate_records(positive))
        missing = embeddings.missing(
            [true_partner.sequence_sha256, *(record.sequence_sha256 for record in candidates)]
        )
        if missing:
            raise EmbeddingCompatibilityError(
                f"Embedding store lacks {len(missing)} eligible sequence hashes; first={missing[0]}"
            )
        target = np.asarray(embeddings.vectors[true_partner.sequence_sha256], dtype=np.float64)
        eligible_ids = {record.protein_id for record in candidates}
        if rank_cache is not None and true_partner.sequence_sha256 in rank_cache:
            candidate, rank = _first_eligible(
                rank_cache[true_partner.sequence_sha256], eligible_ids
            )
            distance = float(
                np.linalg.norm(
                    np.asarray(embeddings.vectors[candidate.sequence_sha256], dtype=np.float64)
                    - target
                )
            )
        else:
            scored = [
                (
                    float(
                        np.linalg.norm(
                            np.asarray(embeddings.vectors[record.sequence_sha256], dtype=np.float64)
                            - target
                        )
                    ),
                    _hash_rank(seed, sampler, positive_id, record.protein_id, schema_version),
                    record,
                )
                for record in candidates
            ]
            scored.sort(key=lambda item: (item[0], item[1]))
            distance, _tie, candidate = scored[0]
        rank = 1
        diagnostics = {
            "rule": "frozen_single_protein_embedding_l2",
            "model_name": embeddings.metadata["model_name"],
            "model_revision": embeddings.metadata["model_revision"],
            "embedding_layer": embeddings.metadata["embedding_layer"],
            "pooling": embeddings.metadata["pooling"],
            "embedding_dimension": embeddings.metadata["embedding_dimension"],
            "embedding_contract_sha256": embeddings.metadata.get("embedding_contract_sha256", ""),
        }
    else:
        raise NegativeContractError(f"Unknown sampler: {sampler!r}")

    if max_distance is not None and distance != "" and float(distance) > max_distance:
        # The closest legal candidate is retained, but the fallback is explicit in every
        # row and in the stage audit. There is no silent relaxation of the contract.
        fallback_status = "closest_eligible_candidate_after_tolerance_failure"
        diagnostics = {**diagnostics, "maximum_distance": max_distance}
    return candidate, rank, distance, diagnostics, fallback_status


def generate_negative_rows(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    sampler: str,
    *,
    seed: int,
    degree_table: Mapping[str, int] | None = None,
    nuisance_scaler: NuisanceScaler | None = None,
    embeddings: EmbeddingStore | None = None,
    schema_version: str = SCHEMA_VERSION,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Generate exactly one synthetic negative for every fixed positive row."""

    if sampler not in SAMPLER_NAMES:
        raise NegativeContractError(f"Unknown sampler: {sampler!r}")
    _validate_schema_version(schema_version)
    degree = degree_table or {}
    scaler = nuisance_scaler or build_nuisance_scaler(universe)
    rank_cache = None
    degree_groups = None
    degree_group_indices = None
    degree_order_cache = None
    distance_selections = None
    if sampler in {"sequence_nuisance_matched", "esm_partner_matched"}:
        distance_selections = _precompute_distance_selections(
            positives,
            universe,
            sampler,
            seed=seed,
            nuisance_scaler=scaler,
            embeddings=embeddings,
        )
    elif sampler == "degree_partner_matched":
        pool = tuple(
            sorted(
                universe.proteins.values(),
                key=lambda item: (item.sequence_sha256, item.protein_id),
            )
        )
        degrees = sorted({int(degree.get(record.protein_id, 0)) for record in pool})
        if schema_version == SCHEMA_VERSION:
            rank_cache = {
                str(target_degree): tuple(
                    sorted(
                        pool,
                        key=lambda record: (
                            abs(int(degree.get(record.protein_id, 0)) - target_degree),
                            _hash_rank(seed, sampler, "degree-target", record.protein_id),
                        ),
                    )
                )
                for target_degree in degrees
            }
        else:
            degree_groups = defaultdict(list)
            for record in pool:
                degree_groups[int(degree.get(record.protein_id, 0))].append(record)
            degree_groups = {
                degree_value: tuple(records) for degree_value, records in degree_groups.items()
            }
            degree_group_indices = {
                degree_value: {record.protein_id: index for index, record in enumerate(records)}
                for degree_value, records in degree_groups.items()
            }
            degree_order_cache = {
                target_degree: tuple(
                    sorted(
                        degree_groups,
                        key=lambda value: (abs(int(value) - target_degree), int(value)),
                    )
                )
                for target_degree in sorted(
                    {
                        int(degree.get(anchor_orientation(positive)[1].protein_id, 0))
                        for positive in positives
                    }
                )
            }
    rows: list[dict[str, Any]] = []
    for positive in positives:
        anchor, true_partner = anchor_orientation(positive)
        if distance_selections is not None:
            candidate, rank, distance, diagnostics = distance_selections[positive.positive_row_id]
            fallback_status = "none"
            if max_distance is not None and float(distance) > max_distance:
                fallback_status = "closest_eligible_candidate_after_tolerance_failure"
                diagnostics = {**diagnostics, "maximum_distance": max_distance}
        else:
            candidate, rank, distance, diagnostics, fallback_status = _select_candidate(
                positive,
                universe,
                sampler,
                seed,
                degree_table=degree,
                nuisance_scaler=scaler,
                embeddings=embeddings,
                rank_cache=rank_cache,
                degree_groups=degree_groups,
                degree_group_indices=degree_group_indices,
                degree_order_cache=degree_order_cache,
                schema_version=schema_version,
                max_distance=max_distance,
            )
        row_id = contract_hash(
            {
                "schema_version": schema_version,
                "sampler": sampler,
                "seed": seed,
                "positive_row_id": positive.positive_row_id,
                "candidate_id": candidate.protein_id,
            }
        )
        rows.append(
            {
                "row_id": f"negative:{row_id[:32]}",
                "split": positive.split,
                "dataset_name": f"{positive.dataset_name}__{sampler}",
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
                "negative_environment": sampler,
                "negative_semantic_class": NEGATIVE_SEMANTIC_CLASS,
                "sampler_version": f"{schema_version}:{sampler}",
                "generation_seed": seed,
                "anchor_orientation_rule": ANCHOR_ORIENTATION_RULE,
                "candidate_universe_id": universe.universe_id,
                "candidate_universe_sha256": universe.universe_sha256,
                "exclusion_registry_name": universe.exclusion_registry.name,
                "exclusion_registry_sha256": universe.exclusion_registry.sha256,
                "source_input_hashes_json": canonical_json(universe.source_input_hashes),
                "eligible_candidate_count": universe.candidate_count(positive),
                "candidate_rank": rank,
                "match_distance": distance,
                "match_diagnostics_json": canonical_json(diagnostics),
                "fallback_status": fallback_status,
                "rejection_reason": "",
            }
        )
    validate_negative_rows(rows, universe=universe)
    return rows


def positive_panel_rows(
    positives: Sequence[PositivePair],
    universe: CandidateUniverse,
    environment: str,
    *,
    seed: int,
    schema_version: str = SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Represent the fixed positive backbone under one environment contract."""

    _validate_schema_version(schema_version)
    rows: list[dict[str, Any]] = []
    for positive in positives:
        anchor, true_partner = anchor_orientation(positive)
        rows.append(
            {
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
                "sampler_version": schema_version,
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
        )
    return rows


def validate_negative_rows(
    rows: Sequence[Mapping[str, Any]], *, universe: CandidateUniverse | None = None
) -> None:
    seen: set[str] = set()
    positive_ids: set[str] = set()
    for row in rows:
        missing = [field for field in PANEL_COLUMNS if field not in row]
        if missing:
            raise NegativeContractError(f"Negative row lacks fields: {missing}")
        row_id = str(row["row_id"])
        if row_id in seen:
            raise NegativeContractError(f"Duplicate negative row ID: {row_id}")
        seen.add(row_id)
        if str(row["interaction_label"]) != "0":
            raise NegativeContractError(f"Negative row has nonzero label: {row_id}")
        if row["negative_semantic_class"] != NEGATIVE_SEMANTIC_CLASS:
            raise NegativeContractError(f"Unsupported synthetic semantic class: {row_id}")
        if row["negative_environment"] not in SAMPLER_NAMES:
            raise NegativeContractError(
                f"Unknown negative environment: {row['negative_environment']}"
            )
        sampler_version = str(row["sampler_version"])
        expected_prefixes = tuple(
            f"{schema_version}:{row['negative_environment']}"
            for schema_version in SUPPORTED_SCHEMA_VERSIONS
        )
        if not sampler_version.startswith(expected_prefixes):
            raise NegativeContractError(f"Unsupported sampler schema: {row_id}")
        if not str(row["positive_row_id"]).strip():
            raise NegativeContractError(f"Negative row lacks positive identity: {row_id}")
        positive_ids.add(str(row["positive_row_id"]))
        if row["anchor_id"] == row["candidate_partner_id"]:
            raise NegativeContractError(f"Self-pair negative: {row_id}")
        if row["true_partner_id"] == row["candidate_partner_id"]:
            raise NegativeContractError(f"Observed positive partner reused as negative: {row_id}")
        for field in (
            "anchor_sequence_sha256",
            "true_partner_sequence_sha256",
            "candidate_sequence_sha256",
            "candidate_universe_sha256",
            "exclusion_registry_sha256",
        ):
            if not _sha256(str(row[field])):
                raise NegativeContractError(f"Invalid SHA-256 in {field}: {row_id}")
        if universe is not None:
            if row["candidate_universe_id"] != universe.universe_id:
                raise NegativeContractError(
                    f"Negative row uses a different candidate universe: {row_id}"
                )
            if row["candidate_universe_sha256"] != universe.universe_sha256:
                raise NegativeContractError(f"Negative row universe hash mismatch: {row_id}")
            if universe.exclusion_registry.excludes(
                str(row["anchor_id"]),
                str(row["candidate_partner_id"]),
                str(row["anchor_sequence_sha256"]),
                str(row["candidate_sequence_sha256"]),
            ):
                raise NegativeContractError(f"Known-positive pair was selected: {row_id}")
        try:
            json.loads(str(row["match_diagnostics_json"]))
            json.loads(str(row["source_input_hashes_json"]))
        except json.JSONDecodeError as error:
            raise NegativeContractError(f"Non-JSON provenance field: {row_id}") from error
    if len(positive_ids) != len(rows):
        raise NegativeContractError(
            "Each positive row may receive at most one negative per sampler"
        )


def validate_panel_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate one balanced panel and its cross-row invariant."""

    seen: set[str] = set()
    positive_ids: set[str] = set()
    negative_ids: set[str] = set()
    environments: set[str] = set()
    universes: set[str] = set()
    sampler_versions: set[str] = set()
    for row in rows:
        missing = [field for field in PANEL_COLUMNS if field not in row]
        if missing:
            raise NegativeContractError(f"Panel row lacks fields: {missing}")
        row_id = str(row["row_id"])
        if row_id in seen:
            raise NegativeContractError(f"Duplicate panel row ID: {row_id}")
        seen.add(row_id)
        environments.add(str(row["negative_environment"]))
        universes.add(str(row["candidate_universe_sha256"]))
        sampler_versions.add(str(row["sampler_version"]).split(":", 1)[0])
        label = str(row["interaction_label"])
        if label == "1":
            positive_ids.add(str(row["positive_row_id"]))
            if row["negative_semantic_class"] != POSITIVE_SEMANTIC_CLASS:
                raise NegativeContractError(f"Positive row has synthetic semantics: {row_id}")
        elif label == "0":
            negative_ids.add(str(row["positive_row_id"]))
        else:
            raise NegativeContractError(f"Panel label is not binary: {row_id}")
    if len(environments) != 1:
        raise NegativeContractError(f"Panel mixes environments: {sorted(environments)}")
    if len(sampler_versions) != 1 or next(iter(sampler_versions)) not in SUPPORTED_SCHEMA_VERSIONS:
        raise NegativeContractError(
            f"Panel mixes or uses unsupported sampler schemas: {sorted(sampler_versions)}"
        )
    if len(universes) != 1:
        raise NegativeContractError("Panel mixes candidate universes")
    if positive_ids != negative_ids:
        raise NegativeContractError(
            f"Panel is not 1:1 by positive row: positives={len(positive_ids)}, "
            f"negatives={len(negative_ids)}"
        )
    if len(rows) != 2 * len(positive_ids):
        raise NegativeContractError("Panel contains duplicate positive or negative identities")


def write_panel_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic compressed or plain panel after contract validation."""

    validate_panel_rows(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name.endswith(".gz"):
        with output.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=PANEL_COLUMNS,
                        extrasaction="ignore",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(
                        {
                            field: "" if row.get(field) is None else row.get(field, "")
                            for field in PANEL_COLUMNS
                        }
                        for row in rows
                    )
    else:
        write_csv(output, PANEL_COLUMNS, rows)


def read_panel_csv(path: str | Path) -> list[dict[str, str]]:
    path_value = Path(path)
    opener = gzip.open if path_value.name.endswith(".gz") else open
    with opener(path_value, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PANEL_COLUMNS:
            raise NegativeContractError(f"Unexpected panel schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    validate_panel_rows(rows)
    return rows


def summarize_sampler_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a compact audit summary for a materialized negative environment."""

    if not rows:
        raise NegativeContractError("Cannot summarize an empty sampler")
    candidates = Counter(str(row["candidate_partner_id"]) for row in rows)
    fallbacks = Counter(str(row["fallback_status"]) for row in rows)
    distances = [float(row["match_distance"]) for row in rows if str(row["match_distance"]) != ""]
    return {
        "rows": len(rows),
        "positive_rows_target": len({str(row["positive_row_id"]) for row in rows}),
        "unique_candidates": len(candidates),
        "maximum_candidate_reuse": max(candidates.values()),
        "mean_candidate_reuse": float(np.mean(list(candidates.values()))),
        "fallback_status_counts": dict(sorted(fallbacks.items())),
        "match_distance": {
            "count": len(distances),
            "min": min(distances) if distances else None,
            "median": float(np.median(distances)) if distances else None,
            "max": max(distances) if distances else None,
        },
        "known_positive_collision_rows": 0,
        "semantic_class": NEGATIVE_SEMANTIC_CLASS,
    }
