"""Common sequence-homology namespaces and conservative graph construction."""

from __future__ import annotations

import shlex
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ppi_shift.artifacts import iter_fasta, sha256_path, utc_now, write_csv, write_yaml


@dataclass(frozen=True)
class SimilarityEdge:
    query: str
    target: str
    fident: float
    alignment_length: int
    qcov: float
    tcov: float
    evalue: float
    bits: float


def parse_similarity_edges(
    path: str | Path,
    *,
    min_identity: float,
    min_coverage: float = 0.8,
) -> list[SimilarityEdge]:
    """Parse MMseqs custom output and retain declared threshold-passing edges."""

    edges: list[SimilarityEdge] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0].lower() in {"query", "query_hash"}:
                continue
            if len(fields) < 8:
                raise ValueError(f"Expected 8 MMseqs fields at {path}:{line_number}")
            try:
                edge = SimilarityEdge(
                    query=fields[0],
                    target=fields[1],
                    fident=float(fields[2]),
                    alignment_length=int(float(fields[3])),
                    qcov=float(fields[4]),
                    tcov=float(fields[5]),
                    evalue=float(fields[6]),
                    bits=float(fields[7]),
                )
            except ValueError as exc:
                raise ValueError(f"Invalid MMseqs edge at {path}:{line_number}") from exc
            if (
                edge.fident >= min_identity
                and edge.qcov >= min_coverage
                and edge.tcov >= min_coverage
            ):
                edges.append(edge)
    return edges


def deduplicate_undirected_edges(edges: list[SimilarityEdge]) -> list[SimilarityEdge]:
    """Remove self/reverse duplicates, retaining the strongest deterministic edge."""

    best: dict[tuple[str, str], SimilarityEdge] = {}
    for edge in edges:
        if edge.query == edge.target:
            continue
        key = tuple(sorted((edge.query, edge.target)))
        previous = best.get(key)
        rank = (edge.fident, edge.qcov, edge.tcov, edge.bits, -edge.alignment_length)
        if previous is None:
            best[key] = edge
            continue
        previous_rank = (
            previous.fident,
            previous.qcov,
            previous.tcov,
            previous.bits,
            -previous.alignment_length,
        )
        if rank > previous_rank or (
            rank == previous_rank and (edge.query, edge.target) < (previous.query, previous.target)
        ):
            best[key] = edge
    return [best[key] for key in sorted(best)]


def connected_components(
    sequence_ids: list[str],
    edges: list[SimilarityEdge],
) -> dict[str, str]:
    """Return deterministic component IDs while preserving singleton sequences."""

    parent = {identifier: identifier for identifier in sequence_ids}

    def find(identifier: str) -> str:
        root = identifier
        while parent[root] != root:
            root = parent[root]
        while parent[identifier] != identifier:
            next_identifier = parent[identifier]
            parent[identifier] = root
            identifier = next_identifier
        return root

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            return
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for edge in edges:
        union(edge.query, edge.target)
    members: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(parent):
        members[find(identifier)].append(identifier)
    assignments: dict[str, str] = {}
    for values in sorted(members.values(), key=lambda value: value[0]):
        component_id = values[0]
        for identifier in values:
            assignments[identifier] = component_id
    return assignments


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_homology_namespace(
    sequence_fasta: str | Path,
    *,
    namespace: str,
    min_identity: float,
    mmseqs_bin: str | Path,
    artifact_root: str | Path,
    repo_root: str | Path = ".",
    threads: int = 8,
    registry_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run the declared all-vs-all MMseqs workflow and materialize one namespace."""

    fasta_path = Path(sequence_fasta)
    if not fasta_path.is_file():
        raise FileNotFoundError(fasta_path)
    mmseqs = str(mmseqs_bin)
    version_result = subprocess.run([mmseqs, "version"], check=True, capture_output=True, text=True)
    version = version_result.stdout.strip() or version_result.stderr.strip()
    sequence_ids = [identifier for identifier, _ in iter_fasta(fasta_path)]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Sequence FASTA identifiers must be unique hashes")

    output_root = Path(artifact_root) / "homology" / namespace
    output_root.mkdir(parents=True, exist_ok=True)
    raw_edges = output_root / "mmseqs_edges.tsv"
    direct_edges = output_root / "direct_edges.tsv"
    components_path = output_root / "components.csv"
    input_hash = sha256_path(fasta_path)
    commands: list[str] = []
    started = time.monotonic()
    with TemporaryDirectory(
        prefix=f"ppi_shift_mmseqs_{namespace}_", dir=str(output_root)
    ) as temp_name:
        temp = Path(temp_name)
        sequence_db = temp / "sequence_db"
        result_db = temp / "result_db"
        work_dir = temp / "tmp"
        createdb = [mmseqs, "createdb", str(fasta_path), str(sequence_db)]
        search = [
            mmseqs,
            "search",
            str(sequence_db),
            str(sequence_db),
            str(result_db),
            str(work_dir),
            "--alignment-mode",
            "3",
            "--cov-mode",
            "0",
            "-c",
            "0.8",
            "--min-seq-id",
            str(min_identity),
            "--max-seqs",
            "1000000",
            "--threads",
            str(threads),
        ]
        convert = [
            mmseqs,
            "convertalis",
            str(sequence_db),
            str(sequence_db),
            str(result_db),
            str(raw_edges),
            "--format-mode",
            "4",
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        ]
        for command in (createdb, search, convert):
            commands.append(" ".join(shlex.quote(value) for value in command))
            _run(command)
    parsed = parse_similarity_edges(raw_edges, min_identity=min_identity, min_coverage=0.8)
    edges = deduplicate_undirected_edges(parsed)
    edge_rows = [
        {
            "sequence_a": min(edge.query, edge.target),
            "sequence_b": max(edge.query, edge.target),
            "fident": edge.fident,
            "alignment_length": edge.alignment_length,
            "qcov": edge.qcov,
            "tcov": edge.tcov,
            "evalue": edge.evalue,
            "bits": edge.bits,
        }
        for edge in edges
    ]
    edge_columns = (
        "sequence_a",
        "sequence_b",
        "fident",
        "alignment_length",
        "qcov",
        "tcov",
        "evalue",
        "bits",
    )
    write_csv(direct_edges, edge_columns, edge_rows, delimiter="\t")
    assignments = connected_components(sequence_ids, edges)
    component_sizes: dict[str, int] = defaultdict(int)
    for component_id in assignments.values():
        component_sizes[component_id] += 1
    component_rows = [
        {
            "sequence_sha256": identifier,
            "component_id": f"{namespace}:{component_id}",
            "component_size": component_sizes[component_id],
        }
        for identifier in sorted(assignments)
    ]
    write_csv(
        components_path, ("sequence_sha256", "component_id", "component_size"), component_rows
    )
    elapsed = time.monotonic() - started
    sizes = sorted(component_sizes.values(), reverse=True)
    manifest_path = Path(repo_root) / "data" / "manifests" / f"mmseqs_{namespace}_manifest.yaml"
    manifest = {
        "manifest_type": "ppi_shift.mmseqs_namespace",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "namespace": namespace,
        "identity_threshold": min_identity,
        "coverage_threshold": 0.8,
        "coverage_mode": 0,
        "alignment_mode": 3,
        "clustering_mode": "connected_components_of_threshold_passing_undirected_edges",
        "mmseqs_version": version,
        "registry_manifest": str(registry_manifest) if registry_manifest else "",
        "input": {"path": str(fasta_path), "sha256": input_hash, "sequences": len(sequence_ids)},
        "commands": commands,
        "outputs": {
            "raw_edges": {"path": str(raw_edges), "sha256": sha256_path(raw_edges)},
            "direct_edges": {
                "path": str(direct_edges),
                "sha256": sha256_path(direct_edges),
                "rows": len(edge_rows),
            },
            "components": {
                "path": str(components_path),
                "sha256": sha256_path(components_path),
                "rows": len(component_rows),
            },
        },
        "statistics": {
            "unique_sequences": len(sequence_ids),
            "passing_direct_edges": len(edge_rows),
            "connected_components": len(component_sizes),
            "singleton_components": sum(size == 1 for size in component_sizes.values()),
            "largest_component_sizes": sizes[:20],
            "missing_sequences": 0,
            "runtime_seconds": elapsed,
        },
    }
    write_yaml(manifest_path, manifest)
    return manifest


def namespace_summary_row(manifest: dict[str, Any]) -> dict[str, Any]:
    """Flatten a namespace manifest for the tracked summary CSV."""

    stats = manifest["statistics"]
    return {
        "namespace": manifest["namespace"],
        "identity_threshold": manifest["identity_threshold"],
        "coverage_threshold": manifest["coverage_threshold"],
        "coverage_mode": manifest["coverage_mode"],
        "alignment_mode": manifest["alignment_mode"],
        "mmseqs_version": manifest["mmseqs_version"],
        "unique_sequences": stats["unique_sequences"],
        "passing_direct_edges": stats["passing_direct_edges"],
        "connected_components": stats["connected_components"],
        "singleton_components": stats["singleton_components"],
        "largest_component_sizes": "|".join(
            str(value) for value in stats["largest_component_sizes"]
        ),
        "missing_sequences": stats["missing_sequences"],
        "runtime_seconds": stats["runtime_seconds"],
        "input_sha256": manifest["input"]["sha256"],
        "direct_edges_sha256": manifest["outputs"]["direct_edges"]["sha256"],
        "components_sha256": manifest["outputs"]["components"]["sha256"],
    }
