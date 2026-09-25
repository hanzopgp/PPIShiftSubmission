"""Audit the official DataSAIL Gold-v4 split and make an independent attempt."""

# Report prose is intentionally kept readable in the generated Markdown.
# ruff: noqa: E501

from __future__ import annotations

import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import (
    iter_delimited,
    sha256_path,
    utc_now,
    write_csv,
    write_fasta,
    write_yaml,
)
from ppi_shift.data import normalize_sequence, sequence_sha256, unordered_pair_key


def _binary_label(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "positive", "pos"}:
        return "1"
    if normalized in {"0", "false", "negative", "neg"}:
        return "0"
    return normalized


def _gold_rows(paths: dict[str, str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_split, path in paths.items():
        for row in iter_delimited(path):
            value = dict(row)
            value["gold_source_split"] = source_split
            value["gold_label"] = _binary_label(row.get("interaction_label", ""))
            value["left_sequence_sha256"] = (
                sequence_sha256(normalize_sequence(row.get("seq1", "")))
                if row.get("seq1", "").strip()
                else ""
            )
            value["right_sequence_sha256"] = (
                sequence_sha256(normalize_sequence(row.get("seq2", "")))
                if row.get("seq2", "").strip()
                else ""
            )
            rows.append(value)
    return rows


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    return unordered_pair_key(row.get("0", "").strip(), row.get("1", "").strip())


def _overlap_stats(
    rows_by_partition: dict[str, list[dict[str, str]]], components: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    partitions = [name for name in ("train", "val", "test") if name in rows_by_partition]
    output: list[dict[str, Any]] = []
    for index, left_name in enumerate(partitions):
        for right_name in partitions[index + 1 :]:
            left_rows, right_rows = rows_by_partition[left_name], rows_by_partition[right_name]
            left_pairs = {_pair_key(row) for row in left_rows}
            right_pairs = {_pair_key(row) for row in right_rows}
            left_proteins = {
                value
                for row in left_rows
                for value in (row.get("0", ""), row.get("1", ""))
                if value
            }
            right_proteins = {
                value
                for row in right_rows
                for value in (row.get("0", ""), row.get("1", ""))
                if value
            }
            left_sequences = {
                value
                for row in left_rows
                for value in (
                    row.get("left_sequence_sha256", ""),
                    row.get("right_sequence_sha256", ""),
                )
                if value
            }
            right_sequences = {
                value
                for row in right_rows
                for value in (
                    row.get("left_sequence_sha256", ""),
                    row.get("right_sequence_sha256", ""),
                )
                if value
            }
            left_sequence_pairs = {
                unordered_pair_key(
                    row.get("left_sequence_sha256", ""), row.get("right_sequence_sha256", "")
                )
                for row in left_rows
                if row.get("left_sequence_sha256") and row.get("right_sequence_sha256")
            }
            right_sequence_pairs = {
                unordered_pair_key(
                    row.get("left_sequence_sha256", ""), row.get("right_sequence_sha256", "")
                )
                for row in right_rows
                if row.get("left_sequence_sha256") and row.get("right_sequence_sha256")
            }

            def entity_sets(
                rows: list[dict[str, str]],
            ) -> tuple[set[str], set[str], dict[str, set[str]]]:
                proteins: set[str] = set()
                sequences: set[str] = set()
                protein_sequences: dict[str, set[str]] = defaultdict(set)
                for row in rows:
                    for protein, sequence in (
                        (row.get("0", ""), row.get("left_sequence_sha256", "")),
                        (row.get("1", ""), row.get("right_sequence_sha256", "")),
                    ):
                        if protein:
                            proteins.add(protein)
                        if sequence:
                            sequences.add(sequence)
                        if protein and sequence:
                            protein_sequences[protein].add(sequence)
                return proteins, sequences, protein_sequences

            left_proteins, left_sequences, left_protein_sequences = entity_sets(left_rows)
            right_proteins, right_sequences, right_protein_sequences = entity_sets(right_rows)
            result: dict[str, Any] = {
                "left_partition": left_name,
                "right_partition": right_name,
                "pair_overlap": len(left_pairs & right_pairs),
                "protein_overlap": len(left_proteins & right_proteins),
                "exact_sequence_overlap": len(left_sequences & right_sequences),
                "exact_sequence_pair_overlap": len(left_sequence_pairs & right_sequence_pairs),
            }
            for namespace, mapping in components.items():
                left_component_by_protein = {
                    protein: {mapping[sequence] for sequence in sequences if sequence in mapping}
                    for protein, sequences in left_protein_sequences.items()
                }
                right_component_by_protein = {
                    protein: {mapping[sequence] for sequence in sequences if sequence in mapping}
                    for protein, sequences in right_protein_sequences.items()
                }
                left_components = {
                    component
                    for values in left_component_by_protein.values()
                    for component in values
                }
                right_components = {
                    component
                    for values in right_component_by_protein.values()
                    for component in values
                }
                shared_components = left_components & right_components
                left_shared_proteins = {
                    protein
                    for protein, values in left_component_by_protein.items()
                    if values & shared_components
                }
                right_shared_proteins = {
                    protein
                    for protein, values in right_component_by_protein.items()
                    if values & shared_components
                }
                left_shared_sequences = {
                    sequence
                    for sequence in left_sequences
                    if mapping.get(sequence) in shared_components
                }
                right_shared_sequences = {
                    sequence
                    for sequence in right_sequences
                    if mapping.get(sequence) in shared_components
                }
                result.update(
                    {
                        f"{namespace}_component_count_left": len(left_components),
                        f"{namespace}_component_count_right": len(right_components),
                        f"{namespace}_shared_component_count": len(shared_components),
                        f"{namespace}_shared_component_protein_count_left": len(
                            left_shared_proteins
                        ),
                        f"{namespace}_shared_component_protein_count_right": len(
                            right_shared_proteins
                        ),
                        f"{namespace}_shared_component_protein_fraction_left": (
                            len(left_shared_proteins) / len(left_proteins) if left_proteins else 0.0
                        ),
                        f"{namespace}_shared_component_protein_fraction_right": (
                            len(right_shared_proteins) / len(right_proteins)
                            if right_proteins
                            else 0.0
                        ),
                        f"{namespace}_shared_component_sequence_count_left": len(
                            left_shared_sequences
                        ),
                        f"{namespace}_shared_component_sequence_count_right": len(
                            right_shared_sequences
                        ),
                        f"{namespace}_shared_component_sequence_fraction_left": (
                            len(left_shared_sequences) / len(left_sequences)
                            if left_sequences
                            else 0.0
                        ),
                        f"{namespace}_shared_component_sequence_fraction_right": (
                            len(right_shared_sequences) / len(right_sequences)
                            if right_sequences
                            else 0.0
                        ),
                    }
                )
                left_families = {
                    unordered_pair_key(
                        mapping[row["left_sequence_sha256"]], mapping[row["right_sequence_sha256"]]
                    )
                    for row in left_rows
                    if row.get("left_sequence_sha256") in mapping
                    and row.get("right_sequence_sha256") in mapping
                }
                right_families = {
                    unordered_pair_key(
                        mapping[row["left_sequence_sha256"]], mapping[row["right_sequence_sha256"]]
                    )
                    for row in right_rows
                    if row.get("left_sequence_sha256") in mapping
                    and row.get("right_sequence_sha256") in mapping
                }
                result[f"{namespace}_family_pair_overlap"] = len(left_families & right_families)
            output.append(result)
    return output


def map_official_split(
    official_path: str | Path,
    gold_paths: dict[str, str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Map official ID/label rows to materialized Gold rows without guessing."""

    gold_index: dict[tuple[tuple[str, str], str], list[dict[str, str]]] = defaultdict(list)
    gold_rows = _gold_rows(gold_paths)
    for row in gold_rows:
        key = (unordered_pair_key(row.get("name1", ""), row.get("name2", "")), row["gold_label"])
        gold_index[key].append(row)
    output_rows: list[dict[str, str]] = []
    unmatched = 0
    ambiguous = 0
    official_rows = list(iter_delimited(official_path))
    official_pair_labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in official_rows:
        official_pair_labels[_pair_key(row)].add(_binary_label(row.get("label", "")))
    for index, row in enumerate(official_rows, start=1):
        key = (_pair_key(row), _binary_label(row.get("label", "")))
        matches = gold_index.get(key, [])
        if len(matches) != 1:
            if not matches:
                unmatched += 1
            else:
                ambiguous += 1
            continue
        gold = matches[0]
        output_rows.append(
            {
                "official_row_number": str(index),
                "left_id": row.get("0", ""),
                "right_id": row.get("1", ""),
                "label": _binary_label(row.get("label", "")),
                "original_split": row.get("split", ""),
                "datasail_assignment": row.get("datasail", ""),
                "gold_source_split": gold["gold_source_split"],
                "left_sequence_sha256": gold["left_sequence_sha256"],
                "right_sequence_sha256": gold["right_sequence_sha256"],
            }
        )
    output_rows.sort(key=lambda row: int(row["official_row_number"]))
    columns = (
        "official_row_number",
        "left_id",
        "right_id",
        "label",
        "original_split",
        "datasail_assignment",
        "gold_source_split",
        "left_sequence_sha256",
        "right_sequence_sha256",
    )
    write_csv(output_path, columns, output_rows)
    mapped_by_assignment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in output_rows:
        if row["datasail_assignment"] in {"train", "val", "test"}:
            mapped_by_assignment[row["datasail_assignment"]].append(
                {"0": row["left_id"], "1": row["right_id"], **row}
            )
    return {
        "official_rows": len(official_rows),
        "official_columns": list(official_rows[0].keys()) if official_rows else [],
        "mapped_rows": len(output_rows),
        "unmatched_rows": unmatched,
        "ambiguous_rows": ambiguous,
        "rows_by_assignment": {
            key: len(value) for key, value in sorted(mapped_by_assignment.items())
        },
        "official_label_counts": dict(
            Counter(_binary_label(row.get("label", "")) for row in official_rows)
        ),
        "official_original_split_counts": dict(
            Counter(row.get("split", "") for row in official_rows)
        ),
        "official_datasail_assignment_counts": dict(
            Counter(row.get("datasail", "") for row in official_rows)
        ),
        "official_unique_proteins": len(
            {
                value
                for row in official_rows
                for value in (row.get("0", ""), row.get("1", ""))
                if value
            }
        ),
        "official_duplicate_identifier_pair_rows": len(official_rows)
        - len({_pair_key(row) for row in official_rows}),
        "official_conflicting_pair_labels": sum(
            len(labels) > 1 for labels in official_pair_labels.values()
        ),
        "mapped_rows_by_assignment": dict(
            Counter(row["datasail_assignment"] for row in output_rows)
        ),
        "mapped_partitions": dict(mapped_by_assignment),
    }


def _make_reproduction_inputs(
    gold_paths: dict[str, str | Path],
    output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    gold_rows = _gold_rows(gold_paths)
    sequences: dict[str, str] = {}
    interactions: set[tuple[str, str]] = set()
    weights: Counter[str] = Counter()
    for row in gold_rows:
        left, right = row.get("name1", ""), row.get("name2", "")
        interactions.add((left, right))
        weights.update((left, right))
        sequences.setdefault(left, normalize_sequence(row.get("seq1", "")))
        sequences.setdefault(right, normalize_sequence(row.get("seq2", "")))
    fasta = output_root / "gold_proteins.fasta"
    write_fasta(
        fasta,
        sorted((identifier, sequence) for identifier, sequence in sequences.items() if sequence),
    )
    interaction_path = output_root / "gold_interactions.tsv"
    write_csv(
        interaction_path,
        ("left", "right"),
        ({"left": left, "right": right} for left, right in sorted(interactions)),
        delimiter="\t",
    )
    weights_path = output_root / "gold_weights.tsv"
    write_csv(
        weights_path,
        ("protein", "weight"),
        ({"protein": key, "weight": weights[key]} for key in sorted(weights)),
        delimiter="\t",
    )
    strat_path = output_root / "gold_stratification.tsv"
    # DataSAIL consumes the first two columns.  This is a deterministic proxy for
    # the balanced interaction stratification described by the addendum; it is not
    # treated as a row-for-row reproduction claim.
    write_csv(
        strat_path,
        ("protein", "stratum"),
        ({"protein": key, "stratum": "all"} for key in sorted(weights)),
        delimiter="\t",
    )
    return interaction_path, fasta, weights_path, strat_path


def _datasail_invocation(datasail_bin: str | Path | None) -> list[str] | None:
    """Resolve either a DataSAIL executable or its Python-module launcher."""

    if datasail_bin:
        candidate = Path(datasail_bin)
        if candidate.is_file():
            if candidate.name.lower().startswith("python"):
                return [str(candidate), "-m", "datasail"]
            return [str(candidate)]
        # DataSAIL 1.2.1 may be installed without a console-script shim.  A
        # configured sibling interpreter is still an explicit, pinned launcher.
        sibling_python = candidate.parent / "python"
        if sibling_python.is_file():
            return [str(sibling_python), "-m", "datasail"]
    executable = shutil.which("datasail")
    if executable:
        return [executable]
    python = shutil.which("python")
    if python:
        return [python, "-m", "datasail"]
    return None


def _parse_reproduction_output(
    output_dir: Path,
    gold_paths: dict[str, str | Path],
    components: dict[str, dict[str, str]],
    official_partitions: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any] | None:
    """Parse and audit DataSAIL's interaction assignment file when present."""

    candidates = [output_dir / "C1e" / "inter.tsv"]
    candidates.extend(
        sorted(path for path in output_dir.rglob("inter.tsv") if path not in candidates)
    )
    assignment_path = next((path for path in candidates if path.is_file()), None)
    if assignment_path is None:
        return None

    gold_rows = _gold_rows(gold_paths)
    gold_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in gold_rows:
        key = unordered_pair_key(row.get("name1", ""), row.get("name2", ""))
        gold_by_pair.setdefault(
            key,
            {
                "left_sequence_sha256": row.get("left_sequence_sha256", ""),
                "right_sequence_sha256": row.get("right_sequence_sha256", ""),
            },
        )

    assignments: dict[tuple[str, str], str] = {}
    duplicate_rows = 0
    conflicting_assignments = 0
    parsed_rows = 0
    for row in iter_delimited(assignment_path):
        left = row.get("E_ID", row.get("left", "")).strip()
        right = row.get("F_ID", row.get("right", "")).strip()
        split = row.get("Split", row.get("split", "")).strip()
        if not left or not right:
            continue
        parsed_rows += 1
        key = unordered_pair_key(left, right)
        previous = assignments.get(key)
        if previous is not None:
            duplicate_rows += 1
            if previous != split:
                conflicting_assignments += 1
        else:
            assignments[key] = split

    partition_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for key, split in assignments.items():
        split_counts[split] += 1
        if split not in {"train", "val", "test"}:
            continue
        sequence_values = gold_by_pair.get(key, {})
        partition_rows[split].append(
            {
                "0": key[0],
                "1": key[1],
                "left_sequence_sha256": sequence_values.get("left_sequence_sha256", ""),
                "right_sequence_sha256": sequence_values.get("right_sequence_sha256", ""),
            }
        )

    expected_pairs = set(gold_by_pair)
    parsed_pairs = set(assignments)
    missing_pairs = expected_pairs - parsed_pairs
    overlaps = _overlap_stats(dict(partition_rows), components)
    comparison: dict[str, Any] = {}
    if official_partitions is not None:
        official_assignments: dict[tuple[str, str], str] = {}
        for partition, rows in official_partitions.items():
            for row in rows:
                key = _pair_key(row)
                if key not in official_assignments:
                    official_assignments[key] = partition
        common_pairs = set(official_assignments) & parsed_pairs
        same_assignment = sum(assignments[key] == official_assignments[key] for key in common_pairs)
        comparison = {
            "official_assigned_rows": len(official_assignments),
            "official_split_counts": {
                partition: len(rows) for partition, rows in sorted(official_partitions.items())
            },
            "reproduction_split_counts": dict(sorted(split_counts.items())),
            "reproduction_rows_for_official_pairs": len(common_pairs),
            "same_assignment_rows": same_assignment,
            "different_assignment_rows": len(common_pairs) - same_assignment,
            "missing_official_pairs": len(set(official_assignments) - parsed_pairs),
            "reproduction_assignment_agreement": (
                same_assignment / len(common_pairs) if common_pairs else None
            ),
        }
    complete = (
        parsed_rows > 0
        and not missing_pairs
        and duplicate_rows == 0
        and conflicting_assignments == 0
    )
    return {
        "assignment_path": str(assignment_path),
        "assignment_sha256": sha256_path(assignment_path),
        "parsed_rows": parsed_rows,
        "expected_unique_pairs": len(expected_pairs),
        "parsed_unique_pairs": len(parsed_pairs),
        "missing_expected_pairs": len(missing_pairs),
        "duplicate_pair_rows": duplicate_rows,
        "conflicting_pair_assignments": conflicting_assignments,
        "split_counts": dict(sorted(split_counts.items())),
        "overlap": overlaps,
        "comparison_to_official": comparison,
        "complete": complete,
    }


def _attempt_reproduction(
    gold_paths: dict[str, str | Path],
    *,
    artifact_root: Path,
    threads: int,
    datasail_bin: str | Path | None = None,
    mmseqs_bin: str | Path | None = None,
    components: dict[str, dict[str, str]] | None = None,
    official_partitions: dict[str, list[dict[str, str]]] | None = None,
    reuse_existing_output: bool = False,
    prior_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    invocation = _datasail_invocation(datasail_bin)
    result: dict[str, Any] = {
        "status": "blocked",
        "reason": "datasail executable not found",
        "requested_version": "1.2.1",
    }
    input_root = artifact_root / "datasail" / "reproduction_inputs"
    interaction_path, fasta_path, weights_path, strat_path = _make_reproduction_inputs(
        gold_paths, input_root
    )
    result["inputs"] = {
        "interactions": {"path": str(interaction_path), "sha256": sha256_path(interaction_path)},
        "fasta": {"path": str(fasta_path), "sha256": sha256_path(fasta_path)},
        "weights": {"path": str(weights_path), "sha256": sha256_path(weights_path)},
        "stratification": {"path": str(strat_path), "sha256": sha256_path(strat_path)},
    }
    output_dir = artifact_root / "datasail" / "reproduction_output"
    if reuse_existing_output:
        result["reused_existing_output"] = True
        result["execution_evidence"] = (
            "prior completed pinned DataSAIL CLI run; output was re-parsed for the updated audit"
        )
        if prior_result:
            for key in ("installed_version", "command", "returncode", "mmseqs_binary"):
                if key in prior_result:
                    result[key] = prior_result[key]
        parsed = _parse_reproduction_output(
            output_dir, gold_paths, components or {}, official_partitions
        )
        if parsed is None:
            result["status"] = "proxy_execution_completed_unparsed"
            result["reason"] = (
                "existing DataSAIL output was requested but no interaction assignment file was found"
            )
        elif parsed["complete"]:
            result["status"] = "proxy_execution_completed"
            result["reason"] = (
                "prior pinned DataSAIL proxy output was re-parsed and its complete interaction assignment was audited"
            )
            result["parsed_output"] = parsed
        else:
            result["status"] = "proxy_execution_partial"
            result["reason"] = (
                "existing DataSAIL proxy output was incomplete or internally inconsistent"
            )
            result["parsed_output"] = parsed
        result["output_files"] = [
            str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()
        ]
        return result
    if invocation is None:
        return result
    environment = os.environ.copy()
    if mmseqs_bin and Path(mmseqs_bin).is_file():
        environment["PATH"] = (
            str(Path(mmseqs_bin).parent) + os.pathsep + environment.get("PATH", "")
        )
        result["mmseqs_binary"] = str(mmseqs_bin)
    version = subprocess.run(
        [*invocation, "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    result["installed_version"] = (version.stdout or version.stderr).strip()
    if "1.2.1" not in result["installed_version"]:
        result["status"] = "blocked"
        result["reason"] = "installed DataSAIL version is not the pinned 1.2.1"
        return result
    output_dir.mkdir(parents=True, exist_ok=True)
    help_result = subprocess.run(
        [*invocation, "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    help_text = help_result.stdout + help_result.stderr
    technique_flag = "--techniques" if "--techniques" in help_text else "--technique"
    output_flag = "--output" if "--output" in help_text else "-o"
    command = [
        *invocation,
        output_flag,
        str(output_dir),
        technique_flag,
        "C1e",
        "--e-type",
        "P",
        "--e-data",
        str(fasta_path),
        "--e-sim",
        "mmseqs",
        "--e-weights",
        str(weights_path),
        "--e-strat",
        str(strat_path),
        "--inter",
        str(interaction_path),
        "--splits",
        "0.7",
        "0.15",
        "0.15",
        "--names",
        "train",
        "val",
        "test",
        "--runs",
        "1",
        "--threads",
        str(threads),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    result.update(
        {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    if completed.returncode:
        result["status"] = "failed"
        result["reason"] = "DataSAIL command failed"
    else:
        parsed = _parse_reproduction_output(
            output_dir, gold_paths, components or {}, official_partitions
        )
        if parsed is None:
            result["status"] = "proxy_execution_completed_unparsed"
            result["reason"] = (
                "DataSAIL proxy completed but no interaction assignment file was found"
            )
        elif parsed["complete"]:
            result["status"] = "proxy_execution_completed"
            result["reason"] = (
                "DataSAIL proxy completed and its complete interaction assignment was parsed and audited"
            )
            result["parsed_output"] = parsed
        else:
            result["status"] = "proxy_execution_partial"
            result["reason"] = (
                "DataSAIL proxy completed but its assignment was incomplete or internally inconsistent"
            )
            result["parsed_output"] = parsed
    result["output_files"] = [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()]
    return result


def audit_datasail_gold(
    official_path: str | Path,
    gold_paths: dict[str, str | Path],
    *,
    component_paths: dict[str, str | Path] | None,
    artifact_root: str | Path,
    repo_root: str | Path = ".",
    official_url: str = "https://zenodo.org/api/records/17376012/files/Gold_PPI_v4_2025_02.csv/content",
    gold_input_doi: str = "10.6084/m9.figshare.21591618.v4",
    concept_doi: str = "10.5281/zenodo.17376011",
    version_doi: str = "10.5281/zenodo.17376012",
    threads: int = 8,
    datasail_bin: str | Path | None = None,
    mmseqs_bin: str | Path | None = None,
    reuse_existing_output: bool = False,
) -> dict[str, Any]:
    """Write official split mapping, namespace overlap audit, and reproduction result."""

    root = Path(artifact_root)
    mapped_path = root / "datasail" / "gold_v4_addendum" / "mapped_official_gold.csv"
    mapped = map_official_split(official_path, gold_paths, output_path=mapped_path)
    components: dict[str, dict[str, str]] = {}
    if component_paths:
        for namespace, path in component_paths.items():
            components[namespace] = {
                row.get("sequence_sha256", ""): row.get("component_id", "")
                for row in iter_delimited(path)
                if row.get("sequence_sha256") and row.get("component_id")
            }
    partitions = mapped.pop("mapped_partitions")
    overlaps = _overlap_stats(partitions, components)
    prior_result: dict[str, Any] = {}
    if reuse_existing_output:
        import yaml

        prior_manifest_path = (
            Path(repo_root) / "data" / "manifests" / "datasail_reproduction_manifest.yaml"
        )
        if prior_manifest_path.is_file():
            prior_payload = yaml.safe_load(prior_manifest_path.read_text(encoding="utf-8"))
            if isinstance(prior_payload, dict) and isinstance(prior_payload.get("result"), dict):
                prior_result = prior_payload["result"]
    reproduction = _attempt_reproduction(
        gold_paths,
        artifact_root=root,
        threads=threads,
        datasail_bin=datasail_bin,
        mmseqs_bin=mmseqs_bin,
        components=components,
        official_partitions=partitions,
        reuse_existing_output=reuse_existing_output,
        prior_result=prior_result,
    )
    official_manifest_path = (
        Path(repo_root) / "data" / "manifests" / "datasail_official_manifest.yaml"
    )
    official_manifest = {
        "manifest_type": "ppi_shift.datasail_official_gold_split",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "concept_doi": concept_doi,
        "version_doi": version_doi,
        "gold_input_doi": gold_input_doi,
        "official_url": official_url,
        "official_download_timestamp_utc": datetime.fromtimestamp(
            Path(official_path).stat().st_mtime,
            timezone.utc,  # noqa: UP017
        )
        .replace(microsecond=0)
        .isoformat(),
        "official_artifact": {
            "path": str(official_path),
            "sha256": sha256_path(official_path),
            "rows": mapped["official_rows"],
            "columns": mapped["official_columns"],
        },
        "gold_inputs": {
            key: {"path": str(value), "sha256": sha256_path(value)}
            for key, value in gold_paths.items()
        },
        "mapped_output": {
            "path": str(mapped_path),
            "sha256": sha256_path(mapped_path),
            "rows": mapped["mapped_rows"],
        },
        "mapping": {
            key: value for key, value in mapped.items() if key not in {"mapped_partitions"}
        },
        "namespace_overlap": overlaps,
        "semantics": "official split assignments are audited as delivered; dropped rows are not silently assigned",
    }
    write_yaml(official_manifest_path, official_manifest)
    reproduction_manifest_path = (
        Path(repo_root) / "data" / "manifests" / "datasail_reproduction_manifest.yaml"
    )
    reproduction_manifest = {
        "manifest_type": "ppi_shift.datasail_reproduction_attempt",
        "manifest_version": 1,
        "created_utc": utc_now(),
        "git_commit": __import__("ppi_shift.artifacts", fromlist=["git_commit"]).git_commit(
            repo_root
        ),
        "requested_version": "1.2.1",
        "procedure": "Pinned DataSAIL v1.2.1 C1e proxy execution on the combined Gold protein universe, weighted by interaction participation, deterministic balanced-proxy stratification, 70/15/15, one run",
        "result": reproduction,
    }
    write_yaml(reproduction_manifest_path, reproduction_manifest)
    report_path = Path(repo_root) / "audit" / "datasail" / "DATASAIL_GOLD_AUDIT.md"
    report_lines = [
        "# DataSAIL Gold-v4 control audit",
        "",
        f"Generated UTC: `{utc_now()}`",
        "",
        "## Official artifact",
        "",
        f"- Official file: `{official_path}`; SHA256 `{sha256_path(official_path)}`; rows `{mapped['official_rows']:,}`.",
        f"- Local retrieval timestamp (filesystem mtime, UTC): `{official_manifest['official_download_timestamp_utc']}`.",
        f"- Source: [Zenodo version DOI](https://doi.org/{version_doi}) (concept DOI `{concept_doi}`); content URL: `{official_url}`.",
        f"- Gold-v4 input DOI: `{gold_input_doi}`.",
        f"- Label counts: `{mapped['official_label_counts']}`.",
        f"- Original Gold split labels: `{mapped['official_original_split_counts']}`.",
        f"- DataSAIL assignment labels (including dropped): `{mapped['official_datasail_assignment_counts']}`.",
        f"- Unique proteins: `{mapped['official_unique_proteins']:,}`; duplicate identifier-pair rows: `{mapped['official_duplicate_identifier_pair_rows']:,}`; conflicting pair labels: `{mapped['official_conflicting_pair_labels']:,}`.",
        "",
        "## Gold-v4 mapping",
        "",
        f"- Deterministic unordered identifier-pair plus binary-label matches: **{mapped['mapped_rows']:,} / {mapped['official_rows']:,}**.",
        f"- Unmatched rows: **{mapped['unmatched_rows']:,}**; ambiguous rows: **{mapped['ambiguous_rows']:,}**.",
        f"- Mapped assigned rows: `{mapped['mapped_rows_by_assignment']}`.",
        "No mismatch was guessed or repaired by sequence similarity.",
        "",
        "## Cross-partition overlap using the PPIShift namespace",
        "",
    ]
    if overlaps:
        overlap_columns = sorted(overlaps[0])
        report_lines.append("| " + " | ".join(overlap_columns) + " |")
        report_lines.append("| " + " | ".join("---" for _ in overlap_columns) + " |")
        for row in overlaps:
            report_lines.append(
                "| " + " | ".join(str(row.get(column, "")) for column in overlap_columns) + " |"
            )
    else:
        report_lines.append(
            "No train/validation/test assignments were available for an overlap comparison."
        )
    report_lines.extend(
        [
            "",
            "## Pinned proxy execution (not an official reproduction)",
            "",
            f"- Status: **{reproduction.get('status', 'unknown')}**.",
            f"- Reason: {reproduction.get('reason', '')}.",
            f"- Requested package version: `{reproduction.get('requested_version', '1.2.1')}`; observed: `{reproduction.get('installed_version', 'not available')}`.",
            f"- Execution evidence: `{reproduction.get('execution_evidence', 'fresh CLI execution')}`.",
            "The proxy inputs and exact attempted command are recorded in `datasail_reproduction_manifest.yaml`. This is an execution/parsing control only, not a successful reproduction claim: the addendum does not publish the authors' complete invocation/configuration.",
            "",
        ]
    )
    comparison = reproduction.get("parsed_output", {}).get("comparison_to_official", {})
    if comparison:
        report_lines.extend(
            [
                f"- Reproduction split counts: `{comparison.get('reproduction_split_counts', {})}`; official assigned counts: `{comparison.get('official_split_counts', {})}`.",
                f"- Official assigned rows compared: `{comparison.get('official_assigned_rows', 0):,}`; reproduced rows available for those pairs: `{comparison.get('reproduction_rows_for_official_pairs', 0):,}`.",
                f"- Same membership assignment: `{comparison.get('same_assignment_rows', 0):,}`; different assignment: `{comparison.get('different_assignment_rows', 0):,}`; missing official pairs: `{comparison.get('missing_official_pairs', 0):,}`; agreement: `{comparison.get('reproduction_assignment_agreement')}`.",
                "Membership differences are expected from the explicitly documented proxy stratification and are reported descriptively, not tuned against PPI performance.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Decision",
            "",
            "The official artifact is retrieved and deterministically mapped when the match counts above are complete. It remains a sensitivity-control split, not the primary PPIShift namespace. The pinned DataSAIL run is recorded as proxy_execution_completed when it executes and parses completely; that status is not an official row-for-row reproduction and is not a prerequisite for the data-layer gate.",
            "",
            f"Tracked manifest: `{official_manifest_path}`; reproduction manifest: `{reproduction_manifest_path}`.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "report_path": str(report_path),
        "official_manifest_path": str(official_manifest_path),
        "reproduction_manifest_path": str(reproduction_manifest_path),
        "official": mapped,
        "overlaps": overlaps,
        "reproduction": reproduction,
    }
