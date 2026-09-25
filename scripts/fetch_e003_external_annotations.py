#!/usr/bin/env python3
"""Run the pinned Bernett UniProt fetch without the unbounded UniParc fallback.

The corrected external sampler uses the frozen source-panel sequences and
records source panels as human when taxonomy is absent. UniParc contributes no
GO annotations and its per-accession fallback can take days for archive-only
IDs. This wrapper therefore preserves the official UniProtKB sequence/GO/
taxonomy retrieval, records archive-only IDs with the declared human taxonomy,
and leaves their GO sets empty. The official fetch module is loaded from the
pinned artifact path so its network/parsing implementation remains the source
of truth.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any


def _official_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("e003_pinned_bernett_fetch_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned Bernett fetch module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-fetch-script", type=Path, required=True)
    parser.add_argument("--protein-list", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--species", type=Path, required=True)
    args = parser.parse_args()
    fetch = _official_module(args.official_fetch_script)
    proteins = fetch.read_protein_list(args.protein_list)
    canonical_map = fetch.build_canonical_map(proteins)
    canonicals = sorted(canonical_map)

    print(
        f"Fetching data for {len(proteins)} proteins ({len(canonicals)} canonical accessions) "
        "from pinned UniProtKB; archive-only UniParc lookup disabled",
        flush=True,
    )
    canon_seqs: dict[str, str] = {}
    canon_go: dict[str, dict[str, set[str]]] = {}
    canon_species: dict[str, str] = {}
    archive_only: list[str] = []
    for index in range(0, len(canonicals), fetch.BATCH_SIZE):
        batch = canonicals[index : index + fetch.BATCH_SIZE]
        try:
            seqs, go, species = fetch.parse_batch(fetch.fetch_batch(batch))
        except fetch.InvalidAccessionBatch:
            seqs, go, species = {}, {}, {}
            for accession in batch:
                try:
                    single = fetch.parse_batch(fetch.fetch_batch([accession]))
                    seqs.update(single[0])
                    go.update(single[1])
                    species.update(single[2])
                except fetch.InvalidAccessionBatch:
                    archive_only.append(accession)
                time.sleep(0.1)
        canon_seqs.update(seqs)
        canon_go.update(go)
        canon_species.update(species)
        print(
            f"  {min(index + fetch.BATCH_SIZE, len(canonicals))}/{len(canonicals)} "
            "canonical accessions fetched",
            flush=True,
        )
        if index + fetch.BATCH_SIZE < len(canonicals):
            time.sleep(0.5)

    # The external source panels are human and the materializer already uses
    # 9606 as its declared fallback. UniParc has no GO fields, so no biological
    # annotation is lost by leaving archive-only GO sets empty.
    for accession in archive_only:
        canon_species[accession] = "9606"
    print(
        f"Archive-only accessions: {len(archive_only)}; assigned source-declared taxon 9606; "
        "GO sets remain empty",
        flush=True,
    )

    all_seqs: dict[str, str] = {}
    all_go: dict[str, dict[str, set[str]]] = {}
    all_species: dict[str, str] = {}
    for canonical, originals in canonical_map.items():
        for accession in originals:
            if canonical in canon_seqs:
                all_seqs[accession] = canon_seqs[canonical]
            all_go[accession] = canon_go.get(
                canonical, {"BP": set(), "MF": set(), "CC": set()}
            )
            all_species[accession] = canon_species.get(canonical, "")

    isoforms = [accession for accession in proteins if "-" in accession]
    for accession in isoforms:
        sequence = fetch.fetch_isoform_sequence(accession)
        if sequence:
            all_seqs[accession] = sequence
        time.sleep(0.1)

    fetch.write_go_tsv(args.go, proteins, all_go)
    fetch.write_species_tsv(args.species, proteins, all_species)
    with args.fasta.open("w", encoding="utf-8") as handle:
        for accession in proteins:
            if accession in all_seqs:
                handle.write(f">{accession}\n{all_seqs[accession]}\n")
    print(f"Written {len(all_seqs)} sequences to {args.fasta}", flush=True)
    print(f"Written GO annotations for {len(proteins)} proteins to {args.go}", flush=True)
    print(f"Written species (taxon IDs) for {len(proteins)} proteins to {args.species}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
