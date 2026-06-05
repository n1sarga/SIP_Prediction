"""Generate controlled negative protein interaction sets.

This orchestrator keeps three sampler families in separate modules:
- bipartite_negative_sampling.py: left/right partition non-edges.
- controlled_negative_sampling.py: random and degree/length-matched non-edges.
- topology_negative_sampling.py: topology-driven CL3 non-edges inspired by UPNA-PPI.
"""

from __future__ import annotations

from pathlib import Path
import itertools
import os

import numpy as np
import pandas as pd

from bipartite_negative_sampling import sample_bipartite_negatives
from controlled_negative_sampling import sample_matched_negatives, sample_random_negatives
from topology_negative_sampling import sample_topology_cl3_negatives


SPECIES = os.getenv("SPECIES", "Diabates")
SEED = int(os.getenv("SEED", "42"))
TOPOLOGY_CANDIDATE_MULTIPLIER = int(os.getenv("TOPOLOGY_CANDIDATE_MULTIPLIER", "200"))
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / SPECIES


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a).strip(), str(b).strip())))


def find_positive_file() -> Path:
    override = os.getenv("POSITIVE_FILE")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = PROCESSED_DIR / path
        if path.exists():
            return path
        raise FileNotFoundError(f"POSITIVE_FILE does not exist: {path}")

    candidates = [
        PROCESSED_DIR / f"{SPECIES}_PPI.csv",
        PROCESSED_DIR / f"{SPECIES}_Cleaned.csv",
        PROCESSED_DIR / f"{SPECIES}_SIPs.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No positive interaction file found under {PROCESSED_DIR}")


def load_available_proteins() -> tuple[set[str], dict[str, int]]:
    proteins_path = PROCESSED_DIR / "Unique_Proteins.csv"
    if not proteins_path.exists():
        return set(), {}

    proteins_df = pd.read_csv(proteins_path).fillna("")
    proteins = set(proteins_df["Protein Identifier"].astype(str).str.strip())
    lengths = {
        str(row["Protein Identifier"]).strip(): len(str(row["Protein Sequence"]).strip())
        for _, row in proteins_df.iterrows()
    }
    return proteins, lengths


def load_positive_pairs() -> tuple[list[tuple[str, str]], list[str], dict[str, int], list[str], list[str]]:
    input_path = find_positive_file()
    print(f"Using positive source: {input_path}")
    available_proteins, sequence_lengths = load_available_proteins()

    df = pd.read_csv(input_path)
    if "Interaction" in df.columns:
        df = df[df["Interaction"] == 1].copy()

    before_count = len(df)
    if available_proteins:
        df = df[
            df["Identifier A"].astype(str).str.strip().isin(available_proteins)
            & df["Identifier B"].astype(str).str.strip().isin(available_proteins)
        ].copy()
        print(f"Filtered interactions to proteins with sequences: {before_count} -> {len(df)}")

    df["Identifier A"] = df["Identifier A"].astype(str).str.strip()
    df["Identifier B"] = df["Identifier B"].astype(str).str.strip()
    left_partition = sorted(set(df["Identifier A"]) - {""})
    right_partition = sorted(set(df["Identifier B"]) - {""})

    pairs = sorted(
        {
            ordered_pair(row["Identifier A"], row["Identifier B"])
            for _, row in df.iterrows()
            if str(row["Identifier A"]).strip() and str(row["Identifier B"]).strip()
        }
    )
    proteins = sorted(set(itertools.chain.from_iterable(pairs)))
    if available_proteins:
        proteins = sorted(set(proteins) & available_proteins)

    print(f"Unique positive unordered pairs: {len(pairs)}")
    print(f"Proteins available for sampling: {len(proteins)}")
    print(f"Bipartite partitions: left={len(left_partition)}, right={len(right_partition)}")
    return pairs, proteins, sequence_lengths, left_partition, right_partition


def write_dataset(
    name: str,
    positives: list[tuple[str, str]],
    negatives: list[tuple[str, str]],
    negative_metadata: pd.DataFrame | None = None,
) -> Path:
    positive_df = pd.DataFrame(positives, columns=["Identifier A", "Identifier B"])
    positive_df["Interaction"] = 1

    if negative_metadata is None:
        negative_df = pd.DataFrame(negatives, columns=["Identifier A", "Identifier B"])
    else:
        negative_df = negative_metadata.copy()
    negative_df["Interaction"] = 0

    metadata_columns = [
        column
        for column in negative_df.columns
        if column not in {"Identifier A", "Identifier B", "Interaction"}
    ]
    for column in metadata_columns:
        positive_df[column] = np.nan

    columns = ["Identifier A", "Identifier B", "Interaction", *metadata_columns]
    output = pd.concat([positive_df[columns], negative_df[columns]], ignore_index=True)
    output_path = PROCESSED_DIR / f"{SPECIES}_All_{name}.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved {name}: {len(positives)} positives + {len(negatives)} negatives -> {output_path}")
    return output_path


def main() -> None:
    rng = np.random.default_rng(SEED)
    positives, proteins, sequence_lengths, left_partition, right_partition = load_positive_pairs()
    if not positives:
        raise RuntimeError("No positive interactions available for negative sampling")

    known_positive_pairs = set(positives)
    n_positive = len(positives)

    try:
        bipartite_1to1, bipartite_metadata_1to1 = sample_bipartite_negatives(
            positives,
            left_partition,
            right_partition,
            n_positive,
            rng,
        )
        bipartite_10to1, bipartite_metadata_10to1 = sample_bipartite_negatives(
            positives,
            left_partition,
            right_partition,
            n_positive * 10,
            rng,
        )
    except RuntimeError as exc:
        bipartite_1to1 = []
        bipartite_metadata_1to1 = pd.DataFrame()
        bipartite_10to1 = []
        bipartite_metadata_10to1 = pd.DataFrame()
        print(f"Skipped bipartite outputs: {exc}")

    random_1to1 = sample_random_negatives(proteins, known_positive_pairs, n_positive, rng)
    random_10to1 = sample_random_negatives(proteins, known_positive_pairs | set(random_1to1), n_positive * 10, rng)
    matched_1to1 = sample_matched_negatives(positives, proteins, sequence_lengths, n_positive, rng)
    degree_matched_10to1 = sample_matched_negatives(positives, proteins, sequence_lengths, n_positive * 10, rng)

    try:
        topology_cl3_10to1, topology_metadata_10to1 = sample_topology_cl3_negatives(
            positives,
            proteins,
            n_positive * 10,
            rng,
            candidate_multiplier=TOPOLOGY_CANDIDATE_MULTIPLIER,
        )
        topology_cl3_1to1 = topology_cl3_10to1[:n_positive]
        topology_metadata_1to1 = topology_metadata_10to1.head(n_positive).copy()
    except RuntimeError as exc:
        topology_cl3_10to1 = []
        topology_metadata_10to1 = pd.DataFrame()
        topology_cl3_1to1 = []
        topology_metadata_1to1 = pd.DataFrame()
        print(f"Skipped topology CL3 outputs: {exc}")

    random_1to1_path = write_dataset("random_1to1", positives, random_1to1)
    if bipartite_1to1:
        write_dataset("bipartite_1to1", positives, bipartite_1to1, bipartite_metadata_1to1)
    if bipartite_10to1:
        write_dataset("bipartite_10to1", positives, bipartite_10to1, bipartite_metadata_10to1)
    write_dataset("random_10to1", positives, random_10to1)
    write_dataset("matched_1to1", positives, matched_1to1)
    write_dataset("degree_matched_10to1", positives, degree_matched_10to1)
    if topology_cl3_1to1:
        write_dataset("topology_cl3_1to1", positives, topology_cl3_1to1, topology_metadata_1to1)
    if topology_cl3_10to1:
        write_dataset("topology_cl3_10to1", positives, topology_cl3_10to1, topology_metadata_10to1)

    compatibility_path = PROCESSED_DIR / f"{SPECIES}_All.csv"
    pd.read_csv(random_1to1_path).to_csv(compatibility_path, index=False)
    print(f"Updated compatibility file: {compatibility_path}")


if __name__ == "__main__":
    main()
