from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_io import jaccard


def _split_has_both_classes(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    return (
        len(train_idx) > 0
        and len(test_idx) > 0
        and len(np.unique(y[train_idx])) == 2
        and len(np.unique(y[test_idx])) == 2
    )


def random_pair_split(y: np.ndarray, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return np.asarray(train_idx), np.asarray(test_idx)


def protein_holdout_split(
    pairs: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    test_protein_frac: float,
    max_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    proteins = np.array(sorted(set(pairs["Identifier A"]) | set(pairs["Identifier B"])))
    n_test = max(2, int(round(len(proteins) * test_protein_frac)))

    for _ in range(max_tries):
        test_proteins = set(rng.choice(proteins, size=n_test, replace=False))
        in_test_a = pairs["Identifier A"].isin(test_proteins).to_numpy()
        in_test_b = pairs["Identifier B"].isin(test_proteins).to_numpy()
        test_idx = np.where(in_test_a & in_test_b)[0]
        train_idx = np.where(~in_test_a & ~in_test_b)[0]
        if _split_has_both_classes(y, train_idx, test_idx):
            return train_idx, test_idx

    raise RuntimeError("Could not create a class-balanced protein holdout split")


def union_find_clusters(
    proteins: list[str],
    kmer_cache: dict[str, frozenset[str]],
    threshold: float,
) -> list[list[str]]:
    parent = {protein: protein for protein in proteins}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, left in enumerate(proteins):
        left_kmers = kmer_cache.get(left, frozenset())
        for right in proteins[i + 1 :]:
            if jaccard(left_kmers, kmer_cache.get(right, frozenset())) >= threshold:
                union(left, right)

    clusters: dict[str, list[str]] = {}
    for protein in proteins:
        clusters.setdefault(find(protein), []).append(protein)
    return list(clusters.values())


def low_similarity_holdout_split(
    pairs: pd.DataFrame,
    y: np.ndarray,
    kmer_cache: dict[str, frozenset[str]],
    seed: int,
    test_protein_frac: float,
    similarity_threshold: float,
    max_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    proteins = sorted(set(pairs["Identifier A"]) | set(pairs["Identifier B"]))
    clusters = union_find_clusters(proteins, kmer_cache, similarity_threshold)
    rng = np.random.default_rng(seed)
    target_size = max(2, int(round(len(proteins) * test_protein_frac)))

    for _ in range(max_tries):
        order = rng.permutation(len(clusters))
        test_proteins: set[str] = set()
        for cluster_idx in order:
            test_proteins.update(clusters[cluster_idx])
            if len(test_proteins) >= target_size:
                break

        in_test_a = pairs["Identifier A"].isin(test_proteins).to_numpy()
        in_test_b = pairs["Identifier B"].isin(test_proteins).to_numpy()
        test_idx = np.where(in_test_a & in_test_b)[0]
        train_idx = np.where(~in_test_a & ~in_test_b)[0]
        if _split_has_both_classes(y, train_idx, test_idx):
            return train_idx, test_idx

    raise RuntimeError("Could not create a class-balanced low-similarity holdout split")


def build_splits(
    split_names: list[str],
    pairs: pd.DataFrame,
    y: np.ndarray,
    kmer_cache: dict[str, frozenset[str]],
    seed: int,
    test_size: float,
    test_protein_frac: float,
    low_similarity_threshold: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    splits = {}
    for split_name in split_names:
        if split_name == "random_pair":
            splits[split_name] = random_pair_split(y, seed, test_size)
        elif split_name == "protein_holdout":
            splits[split_name] = protein_holdout_split(pairs, y, seed, test_protein_frac)
        elif split_name == "low_similarity_holdout":
            splits[split_name] = low_similarity_holdout_split(
                pairs,
                y,
                kmer_cache,
                seed,
                test_protein_frac,
                low_similarity_threshold,
            )
        else:
            raise ValueError(f"Unknown split: {split_name}")
    return splits
