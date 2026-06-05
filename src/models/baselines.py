from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from data_io import jaccard


def pair_similarity_scores(
    pairs: pd.DataFrame,
    kmer_cache: dict[str, frozenset[str]],
    indices: np.ndarray,
) -> np.ndarray:
    scores = []
    for _, row in pairs.iloc[indices].iterrows():
        scores.append(
            jaccard(
                kmer_cache.get(row["Identifier A"], frozenset()),
                kmer_cache.get(row["Identifier B"], frozenset()),
            )
        )
    return np.asarray(scores, dtype=np.float32)


def degree_features_from_train(
    pairs: pd.DataFrame,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
) -> np.ndarray:
    degree = Counter()
    train_pairs = pairs.iloc[train_idx]
    for _, row in train_pairs[train_pairs["Interaction"] == 1].iterrows():
        degree[row["Identifier A"]] += 1
        degree[row["Identifier B"]] += 1

    rows = []
    for _, row in pairs.iloc[target_idx].iterrows():
        d1 = float(degree[row["Identifier A"]])
        d2 = float(degree[row["Identifier B"]])
        rows.append([d1, d2, min(d1, d2), max(d1, d2), abs(d1 - d2), d1 + d2, d1 * d2])
    return np.asarray(rows, dtype=np.float32)
