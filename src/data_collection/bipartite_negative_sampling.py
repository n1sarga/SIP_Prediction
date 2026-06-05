"""Bipartite negative sampling for left/right interaction partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a).strip(), str(b).strip())))


def sample_bipartite_negatives(
    positives: list[tuple[str, str]],
    left_partition: list[str],
    right_partition: list[str],
    count: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    known_positive_pairs = set(positives)
    candidates = sorted(
        {
            ordered_pair(left, right)
            for left in left_partition
            for right in right_partition
            if left != right and ordered_pair(left, right) not in known_positive_pairs
        }
    )
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} bipartite non-edges available, requested {count}")

    selected_idx = rng.choice(len(candidates), size=count, replace=False)
    selected = sorted(candidates[int(index)] for index in selected_idx)
    metadata = pd.DataFrame(selected, columns=["Identifier A", "Identifier B"])
    metadata["negative_source"] = "bipartite_non_edge"
    metadata["left_partition_size"] = len(left_partition)
    metadata["right_partition_size"] = len(right_partition)
    return selected, metadata
