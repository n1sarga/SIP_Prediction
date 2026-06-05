"""Topology-driven CL3 negative sampling inspired by UPNA-PPI."""

from __future__ import annotations

from collections import deque
import itertools

import numpy as np
import pandas as pd


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a).strip(), str(b).strip())))


def build_adjacency(
    positive_pairs: list[tuple[str, str]],
    proteins: list[str],
) -> dict[str, set[str]]:
    adjacency = {protein: set() for protein in proteins}
    for left, right in positive_pairs:
        if left == right:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def l3_path_count(left: str, right: str, adjacency: dict[str, set[str]]) -> int:
    """Count paths left -> n1 -> n2 -> right in the positive PPI graph."""
    right_neighbors = adjacency.get(right, set())
    count = 0
    for left_neighbor in adjacency.get(left, set()):
        count += len(adjacency.get(left_neighbor, set()) & right_neighbors)
    return count


def shortest_path_distance(
    left: str,
    right: str,
    adjacency: dict[str, set[str]],
    max_depth: int = 6,
) -> int:
    if left == right:
        return 0

    visited = {left}
    queue: deque[tuple[str, int]] = deque([(left, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(node, set()):
            if neighbor == right:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return -1


def topology_metadata_row(
    pair: tuple[str, str],
    adjacency: dict[str, set[str]],
    degree: dict[str, int],
    edge_count: int,
    source: str,
) -> dict[str, object]:
    left, right = pair
    degree_left = degree.get(left, 0)
    degree_right = degree.get(right, 0)
    denominator = max(1, 2 * edge_count)
    config_probability = min(1.0, (degree_left * degree_right) / denominator)
    common_neighbors = len(adjacency.get(left, set()) & adjacency.get(right, set()))
    l3_count = l3_path_count(left, right, adjacency)
    return {
        "Identifier A": left,
        "Identifier B": right,
        "negative_source": source,
        "config_probability": config_probability,
        "degree_a": degree_left,
        "degree_b": degree_right,
        "common_neighbor_count": common_neighbors,
        "l3_path_count": l3_count,
        "shortest_path_distance": shortest_path_distance(left, right, adjacency),
    }


def sample_topology_cl3_negatives(
    positives: list[tuple[str, str]],
    proteins: list[str],
    count: int,
    rng: np.random.Generator,
    candidate_multiplier: int = 200,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    graph_positives = [pair for pair in positives if pair[0] != pair[1]]
    if not graph_positives:
        raise RuntimeError("Topology CL3 sampling needs non-self positive PPI pairs")

    known_positive_pairs = set(positives)
    adjacency = build_adjacency(graph_positives, proteins)
    degree = {protein: len(adjacency.get(protein, set())) for protein in proteins}
    edge_count = len(graph_positives)

    candidates: list[tuple[float, float, tuple[str, str]]] = []
    for left, right in itertools.combinations(proteins, 2):
        pair = ordered_pair(left, right)
        if pair in known_positive_pairs:
            continue

        config_probability = min(1.0, (degree.get(left, 0) * degree.get(right, 0)) / max(1, 2 * edge_count))
        candidates.append((config_probability, float(rng.random()), pair))

    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} non-edge candidates available, requested {count}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    bottom_n = min(len(candidates), max(count, count * candidate_multiplier))

    selected_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def collect_from(pool: list[tuple[float, float, tuple[str, str]]], strict_l3: bool) -> None:
        source = "topology_cl3" if strict_l3 else "topology_low_config_fallback"
        for _, _, pair in pool:
            if len(selected_rows) >= count:
                return
            if pair in seen:
                continue

            row = topology_metadata_row(pair, adjacency, degree, edge_count, source)
            if strict_l3 and row["l3_path_count"] != 0:
                continue
            selected_rows.append(row)
            seen.add(pair)

    collect_from(candidates[:bottom_n], strict_l3=True)
    if len(selected_rows) < count and bottom_n < len(candidates):
        collect_from(candidates[bottom_n:], strict_l3=True)
    if len(selected_rows) < count:
        collect_from(candidates, strict_l3=False)

    if len(selected_rows) < count:
        raise RuntimeError(f"Could only sample {len(selected_rows)} topology negatives, requested {count}")

    metadata = pd.DataFrame(selected_rows)
    pairs = list(zip(metadata["Identifier A"], metadata["Identifier B"]))
    strict_count = int((metadata["negative_source"] == "topology_cl3").sum())
    print(
        "Topology CL3 selected "
        f"{strict_count}/{len(metadata)} strict zero-L3 negatives "
        f"(candidate pool={len(candidates)}, bottom_n={bottom_n})"
    )
    return pairs, metadata
