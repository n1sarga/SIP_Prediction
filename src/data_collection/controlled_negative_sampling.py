"""Random and degree/length-matched negative sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
import itertools

import numpy as np


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a).strip(), str(b).strip())))


def sample_random_negatives(
    proteins: list[str],
    known_positive_pairs: set[tuple[str, str]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    negatives: set[tuple[str, str]] = set()
    max_tries = max(50_000, count * 100)

    tries = 0
    while len(negatives) < count and tries < max_tries:
        left, right = rng.choice(proteins, size=2, replace=False)
        pair = ordered_pair(left, right)
        if pair not in known_positive_pairs:
            negatives.add(pair)
        tries += 1

    if len(negatives) < count:
        all_candidates = (
            ordered_pair(left, right)
            for left, right in itertools.combinations(proteins, 2)
        )
        for pair in all_candidates:
            if pair not in known_positive_pairs and pair not in negatives:
                negatives.add(pair)
                if len(negatives) == count:
                    break

    if len(negatives) < count:
        raise RuntimeError(f"Could only sample {len(negatives)} negatives, requested {count}")

    return sorted(negatives)


def degree_bins(positive_pairs: list[tuple[str, str]], proteins: list[str]) -> dict[str, int]:
    degree = Counter()
    for left, right in positive_pairs:
        degree[left] += 1
        degree[right] += 1

    bins = {}
    for protein in proteins:
        value = degree[protein]
        if value == 0:
            bins[protein] = 0
        elif value == 1:
            bins[protein] = 1
        elif value <= 3:
            bins[protein] = 2
        elif value <= 10:
            bins[protein] = 3
        else:
            bins[protein] = 4
    return bins


def length_bins(sequence_lengths: dict[str, int], proteins: list[str]) -> dict[str, int]:
    observed = [sequence_lengths.get(protein, 0) for protein in proteins]
    if not observed or max(observed) == 0:
        return {protein: 0 for protein in proteins}

    quantiles = np.quantile(observed, [0.2, 0.4, 0.6, 0.8])
    return {
        protein: int(np.searchsorted(quantiles, sequence_lengths.get(protein, 0), side="right"))
        for protein in proteins
    }


def sample_matched_negatives(
    positives: list[tuple[str, str]],
    proteins: list[str],
    sequence_lengths: dict[str, int],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    known_positive_pairs = set(positives)
    deg_bin = degree_bins(positives, proteins)
    len_bin = length_bins(sequence_lengths, proteins)

    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for protein in proteins:
        buckets[(deg_bin[protein], len_bin[protein])].append(protein)

    negatives: set[tuple[str, str]] = set()
    positive_cycle = positives * max(1, int(np.ceil(count / max(1, len(positives)))))

    for left, right in positive_cycle:
        if len(negatives) >= count:
            break

        left_bucket = buckets[(deg_bin[left], len_bin[left])]
        right_bucket = buckets[(deg_bin[right], len_bin[right])]

        for _ in range(200):
            neg_left = str(rng.choice(left_bucket))
            neg_right = str(rng.choice(right_bucket))
            if neg_left == neg_right:
                continue
            pair = ordered_pair(neg_left, neg_right)
            if pair not in known_positive_pairs and pair not in negatives:
                negatives.add(pair)
                break

    if len(negatives) < count:
        fallback = sample_random_negatives(
            proteins,
            known_positive_pairs | negatives,
            count - len(negatives),
            rng,
        )
        negatives.update(fallback)

    return sorted(negatives)
