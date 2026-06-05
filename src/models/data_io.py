from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_config import COMPONENT_DIRS, FEATURE_COMPONENTS, PROCESSED_DIR, ROOT, SPECIES


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def protein_id_from_string(value: object) -> str:
    text = str(value).strip()
    if "." in text and text.count(".") == 1:
        return text.split(".")[-1]
    return text


def read_sequences() -> dict[str, str]:
    path = PROCESSED_DIR / "Unique_Proteins.csv"
    df = pd.read_csv(path).fillna("")
    return {
        str(row["Protein Identifier"]).strip(): str(row["Protein Sequence"]).strip().upper()
        for _, row in df.iterrows()
        if str(row["Protein Identifier"]).strip()
    }


def kmer_set(seq: str, k: int = 3) -> frozenset[str]:
    seq = str(seq).upper()
    if len(seq) < k:
        return frozenset()
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def load_component_profiles(component: str) -> tuple[dict[str, np.ndarray], list[str]]:
    folder = COMPONENT_DIRS[component]
    if not folder.exists():
        return {}, []

    profiles: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None

    for path in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue

        if component == "pssm":
            row = df.mean(axis=0, numeric_only=True)
            names = [f"PSSM_{col}" for col in row.index]
        else:
            row = df.iloc[0].drop(labels=["Protein Identifier"], errors="ignore")
            prefix = "AAC" if component == "aac" else "CT"
            names = [f"{prefix}_{col}" for col in row.index]

        values = pd.to_numeric(row, errors="coerce").to_numpy(dtype=np.float32)
        if feature_names is None:
            feature_names = names
        profiles[path.stem] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return profiles, feature_names or []


def build_feature_cache(feature_sets: list[str]) -> dict[str, tuple[dict[str, np.ndarray], list[str]]]:
    needed_components = sorted({component for name in feature_sets for component in FEATURE_COMPONENTS[name]})
    component_cache = {component: load_component_profiles(component) for component in needed_components}
    feature_cache: dict[str, tuple[dict[str, np.ndarray], list[str]]] = {}

    for feature_set in feature_sets:
        components = FEATURE_COMPONENTS[feature_set]
        component_profiles = [component_cache[component][0] for component in components]
        component_names = [component_cache[component][1] for component in components]

        if any(not profiles for profiles in component_profiles):
            feature_cache[feature_set] = ({}, [])
            continue

        common_ids = set.intersection(*(set(profiles) for profiles in component_profiles))
        names = [name for names_part in component_names for name in names_part]
        combined = {
            protein_id: np.concatenate([profiles[protein_id] for profiles in component_profiles])
            for protein_id in common_ids
        }
        feature_cache[feature_set] = (combined, names)

    return feature_cache


def discover_interaction_files(requested: list[str]) -> list[Path]:
    if requested != ["auto"]:
        return [Path(item) if Path(item).is_absolute() else PROCESSED_DIR / item for item in requested]

    candidates = sorted(PROCESSED_DIR.glob(f"{SPECIES}_All*.csv"))
    sip_file = PROCESSED_DIR / f"{SPECIES}_SIPs_All.csv"
    if sip_file.exists():
        candidates.append(sip_file)
    return list(dict.fromkeys(candidates))


def negative_set_name(path: Path) -> str:
    stem = path.stem
    for prefix in (f"{SPECIES}_All_", f"{SPECIES}_SIPs_All_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    if stem in {f"{SPECIES}_All", f"{SPECIES}_SIPs_All"}:
        return "default"
    return stem


def read_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["Identifier A", "Identifier B", "Interaction"]].copy()
    df["Identifier A"] = df["Identifier A"].map(protein_id_from_string)
    df["Identifier B"] = df["Identifier B"].map(protein_id_from_string)
    df["Interaction"] = df["Interaction"].astype(int)
    df["_pair_key"] = df.apply(lambda row: ordered_pair(row["Identifier A"], row["Identifier B"]), axis=1)
    return df.drop_duplicates("_pair_key").drop(columns="_pair_key").reset_index(drop=True)


def prepare_pair_features(
    df_pairs: pd.DataFrame,
    protein_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    kept_pairs = []

    for _, row in df_pairs.iterrows():
        prot_a, prot_b = row["Identifier A"], row["Identifier B"]
        if prot_a not in protein_features or prot_b not in protein_features:
            continue
        rows.append(np.concatenate([protein_features[prot_a], protein_features[prot_b]]))
        labels.append(int(row["Interaction"]))
        kept_pairs.append(row)

    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int), pd.DataFrame()

    x = np.vstack(rows).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, np.asarray(labels, dtype=int), pd.DataFrame(kept_pairs).reset_index(drop=True)
