"""
Compute conjoint triad features for a species-specific protein dataset.
"""

import os
from pathlib import Path

import pandas as pd


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV = ROOT / "data" / "processed" / SPECIES / "Unique_Proteins.csv"
OUTPUT_DIR = ROOT / "artifacts" / "features" / "conjoint_triads" / SPECIES
OUTPUT_CSV = OUTPUT_DIR / "conjoint_triad_vectors.csv"
PARQUET_DIR = OUTPUT_DIR / "Profiles"


def VS(rang: int = 8) -> list[str]:
    vectors = []
    for i in range(1, rang):
        for j in range(1, rang):
            for k in range(1, rang):
                vectors.append(f"VS{k}{j}{i}")
    return vectors


def aa_to_class(aa: str) -> str | None:
    aa = aa.upper()
    if aa in ["A", "G", "V"]:
        return "1"
    if aa in ["I", "L", "F", "P"]:
        return "2"
    if aa in ["Y", "M", "T", "S"]:
        return "3"
    if aa in ["H", "N", "Q", "W"]:
        return "4"
    if aa in ["R", "K"]:
        return "5"
    if aa in ["D", "E"]:
        return "6"
    if aa == "C":
        return "7"
    return None


def frequency(seq: str) -> list[str]:
    triads = []
    seq = str(seq).upper()
    for i in range(len(seq) - 2):
        triad = "VS"
        valid = True
        for j in range(3):
            group = aa_to_class(seq[i + j])
            if group is None:
                valid = False
                break
            triad += group
        if valid:
            triads.append(triad)
    return triads


def freq_dict(vectors: list[str], triads: list[str]) -> dict[str, int]:
    counts = {v: 0 for v in vectors}
    for triad in triads:
        if triad in counts:
            counts[triad] += 1
    return counts


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    vectors = VS(8)
    feature_rows = []

    for _, row in df.iterrows():
        seq_id = row["Protein Identifier"]
        seq = str(row["Protein Sequence"]).strip().upper()

        triad_counts = freq_dict(vectors, frequency(seq))
        pd.DataFrame([triad_counts]).to_parquet(PARQUET_DIR / f"{seq_id}.parquet", index=False)

        triad_counts["Protein Identifier"] = seq_id
        feature_rows.append(triad_counts)

    df_out = pd.DataFrame(feature_rows)
    cols = ["Protein Identifier"] + [v for v in vectors]
    df_out = df_out[cols]
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Conjoint triad features saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
