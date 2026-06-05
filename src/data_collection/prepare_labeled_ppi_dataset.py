"""
Prepare a labelled PPI dataset from raw interaction rows.

Useful for Yeast because raw data already contains positive and negative PPI
labels. Self-pairs are dropped by default so SIP identity does not make the task
trivial.
"""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd


SPECIES = os.getenv("SPECIES", "Yeast")
DROP_SELF_INTERACTIONS = os.getenv("DROP_SELF_INTERACTIONS", "1") == "1"
ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / SPECIES / f"{SPECIES}_All_Interactions.csv"
PROTEINS_FILE = ROOT / "data" / "processed" / SPECIES / "Unique_Proteins.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / SPECIES / f"{SPECIES}_PPI.csv"


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a).strip(), str(b).strip())))


def main() -> None:
    raw = pd.read_csv(RAW_FILE)
    proteins = set(pd.read_csv(PROTEINS_FILE)["Protein Identifier"].astype(str).str.strip())

    df = raw[
        raw["Identifier A"].astype(str).str.strip().isin(proteins)
        & raw["Identifier B"].astype(str).str.strip().isin(proteins)
    ].copy()
    if DROP_SELF_INTERACTIONS:
        df = df[df["Identifier A"].astype(str).str.strip() != df["Identifier B"].astype(str).str.strip()].copy()

    df["Identifier A"] = df["Identifier A"].astype(str).str.strip()
    df["Identifier B"] = df["Identifier B"].astype(str).str.strip()
    df["Interaction"] = df["Interaction"].astype(int)
    df["_pair_key"] = df.apply(lambda row: ordered_pair(row["Identifier A"], row["Identifier B"]), axis=1)

    # If duplicate labels conflict, keep positive evidence.
    df = (
        df.groupby("_pair_key", as_index=False)
        .agg({"Identifier A": "first", "Identifier B": "first", "Interaction": "max"})
        [["Identifier A", "Identifier B", "Interaction"]]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved labelled PPI dataset to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}; labels: {df['Interaction'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
