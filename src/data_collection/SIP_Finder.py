from pathlib import Path

import pandas as pd


SPECIES = "Diabates"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SPECIES
PROCESSED_DIR = ROOT / "data" / "processed" / SPECIES


def main() -> None:
    df = pd.read_csv(RAW_DIR / f"{SPECIES}.csv")

    df = df[
        df["Identifier A"].str.startswith("UniProt", na=False)
        & df["Identifier B"].str.startswith("UniProt", na=False)
    ].copy()

    df["Identifier A"] = (
        df["Identifier A"].str.replace("UniProt", "", regex=False).str.split("-").str[0]
    )
    df["Identifier B"] = (
        df["Identifier B"].str.replace("UniProt", "", regex=False).str.split("-").str[0]
    )

    df = df.drop_duplicates()
    df["Interaction"] = 1

    proteins = df[["Identifier A"]].drop_duplicates()
    proteins = proteins.rename(columns={"Identifier A": "Protein Identifier"})
    proteins["Protein Sequence"] = None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{SPECIES}_Cleaned.csv", index=False)
    proteins.to_csv(PROCESSED_DIR / "Unique_Proteins.csv", index=False)
    print(f"Saved cleaned interactions and unique proteins under: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
