import os
from pathlib import Path

import pandas as pd

from feature_utils import compute_aa_composition, compute_basic_features


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = ROOT / "data" / "processed" / SPECIES / "Unique_Proteins.csv"
OUTPUT_DIR = ROOT / "artifacts" / "features" / "amino_acid_composition" / SPECIES
PROFILE_DIR = OUTPUT_DIR / "Profiles"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(INPUT_FILE)
    aac_data = []
    seqfeat_data = []

    for _, row in df_in.iterrows():
        seq_id = str(row["Protein Identifier"]).strip()
        seq = str(row["Protein Sequence"]).strip().upper()

        aac_feats = compute_aa_composition(seq)
        aac_data.append({"Protein Identifier": seq_id, **aac_feats})
        pd.DataFrame([aac_feats]).to_parquet(PROFILE_DIR / f"{seq_id}.parquet")

        seq_feats = compute_basic_features(seq)
        seq_feats["Protein Identifier"] = seq_id
        seqfeat_data.append(seq_feats)

    df_aac_all = pd.DataFrame(aac_data)
    aac_cols = ["Protein Identifier"] + [c for c in df_aac_all.columns if c != "Protein Identifier"]
    df_aac_all = df_aac_all[aac_cols]
    aac_csv_path = OUTPUT_DIR / "amino_acid_composition_vectors.csv"
    df_aac_all.to_csv(aac_csv_path, index=False)

    df_seqfeat_all = pd.DataFrame(seqfeat_data)
    seq_cols = ["Protein Identifier"] + [c for c in df_seqfeat_all.columns if c != "Protein Identifier"]
    df_seqfeat_all = df_seqfeat_all[seq_cols]
    seqfeat_csv_path = OUTPUT_DIR / "sequence_level_vectors.csv"
    df_seqfeat_all.to_csv(seqfeat_csv_path, index=False)

    print(f"Combined AAC CSV saved at: {aac_csv_path}")
    print(f"AAC parquet profiles saved in: {PROFILE_DIR}")
    print(f"Sequence-level feature CSV saved at: {seqfeat_csv_path}")


if __name__ == "__main__":
    main()
