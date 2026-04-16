import os
from pathlib import Path

import pandas as pd
from Bio.Align import substitution_matrices
from Bio.Blast.Applications import NcbipsiblastCommandline


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[3]
DATASET_FILE = ROOT / "data" / "processed" / SPECIES / "Unique_Proteins.csv"
BLAST_DB = ROOT / "data" / "external" / "blast_db" / "swissprot" / "swissprot"
OUTPUT_DIR = ROOT / "artifacts" / "features" / "pssm" / SPECIES / "Profiles"
TEMP_FASTA = ROOT / "artifacts" / "features" / "pssm" / "temp.fasta"


def run_psiblast(protein_sequence: str, output_pssm_file: Path) -> None:
    TEMP_FASTA.parent.mkdir(parents=True, exist_ok=True)
    with TEMP_FASTA.open("w", encoding="utf-8") as fasta_file:
        fasta_file.write(">temp\n")
        fasta_file.write(protein_sequence)

    psiblast_cline = NcbipsiblastCommandline(
        cmd="psiblast",
        query=str(TEMP_FASTA),
        db=str(BLAST_DB),
        evalue=0.001,
        num_iterations=3,
        out_ascii_pssm=str(output_pssm_file),
        save_pssm_after_last_round=True,
    )
    psiblast_cline()
    TEMP_FASTA.unlink(missing_ok=True)


def parse_pssm(filename: Path) -> pd.DataFrame:
    with filename.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    start = None
    for index, line in enumerate(lines):
        if line.startswith("Last position-specific scoring matrix computed"):
            start = index + 3
            break

    if start is None:
        raise ValueError("PSSM matrix not found in the file")

    pssm_data = []
    for line in lines[start:]:
        if line.strip() == "":
            break
        parts = line.split()
        pssm_data.append([int(x) for x in parts[2:22]])

    columns = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    return pd.DataFrame(pssm_data, columns=columns)


def encode_with_blosum62(protein_sequence: str) -> pd.DataFrame:
    blosum62 = substitution_matrices.load("BLOSUM62")
    encoded = []

    for aa in protein_sequence:
        if aa in blosum62.alphabet:
            encoded.append([blosum62[aa][other] for other in blosum62.alphabet])
        else:
            encoded.append([0] * len(blosum62.alphabet))

    blosum_df = pd.DataFrame(encoded, columns=list(blosum62.alphabet))
    return blosum_df.drop(columns=["B", "Z", "X", "*"], errors="ignore")


def generate_pssm(protein_sequence: str, output_pssm_file: Path) -> pd.DataFrame:
    try:
        run_psiblast(protein_sequence, output_pssm_file)
        return parse_pssm(output_pssm_file)
    except Exception as exc:
        print(f"Falling back to BLOSUM62 encoding for {output_pssm_file.stem}: {exc}")
        return encode_with_blosum62(protein_sequence)


def process_dataset(dataset_file: Path) -> None:
    df = pd.read_csv(dataset_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        identifier = row["Protein Identifier"]
        sequence = row["Protein Sequence"]
        output_pssm_file = OUTPUT_DIR / f"{identifier}.txt"
        pssm_df = generate_pssm(sequence, output_pssm_file)
        pssm_df.to_parquet(OUTPUT_DIR / f"{identifier}.parquet")
        print(f"Processed {identifier}")


if __name__ == "__main__":
    process_dataset(DATASET_FILE)
