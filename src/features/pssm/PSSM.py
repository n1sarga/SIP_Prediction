import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from Bio.Align import substitution_matrices


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[3]
DATASET_FILE = ROOT / "data" / "processed" / SPECIES / "Unique_Proteins.csv"
BLAST_DB = ROOT / "data" / "external" / "blast_db" / "swissprot" / "swissprot"
OUTPUT_DIR = ROOT / "artifacts" / "features" / "pssm" / SPECIES / "Profiles"
OUTPUT_CSV = ROOT / "artifacts" / "features" / "pssm" / SPECIES / "pssm_feature_vectors.csv"
RUNTIME_DIR = Path(tempfile.gettempdir()) / "ppi_prediction_pssm"


def safe_unlink(path: Path, retries: int = 5, delay: float = 0.2) -> None:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == retries - 1:
                return
            time.sleep(delay)


def prepare_blast_db_path() -> Path:
    db_path = BLAST_DB
    if " " not in str(db_path):
        return db_path

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    link_dir = RUNTIME_DIR / "blastdb"
    if not link_dir.exists():
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_dir), str(db_path.parent)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and not link_dir.exists():
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Failed to create BLAST database junction")
    return link_dir / db_path.name


def run_psiblast(protein_sequence: str, temp_fasta: Path, output_pssm_file: Path) -> None:
    temp_fasta.parent.mkdir(parents=True, exist_ok=True)
    blast_db_path = prepare_blast_db_path()
    try:
        with temp_fasta.open("w", encoding="utf-8") as fasta_file:
            fasta_file.write(">temp\n")
            fasta_file.write(protein_sequence)

        subprocess.run(
            [
                "psiblast",
                "-query",
                str(temp_fasta),
                "-db",
                str(blast_db_path),
                "-evalue",
                "0.001",
                "-num_iterations",
                "3",
                "-out_ascii_pssm",
                str(output_pssm_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        safe_unlink(temp_fasta)


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


def generate_pssm(protein_sequence: str, temp_fasta: Path, output_pssm_file: Path) -> pd.DataFrame:
    try:
        run_psiblast(protein_sequence, temp_fasta, output_pssm_file)
        return parse_pssm(output_pssm_file)
    except Exception as exc:
        print(f"Falling back to BLOSUM62 encoding for {output_pssm_file.stem}: {exc}")
        return encode_with_blosum62(protein_sequence)


def process_dataset(dataset_file: Path) -> None:
    df = pd.read_csv(dataset_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    mean_rows = []

    for _, row in df.iterrows():
        identifier = row["Protein Identifier"]
        sequence = row["Protein Sequence"]
        temp_fasta = RUNTIME_DIR / f"{SPECIES.lower()}_{identifier}.fasta"
        temp_pssm = RUNTIME_DIR / f"{SPECIES.lower()}_{identifier}.txt"
        pssm_df = generate_pssm(sequence, temp_fasta, temp_pssm)
        pssm_df.to_parquet(OUTPUT_DIR / f"{identifier}.parquet")
        safe_unlink(temp_pssm)
        mean_row = pssm_df.mean(axis=0).to_dict()
        mean_row["Protein Identifier"] = identifier
        mean_rows.append(mean_row)
        print(f"Processed {identifier}")

    df_out = pd.DataFrame(mean_rows)
    columns = ["Protein Identifier"] + [col for col in df_out.columns if col != "Protein Identifier"]
    df_out = df_out[columns]
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Combined PSSM feature CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    process_dataset(DATASET_FILE)
