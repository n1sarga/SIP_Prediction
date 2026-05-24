from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
import shutil
import subprocess
import tempfile

import pandas as pd
from Bio.Align import substitution_matrices


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[3]
DATASET_FILE = ROOT / "data" / "processed" / SPECIES / "Unique_Proteins.csv"
BLAST_DB = Path(
    os.getenv(
        "BLAST_DB",
        str(ROOT / "data" / "external" / "blast_db" / "swissprot" / "swissprot"),
    )
)
OUTPUT_DIR = ROOT / "artifacts" / "features" / "pssm" / SPECIES / "Profiles"
BLAST_WORK_ROOT = Path(tempfile.gettempdir()) / "test400_blast"
BLAST_DB_LINK = BLAST_WORK_ROOT / f"blast_db_{hashlib.sha1(str(BLAST_DB.parent).encode()).hexdigest()[:10]}"
PSSM_WORKERS = int(os.getenv("PSSM_WORKERS", str(min(4, os.cpu_count() or 1))))
BLAST_NUM_THREADS = os.getenv("BLAST_NUM_THREADS", "1" if PSSM_WORKERS > 1 else "4")
BLAST_NUM_ITERATIONS = os.getenv("BLAST_NUM_ITERATIONS", "3")
BLAST_EVALUE = os.getenv("BLAST_EVALUE", "0.001")
BLAST_MAX_TARGET_SEQS = os.getenv("BLAST_MAX_TARGET_SEQS", "500")


def ensure_blast_db_link() -> None:
    BLAST_WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if BLAST_DB_LINK.exists() or BLAST_DB_LINK.is_symlink():
        if not BLAST_DB_LINK.is_dir():
            raise RuntimeError(f"Blast DB link path is not a directory: {BLAST_DB_LINK}")
        return

    cmd = ["cmd", "/c", "mklink", "/J", str(BLAST_DB_LINK), str(BLAST_DB.parent)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 and not BLAST_DB_LINK.is_dir():
        raise RuntimeError(f"Could not create BLAST DB junction: {result.stderr.strip() or result.stdout.strip()}")


def run_psiblast(protein_sequence: str, protein_id: str) -> Path:
    ensure_blast_db_link()
    work_dir = BLAST_WORK_ROOT / protein_id
    work_dir.mkdir(parents=True, exist_ok=True)
    work_fasta = work_dir / "query.fasta"
    work_pssm = work_dir / "profile.pssm"

    with work_fasta.open("w", encoding="utf-8") as fasta_file:
        fasta_file.write(">temp\n")
        fasta_file.write(protein_sequence)

    cmd = [
        "psiblast",
        "-query",
        str(work_fasta),
        "-db",
        str(BLAST_DB_LINK / BLAST_DB.name),
        "-evalue",
        BLAST_EVALUE,
        "-num_iterations",
        BLAST_NUM_ITERATIONS,
        "-num_threads",
        BLAST_NUM_THREADS,
        "-max_target_seqs",
        BLAST_MAX_TARGET_SEQS,
        "-out",
        os.devnull,
        "-outfmt",
        "6",
        "-out_ascii_pssm",
        str(work_pssm),
        "-save_pssm_after_last_round",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(result.stderr.strip() or f"psiblast exited with {result.returncode}")

    work_fasta.unlink(missing_ok=True)
    return work_pssm


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


def generate_pssm(protein_id: str, protein_sequence: str) -> pd.DataFrame:
    try:
        work_pssm = run_psiblast(protein_sequence, protein_id)
        try:
            return parse_pssm(work_pssm)
        finally:
            shutil.rmtree(work_pssm.parent, ignore_errors=True)
    except Exception as exc:
        print(f"Falling back to BLOSUM62 encoding for {protein_id}: {exc}")
        return encode_with_blosum62(protein_sequence)


def process_protein(record: tuple[str, str]) -> str:
    identifier, sequence = record
    output_parquet = OUTPUT_DIR / f"{identifier}.parquet"
    if output_parquet.exists():
        return identifier

    pssm_df = generate_pssm(identifier, sequence)
    pssm_df.to_parquet(output_parquet)
    return identifier


def process_dataset(dataset_file: Path) -> None:
    df = pd.read_csv(dataset_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_blast_db_link()

    existing = {path.stem for path in OUTPUT_DIR.glob("*.parquet")}
    pending = df[~df["Protein Identifier"].isin(existing)]
    records = [
        (str(row["Protein Identifier"]), str(row["Protein Sequence"]))
        for _, row in pending.iterrows()
    ]

    print(f"PSSM profiles complete: {len(existing)}/{len(df)}")
    print(f"PSSM profiles pending: {len(records)}")
    print(f"PSSM workers: {PSSM_WORKERS}; BLAST threads per worker: {BLAST_NUM_THREADS}")

    if not records:
        return

    if PSSM_WORKERS == 1:
        for index, record in enumerate(records, start=1):
            identifier = process_protein(record)
            print(f"Processed {identifier} ({index}/{len(records)})")
        return

    with ProcessPoolExecutor(max_workers=PSSM_WORKERS) as executor:
        futures = [executor.submit(process_protein, record) for record in records]
        for index, future in enumerate(as_completed(futures), start=1):
            identifier = future.result()
            print(f"Processed {identifier} ({index}/{len(records)})")


if __name__ == "__main__":
    process_dataset(DATASET_FILE)
