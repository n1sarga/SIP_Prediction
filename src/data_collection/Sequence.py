"""
Fetch protein sequences from UniProt and write the filtered protein list.
"""

from pathlib import Path
import csv

import pandas as pd
import requests


SPECIES = "Diabates"
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / SPECIES


def fetch_protein_sequence(uniprot_id: str) -> str | None:
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url, timeout=30)
    if not response.ok:
        print(f"Failed to fetch sequence for UniProt ID: {uniprot_id}")
        return None

    lines = response.text.strip().splitlines()
    if not lines:
        return None
    if lines[0].startswith(">"):
        return "".join(lines[1:])
    return "".join(lines)


def read_csv(file_path: Path) -> list[list[str]]:
    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.reader(csv_file))


def main() -> None:
    input_path = PROCESSED_DIR / "Unique_Proteins.csv"
    output_path = PROCESSED_DIR / "Unique_Proteins.csv"

    data = read_csv(input_path)
    sequences = []
    for row in data:
        if len(row) < 2:
            continue
        pid, pseq = row
        sequence = fetch_protein_sequence(pid)
        if sequence is None and not pseq:
            continue

        sequence_to_use = sequence.strip() if pseq == "" and sequence else pseq.strip()
        if 50 < len(sequence_to_use) < 5000:
            sequences.append(
                {
                    "Protein Identifier": pid,
                    "Protein Sequence": sequence_to_use,
                }
            )
        else:
            print(f"Skipped {pid}")

    df = pd.DataFrame(sequences)
    df.to_csv(output_path, index=False)
    print(f"Sequences with interaction data saved to: {output_path}")


if __name__ == "__main__":
    main()
