import os
import pandas as pd
import numpy as np
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio.Align import substitution_matrices

def run_psiblast(protein_sequence, output_pssm_file, db="C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Scripts/Encoding/blast_db/swissprot"):
    fasta_file = "temp.fasta"
    with open(fasta_file, "w") as f:
        f.write(">temp\n")
        f.write(protein_sequence)

    psiblast_cline = NcbipsiblastCommandline(
        cmd="psiblast",
        query=fasta_file,
        db=db,
        evalue=0.001,
        num_iterations=3,
        out_ascii_pssm=output_pssm_file,
        save_pssm_after_last_round=True
    )
    stdout, stderr = psiblast_cline()

    os.remove(fasta_file)

def parse_pssm(filename):
    with open(filename) as file:
        lines = file.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith("Last position-specific scoring matrix computed"):
            start = i + 3
            break

    if start is None:
        raise ValueError("PSSM matrix not found in the file")

    # Parse the PSSM matrix
    pssm_data = []
    for line in lines[start:]:
        if line.strip() == "":
            break
        parts = line.split()
        pssm_data.append([int(x) for x in parts[2:22]])

    columns = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    pssm_df = pd.DataFrame(pssm_data, columns=columns)

    # Drop B, Z, X and * columns
    pssm_df.drop(columns=["B", "Z", "X", "*"], inplace=True, errors='ignore')

    return pssm_df

def encode_with_blosum62(protein_sequence):
    blosum62 = substitution_matrices.load("BLOSUM62")
    protein_length = len(protein_sequence)
    blosum62_data = []

    for aa in protein_sequence:
        if aa in blosum62.alphabet:
            blosum62_data.append([blosum62[aa][other] for other in blosum62.alphabet])
        else:
            blosum62_data.append([0] * 20)  # Encoding unknown amino acids as 0

    columns = list(blosum62.alphabet)
    blosum62_df = pd.DataFrame(blosum62_data, columns=columns)

    # Drop B, Z, X and * columns
    blosum62_df.drop(columns=["B", "Z", "X", "*"], inplace=True, errors='ignore')

    return blosum62_df

def generate_pssm(protein_sequence, output_pssm_file="pssm.txt"):
    try:
        run_psiblast(protein_sequence, output_pssm_file)
        pssm_df = parse_pssm(output_pssm_file)
    except Exception as e:
        pssm_df = encode_with_blosum62(protein_sequence)

    return pssm_df

def process_dataset(dataset_file):
    df = pd.read_csv(dataset_file)

    output_dir = "C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Yeast_PSSMs"
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        try:
            identifier = row['Protein Identifier']
            sequence = row['Protein Sequence']
            output_pssm_file = f"{output_dir}/{identifier}.txt"
            pssm_df = generate_pssm(sequence, output_pssm_file)
            pssm_df.to_parquet(f"{output_dir}/{identifier}.parquet")
            print(f"Processed {identifier}")
        except KeyError as e:
            print(f"KeyError: {e}")
        except Exception as e:
            print(f"Error processing {identifier}: {e}")

dataset_file = "C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Unique_Proteins.csv"
process_dataset(dataset_file)
