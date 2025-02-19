'''
This code will do the following tasks:

1. Collects Protein Sequences using UniProt REST API
2. Ensures the Golden Standard for the training dataset
'''

import pandas as pd
import requests
import csv

def fetch_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta" # Uniport API
    response = requests.get(url)
    if response.ok:
        # Ignore the garbage section
        lines = response.text.strip().split('\n')
        if lines[0].startswith('>'):
            sequence = ''.join(lines[1:])
        else:
            sequence = ''.join(lines)
        print(f"Fetched: {uniprot_id}")
        return sequence
    else:
        print(f"Failed to fetch sequence for UniProt ID: {uniprot_id}")
        return None

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

def main(file_path):
    data = read_csv(file_path)
    sequences = []
    for row in data:
        pid, pseq = row
        sequence = fetch_protein_sequence(pid)
        sequence_to_use = sequence.strip() if pseq == '' else pseq.strip()

        '''
        The following condition checks if the sequence length is more than 50 residues
        and less than 5000 residues or not.
        '''
        if 50 < len(sequence_to_use) < 5000:
            sequences.append({
                'Protein Identifier': pid,
                'Protein Sequence': sequence_to_use
            })
        else:
            print(f"Skipped {pid}")

    df = pd.DataFrame(sequences)

    output_excel_file = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Unique_Proteins.csv'
    df.to_csv(output_excel_file, index=False)
    print(f"Sequences with interaction data saved to {output_excel_file}")

file_path = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Unique_Proteins.csv'
main(file_path)