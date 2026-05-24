# Predicting Self-Interacting Proteins Using Rotation Forest Classifier

This project implements and evaluates a Rotation Forest classifier for protein-protein interaction prediction using multiple feature extraction techniques. The repository is organized into clear stages for data collection, feature extraction, feature fusion, and model evaluation.

## Workflow

1. Collect protein-protein interaction data and the corresponding protein sequences.
2. Generate negative interactions using a bipartite graph.
3. Generate PSSM profiles for the sequences.
4. Generate feature embeddings using amino acid composition, conjoint triads, and PSSM-based features.
5. Fuse the generated feature sets.
6. Train the Rotation Forest model.
7. Evaluate the model using classification metrics and ROC-AUC.

## Project Layout

```text
PPI Prediction/
|-- src/
|   |-- data_collection/
|   |-- features/
|   |-- fusion/
|   `-- models/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
|-- artifacts/
|   |-- features/
|   `-- fusion/
|-- reports/
|   `-- results/
`-- docs/
```

## Execution Order

Run the scripts in the following order:

If you need to find self-interacting proteins first, run step 1. Otherwise, start from step 2.

1. `python src/data_collection/SIP_Finder.py`
2. `python src/data_collection/Sequence.py`
3. `python src/data_collection/Bipartite_Graph.py`
4. `python src/features/amino_acid_composition/ProteinFeatureExtractor.py`
5. `python src/features/conjoint_triads/conjoint_triads.py`
6. `python src/features/pssm/PSSM.py`
7. `python src/fusion/Fusion.py`
8. `python src/models/RoF_and_Results.py`

You can switch datasets by setting `SPECIES`, for example:

```powershell
$env:SPECIES='Yeast'
python src/features/amino_acid_composition/ProteinFeatureExtractor.py
```

For fusion and model evaluation, you can also choose the fusion type:

```powershell
$env:SPECIES='Yeast'
$env:FUSION_NAME='aac_ct_pssm'
python src/fusion/Fusion.py
python src/models/RoF_and_Results.py
```

Supported fusion names:

- `aac_ct`
- `aac_pssm`
- `ct_pssm`
- `aac_ct_pssm`

## Required External Resources

To execute the project, download and configure the following:

1. NCBI BLAST+ standalone tools
2. Swiss-Prot database

Place external BLAST resources under `data/external/blast_db/`.

## Packages Used

- biopython
- matplotlib
- networkx
- numpy
- pandas
- pyarrow
- requests
- scikit-learn
- scipy

## Notes

- The dataset folder names currently use the spelling `Diabates` to match the existing source material.
- If `psiblast` is unavailable or fails, the PSSM script falls back to BLOSUM62-based encoding.
