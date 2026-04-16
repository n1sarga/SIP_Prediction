# Predicting Self-Interacting Proteins Using Rotation Forest Classifier

This project is based on an undergraduate thesis by Sofia Noor Rafa, Ummay Khadiza Rumpa, and Nisarga Mridha. It implements and evaluates a Rotation Forest classifier for predicting self-interacting proteins in Maize and Yeast using multiple feature extraction techniques. The original Rotation Forest implementation was adapted from Liam-E2.

This README is based on the original GitHub repository README, but the instructions below have been updated to match the restructured local project layout in this folder.

## Workflow

1. Collect protein-protein interaction data and the corresponding protein sequences.
2. Generate negative interactions using a bipartite graph.
3. Generate PSSM profiles for the sequences.
4. Generate feature embeddings using amino acid composition, conjoint triads, and PSSM-based features in the restructured pipeline.
5. Fuse the generated feature sets.
6. Train the Rotation Forest model.
7. Evaluate the model using classification metrics and ROC-AUC.

## Restructured Project Layout

```text
Test - 400/
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

1. `python src/data_collection/SIP_Finder.py`
2. `python src/data_collection/Sequence.py`
3. `python src/data_collection/Bipartite_Graph.py`
4. `python src/features/amino_acid_composition/ProteinFeatureExtractor.py`
5. `python src/features/conjoint_triads/conjoint_triads.py`
6. `python src/features/pssm/PSSM.py`
7. `python src/fusion/Fusion.py`
8. `python src/models/RoF_and_Results.py`

## Required External Resources

To execute the project, download and configure the following:

1. NCBI BLAST+ standalone tools: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/
2. Swiss-Prot database: https://ftp.ncbi.nlm.nih.gov/blast/db/

Place external BLAST resources under `data/external/blast_db/` so the restructured project remains consistent.

## Main Packages Used

- `biopython`
- `networkx`
- `pandas`
- `numpy`
- `scipy`
- `pywt`
- `requests`
- `matplotlib`
- `scikit-learn`

## Notes

- The original repository highlighted transformation methods such as DST, DCT, WT, DHT, and FFT. The restructured local pipeline currently exposes amino acid composition, conjoint triads, PSSM processing, feature fusion, and Rotation Forest modeling through the `src/` layout present in this folder.
- The species label `Diabates` is preserved in the local data paths because that naming already exists in the project files.
- Additional structure details are documented in [docs/project_structure.md](d:/Profiles/Nisarga%20Mridha/Coding/Test%20-%20400/docs/project_structure.md).
