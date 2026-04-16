# Predicting Self-Interacting Proteins Using Rotation Forest Classifier

This project is based on an undergraduate thesis by [Sofia Noor Rafa](https://github.com/sofiautilitarian), [Ummay Khadiza Rumpa](https://github.com/ukrumpa), and Nisarga Mridha under the supervision of [Dr. Shamim H. Ripon](https://fse.ewubd.edu/computer-science-engineering/faculty-view/dshr). It implements and evaluates a Rotation Forest classifier for predicting self-interacting proteins in Maize and Yeast using multiple feature extraction techniques. The original Rotation Forest implementation was adapted from [Liam-E2](https://github.com/Liam-E2/RotationForest).

This README reflects the restructured project layout currently used in the repository.

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
SIP_Prediction/
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

1. NCBI BLAST+ standalone tools: [NCBI BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
2. Swiss-Prot database: [Swiss-Prot DB](https://ftp.ncbi.nlm.nih.gov/blast/db/)

Place external BLAST resources under `data/external/blast_db/`.

## Packages Used

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

- The original repository discussed transformation methods such as DST, DCT, WT, DHT, and FFT. The current restructured repository exposes amino acid composition, conjoint triads, PSSM processing, feature fusion, and Rotation Forest modeling through the `src/` layout.
- The species label `Diabates` is preserved in existing data paths and filenames to avoid breaking the current pipeline.
- Additional structure details are documented in `docs/project_structure.md`.
