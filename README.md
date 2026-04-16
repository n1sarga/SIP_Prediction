# Predicting Self-Interacting Proteins Using Rotation Forest Classifier

This is my university's undergraduate thesis. My thesis members were [Sofia Noor Rafa](https://github.com/sofiautilitarian) and [Ummay Khadiza Rumpa](https://github.com/ukrumpa). My thesis supervisor was [Dr. Shamim H. Ripon](https://fse.ewubd.edu/computer-science-engineering/faculty-view/dshr). In this project, we have implemented and evaluated the Rotation Forest Classifier for predicting interactions in Maize and Yeast using multiple feature extraction techniques. The RoF implementation was taken from [Liam-E2](https://github.com/Liam-E2/RotationForest).

## Workflow

1. Collect protein-protein interaction data and the corresponding protein sequences.
2. Generate negative interactions using a bipartite graph.
3. Generate PSSM profiles for the sequences.
4. Generate feature embeddings using amino acid composition, conjoint triads, and PSSM-based features.
5. Fuse the generated feature sets.
6. Train the Rotation Forest model.
7. Evaluate the model using classification metrics and ROC-AUC.

![System Architecture](https://github.com/user-attachments/assets/b9f671a3-0e0c-43ab-904a-c0df7e249aab)

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

![biopython](https://img.shields.io/badge/Biopython-1.81-green) ![networkx](https://img.shields.io/badge/NetworkX-3.1-yellow) ![pandas](https://img.shields.io/badge/Pandas-2.0.3-blue) ![numpy](https://img.shields.io/badge/Numpy-1.25.0-blue) ![scipy](https://img.shields.io/badge/Scipy-1.11.2-blue) ![pywt](https://img.shields.io/badge/Pywt-1.1.1-blue) ![requests](https://img.shields.io/badge/Requests-2.31.0-red) ![matplotlib](https://img.shields.io/badge/Matplotlib-3.8.0-orange) ![sklearn](https://img.shields.io/badge/scikit--learn-1.3.0-yellowgreen)
