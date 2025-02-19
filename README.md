# Predicting Self-Interacting Proteins using Rotation Forest Classifier

This is my university's undergraduate thesis. My thesis members were [Sofia Noor Rafa](https://github.com/sofiautilitarian) and Ummay Khadiza Rumpa.
In this project, we have implemented and evaluated the Rotation Forest Classifier for predicting interactions in Maize and Yeast using multiple feature extraction techniques. The implementation was taken from [Liam-E2](https://github.com/Liam-E2/RotationForest).

## Workflow:
1. Protein-protein interactions data and corresponding protein sequences collection.
2. Generate negative interactions using a bipartite graph.
3. Generate PSSM profiles for the sequences.
4. Generate feature embeddings using image transformation techniques: Discrete Sine Transformation (DST), Discrete Cosine Transformation (DCT), Wavelet Transformation (WT), Discrete Hilbert Transformation (DHT), and Fast Fourier Transformation (FFT).
5. Train the Rotation Forest Model using the feature sets.
6. Evaluate the models using RoC-AUC and Classification Results.

![System Architecture](https://github.com/user-attachments/assets/b9f671a3-0e0c-43ab-904a-c0df7e249aab)

To execute the project download the following:
1. Download and Install the standalone version of NCBI Blast+ using this link: [NCBI Blast+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
2. Download the Swiss-Prot Database using this link: [Swiss-Prot DB](https://ftp.ncbi.nlm.nih.gov/blast/db/)

## Packages Used:
![biopython](https://img.shields.io/badge/Biopython-1.81-green) ![networkx](https://img.shields.io/badge/NetworkX-3.1-yellow) ![pandas](https://img.shields.io/badge/Pandas-2.0.3-blue) ![numpy](https://img.shields.io/badge/Numpy-1.25.0-blue) ![scipy](https://img.shields.io/badge/Scipy-1.11.2-blue) ![pywt](https://img.shields.io/badge/Pywt-1.1.1-blue) ![requests](https://img.shields.io/badge/Requests-2.31.0-red) ![matplotlib](https://img.shields.io/badge/Matplotlib-3.8.0-orange) ![sklearn](https://img.shields.io/badge/scikit--learn-1.3.0-yellowgreen)
