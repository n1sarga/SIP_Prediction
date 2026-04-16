# Project Structure Notes

## Canonical layout

- `src/`: executable pipeline scripts
- `data/raw/`: original datasets
- `data/processed/`: cleaned interaction and protein tables
- `data/external/`: third-party resources such as the BLAST database
- `artifacts/features/`: generated feature vectors and per-protein profiles
- `artifacts/fusion/`: fused feature sets
- `reports/results/`: evaluation metrics and plots

## Notes

Fusion reads the per-protein PSSM profiles and computes the mean PSSM vector directly while building each fusion output, so no intermediate engineered-PSSM folder is stored on disk.
