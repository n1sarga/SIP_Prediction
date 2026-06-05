# Project Structure Notes

- `src/`: executable pipeline scripts
- `data/raw/`: original datasets
- `data/processed/`: cleaned interaction and protein tables
- `data/external/`: third-party resources such as the BLAST database
- `artifacts/features/`: generated feature vectors and per-protein profiles
- `artifacts/fusion/`: fused feature sets
- `reports/results/`: evaluation metrics, plots, and controlled experiment summaries

Fusion reads the per-protein PSSM profiles and computes the mean PSSM vector directly while building each fusion output.

`src/data_collection/prepare_labeled_ppi_dataset.py` prepares labelled raw PPI data, for example Yeast raw positives/negatives filtered to proteins with generated feature profiles.

`src/models/run_controlled_experiment.py` runs controlled comparisons across feature sets, negative sets, model families, and leakage-aware splits. Model support code is split into `data_io.py`, `splits.py`, `metrics.py`, `classifiers.py`, `baselines.py`, `rotation_forest.py`, and `reporting.py`. `src/models/summarize_experiment_runs.py` combines chunked experiment runs into one report.
