from __future__ import annotations

import argparse
import json

import pandas as pd

from baselines import degree_features_from_train, pair_similarity_scores
from classifiers import fit_predict_model, fit_scaled_logistic
from data_io import (
    build_feature_cache,
    discover_interaction_files,
    kmer_set,
    negative_set_name,
    prepare_pair_features,
    read_interactions,
    read_sequences,
)
from experiment_config import DEFAULT_FEATURE_SETS, DEFAULT_MODELS, DEFAULT_SPLITS, FEATURE_COMPONENTS, REPORTS_DIR, ROOT, SPECIES
from metrics import evaluate_scores
from reporting import save_metric_plot, write_summary
from splits import build_splits


def run(args: argparse.Namespace) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment_dir = REPORTS_DIR / args.run_name
    figures_dir = experiment_dir / "figures"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sequences = read_sequences()
    kmer_cache = {protein: kmer_set(seq) for protein, seq in sequences.items()}
    feature_cache = build_feature_cache(args.feature_sets)
    interaction_files = discover_interaction_files(args.interaction_files)

    results: list[dict[str, object]] = []
    skipped: list[str] = []

    for interaction_file in interaction_files:
        if not interaction_file.exists():
            skipped.append(f"Missing interaction file: {interaction_file}")
            continue

        negative_name = negative_set_name(interaction_file)
        pairs_raw = read_interactions(interaction_file)
        print(f"\nDataset {negative_name}: {len(pairs_raw):,} unique unordered pairs")

        for feature_set in args.feature_sets:
            protein_features, feature_names = feature_cache[feature_set]
            if not protein_features:
                skipped.append(f"{feature_set}: missing component profiles")
                continue

            x, y, pairs = prepare_pair_features(pairs_raw, protein_features)
            if len(y) < 20 or len(set(y)) < 2:
                skipped.append(f"{negative_name}/{feature_set}: not enough labelled pairs after feature filtering")
                continue

            print(f"  Feature set {feature_set}: pairs={len(y):,}, features={x.shape[1]:,}")
            for seed in args.seeds:
                try:
                    split_defs = build_splits(
                        args.splits,
                        pairs,
                        y,
                        kmer_cache,
                        seed,
                        args.test_size,
                        args.test_protein_frac,
                        args.low_similarity_threshold,
                    )
                except RuntimeError as exc:
                    skipped.append(f"{negative_name}/{feature_set}/seed={seed}: {exc}")
                    continue

                for split_name, (train_idx, test_idx) in split_defs.items():
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    x_train = x[train_idx]
                    x_test = x[test_idx]

                    for model_name in args.models:
                        try:
                            if model_name == "similarity":
                                train_scores = pair_similarity_scores(pairs, kmer_cache, train_idx)
                                test_scores = pair_similarity_scores(pairs, kmer_cache, test_idx)
                            elif model_name == "degree":
                                degree_train = degree_features_from_train(pairs, train_idx, train_idx)
                                degree_test = degree_features_from_train(pairs, train_idx, test_idx)
                                train_scores, test_scores = fit_scaled_logistic(degree_train, y_train, degree_test, seed)
                            else:
                                train_scores, test_scores = fit_predict_model(
                                    model_name,
                                    x_train,
                                    y_train,
                                    x_test,
                                    seed,
                                    args.n_trees,
                                )

                            metrics = evaluate_scores(y_train, train_scores, y_test, test_scores)
                            results.append(
                                {
                                    "species": SPECIES,
                                    "negative_set": negative_name,
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "split": split_name,
                                    "seed": seed,
                                    "n_features": len(feature_names) * 2,
                                    "train_size": int(len(train_idx)),
                                    "test_size": int(len(test_idx)),
                                    "test_positives": int((y_test == 1).sum()),
                                    "test_negatives": int((y_test == 0).sum()),
                                    **metrics,
                                }
                            )
                        except Exception as exc:
                            skipped.append(f"{negative_name}/{feature_set}/{split_name}/{model_name}/seed={seed}: {exc}")

    if not results:
        raise RuntimeError("No experiment results were produced")

    results_df = pd.DataFrame(results)
    results_path = experiment_dir / "experiment_results.csv"
    results_df.to_csv(results_path, index=False)

    mean_df = (
        results_df.groupby(["negative_set", "feature_set", "model", "split"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["negative_set", "feature_set", "split", "model"])
    )
    mean_df.to_csv(experiment_dir / "mean_results.csv", index=False)

    metadata = {
        "species": SPECIES,
        "interaction_files": [str(path.relative_to(ROOT)) for path in interaction_files],
        "feature_sets": args.feature_sets,
        "models": args.models,
        "splits": args.splits,
        "seeds": args.seeds,
        "n_trees": args.n_trees,
        "test_size": args.test_size,
        "test_protein_frac": args.test_protein_frac,
        "low_similarity_threshold": args.low_similarity_threshold,
        "skipped": skipped,
    }
    (experiment_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(experiment_dir / "summary.md", results_df, mean_df, metadata)
    save_metric_plot(mean_df, figures_dir / "aupr_by_feature_model.png", metric="aupr")
    save_metric_plot(mean_df, figures_dir / "mcc_by_feature_model.png", metric="mcc")

    print(f"\nSaved detailed results to: {results_path}")
    print(f"Saved summary to: {experiment_dir / 'summary.md'}")
    if skipped:
        print(f"Skipped {len(skipped)} combinations. See run_metadata.json for details.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled PPI prediction experiments.")
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS, choices=sorted(FEATURE_COMPONENTS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=DEFAULT_SPLITS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47])
    parser.add_argument("--interaction-files", nargs="+", default=["auto"])
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--test-protein-frac", type=float, default=0.25)
    parser.add_argument("--low-similarity-threshold", type=float, default=0.2)
    parser.add_argument("--run-name", default="experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
