from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "results" / SPECIES


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def discover_runs(run_names: list[str]) -> list[Path]:
    if run_names:
        return [REPORTS_DIR / name for name in run_names]
    return sorted(
        path
        for path in REPORTS_DIR.glob("experiment_*")
        if path.is_dir() and (path / "experiment_results.csv").exists()
    )


def write_summary(output: Path, combined: pd.DataFrame, mean_df: pd.DataFrame) -> None:
    top_mean = mean_df.sort_values("aupr", ascending=False).head(20)
    hard = mean_df[
        (mean_df["negative_set"].isin(["random_10to1", "degree_matched_10to1"]))
        & (mean_df["split"].isin(["protein_holdout", "low_similarity_holdout"]))
    ].sort_values("aupr", ascending=False).head(20)
    if hard.empty:
        hard = mean_df[mean_df["split"].isin(["protein_holdout", "low_similarity_holdout"])].sort_values(
            "aupr",
            ascending=False,
        ).head(20)

    feature_rank = (
        mean_df.groupby("feature_set", as_index=False)[["aupr", "mcc", "precision_at_k"]]
        .mean(numeric_only=True)
        .sort_values("aupr", ascending=False)
    )
    model_rank = (
        mean_df.groupby("model", as_index=False)[["aupr", "mcc", "precision_at_k", "false_positive_rate"]]
        .mean(numeric_only=True)
        .sort_values("aupr", ascending=False)
    )

    pivot = mean_df.pivot_table(
        index=["negative_set", "feature_set", "model"],
        columns="split",
        values="aupr",
        aggfunc="mean",
    ).reset_index()
    if "random_pair" in pivot and "protein_holdout" in pivot:
        pivot["protein_holdout_drop"] = pivot["random_pair"] - pivot["protein_holdout"]
    if "random_pair" in pivot and "low_similarity_holdout" in pivot:
        pivot["low_similarity_drop"] = pivot["random_pair"] - pivot["low_similarity_holdout"]
    drops = pivot.sort_values(
        [col for col in ["protein_holdout_drop", "low_similarity_drop"] if col in pivot],
        ascending=False,
        na_position="last",
    ).head(20)

    lines = [
        "# Combined Controlled Experiment Summary",
        "",
        "## Scope",
        "",
        f"- Species: `{SPECIES}`",
        f"- Runs combined: `{combined['source_run'].nunique()}`",
        f"- Result rows: `{len(combined)}`",
        f"- Negative sets: `{', '.join(sorted(combined['negative_set'].unique()))}`",
        f"- Feature sets: `{', '.join(sorted(combined['feature_set'].unique()))}`",
        f"- Models: `{', '.join(sorted(combined['model'].unique()))}`",
        f"- Splits: `{', '.join(sorted(combined['split'].unique()))}`",
        "",
        "## Top Mean Runs By AUPR",
        "",
        markdown_table(
            top_mean[
                [
                    "negative_set",
                    "feature_set",
                    "split",
                    "model",
                    "aupr",
                    "auroc",
                    "mcc",
                    "precision_at_k",
                    "false_positive_rate",
                ]
            ]
        ),
        "",
        "## Best Hard-Setting Runs",
        "",
        markdown_table(
            hard[
                [
                    "negative_set",
                    "feature_set",
                    "split",
                    "model",
                    "aupr",
                    "auroc",
                    "mcc",
                    "precision_at_k",
                    "false_positive_rate",
                ]
            ]
        ),
        "",
        "## Mean Feature-Set Ranking",
        "",
        markdown_table(feature_rank),
        "",
        "## Mean Model Ranking",
        "",
        markdown_table(model_rank),
        "",
        "## Largest AUPR Drops From Random-Pair Split",
        "",
        markdown_table(drops),
        "",
        "## Reading The Result",
        "",
        "Hard-setting table most paper-relevant: class imbalance or matched negatives plus protein-level holdout. Strong only in random_pair but weak under holdout = leakage or shortcut learning.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine controlled PPI experiment chunks.")
    parser.add_argument("--run-names", nargs="*", default=[])
    parser.add_argument("--output-name", default="controlled_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args.run_names)
    if not run_dirs:
        raise RuntimeError(f"No experiment run directories found under {REPORTS_DIR}")

    frames = []
    for run_dir in run_dirs:
        path = run_dir / "experiment_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_run"] = run_dir.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_dir = REPORTS_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "combined_results.csv", index=False)

    mean_df = (
        combined.groupby(["negative_set", "feature_set", "model", "split"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["negative_set", "feature_set", "split", "model"])
    )
    mean_df.to_csv(output_dir / "mean_results.csv", index=False)
    write_summary(output_dir / "summary.md", combined, mean_df)

    print(f"Combined {len(run_dirs)} runs into: {output_dir}")
    print(f"Rows: {len(combined)}")


if __name__ == "__main__":
    main()
