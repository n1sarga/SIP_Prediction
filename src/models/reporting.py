from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plot(mean_df: pd.DataFrame, output: Path, metric: str) -> None:
    top = (
        mean_df.groupby(["feature_set", "model"], as_index=False)[metric]
        .mean(numeric_only=True)
        .sort_values(metric, ascending=False)
        .head(20)
    )
    if top.empty:
        return

    labels = [f"{row.feature_set}\n{row.model}" for row in top.itertuples()]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, top[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"Top Mean {metric.upper()} By Feature Set And Model")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


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


def write_summary(path: Path, results: pd.DataFrame, mean_df: pd.DataFrame, metadata: dict[str, object]) -> None:
    top_runs = results.sort_values("aupr", ascending=False).head(15)
    top_mean = mean_df.sort_values("aupr", ascending=False).head(15)

    split_drop = pd.DataFrame()
    if {"random_pair", "protein_holdout"}.issubset(set(mean_df["split"])):
        pivot = mean_df.pivot_table(
            index=["negative_set", "feature_set", "model"],
            columns="split",
            values="aupr",
            aggfunc="mean",
        ).reset_index()
        if "random_pair" in pivot and "protein_holdout" in pivot:
            pivot["protein_holdout_drop"] = pivot["random_pair"] - pivot["protein_holdout"]
        if "low_similarity_holdout" in pivot:
            pivot["low_similarity_drop"] = pivot["random_pair"] - pivot["low_similarity_holdout"]
        split_drop = pivot.sort_values("protein_holdout_drop", ascending=False, na_position="last").head(15)

    skipped = metadata.get("skipped", [])
    skipped_lines = "\n".join(f"- {item}" for item in skipped[:25]) if skipped else "- None"

    content = [
        "# Controlled PPI Experiment Summary",
        "",
        "## Setup",
        "",
        f"- Species: `{metadata['species']}`",
        f"- Feature sets: `{', '.join(metadata['feature_sets'])}`",
        f"- Models: `{', '.join(metadata['models'])}`",
        f"- Splits: `{', '.join(metadata['splits'])}`",
        f"- Seeds: `{', '.join(map(str, metadata['seeds']))}`",
        f"- Trees per tree model: `{metadata['n_trees']}`",
        f"- Low-similarity threshold: `{metadata['low_similarity_threshold']}`",
        "",
        "## Top Individual Runs By AUPR",
        "",
        markdown_table(
            top_runs[
                [
                    "negative_set",
                    "feature_set",
                    "split",
                    "model",
                    "seed",
                    "aupr",
                    "auroc",
                    "mcc",
                    "precision_at_k",
                    "false_positive_rate",
                ]
            ]
        ),
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
        "## AUPR Drop From Random Pair Split",
        "",
        markdown_table(split_drop) if not split_drop.empty else "_Not enough split coverage to compute drops._",
        "",
        "## Interpretation",
        "",
        "Compare random-pair with protein-holdout and low-similarity-holdout. Large drop means leakage or shortcut learning. Degree and similarity baselines show whether model learns biology or dataset structure.",
        "",
        "## Skipped Combinations",
        "",
        skipped_lines,
    ]
    path.write_text("\n".join(content), encoding="utf-8")
