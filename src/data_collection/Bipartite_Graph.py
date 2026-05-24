"""
Generate negative protein interactions from the positive interaction graph.
"""

from pathlib import Path
import itertools
import os

import networkx as nx
import pandas as pd


SPECIES = os.getenv("SPECIES", "Diabates")
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / SPECIES


def main() -> None:
    input_path = PROCESSED_DIR / f"{SPECIES}_Cleaned.csv"
    proteins_path = PROCESSED_DIR / "Unique_Proteins.csv"
    output_path = PROCESSED_DIR / f"{SPECIES}_All.csv"

    df = pd.read_csv(input_path)
    if proteins_path.exists():
        available_proteins = set(pd.read_csv(proteins_path)["Protein Identifier"])
        before_count = len(df)
        df = df[
            df["Identifier A"].isin(available_proteins)
            & df["Identifier B"].isin(available_proteins)
        ].copy()
        print(f"Filtered interactions to proteins with sequences: {before_count} -> {len(df)}")

    positive_interactions = df[df["Interaction"] == 1]

    graph = nx.Graph()
    graph.add_edges_from(positive_interactions[["Identifier A", "Identifier B"]].values)

    all_nodes = set(df["Identifier A"]).union(set(df["Identifier B"]))
    all_edges = set(itertools.combinations(all_nodes, 2))
    negative_interactions = all_edges - set(graph.edges)

    negative_df = pd.DataFrame(list(negative_interactions), columns=["Identifier A", "Identifier B"])
    negative_df["Interaction"] = 0
    sampled_negative_df = negative_df.sample(n=len(positive_interactions), random_state=42)

    all_interactions = pd.concat([positive_interactions, sampled_negative_df], ignore_index=True)
    all_interactions.to_csv(output_path, index=False)
    print(f"Saved balanced interaction dataset to: {output_path}")


if __name__ == "__main__":
    main()
