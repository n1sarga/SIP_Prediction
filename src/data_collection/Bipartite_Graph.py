"""
Generate negative protein interactions from the positive interaction graph.
"""

from pathlib import Path
import itertools

import networkx as nx
import pandas as pd


SPECIES = "Diabates"
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / SPECIES


def main() -> None:
    input_path = PROCESSED_DIR / f"{SPECIES}_Cleaned.csv"
    output_path = PROCESSED_DIR / f"{SPECIES}_All.csv"

    df = pd.read_csv(input_path)
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
