#!/usr/bin/env python3
"""
Calculates the sparsity of the artifact graph.
"""

import json
from pathlib import Path
import argparse


def calculate_sparsity(graph_data_dir: Path):
    """
    Calculates the sparsity of the graph based on node and edge metadata.

    Sparsity = Number of Edges / Max Possible Edges
    """
    node_metadata_path = graph_data_dir / "node_metadata.json"
    edge_metadata_path = graph_data_dir / "edge_metadata.json"

    if not node_metadata_path.exists() or not edge_metadata_path.exists():
        print(f"Error: Metadata files not found in {graph_data_dir}")
        return

    try:
        # Count nodes
        with node_metadata_path.open("r", encoding="utf-8") as f:
            node_metadata = json.load(f)
        num_nodes = len(node_metadata)

        # Count edges
        with edge_metadata_path.open("r", encoding="utf-8") as f:
            edge_metadata = json.load(f)
        num_edges = len(edge_metadata)

    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return
    except MemoryError:
        print("Error: One of the metadata files is too large to fit in memory.")
        print("Alternative methods would be needed for very large graphs.")
        return

    if num_nodes < 2:
        sparsity = 0.0
    else:
        # Assuming an undirected graph where max edges = n * (n - 1) / 2
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        sparsity = num_edges / max_possible_edges if max_possible_edges > 0 else 0

    print("--- Artifact Graph Sparsity ---")
    print(f"  Number of Nodes (N):    {num_nodes}")
    print(f"  Number of Edges (M):    {num_edges}")
    print(f"  Sparsity:                 {sparsity:.8f}")
    print(f"  (This means the graph has {sparsity:.6%} of the maximum possible edges)")
    print("---------------------------------")


def parse_args():
    p = argparse.ArgumentParser(description="Calculate graph sparsity.")
    p.add_argument(
        "--graph-data-dir",
        type=Path,
        default="scripts/output/artifact_graph_data",
        help="Path to the directory containing node and edge metadata.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    calculate_sparsity(a.graph_data_dir)
