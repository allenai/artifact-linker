#!/usr/bin/env python3
"""
Calculates and reports degree statistics for model and dataset nodes in the artifact graph.
"""

import json
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np


def calculate_node_degrees(graph_data_dir: Path):
    """
    Calculates degree statistics for different node types.
    """
    node_metadata_path = graph_data_dir / "node_metadata.json"
    edge_metadata_path = graph_data_dir / "edge_metadata.json"

    if not node_metadata_path.exists() or not edge_metadata_path.exists():
        print(f"Error: Metadata files not found in {graph_data_dir}")
        return

    print("Loading metadata files... (This may take a moment)")
    try:
        with node_metadata_path.open("r", encoding="utf-8") as f:
            node_metadata = json.load(f)
        with edge_metadata_path.open("r", encoding="utf-8") as f:
            edge_metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata files: {e}")
        return

    # 1. Create a map from node ID to node type
    node_id_to_type = {}
    for node_id_str, meta in node_metadata.items():
        node_id = int(node_id_str)
        node_type = meta.get("type")
        if node_type in ["model", "dataset"]:
            node_id_to_type[node_id] = node_type

    # 2. Calculate degrees for all nodes
    degrees = defaultdict(int)
    for edge_key in edge_metadata.keys():
        # Key is a string like '(u, v)', needs parsing.
        try:
            u_str, v_str = edge_key.strip("()").split(",")
            u, v = int(u_str), int(v_str)
            degrees[u] += 1
            degrees[v] += 1
        except ValueError:
            # Handle potential parsing errors for malformed keys
            continue

    # 3. Separate degrees by node type
    model_degrees = []
    dataset_degrees = []
    for node_id, node_type in node_id_to_type.items():
        degree = degrees.get(node_id, 0)  # Use .get for nodes that might have no edges
        if node_type == "model":
            model_degrees.append(degree)
        elif node_type == "dataset":
            dataset_degrees.append(degree)

    # 4. Calculate and print statistics
    print("\n--- Node Degree Statistics ---")
    if model_degrees:
        print_stats("Model Nodes", np.array(model_degrees))
    else:
        print("No model nodes found.")

    if dataset_degrees:
        print_stats("Dataset Nodes", np.array(dataset_degrees))
    else:
        print("No dataset nodes found.")
    print("------------------------------")


def print_stats(name: str, degree_array: np.ndarray):
    """Prints a formatted block of statistics for a degree array."""
    print(f"\n  {name}:")
    print(f"    - Count:          {len(degree_array)}")
    print(f"    - Mean Degree:    {degree_array.mean():.2f}")
    print(f"    - Median Degree:  {np.median(degree_array):.2f}")
    print(f"    - Std Dev:        {degree_array.std():.2f}")
    print(f"    - Min Degree:     {degree_array.min()}")
    print(f"    - Max Degree:     {degree_array.max()}")
    print(f"    - Nodes with 0 degree: {np.sum(degree_array == 0)}")


def parse_args():
    p = argparse.ArgumentParser(description="Calculate node degree statistics.")
    p.add_argument(
        "--graph-data-dir",
        type=Path,
        default="scripts/output/artifact_graph_data",
        help="Path to the directory containing node and edge metadata.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    calculate_node_degrees(a.graph_data_dir)
