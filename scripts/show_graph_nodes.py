#!/usr/bin/env python3

import json
import sys
import os

# Add the parent directory to the path so we can import artifact_graph
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifact_graph.utils.graph_builder import (
    load_artifact_graph_from_json,
    MODEL_NODE,
    DATASET_NODE,
)


def main():
    metric_name = "accuracy"  # Define the metric to filter by
    graph_file = "output/perfect_model_dataset_metrics.json"
    
    print(f"Loading graph from: {graph_file}")
    print(f"Using metric: {metric_name}")
    print("-" * 60)
    
    # Load the graph from JSON file
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )
    
    # Separate model and dataset nodes
    model_nodes = []
    dataset_nodes = []
    
    for node, attrs in G.nodes(data=True):
        if attrs.get('type') == MODEL_NODE:
            model_nodes.append((node, attrs))
        elif attrs.get('type') == DATASET_NODE:
            dataset_nodes.append((node, attrs))
    
    # Sort nodes by downloads (descending)
    model_nodes.sort(key=lambda x: x[1].get('downloads', 0), reverse=True)
    dataset_nodes.sort(key=lambda x: x[1].get('downloads', 0), reverse=True)
    
    print(f"\n📊 GRAPH SUMMARY:")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Model nodes: {len(model_nodes)}")
    print(f"Dataset nodes: {len(dataset_nodes)}")
    
if __name__ == "__main__":
    main()
