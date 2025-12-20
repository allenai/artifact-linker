#!/usr/bin/env python3
"""
Analyzes the community structure of the artifact graph using the Louvain method.
"""

import json
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
from networkx.algorithms.cuts import conductance


def analyze_communities(graph_data_dir: Path):
    """
    Loads the graph, detects communities, and calculates connectivity metrics.
    """
    node_metadata_path = graph_data_dir / "node_metadata.json"
    edge_metadata_path = graph_data_dir / "edge_metadata.json"

    if not node_metadata_path.exists() or not edge_metadata_path.exists():
        print(f"Error: Metadata files not found in {graph_data_dir}")
        return

    print("Loading graph from metadata... (This may take a moment)")
    try:
        # Build the graph using networkx
        G = nx.Graph()
        with node_metadata_path.open("r", encoding="utf-8") as f:
            node_metadata = json.load(f)
            for node_id, meta in node_metadata.items():
                G.add_node(int(node_id), type=meta.get("type", "unknown"))

        with edge_metadata_path.open("r", encoding="utf-8") as f:
            edge_metadata = json.load(f)
            for edge_key in edge_metadata.keys():
                try:
                    u_str, v_str = edge_key.strip("()").split(",")
                    u, v = int(u_str), int(v_str)
                    G.add_edge(u, v)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 1. Detect communities using the Louvain method
    print("\nDetecting communities using the Louvain method...")
    communities = nx_comm.louvain_communities(G)
    num_communities = len(communities)
    print(f"Found {num_communities} communities.")

    # 2. Calculate Modularity
    modularity = nx_comm.modularity(G, communities)

    # 3. Calculate Conductance for each community
    conductances = []
    community_sizes = []
    for comm in communities:
        conductance_val = conductance(G, comm)
        conductances.append(conductance_val)
        community_sizes.append(len(comm))

    # 4. Print results
    print("\n--- Graph Community Analysis ---")
    print(f"  Number of Communities: {num_communities}")
    print(f"  Modularity:            {modularity:.4f} (Higher is better)")

    if community_sizes:
        community_sizes_np = np.array(community_sizes)
        print("\n  Community Size Distribution:")
        print(f"    - Mean:     {community_sizes_np.mean():.2f} nodes")
        print(f"    - Median:   {np.median(community_sizes_np):.2f} nodes")
        print(f"    - Min Size: {community_sizes_np.min()}")
        print(f"    - Max Size: {community_sizes_np.max()}")
    
    if conductances:
        conductances_np = np.array(conductances)
        print("\n  Community Conductance Distribution:")
        print("    (Lower is better, indicating more well-defined communities)")
        print(f"    - Mean:     {conductances_np.mean():.4f}")
        print(f"    - Median:   {np.median(conductances_np):.4f}")
        print(f"    - Min:      {conductances_np.min():.4f}")
        print(f"    - Max:      {conductances_np.max():.4f}")

    # --- New Metric Calculation ---
    print("\n--- Detailed Dataset Degree Analysis ---")
    
    # Get all dataset nodes and their degrees
    dataset_degrees = [d for n, d in G.degree() if G.nodes[n].get('type') == 'dataset']
    
    if dataset_degrees:
        dataset_degrees_np = np.array(dataset_degrees)
        total_edges = G.number_of_edges()
        
        # 1. Median Dataset Degree
        median_degree = np.median(dataset_degrees_np)
        mean_degree = dataset_degrees_np.mean()
        print(f"  Median Dataset Degree: {median_degree:.2f} (vs. Mean {mean_degree:.2f})")
        
        # 2. Top 5 Datasets Edge Share
        # Sort degrees in descending order
        sorted_degrees = np.sort(dataset_degrees_np)[::-1]
        top_5_sum = sorted_degrees[:5].sum()
        # Note: In a bipartite graph (or similar), summing node degrees counts each edge once for that set of nodes.
        # Assuming edges are between model and dataset, total dataset degree sum = total edges.
        # If graph has other edges, we might need to be more careful, but usually sum(degrees)/2 = total_edges.
        # For bipartite dataset-model graph: sum(dataset_degrees) == total_edges
        
        # Calculate share relative to total edges in the graph
        # Check if graph is purely bipartite or close to it to interpret "share of all edges"
        # Simply: (Sum of degrees of top 5 datasets) / (Total edges in graph)
        if total_edges > 0:
            top_5_share = (top_5_sum / total_edges) * 100
            print(f"  Top 5 Datasets Edge Share: {top_5_share:.2f}% ({top_5_sum} / {total_edges} edges)")
        
        # 3. Percentage of Datasets with Degree >= 3
        num_ge_3 = (dataset_degrees_np >= 3).sum()
        pct_ge_3 = (num_ge_3 / len(dataset_degrees_np)) * 100
        print(f"  Datasets with Degree >= 3: {pct_ge_3:.2f}% ({num_ge_3}/{len(dataset_degrees_np)})")
    else:
        print("  No dataset nodes found.")
        
    print("\n----------------------------------")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze graph community structure.")
    p.add_argument(
        "--graph-data-dir",
        type=Path,
        default="scripts/output/artifact_graph_data",
        help="Path to the directory containing node and edge metadata.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    analyze_communities(a.graph_data_dir)
