#!/usr/bin/env python3
"""
Generates a static visualization of the artifact graph, colored by community.
"""

import json
from pathlib import Path
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community as nx_comm
import numpy as np

def visualize_static_graph(graph_data_dir: Path, output_path: Path):
    """
    Loads the graph, detects communities, and saves a static visualization.
    """
    node_metadata_path = graph_data_dir / "node_metadata.json"
    edge_metadata_path = graph_data_dir / "edge_metadata.json"

    if not node_metadata_path.exists() or not edge_metadata_path.exists():
        print(f"Error: Metadata files not found in {graph_data_dir}")
        return

    print("Loading graph from metadata...")
    try:
        G = nx.Graph()
        with node_metadata_path.open("r", encoding="utf-8") as f:
            node_metadata = json.load(f)
            for node_id_str, meta in node_metadata.items():
                node_id = int(node_id_str)
                G.add_node(node_id, type=meta.get("type", "unknown"))

        with edge_metadata_path.open("r", encoding="utf-8") as f:
            edge_metadata = json.load(f)
            for edge_key in edge_metadata.keys():
                try:
                    u_str, v_str = edge_key.strip("()").split(",")
                    G.add_edge(int(u_str), int(v_str))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Detecting communities using the Louvain method...")
    communities = nx_comm.louvain_communities(G, seed=42)
    
    # Create a color map from community id to color
    # Using a colormap to get a wide range of distinct colors
    community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    colors = [community_map[node] for node in G.nodes()]
    cmap = plt.get_cmap('viridis', max(colors) + 1)

    # Assign node sizes
    node_sizes = [200 if G.nodes[node]['type'] == 'dataset' else 30 for node in G.nodes()]
    
    print("Generating layout... (This may take a while for large graphs)")
    # Use a spring layout - it's good for showing community structure
    pos = nx.spring_layout(G, seed=42, iterations=50, k=0.1)

    print("Drawing graph...")
    plt.figure(figsize=(20, 20))
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, cmap=cmap, alpha=0.7)
    
    plt.box(False)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Static graph visualization saved to: {output_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Generate a static visualization of the artifact graph.")
    p.add_argument(
        "--graph-data-dir",
        type=Path,
        default="scripts/output/artifact_graph_data",
        help="Path to the directory containing node and edge metadata.",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default="scripts/output/artifact_graph_visualization.png",
        help="Path to save the output PNG image.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    visualize_static_graph(a.graph_data_dir, a.output_path)
