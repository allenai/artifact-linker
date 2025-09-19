#!/usr/bin/env python3
"""
Visualize model-dataset relationship graph from perfect_model_dataset_metrics.json

Usage:
    python visualize_model_dataset_graph.py [json_file_path] [output_file_path]
    python visualize_model_dataset_graph.py --help

Examples:
    # Basic usage with default filtering
    python visualize_model_dataset_graph.py perfect_model_dataset_metrics.json graph.html

    # No filtering (show all nodes)
    python visualize_model_dataset_graph.py perfect_model_dataset_metrics.json graph.html --no-filter

    # Custom filtering
    python visualize_model_dataset_graph.py perfect_model_dataset_metrics.json graph.html --min-downloads 5000 --max-nodes 200
"""

import argparse
import json
import os
import sys

import networkx as nx

# Add parent directory to path to import graph_visualizer
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from artifact_graph.utils.graph_visualizer import (
    DATASET,
    MODEL,
    visualize_graph_interactive,
)


def load_model_dataset_data(json_path):
    """Load JSON data"""
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("results", [])


def create_model_dataset_graph(results, min_downloads=None, max_nodes=None):
    """
    Create model-dataset bipartite graph from results data

    Args:
        results: List of JSON results
        min_downloads: Minimum download threshold for filtering
        max_nodes: Maximum number of nodes limit (for large datasets)
    """
    G = nx.Graph()

    # Statistics
    model_downloads = {}
    dataset_downloads = {}
    connections = []

    # Collect all connections and download data
    for result in results:
        model_id = result.get("model_id", "")
        dataset_id = result.get("dataset_id", "")
        model_dl = result.get("model_downloads", 0)
        dataset_dl = result.get("dataset_downloads", 0)

        if model_id and dataset_id:
            # Record download counts
            model_downloads[model_id] = model_dl
            dataset_downloads[dataset_id] = dataset_dl

            # Record connections
            connections.append((model_id, dataset_id, model_dl, dataset_dl))

    # Apply filtering conditions
    if min_downloads:
        connections = [
            (m, d, mdl, ddl)
            for m, d, mdl, ddl in connections
            if mdl >= min_downloads or ddl >= min_downloads
        ]

    # Limit number of nodes (select highest download counts)
    if max_nodes and len(connections) > max_nodes:
        # Sort by total download count
        connections.sort(key=lambda x: x[2] + x[3], reverse=True)
        connections = connections[:max_nodes]

    # Build graph
    models_added = set()
    datasets_added = set()

    for model_id, dataset_id, model_dl, dataset_dl in connections:
        # Add model node
        if model_id not in models_added:
            G.add_node(model_id, type=MODEL, downloads=model_dl)
            models_added.add(model_id)

        # Add dataset node
        if dataset_id not in datasets_added:
            G.add_node(dataset_id, type=DATASET, downloads=dataset_dl)
            datasets_added.add(dataset_id)

        # Add edge
        G.add_edge(model_id, dataset_id)

    print("📊 Graph statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  - Models: {len(models_added)}")
    print(f"  - Datasets: {len(datasets_added)}")
    print(f"  Total edges: {G.number_of_edges()}")

    return G


def extract_largest_component(G):
    """
    Extract the largest connected component from the graph

    Args:
        G: NetworkX graph

    Returns:
        NetworkX graph containing only the largest connected component
    """
    if G.number_of_nodes() == 0:
        return G

    # Get all connected components
    connected_components = list(nx.connected_components(G))

    if not connected_components:
        return G

    # Find the largest component
    largest_component = max(connected_components, key=len)

    # Create subgraph with only the largest component
    largest_subgraph = G.subgraph(largest_component).copy()

    # Print component statistics
    print("🔗 Connected component analysis:")
    print(f"  Total components: {len(connected_components)}")
    print(f"  Largest component size: {len(largest_component)} nodes")

    if len(connected_components) > 1:
        component_sizes = sorted([len(cc) for cc in connected_components], reverse=True)
        print(f"  Component sizes: {component_sizes[:10]}...")  # Show top 10

        # Calculate what percentage of nodes are in the largest component
        percentage = (len(largest_component) / G.number_of_nodes()) * 100
        print(f"  Largest component coverage: {percentage:.1f}% of all nodes")

    # Show node type breakdown for largest component
    model_nodes = [n for n in largest_component if G.nodes[n].get("type") == MODEL]
    dataset_nodes = [n for n in largest_component if G.nodes[n].get("type") == DATASET]

    print("  Largest component composition:")
    print(f"  - Models: {len(model_nodes)}")
    print(f"  - Datasets: {len(dataset_nodes)}")
    print(f"  - Edges: {largest_subgraph.number_of_edges()}")

    return largest_subgraph


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model-dataset relationship graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s data.json output.html

  # Only visualize the largest connected component
  %(prog)s data.json output.html --largest-component

  # Custom filtering with largest component
  %(prog)s data.json output.html --min-downloads 5000 --largest-component

  # No filtering but only largest component
  %(prog)s data.json output.html --no-filter --largest-component
        """,
    )

    parser.add_argument(
        "json_path",
        nargs="?",
        default="perfect_model_dataset_metrics.json",
        help="Path to JSON data file (default: perfect_model_dataset_metrics.json)",
    )

    parser.add_argument(
        "output_file",
        nargs="?",
        default="model_dataset_graph.html",
        help="Output HTML file path (default: model_dataset_graph.html)",
    )

    parser.add_argument(
        "--min-downloads", type=int, help="Minimum download count threshold for filtering"
    )

    parser.add_argument("--max-nodes", type=int, help="Maximum number of nodes to display")

    parser.add_argument(
        "--no-filter", action="store_true", help="Disable automatic filtering for large datasets"
    )

    parser.add_argument(
        "--largest-component",
        action="store_true",
        help="Only visualize the largest connected component",
    )

    args = parser.parse_args()

    # Convert to absolute paths
    json_path = args.json_path
    output_file = args.output_file

    if not os.path.isabs(json_path):
        json_path = os.path.join(os.getcwd(), json_path)
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.getcwd(), output_file)

    print(f"📁 Loading data: {json_path}")
    print(f"📄 Output file: {output_file}")
    print("=" * 80)

    # Load data
    results = load_model_dataset_data(json_path)
    if not results:
        print("❌ Failed to load data")
        return

    print(f"📊 Total results: {len(results)}")

    # Determine filtering parameters
    min_downloads = args.min_downloads
    max_nodes = args.max_nodes

    # Apply automatic filtering for large datasets if not disabled
    if not args.no_filter and len(results) > 1000:
        if min_downloads is None:
            min_downloads = 1000
        if max_nodes is None:
            max_nodes = 50000
        print("⚠️  Dataset is large, applying automatic filtering...")
        print(f"   - Minimum downloads: {min_downloads}")
        print(f"   - Maximum nodes: {max_nodes}")
        print("   - Use --no-filter to disable automatic filtering")
    elif args.no_filter:
        print("🔓 Filtering disabled - showing all nodes")

    if min_downloads or max_nodes:
        print("🔽 Filtering enabled:")
        if min_downloads:
            print(f"   - Minimum downloads: {min_downloads}")
        if max_nodes:
            print(f"   - Maximum nodes: {max_nodes}")

    if args.largest_component:
        print("🔗 Largest component mode enabled")

    # Create graph
    G = create_model_dataset_graph(results, min_downloads, max_nodes)

    if G.number_of_nodes() == 0:
        print("❌ No nodes to visualize")
        return

    # Extract largest component if requested
    if args.largest_component:
        print()  # Add blank line for better formatting
        G = extract_largest_component(G)

        if G.number_of_nodes() == 0:
            print("❌ No nodes in largest component to visualize")
            return

    # Visualize
    print("\n🎨 Generating visualization...")
    visualize_graph_interactive(G, output_file)

    if args.largest_component:
        print("\n✅ Visualization complete! Showing largest connected component.")
        print(f"   Open in browser: {output_file}")
    else:
        print(f"\n✅ Visualization complete! Please open in browser: {output_file}")


if __name__ == "__main__":
    main()
