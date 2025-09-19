#!/usr/bin/env python3
"""
Graph Structure Analysis for perfect_model_dataset_metrics.json

Analyzes model-dataset relationships as a bipartite graph and computes:
- Basic graph statistics (nodes, edges)
- Centrality measures
- Connectivity metrics
- Network topology analysis
"""

import argparse
import json
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np


def load_graph_data(json_path):
    """Load JSON data and create bipartite graph"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Create bipartite graph
    G = nx.Graph()

    # Statistics tracking
    stats = {
        "total_entries": len(results),
        "model_downloads": {},
        "dataset_downloads": {},
        "metrics_count": Counter(),
    }

    # Build graph
    for result in results:
        model_id = result.get("model_id", "")
        dataset_id = result.get("dataset_id", "")
        model_dl = result.get("model_downloads", 0)
        dataset_dl = result.get("dataset_downloads", 0)
        metrics = result.get("metrics", {})

        if model_id and dataset_id:
            # Add nodes with attributes
            G.add_node(model_id, bipartite=0, type="model", downloads=model_dl)
            G.add_node(dataset_id, bipartite=1, type="dataset", downloads=dataset_dl)

            # Add edge with metrics as attributes
            G.add_edge(model_id, dataset_id, metrics=metrics)

            # Track statistics
            stats["model_downloads"][model_id] = model_dl
            stats["dataset_downloads"][dataset_id] = dataset_dl

            # Count metric types
            for metric_name in metrics.keys():
                stats["metrics_count"][metric_name] += 1

    return G, stats


def analyze_basic_structure(G, stats):
    """Analyze basic graph structure"""
    analysis = {}

    # Basic counts
    analysis["total_nodes"] = G.number_of_nodes()
    analysis["total_edges"] = G.number_of_edges()

    # Separate model and dataset nodes
    model_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "dataset"]

    analysis["model_nodes"] = len(model_nodes)
    analysis["dataset_nodes"] = len(dataset_nodes)
    analysis["model_dataset_ratio"] = len(model_nodes) / len(dataset_nodes) if dataset_nodes else 0

    # Edge density
    max_possible_edges = len(model_nodes) * len(dataset_nodes)
    analysis["edge_density"] = (
        G.number_of_edges() / max_possible_edges if max_possible_edges > 0 else 0
    )

    # Degree statistics
    degrees = [G.degree(n) for n in G.nodes()]
    analysis["avg_degree"] = np.mean(degrees) if degrees else 0
    analysis["max_degree"] = max(degrees) if degrees else 0
    analysis["min_degree"] = min(degrees) if degrees else 0
    analysis["degree_std"] = np.std(degrees) if degrees else 0

    # Model degrees (how many datasets each model connects to)
    model_degrees = [G.degree(n) for n in model_nodes]
    analysis["avg_model_degree"] = np.mean(model_degrees) if model_degrees else 0
    analysis["max_model_degree"] = max(model_degrees) if model_degrees else 0

    # Dataset degrees (how many models each dataset connects to)
    dataset_degrees = [G.degree(n) for n in dataset_nodes]
    analysis["avg_dataset_degree"] = np.mean(dataset_degrees) if dataset_degrees else 0
    analysis["max_dataset_degree"] = max(dataset_degrees) if dataset_degrees else 0

    return analysis


def analyze_connectivity(G):
    """Analyze graph connectivity"""
    analysis = {}

    # Connected components
    connected_components = list(nx.connected_components(G))
    analysis["num_connected_components"] = len(connected_components)
    analysis["largest_component_size"] = (
        len(max(connected_components, key=len)) if connected_components else 0
    )
    analysis["component_sizes"] = sorted([len(cc) for cc in connected_components], reverse=True)

    # Graph connectivity measures
    analysis["is_connected"] = nx.is_connected(G)
    analysis["is_bipartite"] = nx.is_bipartite(G)

    if analysis["is_connected"]:
        analysis["diameter"] = nx.diameter(G)
        analysis["radius"] = nx.radius(G)
        analysis["average_shortest_path_length"] = nx.average_shortest_path_length(G)
    else:
        # Analyze largest component
        largest_cc = max(connected_components, key=len)
        subgraph = G.subgraph(largest_cc)
        analysis["largest_component_diameter"] = nx.diameter(subgraph)
        analysis["largest_component_radius"] = nx.radius(subgraph)
        analysis["largest_component_avg_path_length"] = nx.average_shortest_path_length(subgraph)

    return analysis


def analyze_node_importance(G):
    """Analyze most important nodes"""
    analysis = {}

    # Most connected models and datasets
    model_nodes = [(n, G.degree(n)) for n, d in G.nodes(data=True) if d.get("type") == "model"]
    dataset_nodes = [(n, G.degree(n)) for n, d in G.nodes(data=True) if d.get("type") == "dataset"]

    analysis["top_connected_models"] = sorted(model_nodes, key=lambda x: x[1], reverse=True)[:10]
    analysis["top_connected_datasets"] = sorted(dataset_nodes, key=lambda x: x[1], reverse=True)[
        :10
    ]

    # Nodes with highest downloads
    model_downloads = [
        (n, d.get("downloads", 0)) for n, d in G.nodes(data=True) if d.get("type") == "model"
    ]
    dataset_downloads = [
        (n, d.get("downloads", 0)) for n, d in G.nodes(data=True) if d.get("type") == "dataset"
    ]

    analysis["top_downloaded_models"] = sorted(model_downloads, key=lambda x: x[1], reverse=True)[
        :10
    ]
    analysis["top_downloaded_datasets"] = sorted(
        dataset_downloads, key=lambda x: x[1], reverse=True
    )[:10]

    return analysis


def analyze_centrality(G):
    """Analyze centrality measures"""
    print("Computing centrality measures...")

    analysis = {}

    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    analysis["top_degree_centrality"] = sorted(
        degree_centrality.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    analysis["top_betweenness_centrality"] = sorted(
        betweenness_centrality.items(), key=lambda x: x[1], reverse=True
    )[:10]

    return analysis


def print_summary(basic, connectivity, importance):
    """Print analysis summary"""
    print("\n" + "=" * 80)
    print("📊 GRAPH STRUCTURE ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n🔢 Basic Statistics:")
    print(f"  Total Nodes: {basic['total_nodes']:,}")
    print(f"  - Model Nodes: {basic['model_nodes']:,}")
    print(f"  - Dataset Nodes: {basic['dataset_nodes']:,}")
    print(f"  Total Edges: {basic['total_edges']:,}")
    print(f"  Model/Dataset Ratio: {basic['model_dataset_ratio']:.2f}")
    print(f"  Edge Density: {basic['edge_density']:.4f}")

    print("\n📈 Degree Statistics:")
    print(f"  Average Degree: {basic['avg_degree']:.2f}")
    print(f"  Max Degree: {basic['max_degree']:,}")
    print(f"  Average Model Connections: {basic['avg_model_degree']:.2f}")
    print(f"  Average Dataset Connections: {basic['avg_dataset_degree']:.2f}")
    print(f"  Most Connected Model: {basic['max_model_degree']} datasets")
    print(f"  Most Connected Dataset: {basic['max_dataset_degree']} models")

    print("\n🔗 Connectivity:")
    print(f"  Is Connected: {connectivity['is_connected']}")
    print(f"  Connected Components: {connectivity['num_connected_components']:,}")
    print(f"  Largest Component: {connectivity['largest_component_size']:,} nodes")
    print(f"  Is Bipartite: {connectivity['is_bipartite']}")

    if connectivity["is_connected"]:
        print(f"  Graph Diameter: {connectivity['diameter']}")
        print(f"  Average Path Length: {connectivity['average_shortest_path_length']:.2f}")
    else:
        print(f"  Largest Component Diameter: {connectivity['largest_component_diameter']}")
        print(
            f"  Largest Component Avg Path: {connectivity['largest_component_avg_path_length']:.2f}"
        )

    # Top connected nodes
    print("\n🏆 Most Connected:")
    print("  Top Models:")
    for i, (model, degree) in enumerate(importance["top_connected_models"][:5], 1):
        print(f"    {i}. {model} ({degree} datasets)")

    print("  Top Datasets:")
    for i, (dataset, degree) in enumerate(importance["top_connected_datasets"][:5], 1):
        print(f"    {i}. {dataset} ({degree} models)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze graph structure of model-dataset relationships"
    )

    parser.add_argument(
        "json_file",
        nargs="?",
        default="perfect_model_dataset_metrics.json",
        help="Path to JSON data file",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="graph_analysis_report.json",
        help="Output file for detailed analysis report",
    )

    args = parser.parse_args()

    try:
        # Load data and create graph
        print(f"📁 Loading data from: {args.json_file}")
        G, stats = load_graph_data(args.json_file)

        print("📊 Starting graph analysis...")

        # Run analyses
        basic = analyze_basic_structure(G, stats)
        connectivity = analyze_connectivity(G)
        importance = analyze_node_importance(G)
        centrality = analyze_centrality(G)

        # Create report
        report = {
            "metadata": {
                "total_entries": stats["total_entries"],
                "graph_type": "bipartite",
            },
            "basic_structure": basic,
            "connectivity_analysis": connectivity,
            "node_importance": importance,
            "centrality_analysis": centrality,
            "raw_statistics": {
                "metric_types": dict(stats["metrics_count"]),
                "total_unique_metrics": len(stats["metrics_count"]),
            },
        }

        # Print summary
        print_summary(basic, connectivity, importance)

        # Save detailed report
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Detailed report saved to: {args.output}")

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
