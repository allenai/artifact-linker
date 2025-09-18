#!/usr/bin/env python3
"""
Visualize the structure of the accuracy graph (model-dataset bipartite graph).
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from artifact_graph.utils.graph_builder import load_artifact_graph_from_json

def create_graph_structure_plot(G: nx.Graph, output_dir: str = "output/plots", 
                               max_nodes: int = 100, layout_type: str = "spring"):
    """Create visualization of the graph structure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get node information
    model_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'model']
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'dataset']
    
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Model nodes: {len(model_nodes)}")
    print(f"Dataset nodes: {len(dataset_nodes)}")
    print(f"Total edges: {G.number_of_edges()}")
    
    # If graph is too large, sample nodes
    if G.number_of_nodes() > max_nodes:
        print(f"Graph too large ({G.number_of_nodes()} nodes), sampling {max_nodes} nodes...")
        
        # Sample nodes by degree (keep highly connected nodes)
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Keep top connected models and datasets
        sample_models = [n for n, d in sorted_nodes[:max_nodes//2] if n in model_nodes][:max_nodes//2]
        sample_datasets = [n for n, d in sorted_nodes[:max_nodes//2] if n in dataset_nodes][:max_nodes//2]
        
        # Create subgraph
        sample_nodes = sample_models + sample_datasets
        G_sub = G.subgraph(sample_nodes).copy()
        
        model_nodes = [n for n in sample_nodes if n in model_nodes]
        dataset_nodes = [n for n in sample_nodes if n in dataset_nodes]
        G = G_sub
        
        print(f"Sampled graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create layout
    if layout_type == "bipartite":
        # Bipartite layout
        pos = {}
        # Place models on left, datasets on right
        for i, model in enumerate(model_nodes):
            pos[model] = (0, i)
        for i, dataset in enumerate(dataset_nodes):
            pos[dataset] = (2, i)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    else:  # spring layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Draw model nodes (blue circles)
    nx.draw_networkx_nodes(G, pos, nodelist=model_nodes, 
                          node_color='lightblue', node_size=100, 
                          node_shape='o', label='Models', alpha=0.8)
    
    # Draw dataset nodes (red squares)
    nx.draw_networkx_nodes(G, pos, nodelist=dataset_nodes, 
                          node_color='lightcoral', node_size=100, 
                          node_shape='s', label='Datasets', alpha=0.8)
    
    plt.title(f'Accuracy Graph Structure\n{len(model_nodes)} Models, {len(dataset_nodes)} Datasets, {G.number_of_edges()} Connections', 
              fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph_structure_{layout_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_degree_distribution_plot(G: nx.Graph, output_dir: str = "output/plots"):
    """Plot degree distribution for models and datasets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'model']
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'dataset']
    
    model_degrees = [G.degree(n) for n in model_nodes]
    dataset_degrees = [G.degree(n) for n in dataset_nodes]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Graph Degree Analysis', fontsize=18)
    
    # Model degree distribution
    axes[0, 0].hist(model_degrees, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('Model Degree Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Degree (Number of Datasets)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Models', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dataset degree distribution
    axes[0, 1].hist(dataset_degrees, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Dataset Degree Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Degree (Number of Models)', fontsize=12)
    axes[0, 1].set_ylabel('Number of Datasets', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top connected models
    top_models = sorted([(n, G.degree(n)) for n in model_nodes], key=lambda x: x[1], reverse=True)[:10]
    model_names = [m[0][:20] + '...' if len(m[0]) > 20 else m[0] for m in top_models]
    model_deg_vals = [m[1] for m in top_models]
    
    axes[1, 0].barh(range(len(model_names)), model_deg_vals, color='lightblue')
    axes[1, 0].set_title('Top 10 Most Connected Models', fontsize=14)
    axes[1, 0].set_xlabel('Number of Datasets', fontsize=12)
    axes[1, 0].set_yticks(range(len(model_names)))
    axes[1, 0].set_yticklabels(model_names, fontsize=10)
    
    # Top connected datasets
    top_datasets = sorted([(n, G.degree(n)) for n in dataset_nodes], key=lambda x: x[1], reverse=True)[:10]
    dataset_names = [d[0][:20] + '...' if len(d[0]) > 20 else d[0] for d in top_datasets]
    dataset_deg_vals = [d[1] for d in top_datasets]
    
    axes[1, 1].barh(range(len(dataset_names)), dataset_deg_vals, color='lightcoral')
    axes[1, 1].set_title('Top 10 Most Connected Datasets', fontsize=14)
    axes[1, 1].set_xlabel('Number of Models', fontsize=12)
    axes[1, 1].set_yticks(range(len(dataset_names)))
    axes[1, 1].set_yticklabels(dataset_names, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degree_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model_degrees': model_degrees,
        'dataset_degrees': dataset_degrees,
        'top_models': top_models,
        'top_datasets': top_datasets
    }

def create_accuracy_distribution_plot(G: nx.Graph, metric_name: str = "accuracy", 
                                    output_dir: str = "output/plots"):
    """Plot distribution of accuracy values on edges."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract accuracy values from edges
    accuracy_values = []
    for u, v, data in G.edges(data=True):
        acc = data.get(metric_name)
        if acc is not None:
            # Normalize if > 1 (percentage to decimal)
            if acc > 1:
                acc = acc / 100
            accuracy_values.append(acc)
    
    plt.figure(figsize=(12, 8))
    
    # Histogram of accuracy values
    plt.subplot(2, 2, 1)
    plt.hist(accuracy_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'Distribution of {metric_name.title()} Values', fontsize=14)
    plt.xlabel(f'{metric_name.title()}', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(accuracy_values, vert=True)
    plt.title(f'{metric_name.title()} Box Plot', fontsize=14)
    plt.ylabel(f'{metric_name.title()}', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Statistics
    plt.subplot(2, 2, 3)
    stats_text = f"""Statistics for {metric_name.title()}:
    
Count: {len(accuracy_values)}
Mean: {np.mean(accuracy_values):.4f}
Median: {np.median(accuracy_values):.4f}
Std: {np.std(accuracy_values):.4f}
Min: {np.min(accuracy_values):.4f}
Max: {np.max(accuracy_values):.4f}
    
Percentiles:
25th: {np.percentile(accuracy_values, 25):.4f}
75th: {np.percentile(accuracy_values, 75):.4f}
90th: {np.percentile(accuracy_values, 90):.4f}
95th: {np.percentile(accuracy_values, 95):.4f}"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_acc = np.sort(accuracy_values)
    cumulative = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
    plt.plot(sorted_acc, cumulative, 'b-', linewidth=2)
    plt.title(f'Cumulative Distribution of {metric_name.title()}', fontsize=14)
    plt.xlabel(f'{metric_name.title()}', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy_values

def main():
    """Main visualization function."""
    print("Loading graph from JSON...")
    
    graph_file = "output/perfect_model_dataset_metrics.json"
    metric_name = "accuracy"
    output_dir = "output/plots"
    
    # Load the graph
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create various visualizations
    print("\n1. Creating graph structure plots...")
    
    # Spring layout (good for smaller graphs)
    create_graph_structure_plot(G, output_dir, max_nodes=5000, layout_type="spring")
    
    # Bipartite layout (shows model-dataset separation clearly)
    create_graph_structure_plot(G, output_dir, max_nodes=5000, layout_type="bipartite")

    print("\n2. Creating degree distribution analysis...")
    degree_stats = create_degree_distribution_plot(G, output_dir)
    
    print("\n3. Creating accuracy distribution analysis...")
    accuracy_values = create_accuracy_distribution_plot(G, metric_name, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("GRAPH STRUCTURE SUMMARY")
    print("="*60)
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.6f}")
    print(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    
    model_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'model']
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'dataset']
    
    print(f"\nModel nodes: {len(model_nodes)}")
    print(f"Dataset nodes: {len(dataset_nodes)}")
    print(f"Average models per dataset: {np.mean(degree_stats['dataset_degrees']):.2f}")
    print(f"Average datasets per model: {np.mean(degree_stats['model_degrees']):.2f}")
    
    print(f"\nAccuracy statistics:")
    print(f"Mean accuracy: {np.mean(accuracy_values):.4f}")
    print(f"Median accuracy: {np.median(accuracy_values):.4f}")
    print(f"Accuracy range: {np.min(accuracy_values):.4f} - {np.max(accuracy_values):.4f}")

if __name__ == "__main__":
    main()
