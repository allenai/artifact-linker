from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

MODEL, DATASET = "model", "dataset"


def visualize_graph_interactive(G: nx.Graph, output_file="model_dataset_graph.html"):
    """Visualize the bipartite graph using PyVis (interactive HTML)."""
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type")
        color = "#87ceeb" if node_type == MODEL else "#90ee90"
        shape = "box"
        net.add_node(node, label=node, color=color, shape=shape)

    for source, target in G.edges():
        net.add_edge(source, target)

    net.set_options(
        """
        var options = {
          "nodes": {
            "font": {
              "size": 18
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
    """
    )

    try:
        import pkg_resources
        from jinja2 import Template

        template_str = pkg_resources.resource_string("pyvis", "templates/template.html").decode(
            "utf-8"
        )
        net.template = Template(template_str)
    except Exception:
        pass

    net.show(output_file)
    print(f"✅ Graph saved to {output_file}")


def visualize_graph_networkx(
    G: nx.Graph,
    output_file: Optional[str] = None,
    layout: str = "spring",
    figsize: Tuple[int, int] = (12, 8),
    node_size_scale: float = 300,
    show_labels: bool = True,
    label_font_size: int = 8,
    title: str = "Model-Dataset Bipartite Graph",
    dpi: int = 300,
    **kwargs,
) -> None:
    """
    Visualize the bipartite graph using NetworkX with matplotlib.

    Args:
        G: NetworkX graph to visualize
        output_file: If provided, save the plot to this file (supports png, pdf, svg, etc.)
        layout: Layout algorithm ('spring', 'bipartite', 'circular', 'random', 'shell')
        figsize: Figure size as (width, height)
        node_size_scale: Base size for nodes
        show_labels: Whether to show node labels
        label_font_size: Font size for node labels
        title: Title for the plot
        dpi: DPI for saved images
        **kwargs: Additional arguments for layout algorithms
    """
    plt.figure(figsize=figsize)

    # Separate nodes by type
    model_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == MODEL]
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == DATASET]

    # Choose layout
    if layout == "bipartite":
        pos = nx.bipartite_layout(G, model_nodes, **kwargs)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50, **kwargs)
    elif layout == "circular":
        pos = nx.circular_layout(G, **kwargs)
    elif layout == "random":
        pos = nx.random_layout(G, **kwargs)
    elif layout == "shell":
        pos = nx.shell_layout(G, nlist=[model_nodes, dataset_nodes], **kwargs)
    else:
        pos = nx.spring_layout(G, **kwargs)

    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    model_sizes = [degrees[node] * node_size_scale for node in model_nodes]
    dataset_sizes = [degrees[node] * node_size_scale for node in dataset_nodes]

    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.5, edge_color="gray")

    # Draw model nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=model_nodes,
        node_color="lightblue",
        node_size=model_sizes,
        alpha=0.8,
        edgecolors="navy",
        linewidths=0.5,
    )

    # Draw dataset nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=dataset_nodes,
        node_color="lightgreen",
        node_size=dataset_sizes,
        alpha=0.8,
        edgecolors="darkgreen",
        linewidths=0.5,
    )

    # Add labels if requested
    if show_labels:
        # For large graphs, only label high-degree nodes
        if len(G.nodes()) > 100:
            # Only label nodes with degree > average
            avg_degree = np.mean(list(degrees.values()))
            high_degree_nodes = [n for n, d in degrees.items() if d > avg_degree]
            labels = {n: n.split("/")[-1] if "/" in n else n for n in high_degree_nodes}
        else:
            labels = {n: n.split("/")[-1] if "/" in n else n for n in G.nodes()}

        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=label_font_size, font_weight="bold"
        )

    # Create legend
    model_patch = mpatches.Patch(color="lightblue", label=f"Models ({len(model_nodes)})")
    dataset_patch = mpatches.Patch(color="lightgreen", label=f"Datasets ({len(dataset_nodes)})")
    plt.legend(handles=[model_patch, dataset_patch], loc="upper right")

    # Set title and remove axes
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"✅ NetworkX graph saved to {output_file}")
    else:
        plt.show()

    plt.close()


def visualize_graph_networkx_subplots(
    G: nx.Graph,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    layouts: list = None,
    **kwargs,
) -> None:
    """
    Visualize the graph using multiple NetworkX layouts in subplots.

    Args:
        G: NetworkX graph to visualize
        output_file: If provided, save the plot to this file
        figsize: Figure size as (width, height)
        layouts: List of layout names to use
        **kwargs: Additional arguments passed to individual visualizations
    """
    if layouts is None:
        layouts = ["spring", "bipartite", "circular", "shell"]

    # Filter layouts to only those that make sense for the graph
    if not nx.is_bipartite(G):
        layouts = [l for l in layouts if l != "bipartite"]

    n_layouts = len(layouts)
    cols = 2
    rows = (n_layouts + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    # Separate nodes by type for consistent coloring
    model_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == MODEL]
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == DATASET]

    for i, layout in enumerate(layouts):
        ax = axes[i]
        plt.sca(ax)

        # Create position layout
        if layout == "bipartite" and nx.is_bipartite(G):
            pos = nx.bipartite_layout(G, model_nodes)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G, nlist=[model_nodes, dataset_nodes])
        else:
            pos = nx.spring_layout(G)

        # Draw the graph
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color="gray")

        # Draw nodes by type
        nx.draw_networkx_nodes(
            G, pos, nodelist=model_nodes, node_color="lightblue", node_size=50, alpha=0.8
        )

        nx.draw_networkx_nodes(
            G, pos, nodelist=dataset_nodes, node_color="lightgreen", node_size=100, alpha=0.8
        )

        ax.set_title(f"{layout.title()} Layout", fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_layouts, len(axes)):
        axes[i].set_visible(False)

    # Add overall title
    fig.suptitle("Model-Dataset Graph: Multiple Layout Comparison", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ NetworkX multi-layout graph saved to {output_file}")
    else:
        plt.show()

    plt.close()


def visualize_graph_networkx_advanced(
    G: nx.Graph,
    output_file: Optional[str] = None,
    color_by: str = "type",
    size_by: str = "degree",
    layout: str = "spring",
    figsize: Tuple[int, int] = (14, 10),
    show_edge_weights: bool = False,
    highlight_nodes: list = None,
    **kwargs,
) -> None:
    """
    Advanced NetworkX visualization with customizable node coloring and sizing.

    Args:
        G: NetworkX graph to visualize
        output_file: If provided, save the plot to this file
        color_by: Node coloring scheme ('type', 'degree', 'downloads', 'centrality')
        size_by: Node sizing scheme ('degree', 'downloads', 'uniform')
        layout: Layout algorithm to use
        figsize: Figure size as (width, height)
        show_edge_weights: Whether to show edge weights
        highlight_nodes: List of nodes to highlight
        **kwargs: Additional arguments for layout algorithms
    """
    plt.figure(figsize=figsize)

    # Calculate layout
    model_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == MODEL]
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == DATASET]

    if layout == "bipartite":
        pos = nx.bipartite_layout(G, model_nodes, **kwargs)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50, **kwargs)
    else:
        pos = nx.spring_layout(G, **kwargs)

    # Determine node colors
    if color_by == "type":
        node_colors = [
            "lightblue" if G.nodes[n].get("type") == MODEL else "lightgreen" for n in G.nodes()
        ]
    elif color_by == "degree":
        degrees = [G.degree(n) for n in G.nodes()]
        node_colors = plt.cm.viridis(np.array(degrees) / max(degrees))
    elif color_by == "downloads":
        downloads = [G.nodes[n].get("downloads", 0) for n in G.nodes()]
        max_downloads = max(downloads) if downloads else 1
        node_colors = plt.cm.plasma(np.array(downloads) / max_downloads)
    else:
        node_colors = "lightgray"

    # Determine node sizes
    if size_by == "degree":
        node_sizes = [G.degree(n) * 50 + 100 for n in G.nodes()]
    elif size_by == "downloads":
        downloads = [G.nodes[n].get("downloads", 0) for n in G.nodes()]
        max_downloads = max(downloads) if downloads else 1
        node_sizes = [(d / max_downloads) * 500 + 100 for d in downloads]
    else:
        node_sizes = 300

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.5, edge_color="gray")

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # Highlight specific nodes if requested
    if highlight_nodes:
        highlight_pos = {n: pos[n] for n in highlight_nodes if n in pos}
        nx.draw_networkx_nodes(
            G,
            highlight_pos,
            nodelist=highlight_nodes,
            node_color="red",
            node_size=1000,
            alpha=0.6,
            edgecolors="darkred",
            linewidths=2,
        )

    # Add labels for important nodes
    if len(G.nodes()) <= 50:
        labels = {n: n.split("/")[-1] if "/" in n else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    # Add edge weights if requested
    if show_edge_weights and G.edges():
        edge_labels = nx.get_edge_attributes(G, "weight")
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title(
        f"Advanced Graph Visualization\nColored by {color_by}, Sized by {size_by}",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ Advanced NetworkX graph saved to {output_file}")
    else:
        plt.show()

    plt.close()


def create_graph_summary_plot(
    G: nx.Graph, output_file: Optional[str] = None, figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Create a comprehensive summary plot with graph visualization and statistics.

    Args:
        G: NetworkX graph to analyze and visualize
        output_file: If provided, save the plot to this file
        figsize: Figure size as (width, height)
    """
    fig = plt.figure(figsize=figsize)

    # Create a grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])

    # Main graph visualization
    ax_main = fig.add_subplot(gs[0, :2])
    plt.sca(ax_main)

    model_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == MODEL]
    dataset_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == DATASET]

    if nx.is_bipartite(G) and model_nodes:
        pos = nx.bipartite_layout(G, model_nodes)
    else:
        pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw graph
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, edge_color="gray")
    nx.draw_networkx_nodes(
        G, pos, nodelist=model_nodes, node_color="lightblue", node_size=100, alpha=0.8
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=dataset_nodes, node_color="lightgreen", node_size=200, alpha=0.8
    )

    ax_main.set_title("Graph Overview", fontsize=14, fontweight="bold")
    ax_main.axis("off")

    # Degree distribution
    ax_degree = fig.add_subplot(gs[0, 2])
    degrees = [G.degree(n) for n in G.nodes()]
    ax_degree.hist(degrees, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax_degree.set_title("Degree Distribution")
    ax_degree.set_xlabel("Degree")
    ax_degree.set_ylabel("Frequency")

    # Basic statistics
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis("off")

    stats_text = f"""
    Graph Statistics:
    • Nodes: {G.number_of_nodes():,}
    • Edges: {G.number_of_edges():,}
    • Density: {nx.density(G):.4f}
    • Connected Components: {nx.number_connected_components(G):,}
    • Average Degree: {np.mean(degrees):.2f}
    • Is Bipartite: {nx.is_bipartite(G)}
    """

    if model_nodes and dataset_nodes:
        stats_text += f"""
    • Models: {len(model_nodes):,}
    • Datasets: {len(dataset_nodes):,}
    • Model/Dataset Ratio: {len(model_nodes)/len(dataset_nodes):.2f}
        """

    ax_stats.text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Component size distribution
    ax_components = fig.add_subplot(gs[2, :])
    component_sizes = [len(c) for c in nx.connected_components(G)]
    ax_components.hist(
        component_sizes,
        bins=min(20, len(component_sizes)),
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    ax_components.set_title("Connected Component Size Distribution")
    ax_components.set_xlabel("Component Size")
    ax_components.set_ylabel("Frequency")

    plt.suptitle("Graph Analysis Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ Graph summary plot saved to {output_file}")
    else:
        plt.show()

    plt.close()
