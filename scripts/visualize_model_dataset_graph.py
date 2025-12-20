#!/usr/bin/env python3
"""
可视化模型-数据集图的脚本
直接调用 artifact_graph.utils.graph_visualizer 中的现有函数
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_visualizer import (
    visualize_graph_interactive,
    visualize_graph_networkx,
)


def create_display_graph(G, node_metadata, max_nodes: int = 1000):
    """
    创建用于显示的图，将整数ID转换为可读的名称
    并可选地进行节点采样以提高可视化性能
    """
    # 如果节点太多，进行采样
    if len(G.nodes()) > max_nodes:
        print(f"图包含 {len(G.nodes())} 个节点，为了可视化性能，采样 {max_nodes} 个节点")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        G_sub = G.subgraph(top_nodes).copy()
    else:
        G_sub = G.copy()

    # 创建一个新图，使用可读的节点名称
    G_display = nx.Graph()

    # 添加节点（使用名称作为节点ID）
    for node_id in G_sub.nodes():
        node_data = node_metadata[node_id]
        name = node_data.get("name", f"Node_{node_id}")
        # 截断长名称以提高可读性
        display_name = name[:30] + "..." if len(name) > 30 else name

        G_display.add_node(
            display_name,
            type=node_data["type"],
            downloads=node_data.get("downloads", 0),
            original_id=node_id,
        )

    # 添加边
    id_to_name = {}
    for node_id in G_sub.nodes():
        node_data = node_metadata[node_id]
        name = node_data.get("name", f"Node_{node_id}")
        display_name = name[:30] + "..." if len(name) > 30 else name
        id_to_name[node_id] = display_name

    for u, v in G_sub.edges():
        u_name = id_to_name[u]
        v_name = id_to_name[v]
        G_display.add_edge(u_name, v_name)

    print(f"创建了用于显示的图：{G_display.number_of_nodes()} 个节点，{G_display.number_of_edges()} 条边")
    return G_display


def plot_basic_statistics(node_metadata, edges, output_dir: str = "output/visualizations"):
    """绘制基本统计图表"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 统计节点类型
    node_types = [data["type"] for data in node_metadata.values()]
    type_counts = Counter(node_types)

    # 统计下载量分布
    model_downloads = [
        data["downloads"] for data in node_metadata.values() if data["type"] == "model"
    ]
    dataset_downloads = [
        data["downloads"] for data in node_metadata.values() if data["type"] == "dataset"
    ]

    # 统计度分布
    node_degrees = defaultdict(int)
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        node_degrees[u] += 1
        node_degrees[v] += 1

    degree_dist = Counter(node_degrees.values())

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model-Dataset Graph Statistics", fontsize=16)

    # 1. 节点类型分布
    axes[0, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct="%1.1f%%")
    axes[0, 0].set_title("Node Type Distribution")

    # 2. 模型下载量分布
    axes[0, 1].hist(model_downloads, bins=50, alpha=0.7, color="blue")
    axes[0, 1].set_xlabel("Downloads")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Model Downloads Distribution")
    axes[0, 1].set_yscale("log")

    # 3. 数据集下载量分布
    axes[1, 0].hist(dataset_downloads, bins=50, alpha=0.7, color="green")
    axes[1, 0].set_xlabel("Downloads")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Dataset Downloads Distribution")
    axes[1, 0].set_yscale("log")

    # 4. 度分布
    degrees = list(degree_dist.keys())
    counts = list(degree_dist.values())
    axes[1, 1].loglog(degrees, counts, "bo-", alpha=0.7)
    axes[1, 1].set_xlabel("Degree")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Degree Distribution (Log-Log)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "graph_statistics.png", dpi=300, bbox_inches="tight")
    print(f"Saved statistics to {output_path}/graph_statistics.png")

    return fig


def create_network_visualizations(G_display, output_dir: str = "output/visualizations"):
    """创建网络可视化，调用现有的可视化函数"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. 创建静态网络图（使用 artifact_graph 的 visualize_graph_networkx）
    static_output = output_path / "network_layout.png"
    print("创建静态网络布局图...")
    visualize_graph_networkx(
        G_display,
        output_file=str(static_output),
        layout="spring",
        figsize=(16, 12),
        node_size_scale=200,
        show_labels=True,
        label_font_size=8,
        title="Model-Dataset Graph Network Layout",
        dpi=300,
    )
    print(f"静态网络图已保存到 {static_output}")

    # 2. 创建交互式图（使用 artifact_graph 的 visualize_graph_interactive）
    try:
        interactive_output = output_path / "interactive_graph.html"
        print("创建交互式网络图...")
        visualize_graph_interactive(G_display, str(interactive_output))
        print(f"交互式网络图已保存到 {interactive_output}")
    except Exception as e:
        print(f"交互式可视化失败: {e}")
        print("请确保已安装 pyvis: pip install pyvis")


def main():
    parser = argparse.ArgumentParser(description="可视化模型-数据集图")
    parser.add_argument("--graph-data-dir", default="output/artifact_graph_data", help="图数据目录")
    parser.add_argument("--output-dir", default="output/visualizations", help="可视化输出目录")
    parser.add_argument("--max-nodes", type=int, default=3000, help="网络可视化中显示的最大节点数")
    parser.add_argument("--skip-stats", action="store_true", help="跳过统计图表生成")
    parser.add_argument("--skip-interactive", action="store_true", help="跳过交互式可视化")

    args = parser.parse_args()

    try:
        # 使用 artifact_graph 的 load_nx_graph 函数加载数据
        print("正在加载图数据...")
        G, node_metadata, edge_metadata = load_nx_graph(args.graph_data_dir)

        print(f"成功加载图：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")

        # 创建输出目录
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 1. 基本统计图表（可选）
        if not args.skip_stats:
            print("正在生成基本统计图表...")
            # 需要构造 edges 数组用于统计
            edges = [[u, v] for u, v in G.edges()]
            plot_basic_statistics(node_metadata, edges, args.output_dir)

        # 2. 创建用于显示的图（将整数ID转换为可读名称）
        print("正在准备显示图...")
        G_display = create_display_graph(G, node_metadata, args.max_nodes)

        # 3. 网络可视化（静态和交互式）
        print("正在创建网络可视化...")
        create_network_visualizations(G_display, args.output_dir)

        print(f"\n✅ 可视化完成！请查看 {args.output_dir} 目录")
        print("生成的文件:")
        if not args.skip_stats:
            print("  - graph_statistics.png: 基本统计图表")
        print("  - network_layout.png: 静态网络布局图")
        if not args.skip_interactive:
            print("  - interactive_graph.html: 交互式网络图")

    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
