#!/usr/bin/env python3
"""
Step 7: Graph splitting with hard negative mining.

Supports multiple hard negative strategies:
- common_neighbors: Nodes sharing common neighbors but no edge
- jaccard: High Jaccard similarity but no edge
- adamic_adar: High Adamic-Adar score but no edge
- preferential_attachment: High PA score but no edge
- degree: High-degree node pairs without edges
- type_aware: Bipartite-aware sampling (model-dataset only)
- random: Standard random negative sampling (baseline)
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.utils.graph_builder import load_nx_graph, load_pyg_graph


def get_existing_edges(G: nx.Graph) -> Set[Tuple[int, int]]:
    """Get all existing edges as a set of tuples."""
    edges = set()
    for u, v in G.edges():
        edges.add((u, v))
        edges.add((v, u))  # For undirected
    return edges


def get_node_types(G: nx.Graph) -> Tuple[List[int], List[int]]:
    """Get model and dataset node IDs."""
    models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
    datasets = [n for n, d in G.nodes(data=True) if d.get("type") == "dataset"]
    return models, datasets


def mine_hard_negatives_common_neighbors(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Mine negatives based on common neighbors."""
    candidates = []
    nodes = list(G.nodes())
    
    # Build common neighbor scores for non-edges
    print("  Computing common neighbor scores...")
    for u in tqdm(nodes, desc="  Common neighbors"):
        neighbors_u = set(G.neighbors(u))
        for v in nodes:
            if u >= v:
                continue
            if (u, v) in existing_edges:
                continue
            neighbors_v = set(G.neighbors(v))
            cn = len(neighbors_u & neighbors_v)
            if cn > 0:
                candidates.append((u, v, cn))
    
    # Sort by common neighbors (descending) and take top
    candidates.sort(key=lambda x: -x[2])
    negatives = [(u, v) for u, v, _ in candidates[:num_negatives]]
    
    # Fill with random if not enough
    if len(negatives) < num_negatives:
        negatives.extend(
            mine_hard_negatives_random(G, pos_edges, num_negatives - len(negatives), existing_edges)
        )
    
    return negatives[:num_negatives]


def mine_hard_negatives_jaccard(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Mine negatives based on Jaccard similarity."""
    candidates = []
    nodes = list(G.nodes())
    
    print("  Computing Jaccard scores...")
    for u in tqdm(nodes, desc="  Jaccard"):
        neighbors_u = set(G.neighbors(u))
        if not neighbors_u:
            continue
        for v in nodes:
            if u >= v:
                continue
            if (u, v) in existing_edges:
                continue
            neighbors_v = set(G.neighbors(v))
            if not neighbors_v:
                continue
            intersection = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            if union > 0 and intersection > 0:
                jaccard = intersection / union
                candidates.append((u, v, jaccard))
    
    candidates.sort(key=lambda x: -x[2])
    negatives = [(u, v) for u, v, _ in candidates[:num_negatives]]
    
    if len(negatives) < num_negatives:
        negatives.extend(
            mine_hard_negatives_random(G, pos_edges, num_negatives - len(negatives), existing_edges)
        )
    
    return negatives[:num_negatives]


def mine_hard_negatives_adamic_adar(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Mine negatives based on Adamic-Adar index."""
    candidates = []
    nodes = list(G.nodes())
    
    print("  Computing Adamic-Adar scores...")
    for u in tqdm(nodes, desc="  Adamic-Adar"):
        neighbors_u = set(G.neighbors(u))
        for v in nodes:
            if u >= v:
                continue
            if (u, v) in existing_edges:
                continue
            neighbors_v = set(G.neighbors(v))
            common = neighbors_u & neighbors_v
            if common:
                aa_score = sum(1 / np.log(G.degree(w) + 1) for w in common if G.degree(w) > 1)
                if aa_score > 0:
                    candidates.append((u, v, aa_score))
    
    candidates.sort(key=lambda x: -x[2])
    negatives = [(u, v) for u, v, _ in candidates[:num_negatives]]
    
    if len(negatives) < num_negatives:
        negatives.extend(
            mine_hard_negatives_random(G, pos_edges, num_negatives - len(negatives), existing_edges)
        )
    
    return negatives[:num_negatives]


def mine_hard_negatives_preferential_attachment(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Mine negatives based on preferential attachment (degree product)."""
    candidates = []
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    
    print("  Computing PA scores...")
    for u in tqdm(nodes, desc="  Preferential Attachment"):
        deg_u = degrees[u]
        if deg_u == 0:
            continue
        for v in nodes:
            if u >= v:
                continue
            if (u, v) in existing_edges:
                continue
            deg_v = degrees[v]
            if deg_v > 0:
                pa_score = deg_u * deg_v
                candidates.append((u, v, pa_score))
    
    candidates.sort(key=lambda x: -x[2])
    negatives = [(u, v) for u, v, _ in candidates[:num_negatives]]
    
    if len(negatives) < num_negatives:
        negatives.extend(
            mine_hard_negatives_random(G, pos_edges, num_negatives - len(negatives), existing_edges)
        )
    
    return negatives[:num_negatives]


def mine_hard_negatives_type_aware(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Mine negatives only between models and datasets (bipartite-aware)."""
    models, datasets = get_node_types(G)
    
    if not models or not datasets:
        print("  Warning: Could not find model/dataset types, falling back to random")
        return mine_hard_negatives_random(G, pos_edges, num_negatives, existing_edges)
    
    print(f"  Mining bipartite negatives: {len(models)} models x {len(datasets)} datasets")
    
    candidates = []
    for m in models:
        for d in datasets:
            if (m, d) not in existing_edges and (d, m) not in existing_edges:
                candidates.append((m, d))
    
    random.shuffle(candidates)
    negatives = candidates[:num_negatives]
    
    if len(negatives) < num_negatives:
        print(f"  Warning: Only found {len(negatives)} type-aware negatives")
    
    return negatives[:num_negatives]


def mine_hard_negatives_random(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Standard random negative sampling."""
    nodes = list(G.nodes())
    negatives = []
    attempts = 0
    max_attempts = num_negatives * 100
    
    while len(negatives) < num_negatives and attempts < max_attempts:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
            negatives.append((u, v))
            existing_edges.add((u, v))
            existing_edges.add((v, u))
        attempts += 1
    
    return negatives


def mine_hard_negatives(
    G: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    num_negatives: int,
    strategy: str,
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Mine hard negatives using specified strategy."""
    strategies = {
        "common_neighbors": mine_hard_negatives_common_neighbors,
        "jaccard": mine_hard_negatives_jaccard,
        "adamic_adar": mine_hard_negatives_adamic_adar,
        "preferential_attachment": mine_hard_negatives_preferential_attachment,
        "type_aware": mine_hard_negatives_type_aware,
        "random": mine_hard_negatives_random,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    
    print(f"  Mining {num_negatives} hard negatives using '{strategy}' strategy...")
    return strategies[strategy](G, pos_edges, num_negatives, existing_edges.copy())


def split_edges(
    edges: List[Tuple[int, int]],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split edges into train/val/test."""
    random.seed(seed)
    edges = list(edges)
    random.shuffle(edges)
    
    n = len(edges)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    test_edges = edges[:n_test]
    val_edges = edges[n_test:n_test + n_val]
    train_edges = edges[n_test + n_val:]
    
    return train_edges, val_edges, test_edges


def save_split(
    split_dir: Path,
    train_edges: np.ndarray,
    pos_edges: List[Tuple[int, int]],
    neg_edges: List[Tuple[int, int]],
    node_metadata: Dict,
):
    """Save a single split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save message-passing edges
    np.savez(split_dir / "edges.npz", edges=train_edges)
    
    # Save positive edges
    pos_arr = np.array(pos_edges, dtype=np.int32).T  # Shape: (2, N)
    np.savez(split_dir / "pos_edges.npz", edges=pos_arr)
    
    # Save negative edges
    neg_arr = np.array(neg_edges, dtype=np.int32).T  # Shape: (2, N)
    np.savez(split_dir / "neg_edges.npz", edges=neg_arr)
    
    # Save combined labels
    all_edges = np.concatenate([pos_arr, neg_arr], axis=1)
    labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
    np.savez(split_dir / "edge_labels.npz", edge_label_index=all_edges, edge_label=labels)
    
    # Save node metadata
    with open(split_dir / "node_metadata.json", "w") as f:
        json.dump(node_metadata, f, indent=2)
    
    return {
        "num_edges": len(train_edges),
        "num_pos": len(pos_edges),
        "num_neg": len(neg_edges),
    }


def main():
    parser = argparse.ArgumentParser(description="Graph splitting with hard negative mining")
    parser.add_argument("--input_dir", type=str, default="./output/artifact_graph_data_v2_1125")
    parser.add_argument("--output_dir", type=str, default="./output/artifact_graph_splits_hard_neg_ratio_5_v2_1125")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--neg_ratio", type=float, default=5.0, help="Negative samples per positive")
    parser.add_argument(
        "--strategy",
        type=str,
        default="preferential_attachment",
        choices=["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment", "type_aware", "random"],
        help="Hard negative mining strategy",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("🚀 Graph Splitting with Hard Negative Mining")
    print("=" * 50)
    print(f"Strategy: {args.strategy}")
    print(f"Neg ratio: {args.neg_ratio}")
    
    # Load graph
    print("\n1. Loading graph...")
    G, node_metadata, _ = load_nx_graph(args.input_dir)
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Get all edges
    all_edges = list(G.edges())
    existing_edges = get_existing_edges(G)
    
    # Split positive edges
    print("\n2. Splitting positive edges...")
    train_pos, val_pos, test_pos = split_edges(
        all_edges, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"   Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")
    
    # Build train graph (for message passing)
    train_edges_arr = np.array(train_pos, dtype=np.int32)
    # Add reverse edges for undirected
    train_edges_undirected = np.concatenate([
        train_edges_arr,
        train_edges_arr[:, ::-1]
    ], axis=0).T  # Shape: (2, 2*N)
    
    # Mine hard negatives for each split
    print("\n3. Mining hard negatives...")
    
    # For train
    print("  [Train]")
    train_neg = mine_hard_negatives(
        G, train_pos, int(len(train_pos) * args.neg_ratio), args.strategy, existing_edges
    )
    
    # For val (use train graph structure)
    print("  [Val]")
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(train_pos)
    val_neg = mine_hard_negatives(
        G_train, val_pos, int(len(val_pos) * args.neg_ratio), args.strategy, existing_edges
    )
    
    # For test (use train+val graph structure)
    print("  [Test]")
    G_train_val = nx.Graph()
    G_train_val.add_nodes_from(G.nodes(data=True))
    G_train_val.add_edges_from(train_pos + val_pos)
    test_neg = mine_hard_negatives(
        G_train_val, test_pos, int(len(test_pos) * args.neg_ratio), args.strategy, existing_edges
    )
    
    # Save splits
    print("\n4. Saving splits...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Save train split
    stats["train"] = save_split(
        output_dir / "train_split",
        train_edges_undirected,
        train_pos,
        train_neg,
        node_metadata,
    )
    
    # Save val split (uses train edges for message passing)
    stats["val"] = save_split(
        output_dir / "val_split",
        train_edges_undirected,
        val_pos,
        val_neg,
        node_metadata,
    )
    
    # Save test split (uses train+val edges for message passing)
    train_val_edges = np.array(train_pos + val_pos, dtype=np.int32)
    train_val_undirected = np.concatenate([
        train_val_edges,
        train_val_edges[:, ::-1]
    ], axis=0).T
    
    stats["test"] = save_split(
        output_dir / "test_split",
        train_val_undirected,
        test_pos,
        test_neg,
        node_metadata,
    )
    
    # Save split info
    split_info = {
        "method": "hard_negative_mining",
        "strategy": args.strategy,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        **stats,
    }
    
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Done! Output: {args.output_dir}")
    print(f"\nStats:")
    for split_name, s in stats.items():
        print(f"  {split_name}: {s['num_pos']} pos, {s['num_neg']} neg")


if __name__ == "__main__":
    main()

