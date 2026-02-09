#!/usr/bin/env python3
"""
Step 7a: Transductive graph splitting for link prediction.

- Uses edge_metadata_normalized.json (edges with metric values only)
- Val/Test edges must have non-empty metrics
- Train edges can have empty metrics
- Negative samples are generated dynamically (full negative)
- Saves per-split edge_metadata_normalized.json so downstream code can read directly
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.utils.graph_builder import load_nx_graph


def load_normalized_edge_metadata(input_dir: Path) -> Dict[str, dict]:
    """
    Load the full normalized edge metadata.

    Returns:
        Raw dict keyed by "u,v" string with full edge info (model_id, dataset_id, metrics, ...).
    """
    edge_file = input_dir / "edge_metadata_normalized.json"
    with open(edge_file) as f:
        return json.load(f)


def separate_edges_by_metrics(
    edge_metadata: Dict[str, dict],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Separate edges by whether they have non-empty metrics.

    Returns:
        edges_with_metrics: Edges that have non-empty metrics (for val/test)
        edges_no_metrics: Edges with empty metrics (train only)
    """
    edges_with_metrics = []
    edges_no_metrics = []

    for edge_key, edge_info in edge_metadata.items():
        u, v = map(int, edge_key.split(","))
        metrics = edge_info.get("metrics", {})

        if metrics:  # Has non-empty metrics
            edges_with_metrics.append((u, v))
        else:
            edges_no_metrics.append((u, v))

    return edges_with_metrics, edges_no_metrics


def split_edges(
    edges_with_metrics: List[Tuple[int, int]],
    edges_no_metrics: List[Tuple[int, int]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split edges into train/val/test.
    
    Val and Test only come from edges_with_metrics.
    Train includes remaining edges_with_metrics + all edges_no_metrics.
    """
    random.seed(seed)
    
    # Shuffle edges with metrics
    edges_with_metrics = list(edges_with_metrics)
    random.shuffle(edges_with_metrics)
    
    n = len(edges_with_metrics)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    test_edges = edges_with_metrics[:n_test]
    val_edges = edges_with_metrics[n_test:n_test + n_val]
    train_edges_from_metrics = edges_with_metrics[n_test + n_val:]
    
    # Train includes edges without metrics too
    train_edges = train_edges_from_metrics + list(edges_no_metrics)
    random.shuffle(train_edges)
    
    return train_edges, val_edges, test_edges


def extract_split_edge_metadata(
    pos_edges: List[Tuple[int, int]],
    full_metadata: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Extract the subset of normalized edge metadata for a given set of edges.

    Args:
        pos_edges: List of (u, v) tuples for this split.
        full_metadata: Full normalized edge metadata keyed by "u,v".

    Returns:
        Dict keyed by "u,v" with the metadata for edges in this split.
    """
    split_meta = {}
    for u, v in pos_edges:
        key = f"{u},{v}"
        if key in full_metadata:
            split_meta[key] = full_metadata[key]
        else:
            # Try reversed direction
            key_rev = f"{v},{u}"
            if key_rev in full_metadata:
                split_meta[key_rev] = full_metadata[key_rev]
    return split_meta


def _save_node_embeddings(input_dir: Path, output_dir: Path, num_nodes: int, seed: int):
    """Save node embeddings to split root so splits are self-contained.

    Copies real embeddings from input_dir and generates random embeddings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy real embeddings
    src = input_dir / "node_embeddings_voyage.npy"
    dst = output_dir / "node_embeddings_voyage.npy"
    if src.exists():
        shutil.copy2(src, dst)
        arr = np.load(dst, allow_pickle=False)
        if hasattr(arr.dtype, "names") and arr.dtype.names and "embedding" in arr.dtype.names:
            dim = arr["embedding"].shape[1]
        else:
            dim = arr.shape[1]
        print(f"   Copied real embeddings → {dst} (dim={dim})")
    else:
        print(f"   ⚠️ Real embeddings not found at {src}, skipping")

    # Generate and save random embeddings (for ablation)
    rng = np.random.RandomState(seed)
    random_emb = rng.randn(num_nodes, 768).astype(np.float32)
    random_dst = output_dir / "node_embeddings_random.npy"
    np.save(random_dst, random_emb)
    print(f"   Generated random embeddings → {random_dst} (dim=768, seed={seed})")


def save_split(
    split_dir: Path,
    message_edges: np.ndarray,
    pos_edges: List[Tuple[int, int]],
    node_metadata: Dict,
    edge_metadata: Dict[str, dict],
):
    """Save a split with edges, node metadata, and edge metadata."""
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Message-passing edges (undirected)
    np.savez(split_dir / "edges.npz", edges=message_edges)
    
    # Positive edges for this split
    pos_arr = np.array(pos_edges, dtype=np.int32).T
    np.savez(split_dir / "pos_edges.npz", edges=pos_arr)
    
    # Node metadata
    with open(split_dir / "node_metadata.json", "w") as f:
        json.dump(node_metadata, f, indent=2)

    # Edge metadata (normalized, only for this split's pos_edges)
    with open(split_dir / "edge_metadata_normalized.json", "w") as f:
        json.dump(edge_metadata, f, indent=2)

    # Count edges with actual metrics
    n_with_metrics = sum(1 for v in edge_metadata.values() if v.get("metrics"))
    return {"num_pos": len(pos_edges), "num_with_metrics": n_with_metrics}


def main():
    parser = argparse.ArgumentParser(description="Transductive graph split for link prediction")
    parser.add_argument("--input-dir", default="../data/artifact_graph_data_v2_1125")
    parser.add_argument("--output-dir", default="../data/artifact_graph_splits_v2_1125_transductive")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("🚀 Transductive Graph Splitting")
    print("=" * 50)
    
    input_dir = Path(args.input_dir)
    
    # Load graph for node metadata
    print("1. Loading graph...")
    G, node_metadata, _ = load_nx_graph(args.input_dir)
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Load full normalized edge metadata
    print("\n2. Loading normalized edge metadata...")
    full_edge_metadata = load_normalized_edge_metadata(input_dir)
    edges_with_metrics, edges_no_metrics = separate_edges_by_metrics(full_edge_metadata)
    print(f"   Total edges: {len(full_edge_metadata)}")
    print(f"   Edges with metrics: {len(edges_with_metrics)}")
    print(f"   Edges without metrics: {len(edges_no_metrics)}")
    
    # Split edges
    print("\n3. Splitting edges...")
    train_pos, val_pos, test_pos = split_edges(
        edges_with_metrics, edges_no_metrics,
        args.val_ratio, args.test_ratio, args.seed
    )
    print(f"   Train: {len(train_pos)} (includes {len(edges_no_metrics)} without metrics)")
    print(f"   Val: {len(val_pos)} (all with metrics)")
    print(f"   Test: {len(test_pos)} (all with metrics)")
    
    # Extract per-split edge metadata from full normalized metadata
    train_edge_meta = extract_split_edge_metadata(train_pos, full_edge_metadata)
    val_edge_meta = extract_split_edge_metadata(val_pos, full_edge_metadata)
    test_edge_meta = extract_split_edge_metadata(test_pos, full_edge_metadata)

    # Build message-passing edges (undirected)
    train_arr = np.array(train_pos, dtype=np.int32)
    train_undirected = np.concatenate([train_arr, train_arr[:, ::-1]], axis=0).T
    
    train_val_arr = np.array(train_pos + val_pos, dtype=np.int32)
    train_val_undirected = np.concatenate([train_val_arr, train_val_arr[:, ::-1]], axis=0).T
    
    # Save
    print("\n4. Saving splits...")
    output_dir = Path(args.output_dir)
    
    stats = {
        "train": save_split(output_dir / "train_split", train_undirected, train_pos, node_metadata, train_edge_meta),
        "val": save_split(output_dir / "val_split", train_undirected, val_pos, node_metadata, val_edge_meta),
        "test": save_split(output_dir / "test_split", train_val_undirected, test_pos, node_metadata, test_edge_meta),
    }
    
    # Save node embeddings to split root (makes splits self-contained)
    _save_node_embeddings(input_dir, output_dir, num_nodes=len(node_metadata), seed=args.seed)

    # Save split info
    split_info = {
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "source": "edge_metadata_normalized.json",
        "negative_sampling": "full (all unconnected model-dataset pairs)",
        "edges_with_metrics": len(edges_with_metrics),
        "edges_no_metrics": len(edges_no_metrics),
        **{f"{k}_pos": v["num_pos"] for k, v in stats.items()},
        **{f"{k}_with_metrics": v["num_with_metrics"] for k, v in stats.items()},
    }
    
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Done! Output: {args.output_dir}")
    print(f"\nSplit sizes:")
    for name, s in stats.items():
        print(f"  {name}: {s['num_pos']} positive edges ({s['num_with_metrics']} with metrics)")
    print("\n📝 Note: Val/Test edges all have metric values")
    print("📝 Note: Per-split edge_metadata_normalized.json saved (from edge_metadata_normalized.json)")
    print("📝 Note: Negative samples will be generated dynamically (full negative)")


if __name__ == "__main__":
    main()
