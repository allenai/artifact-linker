#!/usr/bin/env python3
"""
Step 7b: Inductive graph splitting for link prediction.

- Uses edge_metadata_normalized.json (edges with metric values only)
- Val/Test edges must have non-empty metrics
- Inductive settings: new_models, new_datasets, new_both
- Negative samples are generated dynamically (full negative)
- Saves per-split edge_metadata_normalized.json so downstream code can read directly
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

        if metrics:
            edges_with_metrics.append((u, v))
        else:
            edges_no_metrics.append((u, v))

    return edges_with_metrics, edges_no_metrics


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


def split_nodes(
    node_ids: List[int],
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Split nodes into train/val/test sets."""
    random.seed(seed)
    nodes = list(node_ids)
    random.shuffle(nodes)
    
    n = len(nodes)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    return (
        set(nodes[n_test + n_val:]),  # train
        set(nodes[n_test:n_test + n_val]),  # val
        set(nodes[:n_test]),  # test
    )


def get_edge_split(model_id: int, dataset_id: int, mode: str,
                   train_m: Set, val_m: Set, test_m: Set,
                   train_d: Set, val_d: Set, test_d: Set) -> str:
    """Determine which split an edge belongs to based on node membership."""
    if mode == "new_models":
        if model_id in train_m:
            return "train"
        elif model_id in val_m:
            return "val"
        return "test"
    elif mode == "new_datasets":
        if dataset_id in train_d:
            return "train"
        elif dataset_id in val_d:
            return "val"
        return "test"
    else:  # new_both
        if model_id in test_m or dataset_id in test_d:
            return "test"
        elif model_id in val_m or dataset_id in val_d:
            return "val"
        return "train"


def create_inductive_split(
    node_metadata: Dict,
    edges_with_metrics: List[Tuple[int, int]],
    edges_no_metrics: List[Tuple[int, int]],
    mode: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    """
    Create inductive split.
    
    Val/Test edges must have metrics.
    Train edges can include edges without metrics.
    """
    # Get model and dataset nodes
    model_ids = [int(k) for k, v in node_metadata.items() if v.get("type") == "model"]
    dataset_ids = [int(k) for k, v in node_metadata.items() if v.get("type") == "dataset"]
    
    print(f"Models: {len(model_ids)}, Datasets: {len(dataset_ids)}")
    
    # Split nodes based on mode
    if mode == "new_models":
        train_m, val_m, test_m = split_nodes(model_ids, val_ratio, test_ratio, seed)
        train_d = val_d = test_d = set(dataset_ids)
        print(f"Model split - Train: {len(train_m)}, Val: {len(val_m)}, Test: {len(test_m)}")
    elif mode == "new_datasets":
        train_d, val_d, test_d = split_nodes(dataset_ids, val_ratio, test_ratio, seed)
        train_m = val_m = test_m = set(model_ids)
        print(f"Dataset split - Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)}")
    else:  # new_both
        train_m, val_m, test_m = split_nodes(model_ids, val_ratio, test_ratio, seed)
        train_d, val_d, test_d = split_nodes(dataset_ids, val_ratio, test_ratio, seed + 1)
        print(f"Model split - Train: {len(train_m)}, Val: {len(val_m)}, Test: {len(test_m)}")
        print(f"Dataset split - Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)}")
    
    # Assign edges with metrics to splits
    splits = {"train": [], "val": [], "test": []}
    
    for (u, v) in edges_with_metrics:
        u_type = node_metadata.get(u, {}).get("type")
        v_type = node_metadata.get(v, {}).get("type")
        
        if u_type == "model" and v_type == "dataset":
            model_id, dataset_id = u, v
        elif u_type == "dataset" and v_type == "model":
            model_id, dataset_id = v, u
        else:
            continue
        
        split = get_edge_split(model_id, dataset_id, mode, train_m, val_m, test_m, train_d, val_d, test_d)
        splits[split].append((u, v))
    
    # Add edges without metrics to train only
    train_edges_no_metrics = []
    for (u, v) in edges_no_metrics:
        u_type = node_metadata.get(u, {}).get("type")
        v_type = node_metadata.get(v, {}).get("type")
        
        if u_type == "model" and v_type == "dataset":
            model_id, dataset_id = u, v
        elif u_type == "dataset" and v_type == "model":
            model_id, dataset_id = v, u
        else:
            continue
        
        split = get_edge_split(model_id, dataset_id, mode, train_m, val_m, test_m, train_d, val_d, test_d)
        if split == "train":
            train_edges_no_metrics.append((u, v))
    
    splits["train"].extend(train_edges_no_metrics)
    
    print(f"Edge split:")
    print(f"  Train: {len(splits['train'])} ({len(train_edges_no_metrics)} without metrics)")
    print(f"  Val: {len(splits['val'])} (all with metrics)")
    print(f"  Test: {len(splits['test'])} (all with metrics)")
    
    return {
        "mode": mode,
        "edges": splits,
        "models": {"train": list(train_m), "val": list(val_m), "test": list(test_m)},
        "datasets": {"train": list(train_d), "val": list(val_d), "test": list(test_d)},
        "train_edges_no_metrics": len(train_edges_no_metrics),
    }


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
    msg_edges: np.ndarray,
    pos_edges: List,
    node_meta: Dict,
    edge_metadata: Dict[str, dict],
):
    """Save a single split with edges, node metadata, and edge metadata."""
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / "edges.npz", edges=msg_edges)
    
    if pos_edges:
        np.savez(split_dir / "pos_edges.npz", edges=np.array(pos_edges, dtype=np.int32).T)
    else:
        np.savez(split_dir / "pos_edges.npz", edges=np.array([]).reshape(2, 0))
    
    with open(split_dir / "node_metadata.json", "w") as f:
        json.dump({str(k): v for k, v in node_meta.items()}, f, indent=2)

    # Edge metadata (normalized, only for this split's pos_edges)
    with open(split_dir / "edge_metadata_normalized.json", "w") as f:
        json.dump(edge_metadata, f, indent=2)

    n_with_metrics = sum(1 for v in edge_metadata.values() if v.get("metrics"))
    return {"num_pos": len(pos_edges), "num_with_metrics": n_with_metrics}


def main():
    parser = argparse.ArgumentParser(description="Inductive graph split for link prediction")
    parser.add_argument("--input-dir", default="../data/artifact_graph_data_v2_1125")
    parser.add_argument("--output-dir", default="../data/artifact_graph_splits_v2_1125_inductive")
    parser.add_argument("--mode", choices=["new_models", "new_datasets", "new_both"], default="new_models")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"🚀 Inductive Graph Split ({args.mode})")
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
    
    # Create split
    print("\n3. Creating split...")
    split_data = create_inductive_split(
        node_metadata, edges_with_metrics, edges_no_metrics,
        args.mode, args.val_ratio, args.test_ratio, args.seed
    )
    
    # Extract per-split edge metadata from full normalized metadata
    train_edge_meta = extract_split_edge_metadata(split_data["edges"]["train"], full_edge_metadata)
    val_edge_meta = extract_split_edge_metadata(split_data["edges"]["val"], full_edge_metadata)
    test_edge_meta = extract_split_edge_metadata(split_data["edges"]["test"], full_edge_metadata)

    # Build message-passing edges
    train_arr = np.array(split_data["edges"]["train"], dtype=np.int32) if split_data["edges"]["train"] else np.array([]).reshape(0, 2)
    train_undirected = np.concatenate([train_arr, train_arr[:, ::-1]], axis=0).T if len(train_arr) else np.array([]).reshape(2, 0)
    
    train_val = split_data["edges"]["train"] + split_data["edges"]["val"]
    train_val_arr = np.array(train_val, dtype=np.int32) if train_val else np.array([]).reshape(0, 2)
    train_val_undirected = np.concatenate([train_val_arr, train_val_arr[:, ::-1]], axis=0).T if len(train_val_arr) else np.array([]).reshape(2, 0)
    
    # Save splits
    print("\n4. Saving splits...")
    output_dir = Path(args.output_dir) / args.mode
    
    stats = {
        "train": save_split(output_dir / "train_split", train_undirected, split_data["edges"]["train"], node_metadata, train_edge_meta),
        "val": save_split(output_dir / "val_split", train_undirected, split_data["edges"]["val"], node_metadata, val_edge_meta),
        "test": save_split(output_dir / "test_split", train_val_undirected, split_data["edges"]["test"], node_metadata, test_edge_meta),
    }
    
    # Save node embeddings to split root (makes splits self-contained)
    _save_node_embeddings(input_dir, output_dir, num_nodes=len(node_metadata), seed=args.seed)

    # Save node split info
    with open(output_dir / "node_split.json", "w") as f:
        json.dump({"models": split_data["models"], "datasets": split_data["datasets"]}, f, indent=2)
    
    # Save split info
    split_info = {
        "mode": args.mode,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "source": "edge_metadata_normalized.json",
        "negative_sampling": "full (all unconnected model-dataset pairs)",
        "edges_with_metrics": len(edges_with_metrics),
        "edges_no_metrics": len(edges_no_metrics),
        "train_edges": len(split_data["edges"]["train"]),
        "train_edges_no_metrics": split_data["train_edges_no_metrics"],
        "val_edges": len(split_data["edges"]["val"]),
        "test_edges": len(split_data["edges"]["test"]),
        **{f"{k}_with_metrics": v["num_with_metrics"] for k, v in stats.items()},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Done! Output: {output_dir}")
    print(f"\nSplit sizes:")
    for name, s in stats.items():
        print(f"  {name}: {s['num_pos']} positive edges ({s['num_with_metrics']} with metrics)")
    print("\n📝 Note: Val/Test edges all have metric values")
    print("📝 Note: Per-split edge_metadata_normalized.json saved (from edge_metadata_normalized.json)")
    print("📝 Note: Negative samples will be generated dynamically (full negative)")


if __name__ == "__main__":
    main()
