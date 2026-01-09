#!/usr/bin/env python3
"""
Step 7 (Inductive): Graph splitting for inductive learning.

Inductive settings:
- new_models: Test models are unseen during training
- new_datasets: Test datasets are unseen during training  
- new_both: Some models AND some datasets are unseen
- cold_start: Test edges have at least one unseen node

Key difference from transductive:
- Transductive: All nodes visible, only edges hidden
- Inductive: Some nodes completely hidden during training
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.utils.graph_builder import load_nx_graph


def split_nodes_inductive(
    node_ids: List[int],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Split nodes into train/val/test sets."""
    random.seed(seed)
    nodes = [int(n) for n in node_ids]  # Ensure integers
    random.shuffle(nodes)
    
    n = len(nodes)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    test_nodes = set(nodes[:n_test])
    val_nodes = set(nodes[n_test:n_test + n_val])
    train_nodes = set(nodes[n_test + n_val:])
    
    return train_nodes, val_nodes, test_nodes


def create_inductive_split(
    G,
    node_metadata: Dict,
    edge_metadata: Dict,
    mode: str = "new_models",
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Create inductive split based on mode.
    
    Returns dict with train/val/test edges and node visibility info.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get model and dataset nodes
    model_ids = [int(k) for k, v in node_metadata.items() if v.get("type") == "model"]
    dataset_ids = [int(k) for k, v in node_metadata.items() if v.get("type") == "dataset"]
    
    print(f"Total models: {len(model_ids)}, datasets: {len(dataset_ids)}")
    
    # Get all edges
    all_edges = list(G.edges())
    print(f"Total edges: {len(all_edges)}")
    
    if mode == "new_models":
        # Split models into train/val/test
        train_models, val_models, test_models = split_nodes_inductive(
            model_ids, test_ratio, val_ratio, seed
        )
        train_datasets = set(int(d) for d in dataset_ids)
        val_datasets = set(int(d) for d in dataset_ids)
        test_datasets = set(int(d) for d in dataset_ids)
        
        print(f"Train models: {len(train_models)}, Val: {len(val_models)}, Test: {len(test_models)}")
        
    elif mode == "new_datasets":
        # Split datasets into train/val/test
        train_datasets, val_datasets, test_datasets = split_nodes_inductive(
            dataset_ids, test_ratio, val_ratio, seed
        )
        train_models = set(int(m) for m in model_ids)
        val_models = set(int(m) for m in model_ids)
        test_models = set(int(m) for m in model_ids)
        
        print(f"Train datasets: {len(train_datasets)}, Val: {len(val_datasets)}, Test: {len(test_datasets)}")
        
    elif mode == "new_both":
        # Split both models and datasets
        train_models, val_models, test_models = split_nodes_inductive(
            model_ids, test_ratio, val_ratio, seed
        )
        train_datasets, val_datasets, test_datasets = split_nodes_inductive(
            dataset_ids, test_ratio, val_ratio, seed + 1  # Different seed
        )
        
        print(f"Train models: {len(train_models)}, Test models: {len(test_models)}")
        print(f"Train datasets: {len(train_datasets)}, Test datasets: {len(test_datasets)}")
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Assign edges to splits based on node visibility
    train_edges = []
    val_edges = []
    test_edges = []
    
    for u, v in all_edges:
        u, v = int(u), int(v)  # Convert to int
        u_type = node_metadata.get(u, {}).get("type")
        v_type = node_metadata.get(v, {}).get("type")
        
        # Determine model and dataset
        if u_type == "model" and v_type == "dataset":
            model_id, dataset_id = u, v
        elif u_type == "dataset" and v_type == "model":
            model_id, dataset_id = v, u
        else:
            # Skip non-model-dataset edges
            continue
        
        # Assign based on node visibility
        if mode == "new_models":
            if model_id in train_models:
                train_edges.append((u, v))
            elif model_id in val_models:
                val_edges.append((u, v))
            else:  # test_models
                test_edges.append((u, v))
                
        elif mode == "new_datasets":
            if dataset_id in train_datasets:
                train_edges.append((u, v))
            elif dataset_id in val_datasets:
                val_edges.append((u, v))
            else:  # test_datasets
                test_edges.append((u, v))
                
        elif mode == "new_both":
            # Edge is test if either node is test
            # Edge is val if either node is val (and neither is test)
            # Edge is train if both nodes are train
            model_is_test = model_id in test_models
            dataset_is_test = dataset_id in test_datasets
            model_is_val = model_id in val_models
            dataset_is_val = dataset_id in val_datasets
            
            if model_is_test or dataset_is_test:
                test_edges.append((u, v))
            elif model_is_val or dataset_is_val:
                val_edges.append((u, v))
            else:
                train_edges.append((u, v))
    
    print(f"\nEdge split:")
    print(f"  Train: {len(train_edges)}")
    print(f"  Val: {len(val_edges)}")
    print(f"  Test: {len(test_edges)}")
    
    return {
        "mode": mode,
        "train_edges": train_edges,
        "val_edges": val_edges,
        "test_edges": test_edges,
        "train_models": list(train_models),
        "val_models": list(val_models) if mode != "new_datasets" else list(model_ids),
        "test_models": list(test_models) if mode != "new_datasets" else list(model_ids),
        "train_datasets": list(train_datasets),
        "val_datasets": list(val_datasets) if mode != "new_models" else list(dataset_ids),
        "test_datasets": list(test_datasets) if mode != "new_models" else list(dataset_ids),
    }


def save_inductive_split(
    output_dir: Path,
    split_data: Dict,
    node_metadata: Dict,
    edge_metadata: Dict,
):
    """Save inductive split to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split info
    split_info = {
        "mode": split_data["mode"],
        "num_train_edges": len(split_data["train_edges"]),
        "num_val_edges": len(split_data["val_edges"]),
        "num_test_edges": len(split_data["test_edges"]),
        "num_train_models": len(split_data["train_models"]),
        "num_test_models": len(split_data.get("test_models", [])),
        "num_train_datasets": len(split_data["train_datasets"]),
        "num_test_datasets": len(split_data.get("test_datasets", [])),
    }
    
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    # Save train split
    train_dir = output_dir / "train_split"
    train_dir.mkdir(exist_ok=True)
    
    train_edges_arr = np.array(split_data["train_edges"], dtype=np.int32)
    train_edges_undirected = np.concatenate([
        train_edges_arr,
        train_edges_arr[:, ::-1]
    ], axis=0).T
    np.savez(train_dir / "edges.npz", edges=train_edges_undirected)
    np.savez(train_dir / "pos_edges.npz", edges=train_edges_arr.T)
    
    # Save visible nodes for train
    train_node_ids = set(split_data["train_models"]) | set(split_data["train_datasets"])
    train_node_meta = {
        str(k): v for k, v in node_metadata.items()
        if k in train_node_ids
    }
    with open(train_dir / "node_metadata.json", "w") as f:
        json.dump(train_node_meta, f, indent=2)
    
    # Save val split
    val_dir = output_dir / "val_split"
    val_dir.mkdir(exist_ok=True)
    
    val_edges_arr = np.array(split_data["val_edges"], dtype=np.int32) if split_data["val_edges"] else np.array([]).reshape(0, 2)
    np.savez(val_dir / "edges.npz", edges=train_edges_undirected)  # Use train edges for message passing
    np.savez(val_dir / "pos_edges.npz", edges=val_edges_arr.T if len(val_edges_arr) > 0 else np.array([]).reshape(2, 0))
    
    # Convert node_metadata keys to strings for JSON compatibility
    node_meta_str = {str(k): v for k, v in node_metadata.items()}
    with open(val_dir / "node_metadata.json", "w") as f:
        json.dump(node_meta_str, f, indent=2)  # All nodes visible at val time
    
    # Save test split
    test_dir = output_dir / "test_split"
    test_dir.mkdir(exist_ok=True)
    
    # For test, message passing uses train+val edges
    train_val_edges = split_data["train_edges"] + split_data["val_edges"]
    train_val_arr = np.array(train_val_edges, dtype=np.int32) if train_val_edges else np.array([]).reshape(0, 2)
    train_val_undirected = np.concatenate([
        train_val_arr,
        train_val_arr[:, ::-1]
    ], axis=0).T if len(train_val_arr) > 0 else np.array([]).reshape(2, 0)
    
    test_edges_arr = np.array(split_data["test_edges"], dtype=np.int32)
    np.savez(test_dir / "edges.npz", edges=train_val_undirected)
    np.savez(test_dir / "pos_edges.npz", edges=test_edges_arr.T)
    
    with open(test_dir / "node_metadata.json", "w") as f:
        json.dump(node_meta_str, f, indent=2)
    
    # Save node split info (which nodes are train/val/test)
    node_split = {
        "train_models": split_data["train_models"],
        "val_models": split_data.get("val_models", []),
        "test_models": split_data.get("test_models", []),
        "train_datasets": split_data["train_datasets"],
        "val_datasets": split_data.get("val_datasets", []),
        "test_datasets": split_data.get("test_datasets", []),
    }
    with open(output_dir / "node_split.json", "w") as f:
        json.dump(node_split, f, indent=2)
    
    print(f"\n✅ Saved inductive split to {output_dir}")
    return split_info


def main():
    parser = argparse.ArgumentParser(description="Create inductive graph split")
    parser.add_argument("--input-dir", type=str, default="scripts/output/artifact_graph_data_v2_1125")
    parser.add_argument("--output-dir", type=str, default="scripts/output/artifact_graph_splits_inductive_v2_1125")
    parser.add_argument("--mode", type=str, default="new_models",
                        choices=["new_models", "new_datasets", "new_both"],
                        help="Inductive split mode")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("🚀 Creating Inductive Graph Split")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Test ratio: {args.test_ratio}, Val ratio: {args.val_ratio}")
    
    # Load graph
    print("\n1. Loading graph...")
    G, node_metadata, edge_metadata = load_nx_graph(args.input_dir)
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Create split
    print("\n2. Creating inductive split...")
    split_data = create_inductive_split(
        G, node_metadata, edge_metadata,
        mode=args.mode,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    # Save
    print("\n3. Saving split...")
    output_dir = Path(args.output_dir) / args.mode
    split_info = save_inductive_split(output_dir, split_data, node_metadata, edge_metadata)
    
    print("\n" + "=" * 50)
    print("📊 Split Summary:")
    for k, v in split_info.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()

