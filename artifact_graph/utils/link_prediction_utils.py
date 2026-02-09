#!/usr/bin/env python3
"""Shared utilities for link prediction scripts."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .evaluation_utils import evaluate_binary_classification

Edge = Tuple[int, int]


# =============================================================================
# Common Utilities
# =============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_link_prediction_data(
    G: nx.Graph,
    seed: int = 42,
    max_pairs: int = 0,
) -> Tuple[List[Edge], List[int]]:
    """
    Prepare link prediction dataset using ALL model-dataset pairs.
    
    Positives: Existing edges.
    Negatives: ALL pairs without edges.
    """
    rng = random.Random(seed)

    models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
    datasets = [n for n, d in G.nodes(data=True) if d.get("type") == "dataset"]

    pos_edges: Set[Edge] = set()
    for u, v in G.edges():
        u_type, v_type = G.nodes[u].get("type"), G.nodes[v].get("type")
        if u_type == "model" and v_type == "dataset":
            pos_edges.add((u, v))
        elif u_type == "dataset" and v_type == "model":
            pos_edges.add((v, u))

    neg_edges = [(m, d) for m in models for d in datasets if (m, d) not in pos_edges]

    all_edges = list(pos_edges) + neg_edges
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    combined = list(zip(all_edges, labels))
    rng.shuffle(combined)

    if combined:
        edges, labels = zip(*combined)
    else:
        edges, labels = [], []

    if max_pairs > 0:
        edges, labels = list(edges)[:max_pairs], list(labels)[:max_pairs]
    
    return list(edges), list(labels)


# =============================================================================
# Data Loading
# =============================================================================

def load_link_prediction_data(
    graph_data_dir: str | Path,
    seed: int = 42,
    max_pairs: int = 0,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_link_predictions.json",
    split_dir: str | Path | None = None,
) -> Tuple[Any, Dict, List[Edge], List[int]]:
    """
    Load graph and prepare link prediction data.
    
    Args:
        graph_data_dir: Directory containing graph data.
        seed: Random seed.
        max_pairs: Max pairs to use (0 = all).
        use_gnn_data: Load edges from GNN predictions file.
        gnn_data_path: Path to GNN predictions.
        split_dir: If provided, load test edges from this split directory.
    
    Returns:
        Tuple of (G, node_metadata, edges, labels).
    """
    from .graph_builder import load_nx_graph
    import numpy as np

    G, node_metadata, _ = load_nx_graph(graph_data_dir=str(graph_data_dir))

    if split_dir is not None:
        # Load from pre-defined split (same as GNN)
        split_path = Path(split_dir) / "test_split"
        pos_edges = np.load(split_path / "pos_edges.npz")["edges"]
        
        # Get all positive edges to determine negatives
        all_pos = set()
        for split_name in ["train_split", "val_split", "test_split"]:
            pos_path = Path(split_dir) / split_name / "pos_edges.npz"
            if pos_path.exists():
                pos = np.load(pos_path)["edges"]
                for i in range(pos.shape[1]):
                    all_pos.add((int(pos[0, i]), int(pos[1, i])))
                    all_pos.add((int(pos[1, i]), int(pos[0, i])))
        
        # Build test edges with full negatives
        test_pos = [(int(pos_edges[0, i]), int(pos_edges[1, i])) for i in range(pos_edges.shape[1])]
        
        # Get datasets in test split
        test_datasets = set()
        for u, v in test_pos:
            u_type = node_metadata.get(str(u), {}).get("type") or node_metadata.get(u, {}).get("type")
            v_type = node_metadata.get(str(v), {}).get("type") or node_metadata.get(v, {}).get("type")
            if u_type == "dataset":
                test_datasets.add(u)
            elif v_type == "dataset":
                test_datasets.add(v)
        
        # Get all model IDs
        model_ids = {int(k) for k, v in node_metadata.items() if v.get("type") == "model"}
        
        # Generate negatives for test datasets
        neg_edges = []
        for did in test_datasets:
            for mid in model_ids:
                if (mid, did) not in all_pos and (did, mid) not in all_pos:
                    neg_edges.append((mid, did))
        
        edges = test_pos + neg_edges
        labels = [1] * len(test_pos) + [0] * len(neg_edges)
        
        # Shuffle
        rng = random.Random(seed)
        combined = list(zip(edges, labels))
        rng.shuffle(combined)
        edges, labels = zip(*combined) if combined else ([], [])
        edges, labels = list(edges), list(labels)
        
        if max_pairs > 0:
            edges, labels = edges[:max_pairs], labels[:max_pairs]
        
        return G, node_metadata, edges, labels

    # Legacy: generate edges dynamically
    edges, labels = prepare_link_prediction_data(G, seed=seed, max_pairs=max_pairs)

    if use_gnn_data:
        with open(gnn_data_path, "r") as f:
            gnn_data = json.load(f)
        gnn_edges = gnn_data["test_predictions"]["edges"]
        labels = [edge["ground_truth"] for edge in gnn_edges]
        edges = [(edge["v_id"], edge["u_id"]) for edge in gnn_edges]

    return G, node_metadata, edges, labels


# =============================================================================
# Row Creation
# =============================================================================

def create_link_prediction_row(
    model_id: int,
    dataset_id: int,
    true_label: int,
    node_metadata: Dict,
    predicted_label: Optional[int] = None,
    reason: str = "",
    status: str = "Failed",
) -> Dict[str, Any]:
    """Create a standardized link prediction row."""
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_name": node_metadata.get(model_id, {}).get("name"),
        "dataset_name": node_metadata.get(dataset_id, {}).get("name"),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "reason": reason,
        "status": status,
    }


# =============================================================================
# Collect Valid Results
# =============================================================================

def collect_link_predictions(predictions: List[Dict]) -> Tuple[List[int], List[int], List[float]]:
    """Extract y_true, y_pred, and y_score from successful predictions."""
    y_true, y_pred, y_score = [], [], []
    for row in predictions:
        if row["status"] == "Success" and row["predicted_label"] is not None:
            y_true.append(row["true_label"])
            y_pred.append(row["predicted_label"])
            # Use score if available, otherwise use predicted_label as score
            score = row.get("score", row["predicted_label"])
            y_score.append(float(score) if score is not None else float(row["predicted_label"]))
    return y_true, y_pred, y_score


# =============================================================================
# Metrics
# =============================================================================

def compute_link_prediction_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute link prediction metrics (binary classification + AUC if scores available)."""
    if not y_pred:
        return {}
    
    metrics = evaluate_binary_classification(y_true, y_pred)
    
    # Add AUC if scores are available
    if y_score is not None and len(y_score) == len(y_true):
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            # Check if we have both classes
            if len(set(y_true)) > 1:
                metrics["auc"] = float(roc_auc_score(y_true, y_score))
                metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass  # Skip AUC if calculation fails
    
    return metrics


def print_link_prediction_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
    method_name: str = "Link Prediction",
) -> Dict[str, float]:
    """Compute and print link prediction metrics (same format as GNN)."""
    metrics = compute_link_prediction_metrics(y_true, y_pred, y_score)
    if not metrics:
        print("No valid predictions.")
        return {}

    # Main metrics (same format as GNN evaluator)
    auc = metrics.get("auc", 0.0)
    f1 = metrics.get("f1", 0.0)
    acc = metrics.get("accuracy", 0.0)
    print(f"\n{method_name}_auc {auc:.4f} | {method_name}_f1 {f1:.4f} | {method_name}_acc {acc:.4f}")
    
    # Additional metrics
    print(f"\n--- {method_name} Full Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("-" * 40)
    return metrics


def compute_degree_buckets(predictions: List[Dict], G) -> Dict[str, np.ndarray]:
    """Compute degree-based buckets for predictions."""
    degrees = dict(G.degree())
    valid_edges = [(r["model_id"], r["dataset_id"]) for r in predictions if r["status"] == "Success"]
    if not valid_edges:
        return {}

    valid_edges = np.array(valid_edges)
    u_degs = np.array([degrees.get(n, 0) for n in valid_edges[:, 0]])
    v_degs = np.array([degrees.get(n, 0) for n in valid_edges[:, 1]])
    edge_min_deg = np.minimum(u_degs, v_degs)

    return {
        "Tail (deg<=5)": edge_min_deg <= 5,
        "Medium (5<deg<=20)": (edge_min_deg > 5) & (edge_min_deg <= 20),
        "Head (deg>20)": edge_min_deg > 20,
    }


def print_degree_metrics(predictions: List[Dict], G, method_name: str = "Baseline") -> Dict[str, float]:
    """Print degree-controlled metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true, y_pred, _ = collect_link_predictions(predictions)
    if not y_pred:
        return {}

    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    buckets = compute_degree_buckets(predictions, G)
    if not buckets:
        return {}

    print(f"\n--- Degree-Controlled ({method_name}) ---")
    metrics = {}
    for name, mask in buckets.items():
        if mask.sum() > 0:
            f1 = f1_score(y_true_np[mask], y_pred_np[mask], zero_division=0)
            acc = accuracy_score(y_true_np[mask], y_pred_np[mask])
            print(f"  [{name}] N={mask.sum()} | F1: {f1:.4f} | Acc: {acc:.4f}")
            metrics[f"f1_{name}"] = f1
            metrics[f"acc_{name}"] = acc
    print("-" * 50)
    return metrics


# =============================================================================
# Save
# =============================================================================

def save_link_predictions(predictions: List[Dict], output_path: str | Path) -> Path:
    """Save link predictions to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(convert_numpy_types(predictions), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
