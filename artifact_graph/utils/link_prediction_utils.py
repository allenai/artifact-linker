#!/usr/bin/env python3
"""
Shared utilities for link prediction scripts.

This module provides common functions used across different link prediction methods:
- LLM-based prediction (predict_link_llm.py)
- Baseline/heuristic prediction (predict_link_baseline.py)
- GNN-based prediction (predict_link_gnn.py)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .evaluation_utils import evaluate_binary_classification

if TYPE_CHECKING:
    import networkx as nx

# Type aliases
Edge = Tuple[int, int]
PredictionRow = Dict[str, Any]


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    elif hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    else:
        return obj


def load_prediction_data(
    graph_data_dir: str | Path,
    seed: int = 42,
    max_pairs: int = 5000,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_link_predictions.json",
) -> Tuple[Any, Dict, List[Edge], List[int]]:
    """
    Load graph data and prepare edges for link prediction.

    Args:
        graph_data_dir: Path to graph data directory
        seed: Random seed for reproducibility
        max_pairs: Maximum number of pairs to predict
        use_gnn_data: Whether to use GNN test data instead of sampling
        gnn_data_path: Path to GNN predictions file

    Returns:
        Tuple of (graph, node_metadata, edges, labels)
    """
    # Lazy imports to avoid dependency issues
    from .graph_builder import load_nx_graph
    from .graph_utils import prepare_link_predictor_dataset

    G, node_metadata, _ = load_nx_graph(graph_data_dir=str(graph_data_dir))
    edges, labels = prepare_link_predictor_dataset(G, seed=seed, max_pairs=max_pairs)

    if use_gnn_data:
        with open(gnn_data_path, "r") as f:
            gnn_data = json.load(f)
        gnn_edges = gnn_data["test_predictions"]["edges"]
        labels = [edge["ground_truth"] for edge in gnn_edges]
        edges = [(edge["v_id"], edge["u_id"]) for edge in gnn_edges]

    return G, node_metadata, edges, labels


def create_prediction_row(
    model_id: int,
    dataset_id: int,
    true_label: int,
    node_metadata: Dict,
    predicted_label: Optional[int] = None,
    reason: str = "",
    status: str = "Failed",
) -> PredictionRow:
    """Create a standardized prediction result row."""
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


def save_predictions(
    predictions: List[PredictionRow],
    output_path: str | Path,
    extra_data: Optional[Dict] = None,
) -> Path:
    """
    Save predictions to a JSON file.

    Args:
        predictions: List of prediction rows
        output_path: Output file path
        extra_data: Optional extra data to include

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = convert_numpy_types(predictions)

    if extra_data:
        output_data = {"predictions": serializable, **convert_numpy_types(extra_data)}
    else:
        output_data = serializable

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    return output_path


def collect_valid_predictions(
    predictions: List[PredictionRow],
) -> Tuple[List[int], List[int]]:
    """
    Extract y_true and y_pred from successful predictions.

    Returns:
        Tuple of (y_true, y_pred) lists
    """
    y_true, y_pred = [], []
    for row in predictions:
        if row["status"] == "Success" and row["predicted_label"] is not None:
            y_true.append(row["true_label"])
            y_pred.append(row["predicted_label"])
    return y_true, y_pred


def print_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    method_name: str = "Binary Classification",
) -> Dict[str, float]:
    """
    Compute and print binary classification metrics.

    Returns:
        Dictionary of metrics
    """
    if not y_pred:
        print("No valid predictions produced.")
        return {}

    metrics = evaluate_binary_classification(y_true, y_pred)

    print(f"\n--- {method_name} Metrics ---")
    for k, v in metrics.items():
        print(f"  - {k.capitalize()}: {v:.4f}")
    print("-" * (len(method_name) + 12))

    return metrics


def compute_degree_buckets(
    predictions: List[PredictionRow],
    G,
) -> Dict[str, np.ndarray]:
    """
    Compute degree-based buckets for edge predictions.

    Args:
        predictions: List of prediction rows
        G: NetworkX graph

    Returns:
        Dictionary mapping bucket names to boolean masks
    """
    degrees = dict(G.degree())

    valid_edges = []
    for row in predictions:
        if row["status"] == "Success":
            valid_edges.append((row["model_id"], row["dataset_id"]))

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


def print_degree_controlled_metrics(
    predictions: List[PredictionRow],
    G,
    method_name: str = "Baseline",
) -> Dict[str, float]:
    """
    Print degree-controlled performance metrics.

    Returns:
        Dictionary of bucket-level metrics
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true, y_pred = collect_valid_predictions(predictions)
    if not y_pred:
        return {}

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    buckets = compute_degree_buckets(predictions, G)
    if not buckets:
        return {}

    print(f"\n--- Degree-Controlled Performance ({method_name}) ---")
    metrics = {}

    for name, mask in buckets.items():
        if mask.sum() > 0:
            sub_true = y_true_np[mask]
            sub_pred = y_pred_np[mask]
            sub_f1 = f1_score(sub_true, sub_pred, zero_division=0)
            sub_acc = accuracy_score(sub_true, sub_pred)

            print(f"  [{name}] N={mask.sum()} | F1: {sub_f1:.4f} | Acc: {sub_acc:.4f}")
            metrics[f"f1_{name}"] = sub_f1
            metrics[f"acc_{name}"] = sub_acc

    print("-" * 50)
    return metrics


def create_safe_filename(name: str) -> str:
    """Create a filesystem-safe filename from a string."""
    # Replace problematic characters
    safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_").replace(".", "_")
    return safe
