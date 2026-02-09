#!/usr/bin/env python3
"""Shared utilities for attribute prediction scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from .evaluation_utils import (
    calculate_mae,
    calculate_mape,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
)
from .link_prediction_utils import convert_numpy_types

Edge = Tuple[int, int]


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_attribute_prediction_data(
    G: nx.Graph,
    metric_name: str | None = None,
) -> Tuple[List[Edge], List[float], List[str]]:
    """
    Prepare attribute prediction dataset.
    
    Returns all model-dataset edges with their metric values.
    """
    edges_to_predict: List[Edge] = []
    true_metrics: List[float] = []
    metric_names: List[str] = []

    for u, v, data in G.edges(data=True):
        u_type, v_type = G.nodes[u].get("type"), G.nodes[v].get("type")

        if u_type == "model" and v_type == "dataset":
            edge = (u, v)
        elif v_type == "model" and u_type == "dataset":
            edge = (v, u)
        else:
            continue

        if metric_name is not None:
            if metric_name in data:
                edges_to_predict.append(edge)
                true_metrics.append(float(data[metric_name]))
                metric_names.append(metric_name)
        else:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    edges_to_predict.append(edge)
                    true_metrics.append(float(value))
                    metric_names.append(key)

    return edges_to_predict, true_metrics, metric_names


# =============================================================================
# Data Loading
# =============================================================================

def load_attribute_prediction_data(
    graph_data_dir: str | Path,
    metric_name: str | None = None,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_attribute_predictions.json",
) -> Tuple[Any, Dict, Dict, List, List[float], List[str]]:
    """
    Load graph and prepare attribute prediction data.

    Returns:
        Tuple of (G, node_metadata, edge_metadata, edges, true_metrics, metric_names).
    """
    from .graph_builder import load_nx_graph

    G, node_metadata, edge_metadata = load_nx_graph(graph_data_dir=str(graph_data_dir))
    edges, true_metrics, metric_names = prepare_attribute_prediction_data(G, metric_name)

    if use_gnn_data:
        metric_names = []
        real_edges = []
        with open(gnn_data_path, "r") as f:
            gnn_data = json.load(f)
        gnn_edges = gnn_data["records"]
        true_metrics = [e["ground_truth"] for e in gnn_edges]

        for edge, metric_num in zip(gnn_edges, true_metrics):
            e1 = tuple(edge["input"]["edge"])
            e2 = tuple(edge["input"]["edge"][::-1])
            found_edge = e1 if e1 in edge_metadata else e2 if e2 in edge_metadata else None
            if not found_edge:
                raise ValueError(f"Edge {e1} not found")
            real_edges.append(found_edge)

            for mn, mv in edge_metadata[found_edge]["metrics"].items():
                if abs(mv - metric_num) < 1e-3:
                    metric_names.append(mn)
                    break

        edges = real_edges

    return G, node_metadata, edge_metadata, edges, true_metrics, metric_names


# =============================================================================
# Row Creation
# =============================================================================

def create_attribute_prediction_row(
    model_id: int,
    dataset_id: int,
    metric_name: str,
    true_value: float,
    node_metadata: Dict,
    predicted_value: float | None = None,
    reason: str = "",
    status: str = "Failed",
) -> Dict[str, Any]:
    """Create a standardized attribute prediction row."""
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_name": node_metadata.get(model_id, {}).get("name"),
        "dataset_name": node_metadata.get(dataset_id, {}).get("name"),
        "metric_name": metric_name,
        "true_value": true_value,
        "predicted_value": predicted_value,
        "reason": reason,
        "status": status,
    }


# =============================================================================
# Collect Valid Results
# =============================================================================

def collect_attribute_predictions(predictions: List[Dict]) -> Tuple[List[float], List[float]]:
    """Extract valid predictions and ground truth."""
    pred_values, true_values = [], []
    for row in predictions:
        if row["status"] == "Success" and row.get("predicted_value") is not None:
            pred_values.append(row["predicted_value"])
            true_values.append(row["true_value"])
    return pred_values, true_values


# =============================================================================
# Metrics
# =============================================================================

def compute_attribute_prediction_metrics(
    predictions: List[float],
    ground_truth: List[float],
) -> Dict[str, float]:
    """Compute attribute prediction metrics (regression)."""
    if not predictions:
        return {}
    return {
        "mse": calculate_mse(predictions, ground_truth),
        "mae": calculate_mae(predictions, ground_truth),
        "rmse": calculate_rmse(predictions, ground_truth),
        "mape": calculate_mape(predictions, ground_truth),
        "r_squared": calculate_r_squared(predictions, ground_truth),
    }


def print_attribute_prediction_metrics(
    predictions: List[float],
    ground_truth: List[float],
    method_name: str = "Attribute Prediction",
    total_count: int | None = None,
) -> Dict[str, float]:
    """Compute and print attribute prediction metrics."""
    metrics = compute_attribute_prediction_metrics(predictions, ground_truth)
    if not metrics:
        print("No valid predictions.")
        return {}

    total = total_count or len(predictions)

    print(f"\n--- {method_name} Metrics ---")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    if metrics["mape"] != float("inf"):
        print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r_squared']:.4f}")
    print(f"  Valid: {len(predictions)}/{total}")
    print("-" * 40)

    return metrics


# =============================================================================
# Save
# =============================================================================

def save_attribute_predictions(predictions: List[Dict], output_path: str | Path) -> Path:
    """Save attribute predictions to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(convert_numpy_types(predictions), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
