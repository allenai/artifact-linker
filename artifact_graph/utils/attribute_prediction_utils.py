#!/usr/bin/env python3
"""Shared utilities for attribute prediction scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .evaluation_utils import (
    calculate_mae,
    calculate_mape,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
)
from .link_prediction_utils import convert_numpy_types


def load_attribute_data(
    graph_data_dir: str | Path,
    metric_name: str | None = None,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_attribute_predictions.json",
) -> Tuple[Any, Dict, Dict, List, List[float], List[str]]:
    """
    Load graph and prepare attribute prediction data.

    Returns:
        Tuple of (G, node_metadata, edge_metadata, edges, true_metrics, metric_names)
    """
    from .graph_builder import load_nx_graph
    from .graph_utils import prepare_attribute_predictor_dataset

    G, node_metadata, edge_metadata = load_nx_graph(graph_data_dir=str(graph_data_dir))
    edges, true_metrics, metric_names = prepare_attribute_predictor_dataset(G, metric_name)

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


def create_attribute_row(
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


def collect_valid_attribute_predictions(
    predictions: List[Dict],
) -> Tuple[List[float], List[float]]:
    """Extract valid predictions and ground truth."""
    pred_values, true_values = [], []
    for row in predictions:
        if row["status"] == "Success" and row.get("predicted_value") is not None:
            pred_values.append(row["predicted_value"])
            true_values.append(row["true_value"])
    return pred_values, true_values


def compute_regression_metrics(
    predictions: List[float],
    ground_truth: List[float],
) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        "mse": calculate_mse(predictions, ground_truth),
        "mae": calculate_mae(predictions, ground_truth),
        "rmse": calculate_rmse(predictions, ground_truth),
        "mape": calculate_mape(predictions, ground_truth),
        "r_squared": calculate_r_squared(predictions, ground_truth),
    }


def print_regression_metrics(
    predictions: List[float],
    ground_truth: List[float],
    method_name: str = "Regression",
    total_count: int | None = None,
):
    """Print regression metrics."""
    if not predictions:
        print("No valid predictions produced.")
        return {}

    metrics = compute_regression_metrics(predictions, ground_truth)
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


def save_attribute_predictions(
    predictions: List[Dict],
    output_path: str | Path,
) -> Path:
    """Save attribute predictions to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(convert_numpy_types(predictions), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
