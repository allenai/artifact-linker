#!/usr/bin/env python3
"""Shared utilities for attribute prediction scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .evaluation_utils import (
    evaluate_regression,
)
from .graph_utils import (
    convert_numpy_types,
    load_attribute_graph_from_split,
    load_split_metric_map,
    normalize_model_dataset_edge,
    select_edge_metric_target,
)

Edge = Tuple[int, int]


# =============================================================================
# Data Loading
# =============================================================================

def load_attribute_prediction_data(
    split_dir: str | Path,
    metric_name: str | None = None,
    metric_file: str = "edge_metadata_normalized.json",
) -> Tuple[Any, Dict, Dict, List, List[float], List[str]]:
    """
    Load graph and prepare attribute prediction data.

    Returns:
        Tuple of (G, node_metadata, edge_metadata, edges, true_metrics, metric_names).
    """
    split_path = Path(split_dir)
    G, node_metadata, edge_metadata = load_attribute_graph_from_split(split_path)

    test_pos = Path(split_path) / "test_split" / "pos_edges.npz"
    test_metrics = load_split_metric_map(split_path, split_name="test_split", metric_file=metric_file)

    pos = np.load(test_pos)["edges"]
    edges: List[Edge] = []
    true_metrics: List[float] = []
    metric_names: List[str] = []
    for i in range(pos.shape[1]):
        u, v = int(pos[0, i]), int(pos[1, i])
        edge = normalize_model_dataset_edge(u, v, node_metadata)
        if edge is None:
            continue

        metrics = test_metrics.get((u, v), test_metrics.get((v, u), {}))
        selected = select_edge_metric_target(metrics, metric_name)
        if selected is None:
            continue
        selected_name, selected_value = selected
        edges.append(edge)
        true_metrics.append(selected_value)
        metric_names.append(selected_name)

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
    return evaluate_regression(predictions, ground_truth)


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
