#!/usr/bin/env python3
"""Shared utilities for link prediction scripts."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .evaluation_utils import evaluate_binary_classification
from .graph_utils import (
    Edge,
    collect_all_split_positives,
    convert_numpy_types,
    generate_negative_edges,
    get_all_model_ids,
    get_test_edges_by_dataset,
    load_link_graph_from_split,
)




# =============================================================================
# Data Loading
# =============================================================================

def load_link_prediction_data(
    split_dir: str | Path,
    seed: int = 42,
) -> Tuple[Any, Dict, List[Edge], List[int]]:
    """
    Load graph and prepare link prediction data.
    
    Args:
        split_dir: Root split directory.
        seed: Random seed.
    Returns:
        Tuple of (G, node_metadata, edges, labels).
    """
    split_dir_path = Path(split_dir)
    G, node_metadata = load_link_graph_from_split(split_dir_path)
    _, all_pos_edges = collect_all_split_positives(split_dir_path, node_metadata)
    test_pos, _, test_datasets = get_test_edges_by_dataset(split_dir_path, node_metadata)
    model_ids = get_all_model_ids(node_metadata)
    neg_edges = generate_negative_edges(test_datasets, model_ids, all_pos_edges)

    edges = test_pos + neg_edges
    labels = [1] * len(test_pos) + [0] * len(neg_edges)

    rng = random.Random(seed)
    combined = list(zip(edges, labels))
    rng.shuffle(combined)
    edges, labels = zip(*combined) if combined else ([], [])
    edges, labels = list(edges), list(labels)

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
    """Compute link prediction metrics: ap_auc, mcc, recall (+ accuracy, precision, f1).

    Matches the GNN evaluator metric set.
    """
    if not y_pred:
        return {}

    return evaluate_binary_classification(y_true, y_pred, pred_scores=y_score)


def print_link_prediction_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
    method_name: str = "Link Prediction",
) -> Dict[str, float]:
    """Compute and print link prediction metrics (same format as GNN evaluator)."""
    metrics = compute_link_prediction_metrics(y_true, y_pred, y_score)
    if not metrics:
        print("No valid predictions.")
        return {}

    # Main metrics — same as GNN evaluator: ap_auc, mcc, recall
    ap_auc = metrics.get("ap_auc", 0.0)
    mcc = metrics.get("mcc", 0.0)
    recall = metrics.get("recall", 0.0)
    print(f"\n{method_name}_ap_auc {ap_auc:.4f} | "
          f"{method_name}_mcc {mcc:.4f} | "
          f"{method_name}_recall {recall:.4f}")

    # Full metrics
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
    """Print degree-controlled metrics (ap_auc, mcc, recall — matching GNN evaluator)."""
    from sklearn.metrics import average_precision_score, matthews_corrcoef, recall_score

    y_true, y_pred, y_score = collect_link_predictions(predictions)
    if not y_pred:
        return {}

    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    y_score_np = np.array(y_score) if y_score else None
    buckets = compute_degree_buckets(predictions, G)
    if not buckets:
        return {}

    print(f"\n--- Degree-Controlled ({method_name}) ---")
    metrics = {}
    for name, mask in buckets.items():
        if mask.sum() > 0:
            sub_true = y_true_np[mask]
            sub_pred = y_pred_np[mask]

            mcc = float(matthews_corrcoef(sub_true, sub_pred))
            rec = float(recall_score(sub_true, sub_pred, zero_division=0))
            metrics[f"mcc_{name}"] = mcc
            metrics[f"recall_{name}"] = rec

            ap = 0.0
            if y_score_np is not None and len(set(sub_true)) > 1:
                try:
                    ap = float(average_precision_score(sub_true, y_score_np[mask]))
                except Exception:
                    pass
            metrics[f"ap_auc_{name}"] = ap

            print(f"  [{name}] N={mask.sum()} | AP-AUC: {ap:.4f} | MCC: {mcc:.4f} | Recall: {rec:.4f}")
    print("-" * 50)
    return metrics


# =============================================================================
# Save
# =============================================================================

def save_link_predictions(predictions: List[Dict], output_path: str | Path) -> Path:
    """Save link predictions to JSON.

    For large prediction sets (>100k entries), saves only computed metrics
    instead of all individual predictions to avoid multi-GB files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = convert_numpy_types(predictions)

    if len(converted) > 100_000:
        # Save compact metrics-only summary for large files
        y_true, y_pred, y_score = collect_link_predictions(predictions)
        metrics = compute_link_prediction_metrics(y_true, y_pred, y_score)
        summary = {
            "test_metrics": metrics,
            "num_predictions": len(converted),
            "num_positive": sum(1 for p in converted if p.get("true_label") == 1),
            "num_negative": sum(1 for p in converted if p.get("true_label") == 0),
        }
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2)
    else:
        with output_path.open("w") as f:
            json.dump(converted, f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
