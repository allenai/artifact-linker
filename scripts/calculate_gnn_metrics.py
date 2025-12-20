#!/usr/bin/env python3
"""
Calculates classification metrics from a GNN predictions JSON file.
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(file_path: Path):
    """
    Loads predictions from a JSON file and calculates classification metrics.

    Args:
        file_path: Path to the JSON predictions file.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    with file_path.open("r") as f:
        data = json.load(f)

    # Try to find the list of predictions in common structures
    if isinstance(data, list):
        predictions = data
    elif isinstance(data, dict):
        predictions = data.get("predictions") or data.get("test_predictions", {}).get("edges")
    else:
        predictions = None

    if not predictions:
        print("Error: Could not find a list of predictions in the JSON file.")
        return

    y_true = []
    y_pred = []

    for pred in predictions:
        # Determine true label key
        if "true_label" in pred:
            true_label_key = "true_label"
        elif "ground_truth" in pred:
            true_label_key = "ground_truth"
        else:
            print(f"Skipping prediction due to missing ground truth key: {pred}")
            continue

        # Determine predicted label key
        if "predicted_label" in pred:
            predicted_label_key = "predicted_label"
        elif "prediction_binary" in pred:
            predicted_label_key = "prediction_binary"
        elif "prediction" in pred:
            predicted_label_key = "prediction"
        else:
            print(f"Skipping prediction due to missing prediction key: {pred}")
            continue

        y_true.append(bool(pred[true_label_key]))
        y_pred.append(bool(pred[predicted_label_key]))

    if not y_true:
        print("No valid predictions found to calculate metrics.")
        return

    breakpoint()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("--- GNN Link Prediction Metrics ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("-----------------------------------")


def parse_args():
    p = argparse.ArgumentParser(description="Calculate metrics for GNN predictions.")
    p.add_argument(
        "--file-path",
        type=Path,
        default="scripts/output/final_results/gnn_link_predictions.json",
        help="Path to the GNN predictions JSON file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    calculate_metrics(a.file_path)
