#!/usr/bin/env python3
"""
Calculates regression metrics from a GNN rankings JSON file.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_metrics(file_path: Path):
    """
    Loads rankings from a JSON file and calculates regression metrics.

    Args:
        file_path: Path to the JSON rankings file.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    with file_path.open("r") as f:
        data = json.load(f)

    # Find the list of results
    if isinstance(data, dict):
        results = (
            data.get("results")
            or data.get("detailed_rankings_by_dataset")
            or data.get("records")
        )
    else:
        results = None

    if not results:
        print("Error: Could not find a list of results in the JSON file.")
        return

    y_true = []
    y_pred = []

    for result in results:
        # Find the list of candidates
        candidates = (
            result.get("predicted_ranking")
            or result.get("ranked_candidates")
            or result.get("ranking_candidates")
        )
        if not candidates:
            # Handle the structure of gnn_attribute_predictions.json
            true_val = result.get("ground_truth")
            pred_val = result.get("prediction")
            if true_val is not None and pred_val is not None:
                y_true.append(float(true_val))
                y_pred.append(float(pred_val))
            continue

        for cand in candidates:
            # Find true and predicted values by checking common key names
            true_val = (
                cand.get("ground_truth_score")
                or cand.get("true_value")
                or cand.get("ground_truth_label")
            )
            pred_val = (
                cand.get("predicted_score")
                or cand.get("expected_score")
                or cand.get("predicted_probability")
            )

            if true_val is not None and pred_val is not None:
                y_true.append(float(true_val))
                y_pred.append(float(pred_val))

    if not y_true:
        print("No valid predictions with ground truth and predicted scores found.")
        return

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_np, y_pred_np)

    # Filter out zero values for MAPE calculation to avoid division by zero
    non_zero_mask = y_true_np != 0
    y_true_mape = y_true_np[non_zero_mask]
    y_pred_mape = y_pred_np[non_zero_mask]

    print("--- GNN Ranking Regression Metrics ---")
    print(f"  Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"  Mean Squared Error (MSE):       {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

    if len(y_true_mape) > 0:
        mape = mean_absolute_percentage_error(y_true_mape, y_pred_mape)
        print(f"  Mean Absolute Percentage Error: {mape:.4f}")
        if len(y_true_mape) != len(y_true_np):
            print(
                f"    (calculated on {len(y_true_mape)} of {len(y_true_np)} samples where true value is not zero)"
            )
    else:
        print("  Mean Absolute Percentage Error: Not applicable (all true values are zero)")

    print(f"  R-squared (R^2):                {r2:.4f}")
    print(f"  Mean Absolute Diff:             {mae:.4f}")
    print("--------------------------------------")


def parse_args():
    p = argparse.ArgumentParser(description="Calculate regression metrics for GNN rankings.")
    p.add_argument(
        "--file-path",
        type=Path,
        default="scripts/output/final_results/gnn_attribute_predictions.json",
        help="Path to the GNN rankings JSON file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    calculate_regression_metrics(a.file_path)
