#!/usr/bin/env python3
"""
Calculates ranking metrics (Recall@k and Precision@k) from a GNN rankings JSON file.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def calculate_precision_at_k(predicted_ranking, ground_truth_positives, k):
    """Calculates Precision@k."""
    top_k = predicted_ranking[:k]
    true_positives = len(set(top_k) & set(ground_truth_positives))
    return true_positives / k if k > 0 else 0


def calculate_recall_at_k(predicted_ranking, ground_truth_positives, k):
    """Calculates Recall@k."""
    top_k = predicted_ranking[:k]
    true_positives = len(set(top_k) & set(ground_truth_positives))
    total_positives = len(ground_truth_positives)
    return true_positives / total_positives if total_positives > 0 else 0


def calculate_ranking_metrics(file_path: Path):
    """
    Loads rankings from a JSON file and calculates ranking metrics.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    with file_path.open("r") as f:
        data = json.load(f)

    results = data.get("detailed_rankings_by_dataset", [])
    if not results:
        print("Error: Could not find 'detailed_rankings_by_dataset' in the JSON file.")
        return

    k_values = [1, 3, 5, 10]
    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}

    for result in results:
        candidates = result.get("ranked_candidates", [])
        if not candidates:
            continue

        # Predicted ranking is based on the order in the file
        predicted_ranking = [cand["model_id"] for cand in candidates]
        ground_truth_positives = [
            cand["model_id"] for cand in candidates if cand["ground_truth_label"] == 1
        ]

        if not ground_truth_positives:
            continue  # Skip if there are no positive examples for this dataset

        for k in k_values:
            precision = calculate_precision_at_k(
                predicted_ranking, ground_truth_positives, k
            )
            recall = calculate_recall_at_k(predicted_ranking, ground_truth_positives, k)
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)

    print("--- GNN Link Ranking Metrics ---")
    if not any(all_precisions.values()):
        print("No valid data found to calculate ranking metrics.")
        return

    for k in k_values:
        avg_precision = np.mean(all_precisions[k]) if all_precisions[k] else 0
        print(f"  Precision@{k}: {avg_precision:.4f}")

    for k in k_values:
        avg_recall = np.mean(all_recalls[k]) if all_recalls[k] else 0
        print(f"  Recall@{k}:    {avg_recall:.4f}")
    print("---------------------------------")


def parse_args():
    p = argparse.ArgumentParser(description="Calculate ranking metrics for GNN.")
    p.add_argument(
        "--file-path",
        type=Path,
        default="scripts/output/final_results/gnn_link_rankings.json",
        help="Path to the GNN rankings JSON file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    calculate_ranking_metrics(a.file_path)
