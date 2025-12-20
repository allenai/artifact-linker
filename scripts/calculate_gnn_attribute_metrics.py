#!/usr/bin/env python3
"""
Calculates a comprehensive set of ranking and correlation metrics for GNN attribute rankings.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from artifact_graph.utils.evaluation_utils import (
    calculate_map_continuous,
    calculate_ndcg_standard,
    calculate_ranking_correlation,
)


def calculate_attribute_ranking_metrics(file_path: Path):
    """
    Loads GNN attribute rankings from a JSON file and calculates a suite of metrics.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    with file_path.open("r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("Error: Could not find 'results' in the JSON file.")
        return

    # Initialize lists to store metrics for averaging
    metrics = {
        "ndcg_1": [], "ndcg_3": [], "ndcg_5": [], "ndcg_full": [],
        "map_1": [], "map_3": [], "map_5": [], "map_full": [],
        "hit_1": [], "hit_3": [], "hit_5": [],
        "recall_1": [], "recall_3": [], "recall_5": [],
        "kendall_tau": [], "spearman_rho": [], "pearson_r": [],
    }

    for result in results:
        ranked_models = result.get("predicted_ranking", [])
        if not ranked_models:
            continue

        predicted_items_with_scores = []
        ground_truth = {}
        dataset_id = result["dataset_id"]

        for item in ranked_models:
            model_id = item["model_id"]
            item_key = f"{model_id}_{dataset_id}"
            predicted_items_with_scores.append((item_key, item.get("predicted_score", 0)))
            ground_truth[item_key] = item.get("ground_truth_score", 0)

        if not ground_truth:
            continue

        try:
            # NDCG@k
            metrics["ndcg_1"].append(calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=1))
            metrics["ndcg_3"].append(calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=3))
            metrics["ndcg_5"].append(calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=5))
            metrics["ndcg_full"].append(calculate_ndcg_standard(predicted_items_with_scores, ground_truth))

            # MAP@k
            metrics["map_1"].append(calculate_map_continuous(predicted_items_with_scores, ground_truth, k=1))
            metrics["map_3"].append(calculate_map_continuous(predicted_items_with_scores, ground_truth, k=3))
            metrics["map_5"].append(calculate_map_continuous(predicted_items_with_scores, ground_truth, k=5))
            metrics["map_full"].append(calculate_map_continuous(predicted_items_with_scores, ground_truth))

            # Correlation metrics
            correlation = calculate_ranking_correlation(predicted_items_with_scores, ground_truth)
            for key in ["hit_at_1", "hit_at_3", "hit_at_5", "recall_at_1", "recall_at_3", "recall_at_5", "kendall_tau", "spearman_rho", "pearson_r"]:
                metric_key = key.replace("hit_at_", "hit_").replace("recall_at_", "recall_")
                if key in correlation:
                    metrics[metric_key].append(correlation[key])

        except Exception as e:
            print(f"Warning: Could not calculate metrics for dataset {dataset_id}: {e}")

    print("--- GNN Attribute Ranking Metrics ---")
    if not any(metrics.values()):
        print("No valid data found to calculate metrics.")
        return

    # Print NDCG and MAP
    print("\n  === NDCG & MAP ===")
    for k in [1, 3, 5, "full"]:
        print(f"  NDCG@{k}:    {np.mean(metrics[f'ndcg_{k}']):.4f}")
    for k in [1, 3, 5, "full"]:
        print(f"  MAP@{k}:     {np.mean(metrics[f'map_{k}']):.4f}")
    
    # Print Hit and Recall
    print("\n  === Hit & Recall ===")
    for k in [1, 3, 5]:
        print(f"  Hit@{k}:      {np.mean(metrics[f'hit_{k}']):.4f}")
    for k in [1, 3, 5]:
        print(f"  Recall@{k}:   {np.mean(metrics[f'recall_{k}']):.4f}")
        
    # Print Correlation
    print("\n  === Correlation ===")
    print(f"  Kendall's Tau: {np.mean(metrics['kendall_tau']):.4f}")
    print(f"  Spearman's Rho: {np.mean(metrics['spearman_rho']):.4f}")
    print(f"  Pearson's R:    {np.mean(metrics['pearson_r']):.4f}")
    
    print("\n------------------------------------")


def parse_args():
    p = argparse.ArgumentParser(description="Calculate metrics for GNN attribute rankings.")
    p.add_argument(
        "--file-path",
        type=Path,
        default="scripts/output/final_results/gnn_attribute_rankings.json",
        help="Path to the GNN attribute rankings JSON file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    calculate_attribute_ranking_metrics(a.file_path)
