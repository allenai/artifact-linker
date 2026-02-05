#!/usr/bin/env python3
"""Shared utilities for ranking scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .evaluation_utils import (
    calculate_map_continuous,
    calculate_ndcg_standard,
    calculate_precision_at_k,
    calculate_ranking_correlation,
    calculate_recall_at_k,
)
from .link_prediction_utils import convert_numpy_types


def load_link_ranking_data(
    graph_data_dir: str | Path,
    seed: int = 42,
    candidates_per_dataset: int = 10,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_link_rankings.json",
) -> Tuple[Any, Dict, Dict]:
    """
    Load graph and prepare link ranking data.

    Returns:
        Tuple of (G, node_metadata, ranking_data)
        ranking_data: {dataset_id: (positive_models, negative_models)}
    """
    from .graph_builder import load_nx_graph
    from .graph_utils import prepare_link_ranker_dataset

    G, node_metadata, _ = load_nx_graph(graph_data_dir=str(graph_data_dir))

    if use_gnn_data:
        with open(gnn_data_path, "r") as f:
            gnn_data = json.load(f)
        ranking_data = {}
        for result in gnn_data["detailed_rankings_by_dataset"]:
            dataset_id = result["dataset_id"]
            pos = [c["model_id"] for c in result["ranked_candidates"] if c["ground_truth_label"]]
            neg = [c["model_id"] for c in result["ranked_candidates"] if not c["ground_truth_label"]]
            ranking_data[dataset_id] = (pos, neg)
    else:
        ranking_data = prepare_link_ranker_dataset(G, seed=seed, candidates_per_dataset=candidates_per_dataset)

    return G, node_metadata, ranking_data


def load_attribute_ranking_data(
    graph_data_dir: str | Path,
    metric_name: str | None = None,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_attribute_rankings.json",
) -> Tuple[Any, Dict, Dict, Dict, Dict]:
    """
    Load graph and prepare attribute ranking data.

    Returns:
        Tuple of (G, node_metadata, edge_metadata, ranking_data, dataset_metrics)
    """
    from .graph_builder import load_nx_graph
    from .graph_utils import prepare_attribute_ranker_dataset

    G, node_metadata, edge_metadata = load_nx_graph(graph_data_dir=str(graph_data_dir))

    if use_gnn_data:
        with open(gnn_data_path, "r") as f:
            gnn_data = json.load(f)
        ranking_data = {}
        dataset_metrics = {}
        for result in gnn_data["results"]:
            dataset_id = result["dataset_id"]
            ranking_data[dataset_id] = [(m["model_id"], m["predicted_score"]) for m in result["predicted_ranking"]]
            dataset_metrics[dataset_id] = result["metric_used"]
    else:
        ranking_data, dataset_metrics = prepare_attribute_ranker_dataset(G, metric_name)

    return G, node_metadata, edge_metadata, ranking_data, dataset_metrics


def compute_link_ranking_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute link ranking metrics (Recall@k, Precision@k)."""
    valid = [r for r in results if r and r.get("ranked_model_ids")]
    if not valid:
        return {}

    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})

    for r in valid:
        ranked = r["ranked_model_ids"]
        positives = set(r["positive_models"])

        for k in k_values:
            metrics[f"recall@{k}"].append(calculate_recall_at_k(ranked, positives, k))
            metrics[f"precision@{k}"].append(calculate_precision_at_k(ranked, positives, k))

    return {k: sum(v) / len(v) for k, v in metrics.items() if v}


def compute_attribute_ranking_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute attribute ranking metrics (NDCG, MAP, correlations)."""
    valid = [r for r in results if r and r.get("ranked_models")]
    if not valid:
        return {}

    all_metrics: Dict[str, List[float]] = {
        "kendall_tau": [], "spearman_rho": [], "pearson_r": [],
        **{f"ndcg@{k}": [] for k in k_values}, "ndcg": [],
        **{f"map@{k}": [] for k in k_values}, "map": [],
    }

    for r in valid:
        ranked = r["ranked_models"]
        if len(ranked) < 2:
            continue

        items_with_scores = []
        ground_truth = {}

        for item in ranked:
            key = f"{item['model_id']}_{r['dataset_id']}"
            items_with_scores.append((key, item.get("expected_score", 0)))
            ground_truth[key] = item["true_value"]

        try:
            for k in k_values:
                all_metrics[f"ndcg@{k}"].append(calculate_ndcg_standard(items_with_scores, ground_truth, k=k))
                all_metrics[f"map@{k}"].append(calculate_map_continuous(items_with_scores, ground_truth, k=k))
            all_metrics["ndcg"].append(calculate_ndcg_standard(items_with_scores, ground_truth))
            all_metrics["map"].append(calculate_map_continuous(items_with_scores, ground_truth))

            corr = calculate_ranking_correlation(items_with_scores, ground_truth)
            for key in ["kendall_tau", "spearman_rho", "pearson_r"]:
                if key in corr:
                    all_metrics[key].append(corr[key])
        except Exception:
            pass

    return {k: sum(v) / len(v) for k, v in all_metrics.items() if v}


def print_link_ranking_metrics(results: List[Dict], method_name: str = "Link Ranking"):
    """Print link ranking metrics."""
    metrics = compute_link_ranking_metrics(results)
    valid_count = len([r for r in results if r and r.get("ranked_model_ids")])

    print(f"\n--- {method_name} Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k.upper()}: {v:.4f}")
    print(f"  Valid: {valid_count}/{len(results)}")
    print("-" * 40)
    return metrics


def print_attribute_ranking_metrics(results: List[Dict], method_name: str = "Attribute Ranking"):
    """Print attribute ranking metrics."""
    metrics = compute_attribute_ranking_metrics(results)
    valid_count = len([r for r in results if r and r.get("ranked_models")])

    print(f"\n--- {method_name} Metrics ---")

    # Correlation metrics
    print("  Correlations:")
    for key in ["kendall_tau", "spearman_rho", "pearson_r"]:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    # NDCG/MAP
    print("  NDCG/MAP:")
    for key in sorted(k for k in metrics if k.startswith(("ndcg", "map"))):
        print(f"    {key}: {metrics[key]:.4f}")

    print(f"  Valid: {valid_count}/{len(results)}")
    print("-" * 40)
    return metrics


def save_rankings(
    results: List[Dict],
    output_path: str | Path,
) -> Path:
    """Save rankings to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
