#!/usr/bin/env python3
"""Shared utilities for attribute ranking scripts."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from .evaluation_utils import (
    calculate_map_continuous,
    calculate_ndcg_standard,
    calculate_pairwise_accuracy,
    calculate_ranking_correlation,
    calculate_regret_at_k,
)
from .link_prediction_utils import convert_numpy_types


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_attribute_ranker_dataset(
    G: nx.Graph,
    metric_name: str | None = None,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, str]]:
    """
    Prepare attribute ranking dataset.
    
    For each dataset, find all connected models and their metric values.

    Args:
        G: NetworkX graph with edge attributes.
        metric_name: Metric to use (None = auto-select most frequent per dataset).

    Returns:
        Tuple of (ranking_data, dataset_metrics) where:
        - ranking_data: {dataset_id: [(model_id, metric_value), ...]}
        - dataset_metrics: {dataset_id: metric_name}
    """
    dataset_edges = defaultdict(list)

    for u, v, data in G.edges(data=True):
        u_type, v_type = G.nodes[u].get("type"), G.nodes[v].get("type")
        if u_type == "model" and v_type == "dataset":
            dataset_edges[v].append((u, data))
        elif v_type == "model" and u_type == "dataset":
            dataset_edges[u].append((v, data))

    ranking_data = {}
    dataset_metrics = {}

    for dataset_id, edges in dataset_edges.items():
        if metric_name is None:
            metric_counter = Counter()
            for _, edge_data in edges:
                for key, value in edge_data.items():
                    if isinstance(value, (int, float)) and not key.startswith("_"):
                        metric_counter[key] += 1
            if not metric_counter:
                continue
            selected_metric = metric_counter.most_common(1)[0][0]
        else:
            selected_metric = metric_name

        dataset_data = [
            (model_id, float(edge_data[selected_metric]))
            for model_id, edge_data in edges
            if selected_metric in edge_data
        ]

        if dataset_data:
            dataset_data.sort(key=lambda x: x[1], reverse=True)
            ranking_data[dataset_id] = dataset_data
            dataset_metrics[dataset_id] = selected_metric

    return ranking_data, dataset_metrics


# =============================================================================
# Data Loading
# =============================================================================

def load_attribute_ranking_data(
    graph_data_dir: str | Path,
    metric_name: str | None = None,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_attribute_rankings.json",
) -> Tuple[Any, Dict, Dict, Dict, Dict]:
    """
    Load graph and prepare attribute ranking data.

    Returns:
        Tuple of (G, node_metadata, edge_metadata, ranking_data, dataset_metrics).
    """
    from .graph_builder import load_nx_graph

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


# =============================================================================
# Row Creation
# =============================================================================

def create_attribute_ranking_row(
    dataset_id: int,
    metric_used: str,
    ranked_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a standardized attribute ranking result row."""
    return {
        "dataset_id": dataset_id,
        "metric_used": metric_used,
        "ranked_models": ranked_models,
    }


# =============================================================================
# Collect Valid Results
# =============================================================================

def collect_attribute_rankings(results: List[Dict]) -> List[Dict]:
    """Extract valid ranking results (those with ranked models)."""
    return [r for r in results if r and r.get("ranked_models")]


# =============================================================================
# Metrics
# =============================================================================

def compute_attribute_ranking_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 5, 10, 20, 50, 100],
) -> Dict[str, float]:
    """Compute attribute ranking metrics (NDCG, MAP, correlations, hit@k, regret, pairwise accuracy)."""
    valid = collect_attribute_rankings(results)
    if not valid:
        return {}

    all_metrics: Dict[str, List[float]] = {
        "kendall_tau": [], "spearman_rho": [], "pearson_r": [],
        **{f"ndcg@{k}": [] for k in k_values}, "ndcg": [],
        **{f"map@{k}": [] for k in k_values}, "map": [],
        "hit@1": [], "hit@3": [],
        "top_1_overlap": [], "top_3_overlap": [],
        "regret@1": [],
        "pairwise_accuracy": [],
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

            # --- Additional discriminative metrics ---
            # Hit@k and Top-k overlap from correlation result
            for key in ["hit_at_1", "hit_at_3", "top_1_overlap", "top_3_overlap"]:
                if key in corr:
                    mapped = key.replace("hit_at_", "hit@").replace("_overlap", "_overlap")
                    all_metrics[mapped].append(corr[key])

            # Regret@1: how much worse is the predicted top-1 vs actual best
            all_metrics["regret@1"].append(
                calculate_regret_at_k(items_with_scores, ground_truth, k=1)
            )

            # Pairwise accuracy: fraction of correctly ordered pairs
            all_metrics["pairwise_accuracy"].append(
                calculate_pairwise_accuracy(items_with_scores, ground_truth)
            )

        except Exception:
            pass

    return {k: sum(v) / len(v) for k, v in all_metrics.items() if v}


def print_attribute_ranking_metrics(results: List[Dict], method_name: str = "Attribute Ranking"):
    """Print attribute ranking metrics."""
    metrics = compute_attribute_ranking_metrics(results)
    valid_count = len(collect_attribute_rankings(results))

    print(f"\n--- {method_name} Metrics ---")

    print("  Correlations:")
    for key in ["kendall_tau", "spearman_rho", "pearson_r"]:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    print("  Hit/Overlap:")
    for key in ["hit@1", "hit@3", "top_1_overlap", "top_3_overlap"]:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    print("  Pairwise/Regret:")
    for key in ["pairwise_accuracy", "regret@1"]:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    print("  NDCG/MAP:")
    for key in sorted(k for k in metrics if k.startswith(("ndcg", "map"))):
        print(f"    {key}: {metrics[key]:.4f}")

    print(f"  Valid: {valid_count}/{len(results)}")
    print("-" * 40)
    return metrics


# =============================================================================
# Save
# =============================================================================

def save_attribute_rankings(results: List[Dict], output_path: str | Path) -> Path:
    """Save attribute rankings to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
