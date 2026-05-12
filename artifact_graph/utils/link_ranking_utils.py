#!/usr/bin/env python3
"""Shared utilities for link ranking scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .evaluation_utils import (
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from .graph_utils import (
    collect_all_split_positives,
    load_link_graph_from_split,
    get_all_model_ids,
    get_test_edges_by_dataset,
    convert_numpy_types,
)

# =============================================================================
# Data Loading
# =============================================================================

def load_link_ranking_data(
    split_dir: str | Path,
) -> Tuple[Any, Dict, Dict]:
    """
    Load graph and prepare link ranking data.
    
    Uses ALL models as candidates (full negative) for test datasets in split_dir.

    Returns:
        Tuple of (G, node_metadata, ranking_data).
    """
    split_dir_path = Path(split_dir)
    G, node_metadata = load_link_graph_from_split(split_dir_path)

    all_pos_by_ds, _ = collect_all_split_positives(split_dir_path, node_metadata)
    _, test_pos_by_ds, _ = get_test_edges_by_dataset(split_dir_path, node_metadata)
    all_models = get_all_model_ids(node_metadata)

    ranking_data = {}
    for did, pos_models in test_pos_by_ds.items():
        neg_models = list(all_models - all_pos_by_ds.get(did, set()))
        if neg_models:
            ranking_data[did] = (list(pos_models), neg_models)
    
    print(f"Using test split: {len(ranking_data)} datasets (from {len(test_pos_by_ds)} in test)")

    return G, node_metadata, ranking_data


# =============================================================================
# Row Creation
# =============================================================================

def create_link_ranking_row(
    dataset_id: int,
    positive_models: List[int],
    ranked_model_ids: List[int],
) -> Dict[str, Any]:
    """Create a standardized link ranking result row."""
    return {
        "dataset_id": dataset_id,
        "positive_models": positive_models,
        "ranked_model_ids": ranked_model_ids,
    }


# =============================================================================
# Collect Valid Results
# =============================================================================

def collect_link_rankings(results: List[Dict]) -> List[Dict]:
    """Extract valid ranking results (those with ranked models and positive models)."""
    return [r for r in results if r and r.get("ranked_model_ids") and r.get("positive_models")]


# =============================================================================
# Metrics
# =============================================================================

def compute_link_ranking_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 5, 10, 20, 50, 100],
) -> Dict[str, float]:
    """Compute link ranking metrics (Recall@k, Precision@k, MRR, Hit@k, NDCG@k)."""
    valid = collect_link_rankings(results)
    if not valid:
        return {}

    metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics.update({f"hit@{k}": [] for k in k_values})
    metrics["mrr"] = []

    for r in valid:
        ranked = r["ranked_model_ids"]
        positives = set(r["positive_models"])

        for k in k_values:
            metrics[f"recall@{k}"].append(calculate_recall_at_k(ranked, positives, k))
            metrics[f"precision@{k}"].append(calculate_precision_at_k(ranked, positives, k))
            metrics[f"ndcg@{k}"].append(calculate_ndcg(ranked, positives, k))
            # Hit@k: is any positive model in top-k?
            hit = 1.0 if any(m in positives for m in ranked[:k]) else 0.0
            metrics[f"hit@{k}"].append(hit)

        metrics["mrr"].append(calculate_mrr(ranked, positives))

    return {k: sum(v) / len(v) for k, v in metrics.items() if v}


def print_link_ranking_metrics(results: List[Dict], method_name: str = "Link Ranking"):
    """Print link ranking metrics."""
    metrics = compute_link_ranking_metrics(results)
    valid_count = len(collect_link_rankings(results))

    print(f"\n--- {method_name} Metrics ---")

    # MRR first
    if "mrr" in metrics:
        print(f"  MRR: {metrics['mrr']:.4f}")

    # Group by metric type
    for prefix in ["hit@", "ndcg@", "recall@", "precision@"]:
        keys = sorted([k for k in metrics if k.startswith(prefix)],
                      key=lambda x: int(x.split("@")[1]))
        if keys:
            print(f"  {prefix.rstrip('@').upper()}:")
            for k in keys:
                print(f"    {k}: {metrics[k]:.4f}")

    print(f"  Valid: {valid_count}/{len(results)}")
    print("-" * 40)
    return metrics


# =============================================================================
# Save
# =============================================================================

def save_link_rankings(results, output_path: str | Path) -> Path:
    """Save link rankings to JSON.

    For large result sets, saves only summary metrics to avoid multi-hundred-MB files.
    Accepts either a list of ranking dicts or a dict with a 'results' key.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the ranking list for metric computation
    if isinstance(results, dict) and "results" in results:
        ranking_list = results["results"]
    elif isinstance(results, list):
        ranking_list = results
    else:
        ranking_list = []

    if len(ranking_list) > 100:
        metrics = compute_link_ranking_metrics(ranking_list)
        valid_count = len(collect_link_rankings(ranking_list))
        summary = {
            "test_metrics": metrics,
            "num_queries": len(ranking_list),
            "num_valid": valid_count,
        }
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2)
    else:
        with output_path.open("w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
