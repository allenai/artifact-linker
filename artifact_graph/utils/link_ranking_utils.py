#!/usr/bin/env python3
"""Shared utilities for link ranking scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from .evaluation_utils import (
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from .link_prediction_utils import convert_numpy_types


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_link_ranker_dataset(G: nx.Graph) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    Prepare link ranking dataset using ALL models as candidates.
    
    For each dataset:
    - Positives: Models with edges to this dataset.
    - Negatives: ALL other models.

    Returns:
        Dict mapping dataset_id to (positive_models, negative_models).
    """
    models = set(n for n, d in G.nodes(data=True) if d.get("type") == "model")
    datasets = [n for n, d in G.nodes(data=True) if d.get("type") == "dataset"]

    ranking_data = {}
    for dataset_id in datasets:
        positive_models = [
            n for n in G.neighbors(dataset_id)
            if G.nodes[n].get("type") == "model"
        ]
        if positive_models:
            negative_models = list(models - set(positive_models))
            ranking_data[dataset_id] = (positive_models, negative_models)

    return ranking_data


# =============================================================================
# Data Loading
# =============================================================================

def load_link_ranking_data(
    graph_data_dir: str | Path,
    use_gnn_data: bool = False,
    gnn_data_path: str = "output/final_results/gnn_link_rankings.json",
    split_dir: str | Path | None = None,
) -> Tuple[Any, Dict, Dict]:
    """
    Load graph and prepare link ranking data.
    
    Uses ALL models as candidates (full negative).
    If split_dir is provided, only uses test split datasets for fair comparison.

    Returns:
        Tuple of (G, node_metadata, ranking_data).
    """
    import numpy as np
    from .graph_builder import load_nx_graph

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
    elif split_dir:
        # Use only test split datasets for fair comparison with GNN
        split_dir = Path(split_dir)
        test_pos = np.load(split_dir / "test_split" / "pos_edges.npz")["edges"]
        
        # Build test positive edges by dataset
        test_pos_by_ds = {}
        for i in range(test_pos.shape[1]):
            u, v = int(test_pos[0, i]), int(test_pos[1, i])
            ut = G.nodes.get(u, {}).get("type")
            vt = G.nodes.get(v, {}).get("type")
            if ut == "dataset" and vt == "model":
                test_pos_by_ds.setdefault(u, set()).add(v)
            elif ut == "model" and vt == "dataset":
                test_pos_by_ds.setdefault(v, set()).add(u)
        
        # Build all positive edges by dataset (from full graph)
        all_pos_by_ds = {}
        for u, v in G.edges():
            ut = G.nodes[u].get("type")
            vt = G.nodes[v].get("type")
            if ut == "dataset" and vt == "model":
                all_pos_by_ds.setdefault(u, set()).add(v)
            elif ut == "model" and vt == "dataset":
                all_pos_by_ds.setdefault(v, set()).add(u)
        
        # Get all models for negative candidates
        all_models = {n for n, d in G.nodes(data=True) if d.get("type") == "model"}
        
        # Build ranking data: only datasets in test split
        ranking_data = {}
        for did, pos_models in test_pos_by_ds.items():
            neg_models = list(all_models - all_pos_by_ds.get(did, set()))
            if neg_models:
                ranking_data[did] = (list(pos_models), neg_models)
        
        print(f"Using test split: {len(ranking_data)} datasets (from {len(test_pos_by_ds)} in test)")
    else:
        ranking_data = prepare_link_ranker_dataset(G)

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

def save_link_rankings(results: List[Dict], output_path: str | Path) -> Path:
    """Save link rankings to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)

    print(f"💾 Saved: {output_path}")
    return output_path
