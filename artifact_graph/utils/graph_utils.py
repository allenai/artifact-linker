#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import networkx as nx

Edge = Tuple[int, int]


def prepare_link_predictor_dataset(
    G: nx.Graph,
    seed: int = 42,
    max_pairs: int = 0,
    balance_ratio: float = 1.0,
) -> Tuple[List[Edge], List[int]]:
    rng = random.Random(seed)

    # Extract model and dataset node IDs from graph attributes
    models = [node for node, data in G.nodes(data=True) if data.get("type") == "model"]
    datasets = [node for node, data in G.nodes(data=True) if data.get("type") == "dataset"]

    # Prepare positive edges using integer IDs
    pos_edges: Set[Edge] = set()
    for u, v in G.edges():
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")
        # Ensure consistent order (model, dataset)
        if u_type == "model" and v_type == "dataset":
            pos_edges.add((u, v))

    # Sample negative edges
    num_negatives = int(len(pos_edges) * balance_ratio)
    neg_edges: Set[Edge] = set()

    if models and datasets:
        while len(neg_edges) < num_negatives:
            model_id = rng.choice(models)
            dataset_id = rng.choice(datasets)
            if (model_id, dataset_id) not in pos_edges:
                neg_edges.add((model_id, dataset_id))

    # Combine, label, and shuffle
    all_edges = list(pos_edges) + list(neg_edges)
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    combined = list(zip(all_edges, labels))
    rng.shuffle(combined)

    if combined:
        shuffled_edges, shuffled_labels = zip(*combined)
    else:
        shuffled_edges, shuffled_labels = [], []

    shuffled_edges = list(shuffled_edges)[:max_pairs]
    shuffled_labels = list(shuffled_labels)[:max_pairs]
    return list(shuffled_edges), list(shuffled_labels)


def prepare_attribute_predictor_dataset(
    G: nx.Graph, metric_name: str = None
) -> Tuple[List[Edge], List[float], List[str]]:
    edges_to_predict: List[Edge] = []
    true_metrics: List[float] = []
    metric_names: List[str] = []

    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")

        # Only process model-dataset edges
        if not (
            (u_type == "model" and v_type == "dataset")
            or (v_type == "model" and u_type == "dataset")
        ):
            continue

        # Determine edge order (model, dataset)
        if u_type == "model" and v_type == "dataset":
            edge = (u, v)
        elif v_type == "model" and u_type == "dataset":
            edge = (v, u)
        else:
            continue

        if metric_name is not None:
            # Use specified metric if available
            if metric_name in data:
                edges_to_predict.append(edge)
                true_metrics.append(float(data[metric_name]))
                metric_names.append(metric_name)
        else:
            # Use ALL available numeric metrics
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    edges_to_predict.append(edge)
                    true_metrics.append(float(value))
                    metric_names.append(key)

    if not edges_to_predict:
        if metric_name:
            print(f"Warning: No edges found with the metric '{metric_name}' in the graph.")
        else:
            print("Warning: No edges found with any numeric metrics in the graph.")

    return edges_to_predict, true_metrics, metric_names


def prepare_link_ranker_dataset(
    G: nx.Graph, seed: int = 42, candidates_per_dataset: int = 10
) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    Prepare dataset for link ranking: for each dataset, find positive models and negative candidates.

    Args:
        G: NetworkX graph with integer node IDs.
        seed: Random seed for negative sampling.
        candidates_per_dataset: Number of negative model candidates per dataset.

    Returns:
        Dict mapping dataset_id to (positive_models, negative_candidates) lists.
    """
    import random

    rng = random.Random(seed)

    # Get all models and datasets
    models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
    datasets = [n for n, d in G.nodes(data=True) if d.get("type") == "dataset"]

    ranking_data = {}

    for dataset_id in datasets:
        # Find positive models (connected to this dataset)
        positive_models = []
        for neighbor_id in G.neighbors(dataset_id):
            if G.nodes[neighbor_id].get("type") == "model":
                positive_models.append(neighbor_id)

        # Sample negative models (not connected to this dataset)
        connected_models = set(positive_models)
        available_negatives = [m for m in models if m not in connected_models]

        negative_candidates = rng.sample(
            available_negatives, min(candidates_per_dataset, len(available_negatives))
        )

        if positive_models:  # Only include datasets that have positive connections
            ranking_data[dataset_id] = (positive_models, negative_candidates)

    return ranking_data


def prepare_attribute_ranker_dataset(
    G: nx.Graph, metric_name: str | None = None
) -> tuple[Dict[int, List[Tuple[int, float]]], Dict[int, str]]:
    """
    Prepare dataset for attribute ranking: for each dataset, find all connected models
    and their corresponding metric values.

    Args:
        G: NetworkX graph with integer node IDs and edge attributes.
        metric_name: The name of the edge attribute to extract. If None, uses the most frequent metric per dataset.

    Returns:
        Tuple of (ranking_data, dataset_metrics) where:
        - ranking_data: Dict mapping dataset_id to a list of (model_id, metric_value) tuples
        - dataset_metrics: Dict mapping dataset_id to the metric name used for that dataset
    """
    from collections import Counter, defaultdict

    # First, collect all model-dataset edges and their metrics
    dataset_edges = defaultdict(list)  # dataset_id -> [(model_id, edge_data)]

    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")

        model_id, dataset_id = (None, None)
        if u_type == "model" and v_type == "dataset":
            model_id, dataset_id = u, v
        elif v_type == "model" and u_type == "dataset":
            model_id, dataset_id = v, u

        if model_id is not None and dataset_id is not None:
            dataset_edges[dataset_id].append((model_id, data))

    ranking_data = {}
    dataset_metrics = {}

    global_metric_counter = Counter()

    for dataset_id, edges in dataset_edges.items():
        if metric_name is None:
            # Find the most frequent metric for this specific dataset
            metric_counter = Counter()
            for model_id, edge_data in edges:
                for key, value in edge_data.items():
                    if isinstance(value, (int, float)) and not key.startswith("_"):
                        metric_counter[key] += 1
                        global_metric_counter[key] += 1

            if not metric_counter:
                continue  # Skip this dataset if no numeric metrics

            selected_metric = metric_counter.most_common(1)[0][0]
        else:
            selected_metric = metric_name

        # Collect data for this dataset using the selected metric
        dataset_data = []
        for model_id, edge_data in edges:
            if selected_metric in edge_data:
                metric_value = float(edge_data[selected_metric])
                dataset_data.append((model_id, metric_value))

        if dataset_data:  # Only include if we have data
            # Sort by metric value (descending) for ground truth
            dataset_data.sort(key=lambda x: x[1], reverse=True)
            ranking_data[dataset_id] = dataset_data
            dataset_metrics[dataset_id] = selected_metric

    # Print summary information
    if metric_name is None:
        metric_usage = Counter(dataset_metrics.values())
        print("Auto-selected metrics per dataset:")
        for metric, count in metric_usage.most_common():
            print(f"  - {metric}: used in {count} datasets")
    else:
        print(f"Using specified metric '{metric_name}' for all datasets")
        print(f"Found data for {len(ranking_data)} datasets")

    if not ranking_data:
        warning_msg = "No data found"
        if metric_name:
            warning_msg += f" with metric '{metric_name}'"
        print(f"Warning: {warning_msg}")

    return ranking_data, dataset_metrics


def create_safe_filename(model_name: str) -> str:
    """Create a filesystem-safe filename from model name."""
    return model_name.replace("/", "_").replace(":", "_")
