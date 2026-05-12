#!/usr/bin/env python3
"""
Graph utility functions.

Shared helpers for split loading, edge normalisation, and negative-sample
generation live here so that GNN, LLM and Baseline pipelines all share a
single source of truth.

Data preparation functions are organised in:
- link_prediction_utils.py: prepare_link_predictor_dataset
- attribute_prediction_utils.py: prepare_attribute_predictor_dataset
- ranking_utils.py: prepare_link_ranker_dataset, prepare_attribute_ranker_dataset
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

Edge = Tuple[int, int]


def create_safe_filename(name: str) -> str:
    """Create a filesystem-safe filename from a string."""
    return name.replace("/", "_").replace(":", "_").replace("\\", "_")

# =============================================================================
# Common Utilities
# =============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


# =============================================================================
# Shared Split Data Utilities (used by GNN, LLM, and Baseline)
# =============================================================================

def _get_node_type(node_id: int, node_metadata: Dict) -> Optional[str]:
    """Get node type from metadata (handles both str and int keys)."""
    meta = node_metadata.get(str(node_id), node_metadata.get(node_id, {}))
    return meta.get("type")


def normalize_model_dataset_edge(
    u: int,
    v: int,
    node_metadata: Dict,
) -> Optional[Edge]:
    """Normalize an edge to (model_id, dataset_id) order when possible."""
    u_type = _get_node_type(u, node_metadata)
    v_type = _get_node_type(v, node_metadata)
    if u_type == "model" and v_type == "dataset":
        return (u, v)
    if u_type == "dataset" and v_type == "model":
        return (v, u)
    return None


def load_split_metric_map(
    split_dir: str | Path,
    split_name: str = "test_split",
    metric_file: str = "edge_metadata_normalized.json",
) -> Dict[Edge, Dict[str, float]]:
    """Load split edge metric map from edge_metadata_normalized.json."""
    split_path = Path(split_dir) / split_name
    meta_path = split_path / metric_file
    if not meta_path.exists():
        # Fallback to default
        meta_path = split_path / "edge_metadata_normalized.json"
    if not meta_path.exists():
        return {}

    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics_map: Dict[Edge, Dict[str, float]] = {}
    for key_str, value in raw_meta.items():
        parts = key_str.split(",")
        if len(parts) != 2:
            continue
        u, v = int(parts[0].strip()), int(parts[1].strip())
        metrics = value.get("metrics", {}) if isinstance(value, dict) else {}
        metrics_map[(u, v)] = metrics
    return metrics_map


def select_edge_metric_target(
    metrics: Dict[str, Any],
    metric_name: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    """Select one numeric metric target from an edge metric dict."""
    if metric_name is not None:
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return metric_name, float(value)
        return None

    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def collect_all_split_positives(
    split_dir: Path,
    node_metadata: Dict,
) -> Tuple[Dict[int, Set[int]], Set[Edge]]:
    split_dir = Path(split_dir)
    all_pos_by_ds: Dict[int, Set[int]] = {}
    all_pos_edges: Set[Edge] = set()

    for split_name in ["train_split", "val_split", "test_split"]:
        pos_path = split_dir / split_name / "pos_edges.npz"
        if not pos_path.exists():
            continue
        pos = np.load(pos_path)["edges"]
        for i in range(pos.shape[1]):
            u, v = int(pos[0, i]), int(pos[1, i])
            all_pos_edges.add((u, v))
            all_pos_edges.add((v, u))
            normalized = normalize_model_dataset_edge(u, v, node_metadata)
            if normalized is not None:
                model_id, dataset_id = normalized
                all_pos_by_ds.setdefault(dataset_id, set()).add(model_id)

    return all_pos_by_ds, all_pos_edges


def get_test_edges_by_dataset(
    split_dir: Path,
    node_metadata: Dict,
) -> Tuple[List[Edge], Dict[int, Set[int]], Set[int]]:
    split_dir = Path(split_dir)
    pos_edges = np.load(split_dir / "test_split" / "pos_edges.npz")["edges"]

    test_pos: List[Edge] = []
    test_pos_by_ds: Dict[int, Set[int]] = {}
    test_datasets: Set[int] = set()

    for i in range(pos_edges.shape[1]):
        u, v = int(pos_edges[0, i]), int(pos_edges[1, i])
        normalized = normalize_model_dataset_edge(u, v, node_metadata)
        if normalized is not None:
            model_id, dataset_id = normalized
            test_pos.append(normalized)
            test_pos_by_ds.setdefault(dataset_id, set()).add(model_id)
            test_datasets.add(dataset_id)
        else:
            test_pos.append((u, v))

    return test_pos, test_pos_by_ds, test_datasets


def get_all_model_ids(node_metadata: Dict) -> Set[int]:
    """Get all model node IDs from metadata."""
    return {int(k) for k, v in node_metadata.items() if v.get("type") == "model"}


def generate_negative_edges(
    test_datasets: Set[int],
    all_model_ids: Set[int],
    all_pos_edges: Set[Edge],
) -> List[Edge]:
    neg_edges: List[Edge] = []
    for did in test_datasets:
        for mid in all_model_ids:
            if (mid, did) not in all_pos_edges and (did, mid) not in all_pos_edges:
                neg_edges.append((mid, did))
    return neg_edges


def load_link_graph_from_split(
    split_dir: str | Path,
) -> Tuple[nx.Graph, Dict[int, Dict]]:
    split_path = Path(split_dir)

    node_meta_path = split_path / "train_split" / "node_metadata.json"
    with node_meta_path.open("r") as f:
        node_metadata = {int(k): v for k, v in json.load(f).items()}

    G = nx.Graph()
    for node_id, meta in node_metadata.items():
        G.add_node(node_id, **meta)

    # Add observed train edges — NOT val/test edges
    train_pos_path = split_path / "train_split" / "pos_edges.npz"
    if train_pos_path.exists():
        pos = np.load(train_pos_path)["edges"]
        for i in range(pos.shape[1]):
            u, v = int(pos[0, i]), int(pos[1, i])
            G.add_edge(u, v)

    # For inductive splits, also add support edges so that inductive nodes
    # have neighbourhood context (used by LLM 1-hop and graph-based baselines).
    support_path = split_path / "test_split" / "support_edges.npz"
    n_support = 0
    if support_path.exists():
        sup = np.load(support_path)["edges"]
        for i in range(sup.shape[1]):
            u, v = int(sup[0, i]), int(sup[1, i])
            G.add_edge(u, v)
        n_support = sup.shape[1]

    print(
        f"Loaded observed graph: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        + (f" (incl. {n_support} support edges)" if n_support else " (no support edges)")
    )
    return G, node_metadata


def load_attribute_graph_from_split(
    split_dir: str | Path,
) -> Tuple[nx.Graph, Dict[int, Dict], Dict[Edge, Dict[str, Dict[str, float]]]]:
    split_path = Path(split_dir)

    node_meta_path = split_path / "train_split" / "node_metadata.json"
    with node_meta_path.open("r") as f:
        node_metadata = {int(k): v for k, v in json.load(f).items()}

    G = nx.Graph()
    for node_id, meta in node_metadata.items():
        G.add_node(node_id, **meta)

    # Add observed train edges and metrics — NOT val/test edges
    edge_metadata: Dict[Edge, Dict[str, Dict[str, float]]] = {}
    train_subdir = split_path / "train_split"
    train_pos_path = train_subdir / "pos_edges.npz"
    train_meta_path = train_subdir / "edge_metadata_normalized.json"
    if train_pos_path.exists() and train_meta_path.exists():
        train_metrics = load_split_metric_map(split_path, split_name="train_split")

        pos = np.load(train_pos_path)["edges"]
        for i in range(pos.shape[1]):
            u, v = int(pos[0, i]), int(pos[1, i])
            metrics = train_metrics.get((u, v), train_metrics.get((v, u), {}))
            G.add_edge(u, v, **metrics)
            edge_metadata[(u, v)] = {"metrics": metrics}

    # For inductive splits, also add support edges (with their metrics) so that
    # inductive nodes have neighbourhood context for LLM 1-hop and baselines.
    support_path = split_path / "test_split" / "support_edges.npz"
    n_support = 0
    if support_path.exists():
        # Load support edge metrics from dedicated file (separate from test pos metadata)
        support_metrics: Dict[Edge, Dict[str, float]] = {}
        support_meta_path = split_path / "test_split" / "support_edge_metadata.json"
        if support_meta_path.exists():
            raw_meta = json.loads(support_meta_path.read_text(encoding="utf-8"))
            for key_str, value in raw_meta.items():
                parts = key_str.split(",")
                if len(parts) != 2:
                    continue
                u, v = int(parts[0].strip()), int(parts[1].strip())
                metrics = value.get("metrics", {}) if isinstance(value, dict) else {}
                support_metrics[(u, v)] = metrics

        sup = np.load(support_path)["edges"]
        for i in range(sup.shape[1]):
            u, v = int(sup[0, i]), int(sup[1, i])
            metrics = support_metrics.get((u, v), support_metrics.get((v, u), {}))
            G.add_edge(u, v, **metrics)
            edge_metadata[(u, v)] = {"metrics": metrics}
        n_support = sup.shape[1]

    print(f"Loaded observed attribute graph: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(edge_metadata)} edge metadata entries"
          + (f" (incl. {n_support} support edges)" if n_support else " (no support edges)"))
    return G, node_metadata, edge_metadata
