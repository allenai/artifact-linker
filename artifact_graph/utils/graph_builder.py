import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def _load_graph_raw_data(graph_data_dir: str) -> Tuple[Dict, np.ndarray, Dict]:
    data_path = Path(graph_data_dir)

    with open(data_path / "node_metadata.json", "r") as f:
        node_metadata = {int(k): v for k, v in json.load(f).items()}

    edges_data = np.load(data_path / "edges.npz")
    edges = edges_data["edges"]

    with open(data_path / "edge_metadata.json", "r") as f:
        edge_data = json.load(f)
        edge_metadata = {}
        for key_str, value in edge_data.items():
            node_a, node_b = map(int, key_str.split(","))
            edge_metadata[(node_a, node_b)] = value

    return node_metadata, edges, edge_metadata


def load_nx_graph(
    graph_data_dir: str = "output/artifact_graph_data",
) -> Tuple[nx.Graph, Dict, Dict]:
    node_metadata, edges, edge_metadata = _load_graph_raw_data(graph_data_dir)

    # filtered_edges, normalized_edge_metadata = _filter_and_normalize_edges(
    #    edges, edge_metadata
    # )
    filtered_edges = edges
    normalized_edge_metadata = edge_metadata

    G = nx.Graph()

    int_node_metadata = {int(k): v for k, v in node_metadata.items()}
    for node_id, data in int_node_metadata.items():
        G.add_node(node_id, **data)

    for u, v in filtered_edges:
        edge_meta = normalized_edge_metadata.get((u, v), {})
        metrics = edge_meta.get("metrics", {})
        G.add_edge(u, v, **metrics)

    print(
        f"Loaded NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (after filtering)"
    )
    return G, int_node_metadata, normalized_edge_metadata




def load_pyg_graph(
    graph_data_dir: str = "output/artifact_graph_data",
    metric_key: Optional[str] = None,
    undirected: bool = True,
) -> Tuple[Data, Dict]:
    """
    Load a PyG graph where:
      - edge_index: structural edges for GNN message passing (no edge_attr attached)
      - metric_edge_index / metric_edge_attr: ONLY edges that truly have a metric label

    This avoids any default placeholder (e.g., 0.5). Edges without a real metric are
    used for structure but excluded from supervision tensors.
    """
    data_path = Path(graph_data_dir)
    node_metadata, edges, edge_metadata = _load_graph_raw_data(graph_data_dir)

    # ---- Node embeddings (.npy or .npz) ----
    emb_npy = data_path / "node_embeddings.npy"
    emb_npz = data_path / "node_embeddings.npz"

    if emb_npy.exists():
        arr = np.load(emb_npy, allow_pickle=False)
        # Handle either plain array or structured array with a field named 'embedding'
        if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
            node_embeddings = arr["embedding"] if "embedding" in arr.dtype.names else np.asarray(arr.tolist())
        else:
            node_embeddings = arr
        print(f"✅ Loaded embeddings: {emb_npy}")
    elif emb_npz.exists():
        arr = np.load(emb_npz)
        # Prefer common keys
        for k in ("embeddings", "embedding", "X", "x"):
            if k in arr:
                node_embeddings = arr[k]
                break
        else:
            raise KeyError(f"No embeddings array found in {emb_npz}. Tried keys: embeddings/embedding/X/x")
        print(f"✅ Loaded embeddings: {emb_npz}")
    else:
        print("⚠️ Embeddings not found. Using random fallback.")
        node_embeddings = np.random.randn(len(node_metadata), 768).astype(np.float32)

    x = torch.as_tensor(node_embeddings, dtype=torch.float)

    # ---- Build structural edges + labeled metric edges ----
    structural_edges = []
    labeled_edges = []
    labeled_vals = []

    for u, v in edges:
        u, v = int(u), int(v)
        structural_edges.append([u, v])

        meta = edge_metadata.get((u, v), {})
        metrics: Dict = meta.get("metrics", {})

        # pick the requested metric, or the first available NUMERIC one
        mval = None
        if metric_key is not None:
            mval = metrics.get(metric_key)
        elif metrics:
            # Try to find the first value that looks like a number
            for val_candidate in metrics.values():
                if isinstance(val_candidate, (int, float)) and not isinstance(val_candidate, bool):
                    mval = val_candidate
                    break
                # If it's a string, try to see if it's convertible to float
                if isinstance(val_candidate, str):
                    try:
                        float(val_candidate)
                        mval = val_candidate
                        break
                    except ValueError:
                        continue
            # If still None, fallback to the first value (and let the try-catch below handle it)
            if mval is None:
                mval = next(iter(metrics.values()), None)

        if mval is not None:
            try:
                # Handle list/tuple by taking the first element if applicable, or skip
                if isinstance(mval, (list, tuple)):
                    if len(mval) > 0 and isinstance(mval[0], (int, float, str)):
                         # aggressive fallback: try first element
                         val = float(mval[0])
                    else:
                        continue # Skip complex list
                else:
                    val = float(mval)
                
                if val > 1.0:  # likely a percentage
                    val = val / 100.0
                labeled_edges.append([u, v])
                labeled_vals.append(val)
            except (ValueError, TypeError):
                # Skip values that cannot be converted to float
                continue

    # make undirected if desired (for message passing)
    if undirected:
        structural_edges += [[v, u] for u, v in structural_edges]

    edge_index = torch.tensor(structural_edges, dtype=torch.long).t().contiguous()

    if len(labeled_edges) > 0:
        metric_edge_index = torch.tensor(labeled_edges, dtype=torch.long).t().contiguous()
        metric_edge_attr = torch.tensor(labeled_vals, dtype=torch.float)
    else:
        metric_edge_index = torch.empty((2, 0), dtype=torch.long)
        metric_edge_attr = torch.empty((0,), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    # attach supervision-only tensors
    data.metric_edge_index = metric_edge_index
    data.metric_edge_attr = metric_edge_attr

    print(
        f"Loaded PyG graph: {x.size(0)} nodes, {edge_index.size(1)} structural edges; "
        f"{metric_edge_index.size(1)} labeled edges."
    )
    return data, node_metadata
