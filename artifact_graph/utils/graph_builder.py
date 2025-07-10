from typing import Dict, Optional

import networkx as nx
import torch
from torch_geometric.data import Data

from ..collectors import DatasetCollector, MetricCollector, ModelCollector

MODEL_NODE = "model"
DATASET_NODE = "dataset"


def load_artifact_graph(
    models_dir: str = "output/models/metadata",
    datasets_dir: str = "output/datasets/metadata",
    metrics_dir: str = "output/metrics",
    hf_token: Optional[str] = None,
    min_downloads: int = 1,
    metric_key_substring: str = "acc",
) -> nx.Graph:
    """
    Construct a bipartite graph of models and datasets based on evaluation metrics.

    Nodes:
      - model nodes with attributes {type: 'model', downloads}
      - dataset nodes with attributes {type: 'dataset', downloads}

    Edges:
      - Connect model to dataset if a metric name contains `metric_key_substring`.
      - Edge attribute: {metric_key_substring: metric_value}

    Args:
      models_dir: Directory of model metadata files.
      datasets_dir: Directory of dataset metadata files.
      metrics_dir: Directory of per-model metrics files.
      hf_token: Hugging Face token (currently unused).
      min_downloads: Minimum downloads threshold for filtering.
      metric_key_substring: Substring to match metric names (case-insensitive).

    Returns:
      An undirected NetworkX graph.
    """
    G = nx.Graph()

    # Load metadata and metrics
    models = ModelCollector.load_all_metadata(models_dir, min_downloads)
    datasets = DatasetCollector.load_all_metadata(datasets_dir, min_downloads)
    metrics = MetricCollector.load_all_metrics(metrics_dir)

    # Map normalized dataset name -> dataset ID
    name_map: Dict[str, str] = {
        ds_id.split("/")[-1].lower(): ds_id
        for ds_id in datasets
    }

    for model_id, meta in models.items():
        # Skip models without a linked paper
        tags = meta.get("tags", [])
        if not any(tag.startswith("arxiv:") for tag in tags):
            continue

        model_metrics = metrics.get(model_id, {})
        if not model_metrics:
            continue


        # Inspect dataset metrics for this model
        for ds_key, ds_metrics in model_metrics.items():
            if not isinstance(ds_metrics, dict):
                continue

            ds_norm = ds_key.split("/")[-1].lower()
            ds_id = name_map.get(ds_norm)
            if not ds_id:
                continue

            # Find the first matching metric
            metric_value: Optional[float] = None
            for m_name, m_val in ds_metrics.items():
                if metric_key_substring.lower() in m_name.lower() and isinstance(m_val, (int, float)):
                    metric_value = float(m_val)
                    break

            if metric_value is None:
                continue

            # Normalize percentage values
            if metric_value > 1:
                metric_value /= 100

            # Add dataset node if missing
            if not G.has_node(ds_id):
                G.add_node(
                    model_id,
                    type=MODEL_NODE,
                    downloads=meta.get("downloads", 0)
                )
                G.add_node(
                    ds_id,
                    type=DATASET_NODE,
                    downloads=datasets[ds_id].get("downloads", 0)
                )

                G.add_edge(
                    model_id,
                    ds_id,
                    **{metric_key_substring: metric_value}
                )

    return G


def load_pyg_graph_from_networkx(G: nx.Graph) -> Data:
    """
    Convert a NetworkX bipartite graph to a PyTorch Geometric Data object.

    Node features:
      - One-hot encoding: [1,0] for model, [0,1] for dataset
    Edge attributes:
      - Uses the first available metric value on each edge

    Returns:
      A torch_geometric.data.Data object with:
        - x: node feature tensor
        - edge_index: edge index tensor
        - edge_attr: edge attribute tensor
        - model_names: list of model node IDs
        - dataset_names: list of dataset node IDs
    """
    nodes = list(G.nodes)
    idx_map = {node: idx for idx, node in enumerate(nodes)}
    types = [G.nodes[node]["type"] for node in nodes]

    # Build feature matrix
    x = torch.zeros((len(nodes), 2), dtype=torch.float)
    for idx, t in enumerate(types):
        if t == MODEL_NODE:
            x[idx, 0] = 1.0
        elif t == DATASET_NODE:
            x[idx, 1] = 1.0

    # Build edges and attributes
    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        u_idx, v_idx = idx_map[u], idx_map[v]
        edge_index.append([u_idx, v_idx])
        # Take the first metric value available
        metric_val = next(iter(data.values()), 0.0)
        edge_attr.append(float(metric_val))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Separate node lists by type
    model_names = [node for node in nodes if G.nodes[node]["type"] == MODEL_NODE]
    dataset_names = [node for node in nodes if G.nodes[node]["type"] == DATASET_NODE]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        model_names=model_names,
        dataset_names=dataset_names,
    )
