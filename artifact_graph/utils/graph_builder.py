import networkx as nx
from pathlib import Path
from typing import Optional, Dict

from ..collectors import ModelCollector, DatasetCollector, MetricCollector

MODEL_NODE = "model"
DATASET_NODE = "dataset"


def load_artifact_graph(
    models_dir: str = "output/models/metadata",
    datasets_dir: str = "output/datasets/metadata",
    metrics_dir: str = "output/metrics",
    hf_token: Optional[str] = None,
    min_downloads: int = 1000,
) -> nx.Graph:
    """
    Construct a bipartite graph of models and datasets.

    Nodes:
      - model nodes (type='model', downloads)
      - dataset nodes (type='dataset', downloads)

    Edges:
      - Connect a model to a dataset if evaluation metrics exist
      - Edge attribute: 'metrics' dict

    Args:
      models_dir: directory of model metadata
      datasets_dir: directory of dataset metadata
      metrics_dir: directory of per-model metrics
      hf_token: unused Hugging Face token
      min_downloads: download cutoff for filtering

    Returns:
      An undirected NetworkX graph
    """
    G = nx.Graph()

    # Load metadata and metrics
    models = ModelCollector.load_all_metadata(models_dir, min_downloads)
    datasets = DatasetCollector.load_all_metadata(datasets_dir, min_downloads)
    metrics = MetricCollector.load_all_metrics(metrics_dir)

    # Map normalized dataset names -> full IDs
    name_map: Dict[str, str] = {
        ds_id.split('/')[-1].lower(): ds_id for ds_id in datasets
    }

    for model_id, meta in models.items():
        # Only include models with a linked paper
        if not any(tag.startswith("arxiv:") for tag in meta.get("tags", [])):
            continue

        model_metrics = metrics.get(model_id)
        if not model_metrics:
            continue

        # Add model node
        G.add_node(model_id, type=MODEL_NODE, downloads=meta.get("downloads", 0))

        for ds_key, ds_metrics in model_metrics.items():
            if not isinstance(ds_metrics, dict):
                continue

            norm = ds_key.split('/')[-1].lower()
            ds_id = name_map.get(norm)
            if not ds_id:
                continue

            # Add dataset node if missing
            if not G.has_node(ds_id):
                G.add_node(
                    ds_id,
                    type=DATASET_NODE,
                    downloads=datasets[ds_id].get("downloads", 0),
                )

            # Add edge with metrics
            G.add_edge(model_id, ds_id, metrics=ds_metrics)

    return G


def load_pyg_graph_from_networkx(G: nx.Graph) -> 'Data':
    """
    Convert a NetworkX bipartite graph to a PyTorch Geometric Data object.

    Node features: one-hot for model/dataset types.
    Edge attrs: first metric value as float.

    Returns:
      Data(x, edge_index, edge_attr, model_names, dataset_names)
    """
    from torch_geometric.data import Data
    import torch

    nodes = list(G.nodes)
    idx_map = {n: i for i, n in enumerate(nodes)}
    types = [G.nodes[n]['type'] for n in nodes]

    model_idxs = [i for i, t in enumerate(types) if t == MODEL_NODE]
    dataset_idxs = [i for i, t in enumerate(types) if t == DATASET_NODE]

    # Build feature matrix
    x = torch.zeros((len(nodes), 2))
    x[model_idxs, 0] = 1
    x[dataset_idxs, 1] = 1

    # Build edges and attributes
    edge_list, attr_list = [], []
    for u, v, data in G.edges(data=True):
        u_i, v_i = idx_map[u], idx_map[v]
        edge_list.append([u_i, v_i])
        first_val = next(iter(data.get('metrics', {}).values()), 0.0)
        attr_list.append(float(first_val))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        model_names=[nodes[i] for i in model_idxs],
        dataset_names=[nodes[i] for i in dataset_idxs],
    )
