import os, json, random
import networkx as nx
import torch
from torch_geometric.utils import from_networkx, train_test_split_edges
from torch_geometric.data import Data

MODEL, DATASET = "model", "dataset"

def build_bipartite_graph(data_dir: str,
                          dataset_json: str,
                          metadata_dir: str,
                          min_downloads: int = 1000) -> nx.Graph:
    # 1) Load dataset info
    with open(dataset_json, 'r', encoding='utf-8') as f:
        ds_info = json.load(f)
    # keep only popular datasets
    dataset_names = {
        d['id'].split('/')[-1].lower(): d['downloads']
        for d in ds_info if d['downloads'] > min_downloads
    }

    G = nx.Graph()
    # add dataset nodes
    for name, downloads in dataset_names.items():
        G.add_node(name, type=DATASET, downloads=downloads)

    # add model–dataset edges
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        model_id = fname[:-5]
        # load model metadata
        try:
            with open(os.path.join(metadata_dir, f"{model_id}.json"),
                      'r', encoding='utf-8') as f:
                md = json.load(f)
            if md.get('downloads', 0) < min_downloads:
                continue
        except FileNotFoundError:
            continue

        # load the model→dataset mapping
        mapping = json.load(open(os.path.join(data_dir, fname), encoding='utf-8'))
        for ds in mapping.keys():
            ds_name = ds.split("/")[-1].lower()
            if ds_name not in dataset_names:
                continue
            # add nodes/edge
            G.add_node(model_id, type=MODEL, downloads=md['downloads'])
            G.add_edge(model_id, ds_name)
    return G

def nx_to_pyg_data(G: nx.Graph) -> Data:
    # 2) Convert to PyG Data
    data = from_networkx(G)

    # 3) Build node‐feature matrix
    # We'll encode: [is_model, is_dataset, log(downloads + 1)]
    feats = []
    for node, attrs in G.nodes(data=True):
        is_model = 1 if attrs['type'] == MODEL else 0
        is_dataset = 1 - is_model
        downloads = attrs.get('downloads', 0)
        feats.append([is_model, is_dataset, torch.log1p(torch.tensor(downloads, dtype=torch.float))])
    data.x = torch.stack([torch.tensor(f, dtype=torch.float) for f in feats], dim=0)

    return data

def prepare_link_pred_splits(data: Data, val_ratio=0.1, test_ratio=0.1) -> Data:
    # 4) Split edges and generate negatives
    # This adds:
    #   data.train_pos_edge_index
    #   data.val_pos_edge_index, data.val_neg_edge_index
    #   data.test_pos_edge_index, data.test_neg_edge_index
    return train_test_split_edges(data,
                                  val_ratio=val_ratio,
                                  test_ratio=test_ratio)

# ------------- USAGE -------------
if __name__ == "__main__":
    G = build_bipartite_graph(
        data_dir="eval_datasets_json_download_ranks",
        dataset_json="dataset_info.json",
        metadata_dir="model_metadata_download_ranks"
    )
    print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pyg_data = nx_to_pyg_data(G)
    pyg_data = prepare_link_pred_splits(pyg_data)

    # Now pyg_data has:
    #   .x                 node features
    #   .train_pos_edge_index
    #   .val_pos_edge_index / .val_neg_edge_index
    #   .test_pos_edge_index / .test_neg_edge_index
    #
    # You can feed these into your LinkPredictionGNN:
    #
    #   pos_train, neg_train = pyg_data.train_pos_edge_index, negative_sampling(...)
    #   out_pos, out_neg = model(pyg_data.x, pyg_data.train_pos_edge_index,
    #                            pos_train, neg_train)
    #
    #   … compute loss, backprop, etc.
