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
            # check whether is dict and has accuracy
            acc = None
            if mapping is not None and ds_name in mapping and isinstance(mapping[ds_name], dict):
                if 'accuracy' in mapping[ds_name]:
                    acc = mapping[ds_name]['accuracy']
                elif 'acc' in mapping[ds_name]:
                    acc = mapping[ds_name]['acc']
            if acc is not None:
                try:
                    acc = float(acc)
                    if acc > 1:
                        acc = acc / 100.0
                    if 0 <= acc <= 1:
                        # add nodes/edge with accuracy attribute
                        G.add_node(model_id, type=MODEL)
                        G.add_node(ds_name, type=DATASET)
                        G.add_edge(model_id, ds_name, accuracy=acc)
                except Exception:
                    continue

    return G

def nx_to_pyg_data(G: nx.Graph) -> Data:
    data = from_networkx(G)
    # Build edge_attr manually, matching edge_index order
    edge_attrs = []
    nx_names = {i: n for i, n in enumerate(G.nodes())}
    for u_idx, v_idx in zip(data.edge_index[0], data.edge_index[1]):
        u = nx_names[u_idx.item()]
        v = nx_names[v_idx.item()]
        attr = G.get_edge_data(u, v)
        acc = 0.0
        if attr is not None and 'accuracy' in attr and attr['accuracy'] is not None:
            acc = float(attr['accuracy'])
        edge_attrs.append(acc)
    import torch
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    # Save node names in order
    data.node_names = list(G.nodes())
    # 3) Build node‐feature matrix
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