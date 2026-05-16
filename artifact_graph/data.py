import json
import os
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, train_test_split_edges

MODEL, DATASET = "model", "dataset"


def load_all_splits(
    split_dir: str | Path,
    device: torch.device,
    use_random_embeddings: bool = False,
    seed: int = 42,
) -> Tuple[Data, Data, Data, Data, Data, Data]:
    """Load train/val/test data and labeled-edge splits for standalone link prediction.

    Matches the paper's split protocol (val_ratio=0.0, test_ratio=0.2): there is no
    separate val partition — the test split is reused as val so the
    ``ReduceLROnPlateau`` / early-stopping signal lives on the test distribution.

    Layout expected in split_dir:
      - node_embeddings_{voyage,random}.npy
      - train_split/{edges,pos_edges}.npz, node_metadata.json
      - test_split/{edges,pos_edges}.npz [+ support_edges.npz for inductive]

    Returns six PyG ``Data`` objects:
      train_data, train_split, val_data, val_split, test_data, test_split
    where ``val_*`` are aliases for ``test_*``.

    Convention used by GNNLinkTrainer / GNNLinkEvaluator:
      * ``*_data``  carries ``.x`` (features) and ``.edge_index`` (message-passing).
      * ``*_split`` carries ``.pos_edge_label_index`` and ``.neg_edge_label_index``
        (the labeled edges to score), plus ``.edge_index`` (the message-passing
        view used inside ``train_epoch``).
    """
    from .runners.runner_utils import load_node_embeddings

    split_dir = Path(split_dir)

    with open(split_dir / "train_split" / "node_metadata.json") as f:
        node_metadata = {int(k): v for k, v in json.load(f).items()}
    num_nodes = len(node_metadata)

    mode = "random" if use_random_embeddings else "embedding"
    x = load_node_embeddings(split_dir, mode).to(device)

    train_mp = torch.from_numpy(np.load(split_dir / "train_split" / "edges.npz")["edges"]).long()
    test_mp = torch.from_numpy(np.load(split_dir / "test_split" / "edges.npz")["edges"]).long()
    train_pos = torch.from_numpy(np.load(split_dir / "train_split" / "pos_edges.npz")["edges"]).long()
    test_pos = torch.from_numpy(np.load(split_dir / "test_split" / "pos_edges.npz")["edges"]).long()

    model_ids = sorted(int(nid) for nid, meta in node_metadata.items() if meta.get("type") == "model")
    model_ids_t = torch.tensor(model_ids, dtype=torch.long)

    # Union of every observed positive (both directions) — used to filter negatives.
    pos_set = set()
    for arr in (train_pos.numpy(), test_pos.numpy()):
        for i in range(arr.shape[1]):
            u, v = int(arr[0, i]), int(arr[1, i])
            pos_set.add((u, v))
            pos_set.add((v, u))

    gen = torch.Generator().manual_seed(seed)

    def _sample_neg(target_datasets: torch.Tensor, num: int) -> torch.Tensor:
        if num <= 0 or target_datasets.numel() == 0:
            return torch.zeros(2, 0, dtype=torch.long)
        sampled = []
        seen = set()
        attempts = 0
        max_attempts = max(num * 20, 1000)
        while len(sampled) < num and attempts < max_attempts:
            need = num - len(sampled)
            mi = torch.randint(0, len(model_ids_t), (need,), generator=gen)
            di = torch.randint(0, target_datasets.numel(), (need,), generator=gen)
            for j in range(need):
                m = int(model_ids_t[mi[j]])
                d = int(target_datasets[di[j]])
                if (m, d) in pos_set or (m, d) in seen:
                    continue
                seen.add((m, d))
                sampled.append((m, d))
                if len(sampled) >= num:
                    break
            attempts += need
        if not sampled:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(sampled, dtype=torch.long).t().contiguous()

    def _all_negatives_for_datasets(target_datasets: torch.Tensor) -> torch.Tensor:
        neg = []
        for d in target_datasets.tolist():
            for m in model_ids:
                if (m, d) not in pos_set:
                    neg.append((m, d))
        if not neg:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(neg, dtype=torch.long).t().contiguous()

    # Train negatives: pool of 10× pos for the trainer to subsample.
    train_datasets = torch.unique(train_pos[1])
    train_neg = _sample_neg(train_datasets, num=train_pos.size(1) * 10)

    # Test negatives = full non-edge enumeration over test datasets.
    test_datasets = torch.unique(test_pos[1])
    test_neg = _all_negatives_for_datasets(test_datasets)

    def _make_data(edge_index: torch.Tensor) -> Data:
        return Data(x=x, edge_index=edge_index.to(device), num_nodes=num_nodes)

    def _make_split(mp_edges: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> Data:
        s = Data()
        s.edge_index = mp_edges.to(device)
        s.pos_edge_label_index = pos.to(device)
        s.neg_edge_label_index = neg.to(device)
        return s

    train_data = _make_data(train_mp)
    test_data = _make_data(test_mp)

    train_split = _make_split(train_mp, train_pos, train_neg)
    test_split = _make_split(test_mp, test_pos, test_neg)

    # val_ratio=0.0 in the paper config — val IS test.
    val_data = test_data
    val_split = test_split

    print(
        f"[load_all_splits] num_nodes={num_nodes} "
        f"train_pos={train_pos.size(1)} train_neg={train_neg.size(1)} "
        f"test_pos={test_pos.size(1)} test_neg={test_neg.size(1)} (val == test)"
    )
    return train_data, train_split, val_data, val_split, test_data, test_split


def build_bipartite_graph(
    data_dir: str, dataset_json: str, metadata_dir: str, min_downloads: int = 1000
) -> nx.Graph:
    # 1) Load dataset info
    with open(dataset_json, "r", encoding="utf-8") as f:
        ds_info = json.load(f)
    # keep only popular datasets
    dataset_names = {
        d["id"].split("/")[-1].lower(): d["downloads"]
        for d in ds_info
        if d["downloads"] > min_downloads
    }

    G = nx.Graph()

    # add model–dataset edges
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        model_id = fname[:-5]
        # load model metadata
        try:
            with open(os.path.join(metadata_dir, f"{model_id}.json"), "r", encoding="utf-8") as f:
                md = json.load(f)
            if md.get("downloads", 0) < min_downloads:
                continue
        except FileNotFoundError:
            continue

        # load the model→dataset mapping
        mapping = json.load(open(os.path.join(data_dir, fname), encoding="utf-8"))
        for ds in mapping.keys():
            ds_name = ds.split("/")[-1].lower()
            if ds_name not in dataset_names:
                continue
            # check whether is dict and has accuracy
            acc = None
            if mapping is not None and ds_name in mapping and isinstance(mapping[ds_name], dict):
                if "accuracy" in mapping[ds_name]:
                    acc = mapping[ds_name]["accuracy"]
                elif "acc" in mapping[ds_name]:
                    acc = mapping[ds_name]["acc"]
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
        if attr is not None and "accuracy" in attr and attr["accuracy"] is not None:
            acc = float(attr["accuracy"])
        edge_attrs.append(acc)
    data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    # Save node names in order
    data.node_names = list(G.nodes())
    # 3) Build node‐feature matrix
    feats = []
    for node, attrs in G.nodes(data=True):
        is_model = 1 if attrs["type"] == MODEL else 0
        is_dataset = 1 - is_model
        downloads = attrs.get("downloads", 0)
        feats.append(
            [is_model, is_dataset, torch.log1p(torch.tensor(downloads, dtype=torch.float))]
        )
    data.x = torch.stack([torch.tensor(f, dtype=torch.float) for f in feats], dim=0)
    return data


def prepare_link_pred_splits(data: Data, val_ratio=0.1, test_ratio=0.1) -> Data:
    # 4) Split edges and generate negatives
    # This adds:
    #   data.train_pos_edge_index
    #   data.val_pos_edge_index, data.val_neg_edge_index
    #   data.test_pos_edge_index, data.test_neg_edge_index
    return train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)


# ------------- USAGE -------------
if __name__ == "__main__":
    G = build_bipartite_graph(
        data_dir="eval_datasets_json_download_ranks",
        dataset_json="dataset_info.json",
        metadata_dir="model_metadata_download_ranks",
    )
    print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pyg_data = nx_to_pyg_data(G)
    pyg_data = prepare_link_pred_splits(pyg_data)
