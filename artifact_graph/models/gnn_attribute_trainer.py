#!/usr/bin/env python3
"""GNN training utilities for attribute prediction (regression)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from torch_geometric.utils import degree

from .gnn_link_predictor import GNNLinkPredictor


@dataclass
class AttributeTrainingConfig:
    """Configuration for attribute prediction training."""
    epochs: int = 500
    lr: float = 0.005
    weight_decay: float = 1e-5
    seed: int = 42


@dataclass
class AttributeModelConfig:
    """Configuration for attribute prediction model."""
    in_channels: int
    hidden_channels: int = 128
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.2


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_edge_metadata(graph_data_dir: str | Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load edge metadata with metrics."""
    f = Path(graph_data_dir) / "edge_metadata.json"
    if not f.exists():
        raise FileNotFoundError(f"edge metadata not found: {f}")

    meta = json.loads(f.read_text(encoding="utf-8"))
    mapping = {}

    if isinstance(meta, dict):
        for k, v in meta.items():
            try:
                s = str(k).replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                parts = [t.strip() for t in s.replace(",", " ").split() if t.strip()]
                if len(parts) >= 2:
                    u, w = int(parts[0]), int(parts[1])
                    metrics = v.get("metrics", v) if isinstance(v, dict) else {}
                    mapping[(u, w)] = metrics if isinstance(metrics, dict) else {}
            except (ValueError, TypeError):
                continue
    elif isinstance(meta, list):
        for item in meta.items():
            u = int(item.get("u", item.get("src", -1)))
            w = int(item.get("v", item.get("dst", -1)))
            if u >= 0 and w >= 0:
                mapping[(u, w)] = item.get("metrics", {})

    return mapping


def get_metric_values(
    pos_edges: np.ndarray,
    metadata: Dict,
    metric_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract metric values for positive edges."""
    kept_edges, kept_vals = [], []

    for u, v in pos_edges.T:
        u, v = int(u), int(v)
        val = None
        if (u, v) in metadata and metric_key in metadata[(u, v)]:
            val = metadata[(u, v)][metric_key]
        elif (v, u) in metadata and metric_key in metadata[(v, u)]:
            val = metadata[(v, u)][metric_key]

        if val is not None:
            val = float(val)
            if val > 1.0:
                val /= 100.0
            kept_edges.append([u, v])
            kept_vals.append(val)

    if not kept_edges:
        raise ValueError(f"No edges have metric '{metric_key}'")

    edge_index = torch.tensor(kept_edges, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(kept_vals, dtype=torch.float)
    return edge_index, edge_label


def load_attribute_split(
    split_dir: str | Path,
    graph_data_dir: str | Path,
    metric_key: str,
    forced_x: Optional[torch.Tensor] = None,
) -> Tuple[Data, Data]:
    """Load data for attribute prediction split."""
    p = Path(split_dir)

    edges = torch.from_numpy(np.load(p / "edges.npz")["edges"]).long()
    with open(p / "node_metadata.json") as f:
        num_nodes = len(json.load(f))

    if forced_x is not None:
        x = forced_x
    else:
        emb_path = p.parent.parent / "artifact_graph_data" / "node_embeddings.npy"
        arr = np.load(emb_path, allow_pickle=False)
        if hasattr(arr.dtype, "names") and "embedding" in arr.dtype.names:
            x = torch.from_numpy(arr["embedding"]).float()
        else:
            x = torch.from_numpy(arr).float()

    G = Data(x=x, edge_index=edges, num_nodes=num_nodes)

    pos_all = np.load(p / "pos_edges.npz")["edges"]
    metadata = load_edge_metadata(graph_data_dir)

    # Auto-pick metric if not specified
    if metric_key is None:
        all_keys = set()
        for m in metadata.values():
            all_keys.update(m.keys())
        metric_key = sorted(all_keys)[0] if all_keys else "accuracy"
        print(f"[info] auto-picked metric_key='{metric_key}'")

    edge_index, edge_label = get_metric_values(pos_all, metadata, metric_key)

    split = Data()
    split.edge_label_index = edge_index
    split.edge_label = edge_label
    print(f"[split] {edge_label.numel()}/{pos_all.shape[1]} edges with '{metric_key}'")

    return G, split


class GNNAttributeTrainer:
    """Trainer for GNN attribute prediction (regression)."""

    def __init__(self, model: GNNLinkPredictor, device: torch.device, config: AttributeTrainingConfig):
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.best_mse: Optional[float] = None
        self.best_state: Optional[Dict] = None

    def train_epoch(self, G: Data, split: Data) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        z = self.model.encode(G.x, G.edge_index)
        logits = self.model.decode(z, split.edge_label_index).squeeze(-1)

        # Convert target to logit space
        y = split.edge_label
        y_clipped = torch.clamp(y, min=1e-7, max=1 - 1e-7)
        y_logits = torch.log(y_clipped / (1 - y_clipped))

        loss = F.mse_loss(logits, y_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.detach().cpu())

    def train(self, G_train: Data, split_train: Data, G_val: Data, split_val: Data, verbose: bool = True):
        """Full training loop."""
        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(G_train, split_train)

            eval_freq = 10 if epoch <= 50 else 25
            if epoch % eval_freq == 0 or epoch == self.config.epochs:
                metrics = self.evaluate(G_train, split_val)
                is_best = self.best_mse is None or metrics["mse"] < self.best_mse

                if is_best:
                    self.best_mse = metrics["mse"]
                    self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                if verbose:
                    mark = "⭐" if is_best else ""
                    print(f"epoch {epoch:04d} | loss {loss:.6f} | val_mse {metrics['mse']:.6f} | val_r2 {metrics['r2']:.4f} {mark}")

        if self.best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})
            if verbose:
                print(f"Restored best (val_mse={self.best_mse:.6f})")

    @torch.no_grad()
    def evaluate(self, G: Data, split: Data, return_preds: bool = False):
        """Evaluate model."""
        self.model.eval()
        z = self.model.encode(G.x, G.edge_index)
        logits = self.model.decode(z, split.edge_label_index).squeeze(-1)
        logits_clipped = torch.clamp(logits, min=-10.0, max=10.0)
        y_pred = torch.sigmoid(logits_clipped).cpu().numpy()
        y_true = split.edge_label.cpu().numpy()

        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }

        # Degree-controlled metrics
        node_degrees = degree(G.edge_index[0], G.num_nodes)
        edges = split.edge_label_index
        edge_min_deg = np.minimum(
            node_degrees[edges[0]].cpu().numpy(),
            node_degrees[edges[1]].cpu().numpy()
        )

        for name, mask in [
            ("Tail", edge_min_deg <= 5),
            ("Medium", (edge_min_deg > 5) & (edge_min_deg <= 20)),
            ("Head", edge_min_deg > 20)
        ]:
            if mask.sum() > 0:
                metrics[f"r2_{name}"] = float(r2_score(y_true[mask], y_pred[mask]))

        if not return_preds:
            return metrics

        edges_list = split.edge_label_index.t().cpu().numpy().tolist()
        records = [
            {"input": {"edge": [int(u), int(v)]}, "prediction": float(p), "ground_truth": float(t)}
            for (u, v), p, t in zip(edges_list, y_pred.tolist(), y_true.tolist())
        ]
        return metrics, records

    def save_model(self, path: str | Path, config: AttributeModelConfig):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_config": {
                "in_channels": config.in_channels,
                "hidden_channels": config.hidden_channels,
                "num_layers": config.num_layers,
                "heads": config.heads,
                "dropout": config.dropout,
            },
            "model_state_dict": self.model.state_dict(),
        }, path)


def build_attribute_model(config: AttributeModelConfig, device: torch.device) -> GNNLinkPredictor:
    """Build model for attribute prediction."""
    model = GNNLinkPredictor(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)

    # Xavier initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model
