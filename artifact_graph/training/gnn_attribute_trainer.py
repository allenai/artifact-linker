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
from torch_geometric.data import Data

from ..models.gnn_link_predictor import GNNLinkPredictor
from .gnn_attribute_evaluator import GNNAttributeEvaluator


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


def load_split_edge_metadata(split_dir: str | Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load per-split edge metadata (saved during graph splitting).

    Each split directory should contain an edge_metadata_normalized.json file
    with the normalized edge metadata for that split's positive edges.
    """
    f = Path(split_dir) / "edge_metadata_normalized.json"
    meta = json.loads(f.read_text(encoding="utf-8"))

    mapping = {}
    for k, v in meta.items():
        parts = k.split(",")
        if len(parts) == 2:
            u, w = int(parts[0].strip()), int(parts[1].strip())
            mapping[(u, w)] = v.get("metrics", {}) if isinstance(v, dict) else {}

    return mapping


def load_attribute_split(
    split_dir: str | Path,
    forced_x: Optional[torch.Tensor] = None,
) -> Tuple[Data, Data]:
    """Load data for attribute prediction split.

    Reads edge_metadata_normalized.json directly from split_dir (saved during
    graph splitting from edge_metadata_normalized.json).
    Uses the first numeric metric value as the regression target.
    """
    p = Path(split_dir)

    edges = torch.from_numpy(np.load(p / "edges.npz")["edges"]).long()
    with open(p / "node_metadata.json") as f:
        num_nodes = len(json.load(f))

    if forced_x is not None:
        x = forced_x
    else:
        print(f"[Warning] No forced_x provided, using random embeddings")
        x = torch.randn(num_nodes, 768)

    G = Data(x=x, edge_index=edges, num_nodes=num_nodes)

    pos_all = np.load(p / "pos_edges.npz")["edges"]
    metadata = load_split_edge_metadata(split_dir)

    # Use the first numeric metric value as the regression target
    kept_edges, kept_vals = [], []
    for u, v in pos_all.T:
        u, v = int(u), int(v)
        metrics = metadata.get((u, v), metadata.get((v, u), {}))
        if not metrics:
            continue
        for val in metrics.values():
            if isinstance(val, (int, float)):
                kept_edges.append([u, v])
                kept_vals.append(float(val))
                break

    edge_index = torch.tensor(kept_edges, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(kept_vals, dtype=torch.float)

    split = Data()
    split.edge_label_index = edge_index
    split.edge_label = edge_label
    print(f"[split] {edge_label.numel()}/{pos_all.shape[1]} edges with metrics")

    return G, split


class GNNAttributeTrainer:
    """Trainer for GNN attribute prediction (regression)."""

    def __init__(self, model: GNNLinkPredictor, device: torch.device, config: AttributeTrainingConfig):
        self.model = model
        self.device = device
        self.config = config
        self.evaluator = GNNAttributeEvaluator()
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
                metrics = self.evaluator.evaluate(self.model, G_train, split_val)
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
