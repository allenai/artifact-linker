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

import torch.nn as nn

from ..utils.graph_utils import select_edge_metric_target
from .gnn_attribute_evaluator import GNNAttributeEvaluator


@dataclass
class AttributeTrainingConfig:
    """Configuration for attribute prediction training."""
    epochs: int = 500
    lr: float = 0.005
    weight_decay: float = 1e-5
    seed: int = 42
    neg_ratio: int = 0  # number of negative (unobserved) pairs per positive (0 = disabled)
    neg_target: float = 0.1  # target threshold for unobserved pairs (margin loss)


ATTR_MODEL_TYPES = ("gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy")


@dataclass
class AttributeModelConfig:
    """Configuration for attribute prediction model."""
    in_channels: int
    hidden_channels: int = 128
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.2
    model_type: str = "gatv2"  # one of ATTR_MODEL_TYPES


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_edge_metadata(
    split_dir: str | Path,
    metric_file: str = "edge_metadata_normalized.json",
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load per-split edge metadata (saved during graph splitting).

    Each split directory should contain an edge_metadata_normalized.json file
    with the normalized edge metadata for that split's positive edges.
    """
    f = Path(split_dir) / metric_file
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
    metric_name: Optional[str] = None,
    metric_file: str = "edge_metadata_normalized.json",
) -> Tuple[Data, Data]:
    """Load data for attribute prediction split.

    Reads edge metadata from split_dir (saved during graph splitting).
    Uses one metric target per edge:
      - metric_name if provided
      - otherwise the first numeric metric in sorted-key order.
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
    metadata = load_split_edge_metadata(split_dir, metric_file=metric_file)

    # Use a single metric target per edge (shared with baseline/LLM loading).
    kept_edges, kept_vals = [], []
    for u, v in pos_all.T:
        u, v = int(u), int(v)
        metrics = metadata.get((u, v), metadata.get((v, u), {}))
        if not metrics:
            continue
        selected = select_edge_metric_target(metrics, metric_name=metric_name)
        if selected is not None:
            _, selected_value = selected
            kept_edges.append([u, v])
            kept_vals.append(selected_value)

    edge_index = torch.tensor(kept_edges, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(kept_vals, dtype=torch.float)

    split = Data()
    split.edge_label_index = edge_index
    split.edge_label = edge_label
    print(f"[split] {edge_label.numel()}/{pos_all.shape[1]} edges with metrics")

    return G, split


class GNNAttributeTrainer:
    """Trainer for GNN attribute prediction (regression).

    Works with any model exposing ``encode(x, edge_index)`` and
    ``decode(z, edge_index)`` (GNNLinkPredictor, NCN, NeoGNN, BUDDY, …).

    When neg_ratio > 0, samples unobserved model-dataset pairs each epoch
    and trains them with a low target (neg_target, default 0). This teaches
    the model that arbitrary pairs should have near-zero scores, while only
    observed pairs should have their actual metric values.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: AttributeTrainingConfig):
        self.model = model
        self.device = device
        self.config = config
        self.evaluator = GNNAttributeEvaluator()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.best_mse: Optional[float] = None
        self.best_state: Optional[Dict] = None
        # Cached sets for negative sampling (populated lazily)
        self._pos_set: Optional[set] = None
        self._model_ids: Optional[list] = None
        self._dataset_ids: Optional[list] = None

    def _init_neg_sampling(self, G: Data, split: Data):
        """Initialize node ID lists and positive edge set for negative sampling."""
        if self._pos_set is not None:
            return
        # Build set of positive edges (from both message-passing graph and train split)
        self._pos_set = set()
        ei = G.edge_index.cpu().numpy()
        for u, v in zip(ei[0], ei[1]):
            self._pos_set.add((int(u), int(v)))
        si = split.edge_label_index.cpu().numpy()
        for u, v in zip(si[0], si[1]):
            self._pos_set.add((int(u), int(v)))

        # Identify model and dataset node IDs from edge patterns
        # Models are sources (index 0), datasets are targets (index 1)
        self._model_ids = sorted(set(int(x) for x in si[0]))
        self._dataset_ids = sorted(set(int(x) for x in si[1]))

    def _sample_negatives(self, num_neg: int) -> torch.Tensor:
        """Sample random unobserved model-dataset pairs."""
        neg_edges = []
        attempts = 0
        max_attempts = num_neg * 10
        while len(neg_edges) < num_neg and attempts < max_attempts:
            m = self._model_ids[torch.randint(len(self._model_ids), (1,)).item()]
            d = self._dataset_ids[torch.randint(len(self._dataset_ids), (1,)).item()]
            if (m, d) not in self._pos_set:
                neg_edges.append([m, d])
            attempts += 1
        if not neg_edges:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()

    def train_epoch(self, G: Data, split: Data) -> float:
        """Train for one epoch with optional negative sampling.

        Loss design:
          - Positive pairs: MSE in logit space (standard regression).
          - Negative pairs: margin loss that pushes sigmoid(logit) below neg_target.
            Only penalizes negatives whose predicted score exceeds neg_target.

        This avoids forcing negatives to an extreme logit value, which would
        dominate the loss and collapse the positive regression signal.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        z = self.model.encode(G.x, G.edge_index)

        # Positive edges: regression in logit space
        pos_logits = self.model.decode(z, split.edge_label_index).squeeze(-1)
        y = split.edge_label
        y_clipped = torch.clamp(y, min=1e-7, max=1 - 1e-7)
        y_logits = torch.log(y_clipped / (1 - y_clipped))
        pos_loss = F.mse_loss(pos_logits, y_logits)

        if self.config.neg_ratio > 0:
            self._init_neg_sampling(G, split)
            num_neg = int(split.edge_label_index.size(1) * self.config.neg_ratio)
            neg_edge_index = self._sample_negatives(num_neg).to(self.device)

            if neg_edge_index.size(1) > 0:
                neg_logits = self.model.decode(z, neg_edge_index).squeeze(-1)
                neg_preds = torch.sigmoid(torch.clamp(neg_logits, -10, 10))
                # Hinge-style margin loss: penalize if pred > neg_target
                margin = torch.clamp(neg_preds - self.config.neg_target, min=0.0)
                neg_loss = margin.pow(2).mean()
                loss = pos_loss + neg_loss
            else:
                loss = pos_loss
        else:
            loss = pos_loss

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
                self.model.eval()
                # Always use G_train for validation encoding during training.
                # Support edges are only used at final test time, not during training.
                metrics = self.evaluator.evaluate(self.model, G_train, split_val)
                is_best = self.best_mse is None or metrics["mse"] < self.best_mse

                if is_best:
                    self.best_mse = metrics["mse"]
                    self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                if verbose:
                    mark = "⭐" if is_best else ""
                    print(
                        f"epoch {epoch:04d} | loss {loss:.6f} | "
                        f"val_mse {metrics['mse']:.6f} | val_r_squared {metrics['r_squared']:.4f} {mark}"
                    )
                self.model.train()  # switch back for next training epoch

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
                "model_type": config.model_type,
            },
            "model_state_dict": self.model.state_dict(),
        }, path)


def build_attribute_model(config: AttributeModelConfig, device: torch.device) -> nn.Module:
    """Build model for attribute prediction (factory).

    Supports: gatv2, gcn, ncn, ncnc, neognn, buddy.
    Backbone is derived from model_type: gatv2/gcn use themselves,
    others default to gatv2.
    """
    mt = config.model_type
    # Derive backbone from model_type
    bb = mt if mt in ("gatv2", "gcn") else "gatv2"

    if mt in ("gatv2", "gcn"):
        from ..models.gnn_link_predictor import GNNLinkPredictor
        model: nn.Module = GNNLinkPredictor(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            backbone=bb,
            decoder="mlp",
        )
    elif mt in ("ncn", "ncnc"):
        from ..models.ncn_link_predictor import NCNLinkPredictor
        model = NCNLinkPredictor(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            backbone=bb,
            use_completion=(mt == "ncnc"),
        )
    elif mt == "neognn":
        from ..models.neognn_link_predictor import NeoGNNLinkPredictor
        model = NeoGNNLinkPredictor(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            backbone=bb,
        )
    elif mt == "buddy":
        from ..models.buddy_link_predictor import BUDDYLinkPredictor
        model = BUDDYLinkPredictor(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            backbone=bb,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{mt}'. Choose from {ATTR_MODEL_TYPES}"
        )

    model = model.to(device)

    # Xavier initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model
