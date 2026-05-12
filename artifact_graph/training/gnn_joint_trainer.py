#!/usr/bin/env python3
"""
Joint trainer for link prediction + attribute prediction.

Trains a shared GNN encoder with two decoder heads:
  - Link head: BCE loss on positive/negative edges
  - Attr head: MSE loss (logit space) on positive edges only

The link head's negative sampling teaches the encoder to distinguish
relevant from irrelevant pairs. This signal propagates through the
shared encoder to the attr head, enabling it to implicitly output
lower scores for unobserved pairs.

Best config (from ablation):
  - 2000 epochs with cosine LR schedule (lr=3e-3 → 1e-5)
  - attr_weight=5, neg_ratio=2, dropout=0.2
  - GATv2 backbone, hidden=128, 3 layers, 8 heads
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from ..models.gnn_joint_predictor import GNNJointPredictor
from ..utils.graph_utils import select_edge_metric_target


@dataclass
class JointTrainingConfig:
    epochs: int = 2000
    lr: float = 0.003
    weight_decay: float = 1e-5
    seed: int = 42
    neg_ratio: int = 2
    attr_weight: float = 5.0  # weight of attr loss relative to link loss
    use_cosine_lr: bool = True  # cosine annealing LR schedule
    use_ipw: bool = False  # inverse probability weighting for attr loss


@dataclass
class JointModelConfig:
    in_channels: int
    hidden_channels: int = 128
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.2
    backbone: str = "gatv2"
    model_type: str = "gatv2"  # "gatv2" | "gcn" | "ncn" | "ncnc" | "neognn" | "buddy"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_base_link_model(config: JointModelConfig) -> nn.Module:
    """Build the base link prediction model for any architecture."""
    mt = config.model_type
    bb = mt if mt in ("gatv2", "gcn") else "gatv2"

    if mt in ("gatv2", "gcn"):
        from ..models.gnn_link_predictor import GNNLinkPredictor
        return GNNLinkPredictor(
            in_channels=config.in_channels, hidden_channels=config.hidden_channels,
            num_layers=config.num_layers, heads=config.heads, dropout=config.dropout,
            backbone=bb, decoder="bilinear",
        )
    elif mt in ("ncn", "ncnc"):
        from ..models.ncn_link_predictor import NCNLinkPredictor
        return NCNLinkPredictor(
            in_channels=config.in_channels, hidden_channels=config.hidden_channels,
            num_layers=config.num_layers, heads=config.heads, dropout=config.dropout,
            backbone=bb, use_completion=(mt == "ncnc"),
        )
    elif mt == "neognn":
        from ..models.neognn_link_predictor import NeoGNNLinkPredictor
        return NeoGNNLinkPredictor(
            in_channels=config.in_channels, hidden_channels=config.hidden_channels,
            num_layers=config.num_layers, heads=config.heads, dropout=config.dropout,
            backbone=bb,
        )
    elif mt == "buddy":
        from ..models.buddy_link_predictor import BUDDYLinkPredictor
        return BUDDYLinkPredictor(
            in_channels=config.in_channels, hidden_channels=config.hidden_channels,
            num_layers=config.num_layers, heads=config.heads, dropout=config.dropout,
            backbone=bb,
        )
    else:
        raise ValueError(f"Unknown model_type: {mt}")


def build_joint_model(config: JointModelConfig, device: torch.device, use_heckman: bool = False) -> GNNJointPredictor:
    """Build a joint model: wrap any GNN link predictor with an attr head."""
    base_model = _build_base_link_model(config)
    model = GNNJointPredictor(base_model, dropout=config.dropout, use_heckman=use_heckman)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model.to(device)


class GNNJointTrainer:
    """Joint trainer for link + attribute prediction."""

    def __init__(self, model: GNNJointPredictor, device: torch.device, config: JointTrainingConfig):
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = None
        if config.use_cosine_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-5
            )
        self.best_metric: Optional[float] = None
        self.best_state: Optional[Dict] = None
        self._pos_set: Optional[set] = None
        self._model_ids: Optional[list] = None
        self._dataset_ids: Optional[list] = None

    def _init_neg_sampling(self, G: Data, attr_split: Data):
        if self._pos_set is not None:
            return
        self._pos_set = set()
        ei = G.edge_index.cpu().numpy()
        for u, v in zip(ei[0], ei[1]):
            self._pos_set.add((int(u), int(v)))
        si = attr_split.edge_label_index.cpu().numpy()
        for u, v in zip(si[0], si[1]):
            self._pos_set.add((int(u), int(v)))
        self._model_ids = sorted(set(int(x) for x in si[0]))
        self._dataset_ids = sorted(set(int(x) for x in si[1]))

    def _sample_negatives(self, num_neg: int) -> torch.Tensor:
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

    def train_epoch(self, G: Data, attr_split: Data) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        z = self.model.encode(G.x, G.edge_index)

        pos_edges = attr_split.edge_label_index
        num_pos = pos_edges.size(1)

        # --- Link loss: BCE on pos + neg ---
        self._init_neg_sampling(G, attr_split)
        num_neg = int(num_pos * self.config.neg_ratio)
        neg_edges = self._sample_negatives(num_neg).to(self.device)

        pos_link_logits = self.model.decode_link(z, pos_edges)
        link_loss = F.binary_cross_entropy_with_logits(
            pos_link_logits, torch.ones_like(pos_link_logits)
        )

        if neg_edges.size(1) > 0:
            neg_link_logits = self.model.decode_link(z, neg_edges)
            neg_link_loss = F.binary_cross_entropy_with_logits(
                neg_link_logits, torch.zeros_like(neg_link_logits)
            )
            link_loss = (link_loss + neg_link_loss) / 2

        # --- Attr loss: MSE in logit space (positive edges only) ---
        attr_logits = self.model.decode_attr(z, pos_edges, link_logits=pos_link_logits).squeeze(-1)
        y = attr_split.edge_label
        y_clipped = torch.clamp(y, min=1e-7, max=1 - 1e-7)
        y_logits = torch.log(y_clipped / (1 - y_clipped))
        if self.config.use_ipw:
            # IPW: up-weight "surprising" observations (low link probability)
            p = torch.sigmoid(pos_link_logits.detach())
            w = 1.0 / torch.clamp(p, min=0.1)  # clamp to avoid extreme weights
            w = w / w.mean()  # normalize so weights average to 1
            attr_loss = (w * (attr_logits - y_logits) ** 2).mean()
        else:
            attr_loss = F.mse_loss(attr_logits, y_logits)

        # --- Combined loss ---
        loss = link_loss + self.config.attr_weight * attr_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": float(loss.detach().cpu()),
            "link_loss": float(link_loss.detach().cpu()),
            "attr_loss": float(attr_loss.detach().cpu()),
        }

    @torch.no_grad()
    def evaluate(self, G: Data, attr_split: Data) -> Dict[str, float]:
        """Evaluate on validation/test set (attr regression metrics on positive edges)."""
        self.model.eval()
        z = self.model.encode(G.x, G.edge_index)
        # Pass link logits for Heckman correction (auto-computed if use_heckman)
        attr_logits = self.model.decode_attr(z, attr_split.edge_label_index).squeeze(-1)
        attr_preds = torch.sigmoid(torch.clamp(attr_logits, -10, 10)).cpu().numpy()
        y_true = attr_split.edge_label.cpu().numpy()

        mse = float(np.mean((attr_preds - y_true) ** 2))
        ss_res = np.sum((attr_preds - y_true) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"mse": mse, "r_squared": r2}

    def train(self, G_train: Data, split_train: Data, G_val: Data, split_val: Data, verbose: bool = True):
        for epoch in range(1, self.config.epochs + 1):
            losses = self.train_epoch(G_train, split_train)

            eval_freq = 10 if epoch <= 50 else 25
            if epoch % eval_freq == 0 or epoch == self.config.epochs:
                self.model.eval()
                metrics = self.evaluate(G_train, split_val)
                is_best = self.best_metric is None or metrics["mse"] < self.best_metric

                if is_best:
                    self.best_metric = metrics["mse"]
                    self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                if verbose:
                    mark = "⭐" if is_best else ""
                    print(
                        f"epoch {epoch:04d} | loss {losses['loss']:.4f} "
                        f"(link {losses['link_loss']:.4f} + attr {losses['attr_loss']:.4f}) | "
                        f"val_mse {metrics['mse']:.6f} | val_r2 {metrics['r_squared']:.4f} {mark}"
                    )
                self.model.train()

        if self.best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})
            if verbose:
                print(f"Restored best (val_mse={self.best_metric:.6f})")

    def save_model(self, path: str | Path, config: JointModelConfig):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": "joint",
            "model_config": {
                "in_channels": config.in_channels,
                "hidden_channels": config.hidden_channels,
                "num_layers": config.num_layers,
                "heads": config.heads,
                "dropout": config.dropout,
                "backbone": config.backbone,
                "gnn_model_type": config.model_type,
            },
            "use_heckman": getattr(self.model, "use_heckman", False),
            "model_state_dict": self.model.state_dict(),
        }, path)
