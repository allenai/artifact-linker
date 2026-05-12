#!/usr/bin/env python3
"""GNN training utilities for link prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from ..models.gnn_link_predictor import GNNLinkPredictor
from .gnn_link_evaluator import GNNLinkEvaluator

# Supported model types for link prediction
LINK_MODEL_TYPES = ("gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy")


@dataclass
class LinkTrainingConfig:
    """Configuration for GNN training."""
    epochs: int = 5000
    eval_every: int = 10
    patience: int = 50
    lr: float = 5e-3
    weight_decay: float = 1e-4
    lr_patience: int = 15
    amp: bool = False
    seed: int = 42
    threshold: float = 0.5  # probability threshold for binary classification (F1, precision, recall)
    neg_ratio: Optional[int] = None  # negative:positive ratio for training (e.g. 5 means 1:5); None = use all negatives


@dataclass
class LinkModelConfig:
    """Configuration for GNN model."""
    in_channels: int
    hidden_channels: int = 64
    num_layers: int = 3
    heads: int = 3
    dropout: float = 0.2
    model_type: str = "gatv2"  # one of LINK_MODEL_TYPES


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bce_logits_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Compute binary cross-entropy loss with optional positive weighting."""
    y = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    logits = torch.cat([pos_logits, neg_logits], dim=0)

    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits, y)

    w = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=w)


class GNNLinkTrainer:
    """Trainer for GNN link prediction models.

    Works with any model that exposes ``encode(x, edge_index)`` and
    ``decode(z, edge_index)`` methods (GNNLinkPredictor, NCN, Neo-GNN, BUDDY, …).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: LinkTrainingConfig,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.evaluator = GNNLinkEvaluator(threshold=config.threshold)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.8,
            patience=config.lr_patience,
            min_lr=1e-6,
        )
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=config.amp and device.type == "cuda"
        )

        # Training state
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def _subsample_negatives(self, neg_edge_index: torch.Tensor, num_pos: int) -> torch.Tensor:
        """Subsample negative edges to neg_ratio * num_pos if neg_ratio is set."""
        neg_ratio = self.config.neg_ratio
        if neg_ratio is None:
            return neg_edge_index
        max_neg = neg_ratio * num_pos
        num_neg = neg_edge_index.size(1)
        if num_neg <= max_neg:
            return neg_edge_index
        perm = torch.randperm(num_neg, device=neg_edge_index.device)[:max_neg]
        return neg_edge_index[:, perm]

    def train_epoch(
        self,
        data,
        split,
        pos_weight: Optional[float] = None,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Subsample negatives each epoch for diversity
        num_pos = split.pos_edge_label_index.size(1)
        neg_edges = self._subsample_negatives(split.neg_edge_label_index, num_pos)

        with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
            z = self.model.encode(data.x, split.edge_index)
            pos = self.model.decode(z, split.pos_edge_label_index)
            neg = self.model.decode(z, neg_edges)
            loss = bce_logits_loss(pos, neg, pos_weight)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss.detach().cpu())

    def train(
        self,
        train_data,
        train_split,
        val_data,
        val_split,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Returns:
            Dictionary with training history and best metrics.
        """
        # Calculate positive weight for imbalanced data
        pos_n = train_split.pos_edge_label_index.size(1)
        neg_n = train_split.neg_edge_label_index.size(1)
        # If neg_ratio is set, use the effective ratio for pos_weight
        if self.config.neg_ratio is not None:
            effective_neg = min(neg_n, self.config.neg_ratio * pos_n)
            pos_weight = effective_neg / max(1, pos_n)
            if verbose:
                print(f"Training neg subsample: {neg_n} -> {effective_neg} "
                      f"(ratio 1:{self.config.neg_ratio}, pos={pos_n})")
        else:
            pos_weight = neg_n / max(1, pos_n)

        # Pre-calculate node degrees for degree-controlled evaluation
        node_degrees = degree(train_data.edge_index[0], train_data.num_nodes)

        history = {"losses": [], "val_metrics": []}
        wait = 0

        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(train_data, train_split, pos_weight)
            history["losses"].append(loss)

            if epoch % self.config.eval_every == 0 or epoch == self.config.epochs:
                self.model.eval()
                # Always use train_data for validation encoding during training.
                # Support edges are only used at final test time, not during training.
                with torch.no_grad():
                    z = self.model.encode(train_data.x, train_data.edge_index)

                val_metrics = self.evaluator.evaluate(
                    self.model, z, val_split, node_degrees=node_degrees
                )
                history["val_metrics"].append(val_metrics)

                self.scheduler.step(val_metrics["ap_auc"])

                improved = (
                    self.best_metrics is None
                    or val_metrics["ap_auc"] > self.best_metrics["ap_auc"]
                )

                if improved:
                    self.best_metrics = val_metrics
                    self.best_state = {
                        k: v.cpu() for k, v in self.model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1

                if verbose:
                    self._print_progress(epoch, loss, val_metrics)

                self.model.train()  # switch back for next training epoch

                if wait >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )
            if verbose:
                print(f"Restored best checkpoint (val_ap_auc={self.best_metrics['ap_auc']:.4f})")

        return {
            "history": history,
            "best_metrics": self.best_metrics,
        }

    def _print_progress(self, epoch: int, loss: float, val_metrics: Dict[str, float]):
        """Print training progress."""
        tail_auc = val_metrics.get("ap_auc_Tail (deg<=5)", "")
        head_auc = val_metrics.get("ap_auc_Head (deg>20)", "")

        bucket_info = ""
        if tail_auc:
            bucket_info += f" | Tail_AP {tail_auc:.4f}"
        if head_auc:
            bucket_info += f" | Head_AP {head_auc:.4f}"

        print(
            f"epoch {epoch:04d} | loss {loss:.4f} | "
            f"val_ap_auc {val_metrics['ap_auc']:.4f} | "
            f"val_mcc {val_metrics['mcc']:.4f} | "
            f"val_recall {val_metrics['recall']:.4f}"
            f"{bucket_info}"
        )

    def save_model(self, path: str | Path, model_config: LinkModelConfig):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_config": {
                    "in_channels": model_config.in_channels,
                    "hidden_channels": model_config.hidden_channels,
                    "num_layers": model_config.num_layers,
                    "heads": model_config.heads,
                    "dropout": model_config.dropout,
                    "model_type": model_config.model_type,
                },
                "model_state_dict": self.model.state_dict(),
                "best_val_metrics": self.best_metrics,
            },
            path,
        )
        return path


def build_link_model(config: LinkModelConfig, device: torch.device) -> nn.Module:
    """Build a link-prediction model from config (factory).

    Supports: gatv2, gcn, ncn, ncnc, neognn, buddy.
    Backbone is derived from model_type: gatv2/gcn use themselves,
    others default to gatv2.
    """
    mt = config.model_type
    # Derive backbone from model_type
    bb = mt if mt in ("gatv2", "gcn") else "gatv2"

    if mt in ("gatv2", "gcn"):
        model: nn.Module = GNNLinkPredictor(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            backbone=bb,
            decoder="bilinear"
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
            f"Unknown model_type '{mt}'. Choose from {LINK_MODEL_TYPES}"
        )

    return model.to(device)


def load_link_model(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, float]]:
    """Load a saved link-prediction model checkpoint.

    Returns:
        (model, best_val_metrics)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["model_config"])
    # Backward compatibility: old checkpoints may lack model_type
    if "model_type" not in cfg_dict:
        cfg_dict["model_type"] = "gatv2"
    # Drop legacy backbone field if present (now derived from model_type)
    cfg_dict.pop("backbone", None)
    model_cfg = LinkModelConfig(**cfg_dict)
    model = build_link_model(model_cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("best_val_metrics", {})
