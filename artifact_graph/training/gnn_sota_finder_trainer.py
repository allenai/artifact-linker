#!/usr/bin/env python3
"""
SOTAFinder trainer – Two-phase training for SOTA discovery.

**Phase 1 – Regression** (identical to vanilla NCNC):
    Pure MSE on μ.  Trains encoder + mu_net to convergence.
    Result: NCNC-level regression quality (MSE / R²).

**Phase 2 – Uncertainty calibration** (encoder + mu_net frozen):
    Gaussian NLL on σ.  Only var_net weights are updated.
    The encoder features are stable, so var_net can learn meaningful
    uncertainty (σ ≈ |μ − target|).

**Inference**:
    UCB score = μ + β·σ  promotes uncertain-but-promising candidates
    for SOTA discovery without sacrificing regression quality.

Usage mirrors the existing GNNAttributeTrainer API so it can be
dropped in as a replacement.
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

from ..utils.graph_utils import select_edge_metric_target
from .gnn_sota_finder_evaluator import SOTAFinderEvaluator


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SOTAFinderTrainingConfig:
    """Hyperparameters for SOTAFinder two-phase training."""

    # ---- Phase 1: MSE regression (same as NCNC) ----
    epochs: int = 2000
    lr: float = 0.005
    weight_decay: float = 1e-5
    patience: int = 80          # early stopping patience (eval intervals)
    seed: int = 42

    # ---- Phase 2: Variance calibration (encoder frozen) ----
    epochs_phase2: int = 300    # 0 = skip Phase 2
    lr_phase2: float = 0.001

    # ---- Exploration (inference only) ----
    beta: float = 1.0           # UCB coefficient: score = μ + β·σ


@dataclass
class SOTAFinderModelConfig:
    """Model architecture config."""
    in_channels: int
    hidden_channels: int = 128
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.2
    backbone: str = "gatv2"
    use_completion: bool = True


# --------------------------------------------------------------------------- #
# Seed helper
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data loading (re-uses the attribute split loader format)
# --------------------------------------------------------------------------- #

def load_split_edge_metadata(split_dir: str | Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load per-split edge metadata (normalised metrics)."""
    f = Path(split_dir) / "edge_metadata_normalized.json"
    meta = json.loads(f.read_text(encoding="utf-8"))
    mapping = {}
    for k, v in meta.items():
        parts = k.split(",")
        if len(parts) == 2:
            u, w = int(parts[0].strip()), int(parts[1].strip())
            mapping[(u, w)] = v.get("metrics", {}) if isinstance(v, dict) else {}
    return mapping


def load_sota_finder_split(
    split_dir: str | Path,
    forced_x: Optional[torch.Tensor] = None,
    metric_name: Optional[str] = None,
) -> Tuple[Data, Data]:
    """Load data for SOTAFinder training split.

    Returns (G, split) where:
        G.x, G.edge_index, G.num_nodes  – full graph
        split.edge_label_index           – positive edges with metrics
        split.edge_label                 – metric values  ∈ [0, 1]
    """
    p = Path(split_dir)

    edges = torch.from_numpy(np.load(p / "edges.npz")["edges"]).long()
    with open(p / "node_metadata.json") as f:
        num_nodes = len(json.load(f))

    if forced_x is not None:
        x = forced_x
    else:
        print("[Warning] No forced_x provided, using random embeddings")
        x = torch.randn(num_nodes, 768)

    G = Data(x=x, edge_index=edges, num_nodes=num_nodes)

    pos_all = np.load(p / "pos_edges.npz")["edges"]
    metadata = load_split_edge_metadata(split_dir)

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


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #

def variance_nll_loss(
    mu_detached: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL for variance calibration.

    log_var is already clamped by the model (DualHeadDecoder).
    μ should be detached.
    """
    var = torch.exp(log_var)
    nll = 0.5 * (log_var + (mu_detached - target).pow(2) / var)
    return nll.mean()


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class SOTAFinderTrainer:
    """Two-phase trainer for SOTAFinder (dual-head NCNC).

    Phase 1: Pure MSE on μ (identical to NCNC).
    Phase 2: Gaussian NLL on σ with encoder + μ-head frozen.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: SOTAFinderTrainingConfig,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.evaluator = SOTAFinderEvaluator(beta=config.beta)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.best_mse: Optional[float] = None
        self.best_state: Optional[Dict] = None

    # ----- helpers -------------------------------------------------------- #

    @staticmethod
    def _to_logit(y: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] metric value → logit space."""
        y_clip = torch.clamp(y, min=1e-7, max=1 - 1e-7)
        return torch.log(y_clip / (1 - y_clip))

    # ----- Phase 1: MSE -------------------------------------------------- #

    def _train_step_mse(self, G: Data, split: Data) -> float:
        """Single MSE training step (Phase 1)."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        z = self.model.encode(G.x, G.edge_index)
        mu, _ = self.model.decode(z, split.edge_label_index)
        y_logits = self._to_logit(split.edge_label)

        loss = F.mse_loss(mu, y_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss)

    # ----- Phase 2: Variance NLL ----------------------------------------- #

    def _train_step_var(
        self, G: Data, split: Data, var_optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single variance NLL training step (Phase 2, encoder frozen)."""
        self.model.train()
        var_optimizer.zero_grad(set_to_none=True)

        z = self.model.encode(G.x, G.edge_index)
        mu, log_var = self.model.decode(z, split.edge_label_index)
        y_logits = self._to_logit(split.edge_label)

        loss = variance_nll_loss(mu.detach(), log_var, y_logits)
        loss.backward()
        var_optimizer.step()
        return float(loss)

    # ----- Full training loop --------------------------------------------- #

    def train(
        self,
        G_train: Data,
        split_train: Data,
        G_val: Data,
        split_val: Data,
        verbose: bool = True,
    ):
        """Two-phase training: MSE → freeze → variance NLL."""

        # ================================================================= #
        #  Phase 1:  Pure MSE regression  (identical to NCNC)               #
        # ================================================================= #
        if verbose:
            print(f"[Phase 1] MSE regression ({self.config.epochs} epochs, "
                  f"patience={self.config.patience})")

        wait = 0
        for epoch in range(1, self.config.epochs + 1):
            loss_mu = self._train_step_mse(G_train, split_train)

            eval_freq = 10 if epoch <= 50 else 25
            if epoch % eval_freq == 0 or epoch == self.config.epochs:
                self.model.eval()
                metrics = self.evaluator.evaluate(self.model, G_train, split_val)
                is_best = self.best_mse is None or metrics["mse"] < self.best_mse

                if is_best:
                    self.best_mse = metrics["mse"]
                    self.best_state = {
                        k: v.cpu() for k, v in self.model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1

                if verbose:
                    mark = "⭐" if is_best else ""
                    print(
                        f"  epoch {epoch:04d} | loss_μ {loss_mu:.6f} | "
                        f"val_mse {metrics['mse']:.6f} | "
                        f"val_r² {metrics['r_squared']:.4f} {mark}"
                    )
                self.model.train()

                if wait >= self.config.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(patience={self.config.patience})")
                    break

        # Restore best Phase 1 checkpoint
        if self.best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )
            if verbose:
                print(f"  ✓ Phase 1 best: val_mse={self.best_mse:.6f}")

        # ================================================================= #
        #  Phase 2:  Variance calibration  (encoder + mu_net frozen)        #
        # ================================================================= #
        if self.config.epochs_phase2 <= 0:
            if verbose:
                print("[Phase 2] Skipped (epochs_phase2=0)")
            return

        if verbose:
            print(f"\n[Phase 2] Variance calibration ({self.config.epochs_phase2} epochs, "
                  f"lr={self.config.lr_phase2})")

        # Freeze everything except var_net
        for name, param in self.model.named_parameters():
            param.requires_grad = ("var_net" in name)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        if verbose:
            print(f"  Frozen: {n_frozen:,} params | Trainable (var_net): {n_trainable:,} params")

        var_params = [p for p in self.model.parameters() if p.requires_grad]
        var_optimizer = torch.optim.Adam(var_params, lr=self.config.lr_phase2)

        best_nll = float("inf")
        best_var_state: Optional[Dict] = None

        for epoch in range(1, self.config.epochs_phase2 + 1):
            loss_var = self._train_step_var(G_train, split_train, var_optimizer)

            if epoch % 25 == 0 or epoch == self.config.epochs_phase2:
                self.model.eval()
                metrics = self.evaluator.evaluate(self.model, G_train, split_val)
                nll = metrics.get("nll", float("inf"))
                is_best = nll < best_nll
                if is_best:
                    best_nll = nll
                    best_var_state = {
                        k: v.cpu()
                        for k, v in self.model.state_dict().items()
                        if "var_net" in k
                    }
                if verbose:
                    mark = "⭐" if is_best else ""
                    print(
                        f"  epoch {epoch:04d} | loss_σ {loss_var:.4f} | "
                        f"avg_σ {metrics.get('avg_sigma', 0):.4f} | "
                        f"nll {nll:.4f} {mark}"
                    )
                self.model.train()

        # Restore best var_net weights
        if best_var_state:
            current = self.model.state_dict()
            current.update({k: v.to(self.device) for k, v in best_var_state.items()})
            self.model.load_state_dict(current)
            if verbose:
                print(f"  ✓ Phase 2 best: nll={best_nll:.4f}")

        # Unfreeze all (for checkpoint consistency)
        for param in self.model.parameters():
            param.requires_grad = True

    # ----- checkpoint ----------------------------------------------------- #

    def save_model(self, path: str | Path, config: SOTAFinderModelConfig):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": "sota_finder",
            "model_config": {
                "in_channels": config.in_channels,
                "hidden_channels": config.hidden_channels,
                "num_layers": config.num_layers,
                "heads": config.heads,
                "dropout": config.dropout,
                "backbone": config.backbone,
                "use_completion": config.use_completion,
            },
            "training_config": {
                "epochs": self.config.epochs,
                "epochs_phase2": self.config.epochs_phase2,
                "beta": self.config.beta,
            },
            "model_state_dict": self.model.state_dict(),
        }, path)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_sota_finder(config: SOTAFinderModelConfig, device: torch.device) -> nn.Module:
    """Build SOTAFinder model from config."""
    from ..models.gnn_sota_finder import SOTAFinder

    model = SOTAFinder(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
        backbone=config.backbone,
        use_completion=config.use_completion,
    ).to(device)

    # Xavier initialisation for encoder + mu pathway.
    # Skip var_net (its output layer is zero-initialised in __init__).
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "var_net" not in name:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    return model


def load_sota_finder(
    path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load a saved SOTAFinder checkpoint."""
    from ..models.gnn_sota_finder import SOTAFinder

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = SOTAFinder(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        backbone=cfg.get("backbone", "gatv2"),
        use_completion=cfg.get("use_completion", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
