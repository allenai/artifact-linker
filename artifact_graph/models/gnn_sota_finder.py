#!/usr/bin/env python3
"""
SOTAFinder: Uncertainty-aware GNN for SOTA model discovery.

Built on NCNC (Neural Common Neighbor with Completion) backbone, with a
dual-head decoder that predicts performance mean (μ) and epistemic
uncertainty (log σ²) through **fully decoupled** pathways.

Design principle: the encoder and μ-head are trained purely by MSE (or
optionally asymmetric MSE), exactly like vanilla NCNC.  The variance head
receives **gradient-stopped** features (``feat.detach()``) so that its
training never interferes with the encoder or μ predictions.  This
guarantees regression quality on par with NCNC while adding meaningful
uncertainty estimates for UCB exploration.

Three complementary mechanisms:

  1. Decoupled Uncertainty Estimation – separate variance pathway with
     gradient isolation trains σ without affecting μ quality.
  2. Semi-supervised Ranking Constraint – pairwise margin loss between
     observed best and randomly sampled unobserved candidates.
  3. Uncertainty-aware Exploration Scoring – UCB-style inference score
     μ + β·σ that automatically promotes high-potential unknowns.

Interface (compatible with existing GNNAttributeTrainer / Evaluator):
    encode(x, edge_index, num_nodes=None) → z
    decode(z, edge_index)                → (mu, log_var)
    decode_score(z, edge_index, beta)    → exploration score
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .gnn_encoder import GNNEncoder


# --------------------------------------------------------------------------- #
# Dual-head decoder
# --------------------------------------------------------------------------- #

class DualHeadDecoder(nn.Module):
    """Predict both μ (mean logit) and log σ² (log-variance) for each edge.

    **Key design**: the μ and σ pathways are fully separated.  The variance
    head receives ``feat.detach()`` so its gradients never flow back into
    the encoder or any upstream modules.  This ensures that uncertainty
    estimation cannot interfere with the mean prediction quality.

    Args:
        in_dim:  Input dimension (concatenated src + dst + cn features).
        hidden:  Hidden layer width.
        dropout: Dropout rate.
    """

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        # ---- Mean pathway (receives full gradients → trains encoder) ----
        self.mu_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # ---- Variance pathway (receives detached features → isolated) ----
        self.var_net = nn.Sequential(
            nn.Linear(in_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Zero-init variance output → log_var ≈ 0, σ ≈ 1 at start
        nn.init.zeros_(self.var_net[-1].weight)
        nn.init.zeros_(self.var_net[-1].bias)

    def forward(self, feat: torch.Tensor):
        """Return (mu, log_var) each of shape (num_edges,).

        Gradients from log_var do NOT propagate through feat into the
        encoder thanks to the .detach() call.  log_var is clamped to
        prevent numerical overflow in downstream exp() calls.
        """
        mu = self.mu_net(feat).squeeze(-1)
        log_var = self.var_net(feat.detach()).squeeze(-1)
        # Clamp to [-6, 2] → σ ∈ [~0.05, ~2.7], prevents inf in exp()
        log_var = torch.clamp(log_var, min=-6.0, max=2.0)
        return mu, log_var


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class SOTAFinder(nn.Module):
    """Uncertainty-aware NCNC model for SOTA discovery.

    Args:
        in_channels:      Input feature dimension.
        hidden_channels:  GNN hidden dimension.
        num_layers:       Number of GNN layers.
        heads:            Attention heads (GATv2 backbone).
        dropout:          Dropout rate.
        backbone:         ``"gatv2"`` or ``"gcn"`` for the shared encoder.
        use_completion:   Enable NCNC virtual-neighbor completion (default True).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        backbone: str = "gatv2",
        use_completion: bool = True,
    ):
        super().__init__()
        self.use_completion = use_completion
        self.dropout = dropout

        # ---- Shared GNN encoder (same as NCNC) ----
        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            backbone=backbone,
        )
        h = self.encoder.out_channels

        # ---- Common-neighbor aggregator ----
        self.cn_mlp = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

        # ---- (NCNC) Virtual-neighbor completion ----
        if use_completion:
            self.completion = nn.Sequential(
                nn.Linear(h * 2, h),
                nn.ReLU(),
                nn.Linear(h, h),
            )
            predictor_in = h * 2 + h * 2   # src+dst + cn+virtual_cn
        else:
            predictor_in = h * 2 + h        # src+dst + cn

        # ---- Dual-head decoder (mu + log_var) ----
        self.predictor = DualHeadDecoder(
            in_dim=predictor_in,
            hidden=hidden_channels,
            dropout=dropout,
        )

        # Cached sparse adjacency (rebuilt only when edge_index changes)
        self._adj: sp.csr_matrix | None = None
        self._num_nodes: int = 0
        self._cache_key: tuple | None = None

    # ------------------------------------------------------------------ #
    # Sparse adjacency caching (identical to NCNC)
    # ------------------------------------------------------------------ #
    def _maybe_rebuild_adj(self, edge_index: torch.Tensor, num_nodes: int):
        key = (edge_index.data_ptr(), edge_index.shape[1])
        if key == self._cache_key:
            return
        ei = edge_index.cpu().numpy()
        rows = np.concatenate([ei[0], ei[1]])
        cols = np.concatenate([ei[1], ei[0]])
        data = np.ones(len(rows), dtype=np.float32)
        self._adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        self._adj.data[:] = 1.0
        self._num_nodes = num_nodes
        self._cache_key = key

    def _get_cn_embeddings(self, z: torch.Tensor, pred_edges: torch.Tensor) -> torch.Tensor:
        src_np = pred_edges[0].cpu().numpy()
        dst_np = pred_edges[1].cpu().numpy()
        A = self._adj

        cn_matrix = A[src_np].multiply(A[dst_np])
        cn_counts = np.asarray(cn_matrix.sum(axis=1)).flatten()

        cn_coo = cn_matrix.tocoo()
        if cn_coo.nnz > 0:
            indices = torch.from_numpy(
                np.vstack([cn_coo.row, cn_coo.col]).astype(np.int64)
            )
            values = torch.from_numpy(cn_coo.data.astype(np.float32))
            cn_sparse = torch.sparse_coo_tensor(
                indices, values, cn_matrix.shape
            ).to(z.device)
            cn_sum = torch.sparse.mm(cn_sparse, z)
        else:
            cn_sum = torch.zeros(len(src_np), z.size(1), device=z.device)

        cn_counts_t = (
            torch.from_numpy(cn_counts.astype(np.float32))
            .to(z.device)
            .unsqueeze(1)
        )
        cn_mean = cn_sum / cn_counts_t.clamp(min=1)

        # Fallback to midpoint for edges without common neighbors
        no_cn_mask = cn_counts_t.squeeze(1) == 0
        if no_cn_mask.any():
            z_src = z[pred_edges[0]]
            z_dst = z[pred_edges[1]]
            cn_mean[no_cn_mask] = (z_src[no_cn_mask] + z_dst[no_cn_mask]) / 2

        return cn_mean

    # ------------------------------------------------------------------ #
    # encode / decode interface
    # ------------------------------------------------------------------ #
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """Encode node features → node embeddings z."""
        n = num_nodes or x.size(0)
        z = self.encoder(x, edge_index, num_nodes=num_nodes)
        self._maybe_rebuild_adj(edge_index, n)
        return z

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode edges → (mu, log_var) in logit space.

        Returns:
            mu:      Mean prediction logits, shape (num_edges,).
            log_var: Log-variance (epistemic uncertainty), shape (num_edges,).
        """
        src, dst = z[edge_index[0]], z[edge_index[1]]
        cn_z = self._get_cn_embeddings(z, edge_index)
        cn_feat = self.cn_mlp(cn_z)

        if self.use_completion:
            virtual_cn = self.completion(torch.cat([src, dst], dim=-1))
            virtual_cn_feat = self.cn_mlp(virtual_cn)
            combined = torch.cat([src, dst, cn_feat, virtual_cn_feat], dim=-1)
        else:
            combined = torch.cat([src, dst, cn_feat], dim=-1)

        return self.predictor(combined)

    def decode_score(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """UCB exploration score: μ + β · σ  (in logit space).

        Higher beta → more exploration (favour uncertain candidates).
        """
        mu, log_var = self.decode(z, edge_index)
        sigma = torch.exp(0.5 * log_var)       # std-dev
        return mu + beta * sigma

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        prediction_edge_index: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index, num_nodes=num_nodes)
        if prediction_edge_index is not None:
            return self.decode(z, prediction_edge_index)
        return z
