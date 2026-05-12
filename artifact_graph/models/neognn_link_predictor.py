#!/usr/bin/env python3
"""
Neo-GNN: Neighborhood Overlap-aware GNN for link prediction.

Explicitly models neighborhood overlap structure by computing hand-crafted
structural features (common neighbors, Adamic-Adar, Resource Allocation, etc.)
and combining them with learned node representations.

Reference:
    Yun et al. "Neo-GNN: Neighborhood Overlap-aware Graph Neural Networks
    for Link Prediction" (NeurIPS 2021)

Interface:
    encode(x, edge_index, num_nodes=None) → z   (also caches adjacency)
    decode(z, edge_index) → logits               (uses cached adjacency)

Now uses the shared GNNEncoder backbone so that all models have aligned
parameter counts and the comparison is fair.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .gnn_encoder import GNNEncoder

_NUM_STRUCT_FEATURES = 6  # cn_count, aa, ra, deg_src, deg_dst, pa


class NeoGNNLinkPredictor(nn.Module):
    """
    Neo-GNN link predictor.

    Args:
        in_channels:      Input feature dimension.
        hidden_channels:  Hidden layer dimension.
        num_layers:       Number of encoder layers.
        heads:            Number of attention heads (GATv2 only).
        dropout:          Dropout rate.
        backbone:         ``"gatv2"`` or ``"gcn"`` — shared encoder type.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        backbone: str = "gatv2",
    ):
        super().__init__()
        self.dropout = dropout

        # ---- Shared GNN encoder ----
        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            backbone=backbone,
        )
        h = self.encoder.out_channels

        # ---- Structural feature encoder ----
        self.struct_mlp = nn.Sequential(
            nn.Linear(_NUM_STRUCT_FEATURES, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

        # ---- Node feature combiner ----
        self.node_mlp = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
        )

        # ---- Final predictor ----
        self.predictor = nn.Sequential(
            nn.Linear(h * 2, h),  # node_feat + struct_feat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

        # Cached sparse adjacency
        self._adj: sp.csr_matrix | None = None
        self._degrees: np.ndarray | None = None
        self._num_nodes: int = 0
        self._cache_key: tuple | None = None

    # --------------------------------------------------------------------- #
    # Sparse adjacency caching & vectorised structural features
    # --------------------------------------------------------------------- #
    def _maybe_rebuild_adj(self, edge_index: torch.Tensor, num_nodes: int):
        """Build scipy CSR adjacency matrix; skip if edge_index unchanged."""
        key = (edge_index.data_ptr(), edge_index.shape[1])
        if key == self._cache_key:
            return
        ei = edge_index.cpu().numpy()
        rows = np.concatenate([ei[0], ei[1]])
        cols = np.concatenate([ei[1], ei[0]])
        data = np.ones(len(rows), dtype=np.float32)
        self._adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        self._adj.data[:] = 1.0  # binarize
        self._degrees = np.asarray(self._adj.sum(axis=1)).flatten()
        self._num_nodes = num_nodes
        self._cache_key = key

    def _compute_structural_features(self, pred_edges: torch.Tensor) -> torch.Tensor:
        """Compute per-edge structural features via sparse matrix ops.

        Returns tensor of shape (num_edges, 6):
        [cn_count, adamic_adar, resource_allocation, deg_src, deg_dst, pref_attach]
        """
        src = pred_edges[0].cpu().numpy()
        dst = pred_edges[1].cpu().numpy()
        A = self._adj
        deg = self._degrees

        cn_matrix = A[src].multiply(A[dst])

        # 1. Common-neighbor count
        cn_count = np.asarray(cn_matrix.sum(axis=1)).flatten()

        # 2. Adamic-Adar
        log_deg = np.log(deg + 1.0)
        aa_w = np.where(deg > 1, 1.0 / log_deg, 0.0)
        aa = np.asarray(cn_matrix.dot(aa_w.reshape(-1, 1))).flatten()

        # 3. Resource Allocation
        ra_w = 1.0 / (deg + 1.0)
        ra = np.asarray(cn_matrix.dot(ra_w.reshape(-1, 1))).flatten()

        # 4 & 5. Degrees of src and dst
        deg_src = deg[src]
        deg_dst = deg[dst]

        # 6. Preferential Attachment
        pa = deg_src * deg_dst

        feats = np.stack([cn_count, aa, ra, deg_src, deg_dst, pa], axis=1).astype(np.float32)
        return torch.from_numpy(feats).to(pred_edges.device)

    # --------------------------------------------------------------------- #
    # encode / decode
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
        n = num_nodes or x.size(0)
        z = self.encoder(x, edge_index, num_nodes=num_nodes)
        self._maybe_rebuild_adj(edge_index, n)
        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = z[edge_index[0]], z[edge_index[1]]
        struct_feats = self._compute_structural_features(edge_index)
        struct_feat = self.struct_mlp(struct_feats)
        node_feat = self.node_mlp(torch.cat([src, dst], dim=-1))
        combined = torch.cat([node_feat, struct_feat], dim=-1)
        return self.predictor(combined).squeeze(-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        prediction_edge_index: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index, num_nodes=num_nodes)
        if prediction_edge_index is not None:
            return self.decode(z, prediction_edge_index)
        return z
