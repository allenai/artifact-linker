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

Performance:
    Adjacency is stored as a scipy CSR sparse matrix with caching across epochs.
    All six structural features (CN count, AA, RA, deg_src, deg_dst, PA) are
    computed via vectorised sparse-matrix operations — no Python per-edge loops.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm

_NUM_STRUCT_FEATURES = 6  # cn_count, aa, ra, deg_src, deg_dst, pa


class NeoGNNLinkPredictor(nn.Module):
    """
    Neo-GNN link predictor.

    Args:
        in_channels:      Input feature dimension.
        hidden_channels:  Hidden layer dimension.
        num_layers:       Number of GCN encoder layers.
        dropout:          Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # ---- GCN encoder ----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        self.norms.append(GraphNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))
            self.norms.append(GraphNorm(hidden_channels))

        # ---- Structural feature encoder ----
        self.struct_mlp = nn.Sequential(
            nn.Linear(_NUM_STRUCT_FEATURES, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ---- Node feature combiner ----
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
        )

        # ---- Final predictor ----
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # node_feat + struct_feat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
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

        # CN indicator matrix: (batch, num_nodes)
        cn_matrix = A[src].multiply(A[dst])

        # 1. Common-neighbor count
        cn_count = np.asarray(cn_matrix.sum(axis=1)).flatten()

        # 2. Adamic-Adar: Σ_{w ∈ CN} 1/log(deg(w)+1)  (only w with deg > 1)
        log_deg = np.log(deg + 1.0)
        aa_w = np.where(deg > 1, 1.0 / log_deg, 0.0)
        aa = np.asarray(cn_matrix.dot(aa_w.reshape(-1, 1))).flatten()

        # 3. Resource Allocation: Σ_{w ∈ CN} 1/(deg(w)+1)
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
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        self._maybe_rebuild_adj(edge_index, n)
        return h

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
