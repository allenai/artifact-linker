#!/usr/bin/env python3
"""
BUDDY: Subgraph sketching for scalable link prediction.

Uses MinHash-like sketches to summarise node neighborhoods, providing a
scalable approximation of subgraph-based structural features.

Reference:
    Chamberlain et al. "Graph Neural Networks for Link Prediction with
    Subgraph Sketching" (ICLR 2023)

Interface:
    encode(x, edge_index, num_nodes=None) → z   (also caches adjacency & sketches)
    decode(z, edge_index) → logits               (uses cached sketches)

Performance:
    Adjacency is stored as a scipy CSR sparse matrix with caching across epochs.
    Per-node sketches are pre-computed once during ``encode`` using vectorised
    numpy operations. Edge-level sketch lookup in ``decode`` uses numpy fancy
    indexing (no Python per-edge loop).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm


class BUDDYLinkPredictor(nn.Module):
    """
    BUDDY link predictor.

    Args:
        in_channels:      Input feature dimension.
        hidden_channels:  Hidden layer dimension.
        num_layers:       Number of GCN encoder layers.
        dropout:          Dropout rate.
        sketch_dim:       Dimensionality of each MinHash sketch vector.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sketch_dim: int = 32,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.sketch_dim = sketch_dim

        # ---- GCN encoder ----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        self.norms.append(GraphNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))
            self.norms.append(GraphNorm(hidden_channels))

        # ---- Sketch encoder ----
        self.sketch_encoder = nn.Sequential(
            nn.Linear(sketch_dim * 2, hidden_channels),  # src_sketch + dst_sketch
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ---- Node feature encoder ----
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
        )

        # ---- Final predictor ----
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # node_feat + sketch_feat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Cached adjacency & per-node sketches
        self._node_sketches: np.ndarray | None = None  # (num_nodes, sketch_dim)
        self._num_nodes: int = 0
        self._cache_key: tuple | None = None

        # Deterministic hash parameters (sketch_dim independent hash functions)
        rng = np.random.RandomState(42)
        self._hash_a = rng.randint(1, 10_000, size=sketch_dim).astype(np.int64)
        self._hash_b = rng.randint(0, 10_000, size=sketch_dim).astype(np.int64)
        self._hash_p = np.int64(10_007)

    # --------------------------------------------------------------------- #
    # Adjacency & sketch caching
    # --------------------------------------------------------------------- #
    def _maybe_rebuild_sketches(self, edge_index: torch.Tensor, num_nodes: int):
        """Build per-node MinHash sketches; skip if edge_index unchanged."""
        key = (edge_index.data_ptr(), edge_index.shape[1])
        if key == self._cache_key:
            return
        ei = edge_index.cpu().numpy()
        rows = np.concatenate([ei[0], ei[1]])
        cols = np.concatenate([ei[1], ei[0]])
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

        sd = self.sketch_dim
        sketches = np.zeros((num_nodes, sd), dtype=np.float32)

        # For each node, compute MinHash over its neighbors
        # Use CSR structure for efficient neighbor access
        for node in range(num_nodes):
            start, end = A.indptr[node], A.indptr[node + 1]
            if start == end:
                continue
            neighbors = A.indices[start:end].astype(np.int64)
            # Vectorised: (sketch_dim, num_neighbors) hash values
            hashes = (self._hash_a[:, None] * neighbors[None, :] + self._hash_b[:, None]) % self._hash_p
            sketches[node] = hashes.min(axis=1) / float(self._hash_p)

        self._node_sketches = sketches
        self._num_nodes = num_nodes
        self._cache_key = key

    def _get_edge_sketches(self, pred_edges: torch.Tensor) -> torch.Tensor:
        """Concatenate src and dst sketches via numpy fancy indexing."""
        assert self._node_sketches is not None
        src = pred_edges[0].cpu().numpy()
        dst = pred_edges[1].cpu().numpy()
        # Fancy indexing: (batch, sketch_dim) each, then hstack → (batch, 2*sketch_dim)
        src_sk = self._node_sketches[src]
        dst_sk = self._node_sketches[dst]
        combined = np.concatenate([src_sk, dst_sk], axis=1)
        return torch.from_numpy(combined).to(pred_edges.device)

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
        self._maybe_rebuild_sketches(edge_index, n)
        return h

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = z[edge_index[0]], z[edge_index[1]]
        sketches = self._get_edge_sketches(edge_index)
        sketch_feat = self.sketch_encoder(sketches)
        node_feat = self.node_encoder(torch.cat([src, dst], dim=-1))
        combined = torch.cat([node_feat, sketch_feat], dim=-1)
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
