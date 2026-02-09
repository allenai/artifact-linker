#!/usr/bin/env python3
"""
Neural Common Neighbor (NCN / NCNC) link predictor.

NCN aggregates common neighbor embeddings as extra features for edge prediction.
NCNC extends NCN with virtual neighbor completion for node pairs with few/no
common neighbors.

Reference:
    Wang et al. "Neural Common Neighbor with Completion for Link Prediction"
    (ICLR 2024)

Interface:
    encode(x, edge_index, num_nodes=None) → z   (also caches adjacency)
    decode(z, edge_index) → logits               (uses cached adjacency)

Performance:
    Adjacency is built as a scipy CSR sparse matrix once per unique edge_index
    (cached across epochs). Common-neighbor embeddings are computed via a single
    sparse-dense matmul instead of per-edge Python loops.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm


class NCNLinkPredictor(nn.Module):
    """
    NCN / NCNC link predictor.

    Args:
        in_channels:      Input feature dimension.
        hidden_channels:  Hidden layer dimension.
        num_layers:       Number of GCN encoder layers.
        dropout:          Dropout rate.
        use_completion:   If True, enable virtual-neighbor completion (NCNC).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_completion: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.use_completion = use_completion

        # ---- GCN encoder ----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        self.norms.append(GraphNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))
            self.norms.append(GraphNorm(hidden_channels))

        # ---- Common-neighbor aggregator ----
        self.cn_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ---- (NCNC) Virtual-neighbor completion ----
        if use_completion:
            self.completion = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            predictor_in = hidden_channels * 2 + hidden_channels * 2  # src+dst + cn+virtual_cn
        else:
            predictor_in = hidden_channels * 2 + hidden_channels  # src+dst + cn

        # ---- Final predictor ----
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Cached sparse adjacency (rebuilt only when edge_index changes)
        self._adj: sp.csr_matrix | None = None
        self._num_nodes: int = 0
        self._cache_key: tuple | None = None  # (data_ptr, shape) for cache invalidation

    # --------------------------------------------------------------------- #
    # Sparse adjacency caching
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
        # Deduplicate (handles repeated edges)
        self._adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        self._adj.data[:] = 1.0  # binarize
        self._num_nodes = num_nodes
        self._cache_key = key

    def _get_cn_embeddings(self, z: torch.Tensor, pred_edges: torch.Tensor) -> torch.Tensor:
        """Compute mean common-neighbor embeddings via sparse matmul.

        For each prediction edge (u, v), the common-neighbor indicator is
        ``A[u] ∘ A[v]`` (element-wise product of adjacency rows). Stacking
        these into a sparse matrix and multiplying by ``z`` yields the sum of
        CN embeddings in one shot.
        """
        src_np = pred_edges[0].cpu().numpy()
        dst_np = pred_edges[1].cpu().numpy()
        A = self._adj

        # CN indicator matrix: (batch, num_nodes), entry (i, w) = 1 iff w ∈ CN(src[i], dst[i])
        cn_matrix = A[src_np].multiply(A[dst_np])
        cn_counts = np.asarray(cn_matrix.sum(axis=1)).flatten()

        # Sparse → torch sparse for matmul with z
        cn_coo = cn_matrix.tocoo()
        if cn_coo.nnz > 0:
            indices = torch.from_numpy(
                np.vstack([cn_coo.row, cn_coo.col]).astype(np.int64)
            )
            values = torch.from_numpy(cn_coo.data.astype(np.float32))
            cn_sparse = torch.sparse_coo_tensor(
                indices, values, cn_matrix.shape
            ).to(z.device)
            cn_sum = torch.sparse.mm(cn_sparse, z)  # (batch, hidden)
        else:
            cn_sum = torch.zeros(len(src_np), z.size(1), device=z.device)

        # Mean aggregation (divide by CN count)
        cn_counts_t = torch.from_numpy(cn_counts.astype(np.float32)).to(z.device).unsqueeze(1)
        cn_mean = cn_sum / cn_counts_t.clamp(min=1)

        # For edges without common neighbors, use midpoint of src and dst
        no_cn_mask = cn_counts_t.squeeze(1) == 0
        if no_cn_mask.any():
            z_src = z[pred_edges[0]]
            z_dst = z[pred_edges[1]]
            cn_mean[no_cn_mask] = (z_src[no_cn_mask] + z_dst[no_cn_mask]) / 2

        return cn_mean

    # --------------------------------------------------------------------- #
    # encode / decode interface (compatible with GNNLinkTrainer)
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
        n = num_nodes or x.size(0)
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        # Cache adjacency for decode (skipped if edge_index unchanged)
        self._maybe_rebuild_adj(edge_index, n)
        return h

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = z[edge_index[0]], z[edge_index[1]]
        cn_z = self._get_cn_embeddings(z, edge_index)
        cn_feat = self.cn_mlp(cn_z)

        if self.use_completion:
            virtual_cn = self.completion(torch.cat([src, dst], dim=-1))
            virtual_cn_feat = self.cn_mlp(virtual_cn)
            combined = torch.cat([src, dst, cn_feat, virtual_cn_feat], dim=-1)
        else:
            combined = torch.cat([src, dst, cn_feat], dim=-1)

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
