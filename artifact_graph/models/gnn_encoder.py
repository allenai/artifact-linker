#!/usr/bin/env python3
"""
Shared GNN encoder backbone used by all link-prediction models.

Supports GATv2 and GCN backbones with optional JumpingKnowledge aggregation,
residual connections, edge dropout, and self-loops.  Every model (GATv2, GCN,
NCN, NCNC, NeoGNN, BUDDY) should instantiate this encoder so that parameter
counts are aligned and the comparison is fair.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm, JumpingKnowledge
from torch_geometric.utils import dropout_adj, add_self_loops


class GNNEncoder(nn.Module):
    """
    Configurable GNN encoder backbone.

    After ``forward``, produces a node embedding matrix of shape
    ``(num_nodes, out_channels)`` where ``out_channels`` equals
    ``hidden_channels`` regardless of the backbone or JK mode.

    Args:
        in_channels:     Input feature dimension.
        hidden_channels: Hidden / output dimension (final embedding dim).
        num_layers:      Number of message-passing layers (>= 1).
        heads:           Number of attention heads (GATv2 only).
        dropout:         Feature dropout rate.
        backbone:        ``"gatv2"`` or ``"gcn"``.
        jk_mode:         JumpingKnowledge mode: ``"cat"`` | ``"max"`` | ``"last"``.
        p_edge_dropout:  Probability of dropping edges during training.
        add_selfloop:    Whether to add self-loops to edge_index.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        backbone: str = "gatv2",
        jk_mode: str = "cat",
        p_edge_dropout: float = 0.0,
        add_selfloop: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        self.drop_edge = p_edge_dropout
        self.add_selfloop = add_selfloop
        self.jk_mode = jk_mode
        self.feature_dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        dims: list[int] = []

        if backbone == "gatv2":
            out_each = hidden_channels * heads
            self.convs.append(
                GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(GraphNorm(out_each))
            self.acts.append(nn.PReLU())
            dims.append(out_each)

            for _ in range(num_layers - 2):
                self.convs.append(
                    GATv2Conv(out_each, hidden_channels, heads=heads, dropout=dropout, concat=True)
                )
                self.norms.append(GraphNorm(out_each))
                self.acts.append(nn.PReLU())
                dims.append(out_each)

            if num_layers > 1:
                self.convs.append(
                    GATv2Conv(out_each, hidden_channels, heads=1, dropout=dropout, concat=True)
                )
                self.norms.append(GraphNorm(hidden_channels))
                self.acts.append(nn.PReLU())
                dims.append(hidden_channels)

        elif backbone == "gcn":
            # Use hidden_channels * heads as effective width to match GATv2 param count
            gcn_dim = hidden_channels * heads
            self.convs.append(GCNConv(in_channels, gcn_dim, cached=False, normalize=True))
            self.norms.append(GraphNorm(gcn_dim))
            self.acts.append(nn.PReLU())
            dims.append(gcn_dim)

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(gcn_dim, gcn_dim, cached=False, normalize=True))
                self.norms.append(GraphNorm(gcn_dim))
                self.acts.append(nn.PReLU())
                dims.append(gcn_dim)

            if num_layers > 1:
                self.convs.append(GCNConv(gcn_dim, hidden_channels, cached=False, normalize=True))
                self.norms.append(GraphNorm(hidden_channels))
                self.acts.append(nn.PReLU())
                dims.append(hidden_channels)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # JumpingKnowledge
        if jk_mode in ("cat", "max"):
            self.jk = JumpingKnowledge(mode=jk_mode)
        else:
            self.jk = None

        if jk_mode == "cat":
            self.proj = nn.Linear(sum(dims), hidden_channels)
            self._out_channels = hidden_channels
        elif jk_mode == "max":
            self.proj = nn.Identity()
            self._out_channels = max(dims)
        else:  # "last"
            self.proj = nn.Identity()
            self._out_channels = dims[-1]

    @property
    def out_channels(self) -> int:
        """Output embedding dimension."""
        return self._out_channels

    @torch.no_grad()
    def _with_selfloops(self, edge_index, num_nodes):
        if not self.add_selfloop:
            return edge_index
        out, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
        if self.training and self.drop_edge > 0:
            edge_index, _ = dropout_adj(edge_index, p=self.drop_edge, force_undirected=False, training=True)
        if num_nodes is not None:
            edge_index = self._with_selfloops(edge_index, num_nodes)

        xs, h = [], x
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = act(h)
            h = self.feature_dropout(h)
            # Residual connection when dimensions match
            if h_in.shape[-1] == h.shape[-1]:
                h = h + h_in
            xs.append(h)

        if self.jk is not None:
            h = self.jk(xs)
        else:
            h = xs[-1]
        return self.proj(h)
