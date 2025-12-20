#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm, JumpingKnowledge
from torch_geometric.utils import dropout_adj, add_self_loops


class EdgePredictor(nn.Module):
    def __init__(self, dim, mode="bilinear", mlp_hidden=128, temperature=1.0, dropout=0.2):
        super().__init__()
        self.mode, self.temperature = mode, temperature
        if mode == "bilinear":
            self.bilinear = nn.Bilinear(dim, dim, 1, bias=True)
        elif mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(dim * 2, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )
        elif mode == "linear-cat":
            self.lin = nn.Linear(dim * 2, 1)

    def forward(self, z, edge_index):
        src, dst = z[edge_index[0]], z[edge_index[1]]
        if self.mode == "dot":
            return (src * dst).sum(-1) / self.temperature
        if self.mode == "cosine":
            return F.cosine_similarity(src, dst, dim=-1) / self.temperature
        if self.mode == "bilinear":
            return self.bilinear(src, dst).squeeze(-1)
        if self.mode == "mlp":
            return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)
        if self.mode == "linear-cat":
            return self.lin(torch.cat([src, dst], dim=-1)).squeeze(-1)
        raise ValueError(self.mode)


class GNNLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        num_layers=3,
        heads=4,
        dropout=0.2,
        backbone="gatv2",          # "gatv2" | "gcn"
        jk_mode="cat",             # "cat" | "max" | "last"
        p_edge_dropout=0.0,
        add_selfloop=True,
        decoder="bilinear",        # "dot" | "cosine" | "bilinear" | "mlp" | "linear-cat"
    ):
        super().__init__()
        assert num_layers >= 1
        self.drop_edge = p_edge_dropout
        self.add_selfloop = add_selfloop
        self.jk_mode = jk_mode
        self.feature_dropout = nn.Dropout(dropout)

        self.convs, self.norms, self.acts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        dims = []

        if backbone == "gatv2":
            out_each = hidden_channels * heads
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
            self.norms.append(GraphNorm(out_each)); self.acts.append(nn.PReLU()); dims.append(out_each)
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(out_each, hidden_channels, heads=heads, dropout=dropout, concat=True))
                self.norms.append(GraphNorm(out_each)); self.acts.append(nn.PReLU()); dims.append(out_each)
            if num_layers > 1:
                self.convs.append(GATv2Conv(out_each, hidden_channels, heads=1, dropout=dropout, concat=True))
                self.norms.append(GraphNorm(hidden_channels)); self.acts.append(nn.PReLU()); dims.append(hidden_channels)
        elif backbone == "gcn":
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))
            self.norms.append(GraphNorm(hidden_channels)); self.acts.append(nn.PReLU()); dims.append(hidden_channels)
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
                self.norms.append(GraphNorm(hidden_channels)); self.acts.append(nn.PReLU()); dims.append(hidden_channels)
        else:
            raise ValueError(backbone)

        if jk_mode in ("cat", "max"):
            self.jk = JumpingKnowledge(mode=jk_mode)
        else:
            self.jk = None

        if jk_mode == "cat":
            self.proj = nn.Linear(sum(dims), hidden_channels)
            final_dim = hidden_channels
        elif jk_mode == "max":
            self.proj = nn.Identity()
            final_dim = max(dims)
        else:  # "last"
            self.proj = nn.Identity()
            final_dim = dims[-1]

        self.edge_predictor = EdgePredictor(final_dim, mode=decoder, dropout=dropout)

    @torch.no_grad()
    def _with_selfloops(self, edge_index, num_nodes):
        if not self.add_selfloop:
            return edge_index
        out, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return out

    def encode(self, x, edge_index, num_nodes=None):
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
            if h_in.shape[-1] == h.shape[-1]:
                h = h + h_in
            xs.append(h)

        if self.jk is not None:
            h = self.jk(xs)
        else:
            h = xs[-1]
        return self.proj(h)

    def decode(self, z, edge_index):
        return self.edge_predictor(z, edge_index)

    def forward(self, x, edge_index, prediction_edge_index=None, num_nodes=None):
        z = self.encode(x, edge_index, num_nodes=num_nodes)
        return self.decode(z, prediction_edge_index) if prediction_edge_index is not None else z
