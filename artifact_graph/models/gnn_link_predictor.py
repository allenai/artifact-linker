#!/usr/bin/env python3
"""
GATv2 / GCN link predictor using shared GNNEncoder backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import GNNEncoder


class EdgePredictor(nn.Module):
    def __init__(self, dim, mode="bilinear", mlp_hidden=128, temperature=1.0, dropout=0.2, extra_dim=0):
        super().__init__()
        self.mode, self.temperature = mode, temperature
        if mode == "bilinear":
            self.bilinear = nn.Bilinear(dim, dim, 1, bias=True)
        elif mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(dim * 2 + extra_dim, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )
        elif mode == "linear-cat":
            self.lin = nn.Linear(dim * 2, 1)

    def forward(self, z, edge_index, extra_features=None):
        src, dst = z[edge_index[0]], z[edge_index[1]]
        if self.mode == "dot":
            return (src * dst).sum(-1) / self.temperature
        if self.mode == "cosine":
            return F.cosine_similarity(src, dst, dim=-1) / self.temperature
        if self.mode == "bilinear":
            return self.bilinear(src, dst).squeeze(-1)
        if self.mode == "mlp":
            cat = torch.cat([src, dst], dim=-1)
            if extra_features is not None:
                cat = torch.cat([cat, extra_features], dim=-1)
            return self.mlp(cat).squeeze(-1)
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
        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            backbone=backbone,
            jk_mode=jk_mode,
            p_edge_dropout=p_edge_dropout,
            add_selfloop=add_selfloop,
        )
        self.edge_predictor = EdgePredictor(
            self.encoder.out_channels, mode=decoder, dropout=dropout,
        )

    def encode(self, x, edge_index, num_nodes=None):
        return self.encoder(x, edge_index, num_nodes=num_nodes)

    def decode(self, z, edge_index):
        return self.edge_predictor(z, edge_index)

    def forward(self, x, edge_index, prediction_edge_index=None, num_nodes=None):
        z = self.encode(x, edge_index, num_nodes=num_nodes)
        return self.decode(z, prediction_edge_index) if prediction_edge_index is not None else z
