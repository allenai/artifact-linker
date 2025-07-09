import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, num_layers=2, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        # first layer: project in_channels → hidden_channels via multiple heads
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        )
        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
            )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin = nn.Linear(hidden_channels * 2, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=1)
        return torch.sigmoid(self.lin(h)).view(-1)


class LinkPredictionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, num_layers=2, dropout=0.6):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, heads, num_layers, dropout)
        self.predictor = LinkPredictor(hidden_channels)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        # encode node features
        z = self.encoder(x, edge_index)

        # score positive and negative edges
        pos_scores = self.predictor(z, pos_edge_index)
        neg_scores = self.predictor(z, neg_edge_index)
        return pos_scores, neg_scores