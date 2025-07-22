import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv

class GNNLinkPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats=1):
        super().__init__()
        self.encoder_conv_1 = GeneralConv(in_feats, hidden_feats)
        self.encoder_conv_2 = GeneralConv(hidden_feats, hidden_feats)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        
        # MLP for edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, x, edge_index):
        # Encode nodes
        x = F.relu(self.bn1(self.encoder_conv_1(x, edge_index)))
        x = self.bn2(self.encoder_conv_2(x, edge_index))
        
        return x

    def predict_accuracy(self, x, edge_index_to_predict):
        # Get embeddings for source and destination nodes
        src_nodes = x[edge_index_to_predict[0]]
        dst_nodes = x[edge_index_to_predict[1]]
        
        # Concatenate node embeddings
        edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
        
        # Predict accuracy using MLP
        predicted_accuracy = self.edge_predictor(edge_features).squeeze(-1)
        return predicted_accuracy
