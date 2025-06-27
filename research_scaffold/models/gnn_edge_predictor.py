import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.nn import GeneralConv
from .utils import build_bipartite_graph, nx_to_pyg_data
from ..sampler import EdgeBatchSampler


class FeatAlign(nn.Module):
    def __init__(self, model_feat_dim, common_dim):
        super().__init__()
        self.model_transform = nn.Linear(model_feat_dim, common_dim)
        self.dataset_transform = nn.Linear(model_feat_dim, common_dim)

    def forward(self, dataset_feats, model_feats):
        aligned_dataset_feats = self.dataset_transform(dataset_feats)
        aligned_model_feats = self.model_transform(model_feats)
        return torch.cat([aligned_dataset_feats, aligned_model_feats], 0)

class GNNEdgePredictor(nn.Module):
    def __init__(self, model_feat_dim, hidden_feats, in_edges):
        super().__init__()
        self.in_edges = in_edges
        self.model_align = FeatAlign(model_feat_dim, hidden_feats)
        self.encoder_conv_1 = GeneralConv(hidden_feats, hidden_feats, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(hidden_feats, hidden_feats, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.bn2 = nn.BatchNorm1d(hidden_feats)

    def forward(self, dataset_feats, model_feats, edge_index, edge_mask, edge_can_see, edge_weight):
        edge_index_mask = edge_index[:, edge_can_see]
        edge_index_predict = edge_index[:, edge_mask]
        breakpoint()
        edge_weight_mask = F.relu(self.edge_mlp(edge_weight[edge_can_see]))
        x_ini = self.model_align(dataset_feats, model_feats)
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        # Regression output: no sigmoid
        edge_predict = (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1)
        return edge_predict

class GNNTrainer:
    def __init__(self, model, optimizer, loss_fn, config, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

    def train(self, data, sampler):
        self.model.train()
        total_loss = 0
        for batch_mask, edge_can_see in sampler:
            preds = self.model(
                data.dataset_feats.to(self.device),
                data.model_feats.to(self.device),
                data.edge_index.to(self.device),
                edge_mask=batch_mask.to(self.device),
                edge_can_see=edge_can_see.to(self.device),
                edge_weight=data.edge_label.unsqueeze(-1).to(self.device)
            )
            labels = data.edge_label[batch_mask].to(self.device)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / sampler.batch_size

    def evaluate(self, data, mask):
        self.model.eval()
        edge_can_see = ~mask
        with torch.no_grad():
            preds = self.model(
                data.dataset_feats.to(self.device),
                data.model_feats.to(self.device),
                data.edge_index.to(self.device),
                edge_mask=mask.to(self.device),
                edge_can_see=edge_can_see.to(self.device),
                edge_weight=data.edge_label.unsqueeze(-1).to(self.device)
            )
            labels = data.edge_label[mask].to(self.device)
            mse = self.loss_fn(preds, labels).item()
        return mse

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build graph and convert to PyG
    G = build_bipartite_graph(
        data_dir="../data/eval_datasets_json_download_ranks",
        dataset_json="../data/dataset_info.json",
        metadata_dir="../data/model_metadata_download_ranks"
    )
    data = nx_to_pyg_data(G)

    # Feature dimensions
    model_dim = 64
    # Initialize features by node type
    num_dataset = data.is_dataset.sum().item()
    num_model = data.is_model.sum().item()
    data.dataset_feats = torch.randn((num_dataset, model_dim))
    data.model_feats = torch.randn((num_model, model_dim))

    num_edges = data.edge_index.size(1)
    # Train/val split on existing edges
    train_mask = torch.rand(num_edges) < 0.8
    data.train_mask = train_mask
    data.val_mask = ~train_mask

    # Setup model, optimizer, and MSE loss
    model = GNNEdgePredictor(model_feat_dim=model_dim, hidden_feats=64, in_edges=1).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    config = {'train_epoch': 50, 'batch_size': 4, 'train_mask_rate': 0.2}

    trainer = GNNTrainer(model, optimizer, loss_fn, config, device)
    sampler = EdgeBatchSampler(edge_mask=data.train_mask, batch_size=config['batch_size'], mask_rate=config['train_mask_rate'])

    # Training loop
    for epoch in range(config['train_epoch']):
        train_loss = trainer.train(data, sampler)
        val_mse = trainer.evaluate(data, data.val_mask)
        print(f"[Epoch {epoch:03d}] Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f}")

if __name__ == "__main__":
    main()
