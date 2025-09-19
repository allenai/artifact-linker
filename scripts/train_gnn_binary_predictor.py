#!/usr/bin/env python3

import json
import random

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW

from artifact_graph.models.gnn_link_predictor import GNNBinaryLinkPredictor
from artifact_graph.utils.graph_builder import (
    load_artifact_graph_from_json,
    load_pyg_graph_from_networkx,
)


class GNNBinaryTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, data, train_edges, train_labels, batch_size):
        self.model.train()
        total_loss = 0

        if train_edges.size(1) == 0:
            return 0.0

        for i in range(0, train_edges.size(1), batch_size):
            # Get batch
            batch_edges = train_edges[:, i : i + batch_size]
            batch_labels = train_labels[i : i + batch_size]

            # Forward pass
            node_embeddings = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            preds = self.model.predict_link_probability(
                node_embeddings, batch_edges.to(self.device)
            )

            # Compute loss
            loss = self.loss_fn(preds, batch_labels.to(self.device))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / (train_edges.size(1) / batch_size)

    def evaluate(self, data, eval_edges, eval_labels):
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            preds = self.model.predict_link_probability(node_embeddings, eval_edges.to(self.device))
            loss = self.loss_fn(preds, eval_labels.to(self.device)).item()

        return loss

    def predict(self, data, prediction_edges):
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            predicted_probs = self.model.predict_link_probability(
                node_embeddings, prediction_edges.to(self.device)
            )
        return predicted_probs.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_name = "accuracy"  # Define the metric to use for loading

    # 1. Load graph using load_artifact_graph_from_json function
    graph_file = "output/perfect_model_dataset_metrics.json"
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 2. Convert to PyG data object
    data = load_pyg_graph_from_networkx(G)

    # 3. Create initial node features
    feature_dim = data.x.size(1)

    # 4. Prepare positive edges (existing connections)
    positive_edges = data.edge_index
    print(f"Found {positive_edges.size(1)} positive edges from the graph.")

    # 5. Generate negative edges (non-existing connections)
    all_models = [i for i, node_type in enumerate(data.node_type) if node_type == 0]
    all_datasets = [i for i, node_type in enumerate(data.node_type) if node_type == 1]
    existing_edges_set = set(map(tuple, positive_edges.t().tolist()))

    negative_edges = []
    num_negative_samples = positive_edges.size(1)  # 1:1 ratio

    print(f"Generating {num_negative_samples} negative samples...")
    while len(negative_edges) < num_negative_samples:
        model_idx = random.choice(all_models)
        dataset_idx = random.choice(all_datasets)
        if (model_idx, dataset_idx) not in existing_edges_set and (
            dataset_idx,
            model_idx,
        ) not in existing_edges_set:
            negative_edges.append([model_idx, dataset_idx])

    negative_edges = torch.tensor(negative_edges).t()
    print("Negative sampling complete.")

    # 6. Combine positive and negative edges
    all_edges = torch.cat([positive_edges, negative_edges], dim=1)
    all_labels = torch.cat(
        [
            torch.ones(positive_edges.size(1)),  # Positive labels
            torch.zeros(negative_edges.size(1)),  # Negative labels
        ]
    )

    # 7. Shuffle the data
    perm = torch.randperm(all_edges.size(1))
    all_edges = all_edges[:, perm]
    all_labels = all_labels[perm]

    print(f"Total edges for training: {all_edges.size(1)} (50% positive, 50% negative)")

    # 8. Setup model, optimizer, and loss
    model = GNNBinaryLinkPredictor(in_feats=feature_dim, hidden_feats=256).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    trainer = GNNBinaryTrainer(model, optimizer, loss_fn, device)

    # 9. Training loop
    print("Starting training...")
    if all_edges.size(1) > 2:
        # Split data: 70% train, 15% validation, 15% test
        train_size = int(0.7 * all_edges.size(1))
        val_size = int(0.15 * all_edges.size(1))

        train_edges = all_edges[:, :train_size]
        val_edges = all_edges[:, train_size : train_size + val_size]
        test_edges = all_edges[:, train_size + val_size :]

        train_labels = all_labels[:train_size]
        val_labels = all_labels[train_size : train_size + val_size]
        test_labels = all_labels[train_size + val_size :]

        # Print test set info
        print("\n--- Test Set Info ---")
        print(f"Test edges: {test_edges.size(1)}")
        print(f"Positive test edges: {test_labels.sum().item()}")
        print(f"Negative test edges: {(test_labels == 0).sum().item()}")
        print("--------------------\n")

        for epoch in range(50):
            train_loss = trainer.train(data, train_edges, train_labels, batch_size=64)
            val_loss = trainer.evaluate(data, val_edges, val_labels)
            print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 10. Evaluate on test set
        print("\nEvaluating GNN Binary Predictor on test set...")
        predicted_probs = trainer.predict(data, test_edges)
        predicted_labels = (predicted_probs > 0.5).astype(int)
        true_labels = test_labels.numpy()

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        print("\n--- Binary Classification Metrics ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("---------------------------------------")

        # 11. Save results
        results_to_save = []
        node_names = data.node_names_ordered

        for i in range(test_edges.size(1)):
            src_idx = test_edges[0, i].item()
            dst_idx = test_edges[1, i].item()
            src_name = node_names[src_idx]
            dst_name = node_names[dst_idx]
            model_name = src_name if data.node_type[src_idx] == 0 else dst_name
            dataset_name = dst_name if data.node_type[src_idx] == 0 else src_name

            results_to_save.append(
                {
                    "model_id": model_name,
                    "dataset_id": dataset_name,
                    "true_label": int(true_labels[i]),
                    "predicted_label": int(predicted_labels[i]),
                    "predicted_probability": float(predicted_probs[i]),
                    "reason": "Predicted by GNN Binary Classifier",
                    "status": "Success",
                }
            )

        output_file = "output/gnn_binary_predictions.json"
        with open(output_file, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nPredictions saved to {output_file}")

    else:
        print("Not enough data for train/val/test split. Skipping.")


if __name__ == "__main__":
    main()
