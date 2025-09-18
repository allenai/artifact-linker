#!/usr/bin/env python3

import json

import torch
import torch.nn as nn
from torch.optim import AdamW

from artifact_graph.models.gnn_link_predictor import GNNLinkPredictor
from artifact_graph.utils.graph_builder import (
    load_artifact_graph_from_json,
    load_pyg_graph_from_networkx,
)


class GNNTrainer:
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
            preds = self.model.predict_accuracy(node_embeddings, batch_edges.to(self.device))

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
            preds = self.model.predict_accuracy(node_embeddings, eval_edges.to(self.device))
            mse = self.loss_fn(preds, eval_labels.to(self.device)).item()

        return mse

    def predict(self, data, prediction_edges):
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            predicted_accuracies = self.model.predict_accuracy(
                node_embeddings, prediction_edges.to(self.device)
            )
        return predicted_accuracies.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_name = "accuracy"  # Define the metric to predict and use for loading

    # 1. Build graph using load_artifact_graph_from_json function
    graph_file = "output/perfect_model_dataset_metrics.json"
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 2. Convert to PyG data object
    data = load_pyg_graph_from_networkx(G)

    # 3. Create initial node features (e.g., based on node type)
    # Using 'x' from data object which is one-hot encoded
    feature_dim = data.x.size(1)

    # 4. Prepare edges and labels
    # Since we load from the JSON, all edges in G should have the metric.
    all_metrics = [d.get(metric_name) for u, v, d in G.edges(data=True)]

    # Create a mask for edges that have a valid metric value (not None)
    valid_mask = torch.tensor([m is not None for m in all_metrics])
    known_edges = data.edge_index[:, valid_mask]

    # For valid edges, get their labels and normalize them if they are > 1 (e.g., percentages)
    known_labels = torch.tensor(
        [m / 100.0 if m > 1.0 else m for m in all_metrics if m is not None],
        dtype=torch.float,
    )

    # The original `prediction_edges` (for edges without metrics) is now empty.
    # It will be redefined later from the split of known_edges.

    # Setup model, optimizer, and loss
    model = GNNLinkPredictor(in_feats=feature_dim, hidden_feats=256).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    trainer = GNNTrainer(model, optimizer, loss_fn, device)

    # 5. Training loop
    print("Starting training...")
    if known_edges.size(1) > 2:  # Need at least 3 edges for a train/val/predict split
        perm = torch.randperm(known_edges.size(1))

        # Split data: 70% train, 15% validation, 15% prediction (for LLM eval)
        train_size = int(0.7 * perm.numel())
        val_size = int(0.15 * perm.numel())

        train_edges = known_edges[:, perm[:train_size]]
        val_edges = known_edges[:, perm[train_size : train_size + val_size]]
        prediction_edges = known_edges[:, perm[train_size + val_size :]]

        train_labels = known_labels[perm[:train_size]]
        val_labels = known_labels[perm[train_size : train_size + val_size]]
        prediction_labels = known_labels[perm[train_size + val_size :]]

        # Print the dev split (validation set)
        print("\n--- Validation (Dev) Set Links ---")
        node_names = data.node_names_ordered
        for i in range(val_edges.size(1)):
            src_idx = val_edges[0, i].item()
            dst_idx = val_edges[1, i].item()
            src_name = node_names[src_idx]
            dst_name = node_names[dst_idx]
            model_name = src_name if data.node_type[src_idx] == 0 else dst_name
            dataset_name = dst_name if data.node_type[src_idx] == 0 else src_name
            print(f"  - Model: {model_name}, Dataset: {dataset_name}")
        print("------------------------------------\n")

        # Save the prediction set for the LLM predictor
        validation_data_for_llm = []
        for i in range(prediction_edges.size(1)):
            src_idx = prediction_edges[0, i].item()
            dst_idx = prediction_edges[1, i].item()
            src_name = node_names[src_idx]
            dst_name = node_names[dst_idx]
            model_name = src_name if data.node_type[src_idx] == 0 else dst_name
            dataset_name = dst_name if data.node_type[src_idx] == 0 else src_name
            validation_data_for_llm.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "accuracy": prediction_labels[i].item(),
                }
            )

        with open("output/gnn_validation_set.json", "w") as f:
            json.dump(validation_data_for_llm, f, indent=2)
        print(
            f"Saved {len(validation_data_for_llm)} edges for LLM validation to output/gnn_validation_set.json"
        )

        for epoch in range(50):
            train_loss = trainer.train(data, train_edges, train_labels, batch_size=64)
            val_mse = trainer.evaluate(data, val_edges, val_labels)
            print(f"[Epoch {epoch:03d}] Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f}")
    else:
        print("Not enough labeled data for train/val/predict split. Skipping.")
        prediction_edges = torch.empty((2, 0), dtype=torch.long)

    # 6. Predict accuracies for the held-out prediction set
    if prediction_edges.size(1) > 0:
        print("\nEvaluating GNN on the held-out prediction set...")
        predicted_accuracies = trainer.predict(data, prediction_edges)

        # 7. Evaluate and print prediction results
        # Cast to float64 for a more precise MSE calculation, matching the evaluation script
        prediction_mse = nn.functional.mse_loss(
            torch.tensor(predicted_accuracies, dtype=torch.float64),
            prediction_labels.to(torch.float64),
        ).item()
        print(f"\nGNN Prediction set MSE: {prediction_mse:.4f}")

        print("\n--- GNN Prediction Results ---")
        node_names = data.node_names_ordered
        results_to_save = []
        for i in range(prediction_edges.size(1)):
            src_idx = prediction_edges[0, i].item()
            dst_idx = prediction_edges[1, i].item()
            src_name = node_names[src_idx]
            dst_name = node_names[dst_idx]
            model_name = src_name if data.node_type[src_idx] == 0 else dst_name
            dataset_name = dst_name if data.node_type[src_idx] == 0 else src_name
            pred_acc = predicted_accuracies[i]
            true_acc = prediction_labels[i].item()
            print(
                f"  - Model: {model_name}, Dataset: {dataset_name} -> Predicted: {pred_acc:.4f}, Actual: {true_acc:.4f}"
            )
            results_to_save.append(
                {
                    "model_id": model_name,
                    "dataset_id": dataset_name,
                    "metric_name": metric_name,
                    "true_metric": true_acc,
                    "predicted_metric": float(pred_acc),
                    "reason": "Predicted by GNN",
                    "status": "Success",
                }
            )
        print("------------------------------")

        # 8. Save results to a file
        output_file = "output/gnn_predictions.json"
        with open(output_file, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nPredictions saved to {output_file}")

    else:
        print("\nNo edges in the prediction set.")


if __name__ == "__main__":
    main()
