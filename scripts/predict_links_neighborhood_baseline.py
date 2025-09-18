#!/usr/bin/env python3
"""
Neighborhood-based baseline predictor for binary link prediction.
Uses local neighborhood averaging to predict whether a model-dataset connection exists.
"""

import json
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from artifact_graph.utils.graph_builder import load_artifact_graph_from_json


class NeighborhoodBaselinePredictor:
    def __init__(self, G, metric_name="accuracy"):
        self.G = G
        self.metric_name = metric_name
        
    def get_model_neighborhood_features(self, model_name):
        """Get average performance of a model across all its connected datasets."""
        neighbors = list(self.G.neighbors(model_name))
        dataset_neighbors = [n for n in neighbors if self.G.nodes[n].get("type") == "dataset"]
        
        if not dataset_neighbors:
            return 0.0, 0  # No datasets connected
        
        total_performance = 0.0
        valid_connections = 0
        
        for dataset in dataset_neighbors:
            if self.metric_name in self.G[model_name][dataset]:
                total_performance += self.G[model_name][dataset][self.metric_name]
                valid_connections += 1
        
        if valid_connections == 0:
            return 0.0, 0
            
        return total_performance / valid_connections, valid_connections
    
    def get_dataset_neighborhood_features(self, dataset_name):
        """Get average performance of a dataset across all its connected models."""
        neighbors = list(self.G.neighbors(dataset_name))
        model_neighbors = [n for n in neighbors if self.G.nodes[n].get("type") == "model"]
        
        if not model_neighbors:
            return 0.0, 0  # No models connected
        
        total_performance = 0.0
        valid_connections = 0
        
        for model in model_neighbors:
            if self.metric_name in self.G[model][dataset_name]:
                total_performance += self.G[model][dataset_name][self.metric_name]
                valid_connections += 1
        
        if valid_connections == 0:
            return 0.0, 0
            
        return total_performance / valid_connections, valid_connections
    
    def predict_connection_probability(self, model_name, dataset_name):
        """
        Predict the probability of a connection existing between model and dataset.
        Uses neighborhood similarity and performance patterns.
        """
        # Get neighborhood features
        model_avg_perf, model_connections = self.get_model_neighborhood_features(model_name)
        dataset_avg_perf, dataset_connections = self.get_dataset_neighborhood_features(dataset_name)
        
        # If either has no connections, use a low probability
        if model_connections == 0 or dataset_connections == 0:
            return 0.1
        
        # Calculate similarity score based on performance patterns
        # Models and datasets with similar performance patterns are more likely to connect
        performance_similarity = 1.0 / (1.0 + abs(model_avg_perf - dataset_avg_perf))
        
        # Connection density factor (more connections = higher probability)
        model_density = min(model_connections / 10.0, 1.0)  # Normalize to 0-1
        dataset_density = min(dataset_connections / 10.0, 1.0)
        
        # Combine factors
        connection_prob = (performance_similarity * 0.6 + 
                          model_density * 0.2 + 
                          dataset_density * 0.2)
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, connection_prob))
    
    def predict_batch(self, edge_pairs):
        """Predict connection probabilities for a batch of model-dataset pairs."""
        predictions = []
        
        for model_name, dataset_name in tqdm(edge_pairs, desc="Predicting with Neighborhood Baseline"):
            try:
                prob = self.predict_connection_probability(model_name, dataset_name)
                predictions.append({
                    "prediction": prob > 0.5,  # Binary prediction
                    "probability": prob,
                    "reason": f"Neighborhood similarity: model_avg={self.get_model_neighborhood_features(model_name)[0]:.3f}, dataset_avg={self.get_dataset_neighborhood_features(dataset_name)[0]:.3f}"
                })
            except Exception as e:
                print(f"Error predicting for ({model_name}, {dataset_name}): {e}")
                predictions.append(None)
        
        return predictions


def main():
    metric_name = "accuracy"  # Define the metric to use for neighborhood context
    output_file = f"output/neighborhood_baseline_predictions.json"

    # 1. Load the graph from JSON file
    graph_file = "output/perfect_model_dataset_metrics.json"
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )

    # 2. Extract positive edges directly from the graph to ensure consistency
    all_models_in_graph = {n for n, d in G.nodes(data=True) if d["type"] == "model"}
    all_datasets_in_graph = {n for n, d in G.nodes(data=True) if d["type"] == "dataset"}
    
    positive_edges = set()
    for u, v in G.edges():
        if u in all_models_in_graph and v in all_datasets_in_graph:
            positive_edges.add((u, v))
        elif v in all_models_in_graph and u in all_datasets_in_graph:
            positive_edges.add((v, u)) # Ensure correct order
    positive_edges = list(positive_edges)

    print(f"Extracted {len(positive_edges)} positive edges from the filtered graph.")

    # 3. Perform negative sampling
    all_models = list(all_models_in_graph)
    all_datasets = list(all_datasets_in_graph)
    existing_edges_set = set(positive_edges)
    negative_edges = []
    num_negative_samples = len(positive_edges)  # 1:1 ratio

    print(f"Generating {num_negative_samples} negative samples...")
    while len(negative_edges) < num_negative_samples:
        model = random.choice(all_models)
        dataset = random.choice(all_datasets)
        if (model, dataset) not in existing_edges_set:
            negative_edges.append((model, dataset))
    print("Negative sampling complete.")

    # 4. Combine and prepare data for predictor
    edges_to_predict = positive_edges + negative_edges
    true_labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    # Shuffle the data
    combined = list(zip(edges_to_predict, true_labels))
    random.shuffle(combined)
    edges_to_predict, true_labels = zip(*combined) if combined else ([], [])

    print(f"Total pairs to predict: {len(edges_to_predict)}")

    # 5. Initialize predictor
    predictor = NeighborhoodBaselinePredictor(G, metric_name)

    # 6. Predict
    predicted_links = predictor.predict_batch(edges_to_predict)

    # 7. Evaluate and Print results
    results_to_save = []
    valid_predictions = []
    valid_true_labels = []

    for result, true_label in zip(predicted_links, true_labels):
        if result and result.get("prediction") is not None:
            valid_predictions.append(1 if result["prediction"] else 0)
            valid_true_labels.append(true_label)

    if not valid_predictions:
        print("Could not generate any valid predictions.")
    else:
        # Calculate metrics
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        precision = precision_score(valid_true_labels, valid_predictions)
        recall = recall_score(valid_true_labels, valid_predictions)
        f1 = f1_score(valid_true_labels, valid_predictions)

        print("\n--- Neighborhood Baseline Binary Classification Metrics ---")
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1 Score:  {f1:.4f}")
        print("--------------------------------------------------------")

        print("\n--- Neighborhood Baseline Prediction Results ---")
        for i in range(len(edges_to_predict)):
            model, dataset = edges_to_predict[i]
            true_label = true_labels[i]
            prediction_result = predicted_links[i]

            result_item = {
                "model_id": model,
                "dataset_id": dataset,
                "true_label": true_label,
                "predicted_label": None,
                "predicted_probability": None,
                "reason": "",
                "status": "Failed",
            }

            if prediction_result and prediction_result.get("prediction") is not None:
                pred_label = 1 if prediction_result["prediction"] else 0
                prob = prediction_result.get("probability", 0.0)
                reason = prediction_result.get("reason", "")
                print(
                    f"  - Model: {model}, Dataset: {dataset} -> Predicted: {pred_label} (prob: {prob:.3f}), Actual: {true_label}"
                )
                result_item["predicted_label"] = pred_label
                result_item["predicted_probability"] = prob
                result_item["reason"] = reason
                result_item["status"] = "Success"
            else:
                print(f"  - Model: {model}, Dataset: {dataset} -> Prediction Failed")

            results_to_save.append(result_item)
        print("------------------------------------------------")

    # 8. Save results to a file
    with open(output_file, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    main()




