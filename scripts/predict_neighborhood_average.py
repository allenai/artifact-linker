#!/usr/bin/env python3
"""
Predict accuracy using neighborhood averages from the artifact graph.
"""

import json
import numpy as np
from tqdm import tqdm
import networkx as nx
from typing import List, Tuple, Dict, Optional

from artifact_graph.utils.graph_builder import (
    load_artifact_graph,
    load_artifact_graph_from_json,
)


class NeighborhoodPredictor:
    """Predict metrics using neighborhood averages in the graph."""
    
    def __init__(self, graph: nx.Graph, metric_name: str = "accuracy"):
        self.graph = graph
        self.metric_name = metric_name
    
    def get_node_neighbors(self, node: str, node_type: str) -> List[str]:
        """Get neighbors of a specific type for a given node."""
        neighbors = []
        for neighbor in self.graph.neighbors(node):
            neighbor_data = self.graph.nodes[neighbor]
            if neighbor_data.get('type') == node_type:
                neighbors.append(neighbor)
        return neighbors
    
    def get_edge_metric(self, model: str, dataset: str) -> Optional[float]:
        """Get the metric value for a model-dataset edge."""
        if self.graph.has_edge(model, dataset):
            edge_data = self.graph.edges[model, dataset]
            score = edge_data.get(self.metric_name)
            if score is not None and score > 1:
                # Normalize percentage values (e.g., 85.5 -> 0.855)
                score = score / 100
            return score
        return None
    
    def predict_model_dataset_metric(self, model: str, dataset: str) -> Dict:
        """Predict metric for a model-dataset pair using neighborhood averages."""
        
        # Method 1: Average of different models on the same dataset
        dataset_neighbors = self.get_node_neighbors(dataset, 'model')
        model_scores_on_dataset = []
        for neighbor_model in dataset_neighbors:
            if neighbor_model != model:  # Exclude the target model
                score = self.get_edge_metric(neighbor_model, dataset)
                if score is not None:
                    model_scores_on_dataset.append(score)
        
        # Method 2: Global average of the metric (fallback)
        all_scores = []
        for edge in self.graph.edges(data=True):
            score = edge[2].get(self.metric_name)
            if score is not None:
                all_scores.append(score)
        
        predictions = {}
        reasons = []
        
        # Calculate predictions using different methods
        if model_scores_on_dataset:
            predictions['dataset_neighborhood'] = np.mean(model_scores_on_dataset)
            reasons.append(f"Dataset neighborhood: {len(model_scores_on_dataset)} similar models")
        
        if all_scores:
            predictions['global_average'] = np.mean(all_scores)
            reasons.append(f"Global average: {len(all_scores)} total scores")
        
        # Choose the best prediction method (prioritize dataset neighborhood)
        if 'dataset_neighborhood' in predictions:
            final_prediction = predictions['dataset_neighborhood']
            method = "dataset_neighborhood"
        elif 'global_average' in predictions:
            final_prediction = predictions['global_average']
            method = "global_average"
        else:
            final_prediction = None
            method = "no_data"
        
        return {
            'prediction': final_prediction,
            'method': method,
            'all_predictions': predictions,
            'reason': "; ".join(reasons),
            'dataset_neighbors_count': len(model_scores_on_dataset),
            'model_neighbors_count': 0,  # Not used anymore
            'global_scores_count': len(all_scores),
            'model_scores_on_dataset': model_scores_on_dataset  # Raw scores from neighbor models
        }


def main():
    metric_name = "accuracy"  # Define the metric to predict and evaluate
    output_file = f"output/neighborhood_predictions_{metric_name}.json"
    
    # 1. Load the graph from JSON file
    graph_file = "output/perfect_model_dataset_metrics.json"
    print(f"Loading graph from {graph_file}...")
    
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )
    
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # 2. Load the dataset for prediction
    try:
        with open(graph_file, "r") as f:
            data = json.load(f)
        prediction_data = data.get("results", [])
    except FileNotFoundError:
        print(f"Error: Input file not found at {graph_file}")
        exit()
    
    if not prediction_data:
        print("Prediction data is empty. Nothing to predict.")
        exit()
    
    # 3. Prepare data for prediction, filtering for the specified metric
    edges_to_predict = []
    true_metrics = []
    
    for item in prediction_data:
        metric_value = item.get("metrics", {}).get(metric_name)
        
        if metric_value is not None:
            # Handle cases where the metric value is a dictionary
            if isinstance(metric_value, dict):
                if "score" in metric_value:
                    metric_value = metric_value["score"]
                elif "value" in metric_value:
                    metric_value = metric_value["value"]
                else:
                    print(
                        f"Warning: Metric '{metric_name}' for model {item['model_id']} is a complex dictionary. Skipping."
                    )
                    continue
            
            try:
                metric_float = float(metric_value)
                edges_to_predict.append((item["model_id"], item["dataset_id"]))
                true_metrics.append(metric_float)
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert value '{metric_value}' for metric '{metric_name}'. Skipping."
                )
                continue
    
    if not edges_to_predict:
        print(f"No data found with the metric '{metric_name}' in {graph_file}.")
        exit()
    
    print(f"Loaded {len(edges_to_predict)} pairs with metric '{metric_name}' from {graph_file}")
    
    # 4. Initialize predictor
    predictor = NeighborhoodPredictor(G, metric_name)
    
    # 5. Make predictions
    print("Making neighborhood-based predictions...")
    results_to_save = []
    valid_predictions = []
    valid_true_metrics = []
    
    for model, dataset in tqdm(edges_to_predict, desc="Predicting"):
        true_metric = true_metrics[len(results_to_save)]  # Get corresponding true metric
        
        # Make prediction
        prediction_result = predictor.predict_model_dataset_metric(model, dataset)
        
        result_item = {
            "model_id": model,
            "dataset_id": dataset,
            "metric_name": metric_name,
            "true_metric": true_metric,
            "predicted_metric": prediction_result.get('prediction'),
            "prediction_method": prediction_result.get('method'),
            "all_predictions": prediction_result.get('all_predictions', {}),
            "reason": prediction_result.get('reason', ''),
            "dataset_neighbors_count": prediction_result.get('dataset_neighbors_count', 0),
            "model_neighbors_count": prediction_result.get('model_neighbors_count', 0),
            "global_scores_count": prediction_result.get('global_scores_count', 0),
            "model_scores_on_dataset": prediction_result.get('model_scores_on_dataset', []),
            "status": "Success" if prediction_result.get('prediction') is not None else "Failed"
        }
        
        results_to_save.append(result_item)
        
        # Collect valid predictions for evaluation
        if prediction_result.get('prediction') is not None:
            valid_predictions.append(prediction_result['prediction'])
            valid_true_metrics.append(true_metric)
    
    # 6. Evaluate results
    mse = rmse = mae = correlation = None
    method_counts = {}
    
    if len(valid_predictions) > 0:
        valid_predictions = np.array(valid_predictions)
        valid_true_metrics = np.array(valid_true_metrics)
        
        # Calculate metrics
        mse = np.mean((valid_predictions - valid_true_metrics) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(valid_predictions - valid_true_metrics))
        
        # Calculate correlation
        correlation = np.corrcoef(valid_predictions, valid_true_metrics)[0, 1]
        
        print(f"\n--- Neighborhood Prediction Results ---")
        print(f"Valid predictions: {len(valid_predictions)}/{len(edges_to_predict)}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Correlation coefficient: {correlation:.6f}")
        
        # Method breakdown
        method_counts = {}
        method_mses = {}
        for result in results_to_save:
            if result['status'] == 'Success':
                method = result['prediction_method']
                method_counts[method] = method_counts.get(method, 0) + 1
                
                if method not in method_mses:
                    method_mses[method] = []
                
                pred = result['predicted_metric']
                true = result['true_metric']
                method_mses[method].append((pred - true) ** 2)
        
        print(f"\n--- Method Breakdown ---")
        for method, count in method_counts.items():
            avg_mse = np.mean(method_mses[method]) if method in method_mses else 0
            print(f"{method}: {count} predictions, MSE: {avg_mse:.6f}")
        
        # Show some examples
        print(f"\n--- Example Predictions ---")
        for i, result in enumerate(results_to_save[:5]):
            if result['status'] == 'Success':
                print(f"Model: {result['model_id']}")
                print(f"Dataset: {result['dataset_id']}")
                print(f"True: {result['true_metric']:.4f}, Predicted: {result['predicted_metric']:.4f}")
                print(f"Method: {result['prediction_method']}")
                print(f"Neighbors: Dataset={result['dataset_neighbors_count']}, Model={result['model_neighbors_count']}")
                print("---")
    else:
        print("No valid predictions generated.")
    
    # 7. Save results
    summary = {
        "metric_name": metric_name,
        "total_pairs": len(edges_to_predict),
        "valid_predictions": len(valid_predictions),
        "failed_predictions": len(edges_to_predict) - len(valid_predictions),
        "evaluation_metrics": {
            "mse": float(mse) if mse is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
            "mae": float(mae) if mae is not None else None,
            "correlation": float(correlation) if correlation is not None else None
        } if len(valid_predictions) > 0 else None,
        "method_breakdown": method_counts if len(valid_predictions) > 0 else None,
        "predictions": results_to_save
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
