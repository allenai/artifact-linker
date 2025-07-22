#!/usr/bin/env python3

import json
import numpy as np
from artifact_graph.utils.graph_builder import load_artifact_graph
from artifact_graph.models.llm_link_predictor import LLMLinkPredictor

def main():
    # 1. Load the graph (still needed for neighborhood context)
    G = load_artifact_graph(
        models_dir="output/models/metadata",
        datasets_dir="output/datasets/metadata",
        metrics_dir="output/metrics",
    )
    print(f"Graph for context lookup built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 2. Load the validation set created by the GNN script
    validation_file = "output/gnn_validation_set.json"
    try:
        with open(validation_file, "r") as f:
            validation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Validation file not found at {validation_file}")
        print("Please run the GNN training script first to generate the validation set.")
        exit()
        
    if not validation_data:
        print("Validation data is empty. Nothing to predict.")
    else:
        print(f"Loaded {len(validation_data)} validation pairs from {validation_file}")
        
        # Prepare data for predictor
        edges_to_predict = [(item["model"], item["dataset"]) for item in validation_data]
        true_accuracies = [item["accuracy"] for item in validation_data]

        print("\n--- LLM Prediction Set Links ---")
        for model, dataset in edges_to_predict:
            print(f"  - Model: {model}, Dataset: {dataset}")
        print("------------------------------------\n")

        # 3. Initialize predictor
        predictor = LLMLinkPredictor(model_name="gpt-4o")
        
        # 4. Predict
        predicted_accuracies = predictor.predict(
            edges_to_predict,
            G,
            model_dir="output/models",
            dataset_dir="output/datasets",
            mode="simple"
        )
        
        # 5. Evaluate and Print results
        # Filter out None values from predictions and corresponding true values
        valid_predictions = []
        valid_true_accuracies = []
        for pred, true_acc in zip(predicted_accuracies, true_accuracies):
            if pred is not None:
                valid_predictions.append(pred)
                valid_true_accuracies.append(true_acc)

        if not valid_predictions:
            print("Could not generate any valid predictions.")
        else:
            # Calculate MSE
            mse = np.mean((np.array(valid_predictions) - np.array(valid_true_accuracies))**2)
            print(f"\nOverall Mean Squared Error (MSE) on validation set: {mse:.4f}")

            print("\n--- LLM Prediction Results ---")
            # We iterate through the original list to match predictions with their pairs
            valid_idx = 0
            for i in range(len(edges_to_predict)):
                model, dataset = edges_to_predict[i]
                true_acc = true_accuracies[i]
                # Check if the prediction was valid for this pair
                if predicted_accuracies[i] is not None:
                    pred_acc = valid_predictions[valid_idx]
                    print(f"  - Model: {model}, Dataset: {dataset} -> Predicted: {pred_acc:.4f}, Actual: {true_acc:.4f}")
                    valid_idx += 1
                else:
                    print(f"  - Model: {model}, Dataset: {dataset} -> Prediction Failed")
            print("------------------------------")


if __name__ == "__main__":
    main() 