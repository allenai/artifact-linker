#!/usr/bin/env python3

import json

import numpy as np
from tqdm import tqdm

from artifact_graph.models.llm_link_predictor import LLMLinkPredictor
from artifact_graph.utils.graph_builder import (
    load_artifact_graph,
    load_artifact_graph_from_json,
)


def main():
    mode = "zero-shot"  # Define the mode for prediction: "zero-shot", "simple", "neighborhood"
    metric_name = "accuracy"  # Define the metric to predict and evaluate
    output_file = f"output/llm_predictions_{mode}.json"  # Output file will be named based on the mode

    # 1. Load the graph from JSON file (for neighborhood context)
    # The graph is built from the same data we'll use for prediction.
    graph_file = "output/perfect_model_dataset_metrics.json"
    G = load_artifact_graph_from_json(
        json_file=graph_file,
        min_downloads=1,
        metric_key=metric_name,
    )

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

    # Prepare data for predictor, filtering for the specified metric
    edges_to_predict = []
    true_metrics = []
    for item in prediction_data:
        metric_value = item.get("metrics", {}).get(metric_name)

        if metric_value is not None:
            # Handle cases where the metric value is a dictionary (e.g., {'score': 0.9, 'std': 0.1})
            if isinstance(metric_value, dict):
                # Prioritize 'score', then 'value', as common keys for the primary metric
                if "score" in metric_value:
                    metric_value = metric_value["score"]
                elif "value" in metric_value:
                    metric_value = metric_value["value"]
                else:
                    # If we can't find a primary key, we have to skip this record.
                    print(
                        f"Warning: Metric '{metric_name}' for model {item['model_id']} is a complex dictionary without a 'score' or 'value' key. Skipping."
                    )
                    continue

            try:
                # Ensure the final metric value can be cast to a float before adding it.
                metric_float = float(metric_value)
                edges_to_predict.append((item["model_id"], item["dataset_id"]))
                true_metrics.append(metric_float)
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert value '{metric_value}' for metric '{metric_name}' of model {item['model_id']} to float. Skipping."
                )
                continue

    if not edges_to_predict:
        print(f"No data found with the metric '{metric_name}' in {graph_file}.")
        exit()

    print(f"Loaded {len(edges_to_predict)} pairs with metric '{metric_name}' from {graph_file}")

    print("\n--- LLM Prediction Set Links ---")
    for model, dataset in edges_to_predict:
        print(f"  - Model: {model}, Dataset: {dataset}")
    print("------------------------------------\n")

    # 3. Initialize predictor
    predictor = LLMLinkPredictor(model_name="gpt-4o")

    # 4. Predict
    predicted_metrics = predictor.predict(
        edges_to_predict,
        G,
        model_dir="output/models",
        dataset_dir="output/datasets",
        mode=mode,
        metric_name=metric_name,
    )

    # 5. Evaluate and Print results
    results_to_save = []
    valid_predictions = []
    valid_true_metrics = []

    # Unpack the results and filter out None values
    for result, true_metric in zip(predicted_metrics, true_metrics):
        if result and result.get("prediction") is not None:
            valid_predictions.append(result["prediction"])
            valid_true_metrics.append(true_metric)

    if not valid_predictions:
        print("Could not generate any valid predictions.")
    else:
        # Calculate MSE
        mse = np.mean((np.array(valid_predictions) - np.array(valid_true_metrics)) ** 2)
        print(f"\nOverall Mean Squared Error (MSE) on validation set: {mse:.4f}")

        print("\n--- LLM Prediction Results ---")
        # We iterate through the original list to match predictions with their pairs
        valid_idx = 0
        for i in range(len(edges_to_predict)):
            model, dataset = edges_to_predict[i]
            true_metric = true_metrics[i]
            prediction_result = predicted_metrics[i]  # This is now a dict or None

            result_item = {
                "model_id": model,
                "dataset_id": dataset,
                "metric_name": metric_name,
                "true_metric": true_metric,
                "predicted_metric": None,
                "reason": "",
                "status": "Failed",
            }

            # Check if the prediction was valid for this pair
            if prediction_result and prediction_result.get("prediction") is not None:
                pred_metric = prediction_result["prediction"]
                reason = prediction_result.get("reason", "")
                print(
                    f"  - Model: {model}, Dataset: {dataset} -> Predicted: {pred_metric:.4f}, Actual: {true_metric:.4f}"
                )
                result_item["predicted_metric"] = pred_metric
                result_item["reason"] = reason
                result_item["status"] = "Success"
                valid_idx += 1
            else:
                print(f"  - Model: {model}, Dataset: {dataset} -> Prediction Failed")

            results_to_save.append(result_item)
        print("------------------------------")

    # 6. Save results to a file
    with open(output_file, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    main()
