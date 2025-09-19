#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from artifact_graph.models.llm_link_predictor import LLMLinkPredictor
from artifact_graph.utils.graph_builder import load_artifact_graph_from_json

Edge = Tuple[str, str]


def _load_summaries(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
        return data if isinstance(data, dict) else {}


def _extract_prediction_data(graph_file: Path, metric_name: str) -> Tuple[List[Edge], List[float]]:
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

    edges_to_predict = []
    true_metrics = []
    for item in prediction_data:
        metric_value = item.get("metrics", {}).get(metric_name)

        if metric_value is not None:
            if isinstance(metric_value, dict):
                if "score" in metric_value:
                    metric_value = metric_value["score"]
                elif "value" in metric_value:
                    metric_value = metric_value["value"]
                else:
                    print(
                        f"Warning: Metric '{metric_name}' for model {item['model_id']} is a complex dictionary without a 'score' or 'value' key. Skipping."
                    )
                    continue

            try:
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

    return edges_to_predict, true_metrics


def evaluate(true_metrics: List[float], pred_metrics: List[float]) -> Dict[str, float]:
    mse = np.mean((np.array(pred_metrics) - np.array(true_metrics)) ** 2)
    return {"mse": float(mse)}


def run(
    graph_file: Path,
    summaries_file: Path,
    model_name: str,
    mode: str,
    metric_name: str,
    max_pairs: int,
):
    # Create a valid filename from the model name
    safe_model_name = model_name.replace("/", "_")
    output_file = Path(
        f"output/llm_attribute_predictions_{mode}_{safe_model_name}_{metric_name}.json"
    )

    # 1. Load data
    edges_to_predict, true_metrics = _extract_prediction_data(graph_file, metric_name)
    summaries = _load_summaries(summaries_file)

    G = None
    if mode == "neighborhood":
        G = load_artifact_graph_from_json(
            json_file=str(graph_file),
            min_downloads=1,
            metric_key=metric_name,
        )

    if max_pairs > 0:
        edges_to_predict = edges_to_predict[:max_pairs]
        true_metrics = true_metrics[:max_pairs]

    print(f"Total pairs to predict: {len(edges_to_predict)} (mode={mode}, metric={metric_name})")

    # 2. Predict
    predictor = LLMLinkPredictor(model_name=model_name)
    predicted_results = predictor.predict(
        edges_to_predict,
        G=G,
        mode=mode,
        metric_name=metric_name,
        summaries=summaries,
    )

    # 3. Evaluate
    out_rows = []
    valid_predictions = []
    valid_true_metrics = []

    for edge, true_metric, result in zip(edges_to_predict, true_metrics, predicted_results):
        model, dataset = edge
        row = {
            "model_id": model,
            "dataset_id": dataset,
            "metric_name": metric_name,
            "true_metric": true_metric,
            "predicted_metric": None,
            "reason": "",
            "status": "Failed",
        }
        if result and result.get("prediction") is not None:
            pred_metric = result["prediction"]
            valid_predictions.append(pred_metric)
            valid_true_metrics.append(true_metric)
            row.update(
                {
                    "predicted_metric": pred_metric,
                    "reason": result.get("reason", ""),
                    "status": "Success",
                }
            )
        out_rows.append(row)

    if valid_predictions:
        metrics = evaluate(valid_true_metrics, valid_predictions)
        print("\n--- Regression Metrics ---")
        for k, v in metrics.items():
            print(f"  - {k.upper()}: {v:.4f}")
        print("--------------------------")
    else:
        print("No valid predictions produced.")

    # 4. Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(out_rows, f, indent=2)
    print(f"\nPredictions saved to {output_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-file", default="output/perfect_model_dataset_metrics.json")
    p.add_argument("--summaries-file", default="output/readme_summaries.json")
    p.add_argument(
        "--model",
        default="openai/gpt-4o",
        choices=[
            "openai/gpt-4o",
            "openai/gpt-3.5-turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
    )
    p.add_argument(
        "--mode",
        choices=["zero-shot", "simple", "neighborhood"],
        default="simple",
    )
    p.add_argument("--metric", default="accuracy")
    p.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Cap the number of pairs to predict (0 for no limit)",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_file=Path(a.graph_file),
        summaries_file=Path(a.summaries_file),
        model_name=a.model,
        mode=a.mode,
        metric_name=a.metric,
        max_pairs=a.max_pairs,
    )
