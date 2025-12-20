#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from artifact_graph.models.baseline_attribute_predictor import (
    BaselineAttributePredictor,
)
from artifact_graph.utils.evaluation_utils import (
    calculate_mae,
    calculate_mape,
    calculate_mean_absolute_difference,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
)
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import prepare_attribute_predictor_dataset


def run(graph_data_dir: Path, mode: str, metric_name: str | None, use_gnn_data: bool = False):
    """Run the baseline attribute prediction and evaluation."""
    metric_str = metric_name if metric_name else "all"
    output_file = Path(
        f"output/final_results/baseline_attribute_predictions_mode_{mode}_metric_{metric_str}{'_gnn' if use_gnn_data else ''}.json"
    )

    G, node_metadata, edge_metadata = load_nx_graph(str(graph_data_dir))
    edges, true_metrics, metric_names = prepare_attribute_predictor_dataset(
        G, metric_name=metric_name
    )

    if use_gnn_data:
        metric_names = []
        real_edges_to_predict = []
        # for the same with GNN attribute prediction
        with open("output/final_results/gnn_attribute_predictions.json", "r") as f:
            gnn_attribute_predictions = json.load(f)
        edges_to_predict = gnn_attribute_predictions["records"]
        true_metrics = [edge["ground_truth"] for edge in edges_to_predict]
        for edge, metric_num in zip(edges_to_predict, true_metrics):
            edges_1 = tuple(edge["input"]["edge"])
            edges_2 = tuple(edge["input"]["edge"][::-1])
            if edges_1 in edge_metadata:
                edges = edges_1
                real_edges_to_predict.append(edges)
            elif edges_2 in edge_metadata:
                edges = edges_2
                real_edges_to_predict.append(edges)
            else:
                raise ValueError(f"Edge {edges} not found in edge_metadata")
            metrics = edge_metadata[edges]['metrics']
            for metric_name, metric_value in metrics.items():
                if abs(metric_value - metric_num) < 1e-3:
                    metric_names.append(metric_name)
                    break
        assert len(metric_names) == len(edges_to_predict)
        edges = real_edges_to_predict
        # end for GNN

    if not edges:
        print("No edges found for the specified metric to predict.")
        return

    print(f"Total attributes to predict: {len(edges)} (mode={mode}, metric={metric_str})")

    predictor = BaselineAttributePredictor(mode=mode)

    out_rows = []
    predictions = []
    ground_truth = []

    for edge, true_metric, current_metric in tqdm(
        zip(edges, true_metrics, metric_names), total=len(edges)
    ):
        model_id, dataset_id = edge
        result = predictor.predict(
            model_id=model_id,
            dataset_id=dataset_id,
            G=G,
            node_metadata=node_metadata,
            edge_metadata=edge_metadata,
            metric_name=current_metric,
        )

        row: Dict[str, Any] = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "model_name": node_metadata.get(model_id, {}).get("name"),
            "dataset_name": node_metadata.get(dataset_id, {}).get("name"),
            "metric_name": current_metric,
            "true_value": true_metric,
            "predicted_value": None,
            "reason": "",
            "status": "Failed",
        }

        if result and (result.get("prediction") is not None):
            predicted_value = result["prediction"]
            predictions.append(predicted_value)
            ground_truth.append(true_metric)
            row.update(
                {
                    "predicted_value": predicted_value,
                    "reason": result.get("reason", ""),
                    "status": "Success",
                }
            )
        out_rows.append(row)

    if predictions:
        print("\n--- Regression Metrics (Baseline) ---")
        metrics = {
            "MSE": calculate_mse(predictions, ground_truth),
            "MAE": calculate_mae(predictions, ground_truth),
            "RMSE": calculate_rmse(predictions, ground_truth),
            "MAPE": calculate_mape(predictions, ground_truth),
            "R-squared": calculate_r_squared(predictions, ground_truth),
            "Mean Absolute Diff": calculate_mean_absolute_difference(predictions, ground_truth),
        }
        for name, value in metrics.items():
            print(f"  - {name}: {value:.4f}")
        print("---------------------------------------------")
    else:
        print("No valid predictions were made.")

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        elif hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        else:
            return obj

    serializable_rows = convert_numpy_types(out_rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(serializable_rows, f, indent=2)
    print(f"\nPredictions saved to {output_file}")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run Baseline Attribute Predictor")
    p.add_argument("--graph-data-dir", default="output/artifact_graph_data")
    p.add_argument(
        "--mode",
        choices=["global_average", "dataset_average"],
        default="dataset_average",
        help="The baseline strategy to use.",
    )
    p.add_argument(
        "--metric",
        default=None,
        help="Specific metric to predict. If not provided, predicts all available metrics.",
    )
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN attribute predictions data")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_data_dir=Path(a.graph_data_dir),
        mode=a.mode,
        metric_name=a.metric,
        use_gnn_data=a.use_gnn_data,
    )
