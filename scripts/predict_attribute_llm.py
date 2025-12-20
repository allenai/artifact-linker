#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from artifact_graph.models.llm_attribute_predictor import LLMAttributePredictor
from artifact_graph.utils.evaluation_utils import (
    calculate_mae,
    calculate_map_continuous,
    calculate_mape,
    calculate_mean_absolute_difference,
    calculate_mse,
    calculate_ndcg_standard,
    calculate_r_squared,
    calculate_ranking_correlation,
    calculate_rmse,
)
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import prepare_attribute_predictor_dataset

Edge = Tuple[int, int]


def predict_single_attribute(
    edge: Edge,
    true_metric: float,
    current_metric_name: str,
    predictor: LLMAttributePredictor,
    G,
    node_metadata: Dict,
    edge_metadata: Dict,
) -> Dict:
    """Predict a single attribute and return the result."""
    model_id, dataset_id = edge

    row = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "metric_name": current_metric_name,
        "true_metric": true_metric,
        "predicted_metric": None,
        "reason": "",
        "status": "Failed",
    }

    # Predict this single edge-metric pair
    result = predictor.predict(
        model_id,
        dataset_id,
        G=G,
        node_metadata=node_metadata,
        edge_metadata=edge_metadata,
        metric_name=current_metric_name,
    )

    if result and result.get("prediction") is not None:
        pred_metric = result["prediction"]
        row.update(
            {
                "predicted_metric": pred_metric,
                "reason": result.get("reason", ""),
                "status": "Success",
            }
        )

    return row


def run(
    graph_data_dir: Path,
    model_name: str,
    hops: int,
    use_info: bool,
    metric_name: str | None,
    max_pairs: int,
    max_workers: int = 4,
    use_gnn_data: bool = False,
):
    safe_model_name = model_name.replace("/", "_")
    metric_suffix = "all_metrics" if metric_name is None else metric_name
    output_file = Path(
        f"output/final_results/llm_attribute_predictions_{hops}hop_{safe_model_name}_{metric_suffix}{'_gnn' if use_gnn_data else ''}.json"
    )

    G, node_metadata, edge_metadata = load_nx_graph(graph_data_dir=str(graph_data_dir))
    edges_to_predict, true_metrics, metric_names = prepare_attribute_predictor_dataset(
        G, metric_name
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
        edges_to_predict = real_edges_to_predict
        # end for GNN

    if not edges_to_predict:
        return

    if max_pairs > 0:
        edges_to_predict = edges_to_predict[:max_pairs]
        true_metrics = true_metrics[:max_pairs]
        metric_names = metric_names[:max_pairs]

    if metric_name is None:
        print(
            f"Total pairs to predict: {len(edges_to_predict)} across {len(set(metric_names))} different metrics (hops={hops}, use_info={use_info})"
        )
        print(f"Metrics found: {sorted(set(metric_names))}")
    else:
        print(
            f"Total pairs to predict: {len(edges_to_predict)} (hops={hops}, use_info={use_info}, metric={metric_name})"
        )

    print(f"Using {max_workers} parallel workers for predictions")

    predictor = LLMAttributePredictor(model_name=model_name, hop_number=hops, use_info=use_info)

    # Create prediction function with fixed parameters
    predict_func = partial(
        predict_single_attribute,
        predictor=predictor,
        G=G,
        node_metadata=node_metadata,
        edge_metadata=edge_metadata,
    )

    out_rows = []
    valid_predictions = []
    valid_true_metrics = []

    # Prepare tasks for parallel processing
    tasks = list(zip(edges_to_predict, true_metrics, metric_names))

    # Process predictions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(predict_func, edge, true_metric, current_metric_name): (
                edge,
                true_metric,
                current_metric_name,
            )
            for edge, true_metric, current_metric_name in tasks
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Predicting attributes",
        ):
            try:
                row = future.result()
                out_rows.append(row)

                # Collect successful predictions for evaluation
                if row["status"] == "Success" and row["predicted_metric"] is not None:
                    valid_predictions.append(row["predicted_metric"])
                    valid_true_metrics.append(row["true_metric"])

            except Exception as e:
                edge, true_metric, current_metric_name = future_to_task[future]
                model_id, dataset_id = edge
                print(
                    f"Error predicting attribute ({model_id}, {dataset_id}, {current_metric_name}): {e}"
                )
                # Add failed result
                row = {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "metric_name": current_metric_name,
                    "true_metric": true_metric,
                    "predicted_metric": None,
                    "reason": f"Error: {str(e)}",
                    "status": "Failed",
                }
                out_rows.append(row)

    if valid_predictions:
        # Calculate comprehensive regression metrics
        metrics = {
            "mse": calculate_mse(valid_predictions, valid_true_metrics),
            "mae": calculate_mae(valid_predictions, valid_true_metrics),
            "rmse": calculate_rmse(valid_predictions, valid_true_metrics),
            "mape": calculate_mape(valid_predictions, valid_true_metrics),
            "r_squared": calculate_r_squared(valid_predictions, valid_true_metrics),
            "mean_abs_diff": calculate_mean_absolute_difference(
                valid_predictions, valid_true_metrics
            ),
        }

        print("\n--- Regression Metrics ---")
        print(f"  - MSE (Mean Squared Error): {metrics['mse']:.4f}")
        print(f"  - MAE (Mean Absolute Error): {metrics['mae']:.4f}")
        print(f"  - RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        if metrics["mape"] != float("inf"):
            print(f"  - MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
        else:
            print("  - MAPE: Undefined (zero true values)")
        print(f"  - R² (R-squared): {metrics['r_squared']:.4f}")
        print(f"  - Mean Absolute Difference: {metrics['mean_abs_diff']:.4f}")
        print(f"  - Valid predictions: {len(valid_predictions)}/{len(out_rows)}")
        print("--------------------------")

        # Calculate ranking metrics by grouping predictions by dataset
        print("\n--- Ranking Metrics ---")

        # Group predictions by dataset
        dataset_predictions = {}
        for row in out_rows:
            if row["status"] == "Success" and row["predicted_metric"] is not None:
                dataset_id = row["dataset_id"]
                if dataset_id not in dataset_predictions:
                    dataset_predictions[dataset_id] = []

                dataset_predictions[dataset_id].append(
                    {
                        "model_id": row["model_id"],
                        "predicted_score": row["predicted_metric"],
                        "true_score": row["true_metric"],
                    }
                )

        # Calculate ranking metrics for each dataset
        all_ndcg_1 = []
        all_ndcg_3 = []
        all_ndcg_5 = []
        all_ndcg_10 = []
        all_ndcg_full = []
        all_map_1 = []
        all_map_3 = []
        all_map_5 = []
        all_map_10 = []
        all_map_full = []
        all_kendall_tau = []
        all_spearman = []

        for dataset_id, predictions in dataset_predictions.items():
            if len(predictions) < 2:  # Need at least 2 items to rank
                continue

            # Sort by predicted scores (descending for ranking)
            predictions_sorted = sorted(
                predictions, key=lambda x: x["predicted_score"], reverse=True
            )

            # Create predicted ranking and ground truth
            predicted_items_with_scores = []
            ground_truth = {}

            for pred in predictions_sorted:
                model_id = pred["model_id"]
                item_key = f"{model_id}_{dataset_id}"
                predicted_items_with_scores.append((item_key, pred["predicted_score"]))
                ground_truth[item_key] = pred["true_score"]

            try:
                # NDCG@k metrics
                ndcg_1 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=1)
                ndcg_3 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=3)
                ndcg_5 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=5)
                ndcg_10 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=10)
                ndcg_full = calculate_ndcg_standard(predicted_items_with_scores, ground_truth)

                all_ndcg_1.append(ndcg_1)
                all_ndcg_3.append(ndcg_3)
                all_ndcg_5.append(ndcg_5)
                all_ndcg_10.append(ndcg_10)
                all_ndcg_full.append(ndcg_full)

                # MAP@k metrics
                map_1 = calculate_map_continuous(predicted_items_with_scores, ground_truth, k=1)
                map_3 = calculate_map_continuous(predicted_items_with_scores, ground_truth, k=3)
                map_5 = calculate_map_continuous(predicted_items_with_scores, ground_truth, k=5)
                map_10 = calculate_map_continuous(predicted_items_with_scores, ground_truth, k=10)
                map_full = calculate_map_continuous(predicted_items_with_scores, ground_truth)

                all_map_1.append(map_1)
                all_map_3.append(map_3)
                all_map_5.append(map_5)
                all_map_10.append(map_10)
                all_map_full.append(map_full)

                # Ranking correlation
                correlation_metrics = calculate_ranking_correlation(
                    predicted_items_with_scores, ground_truth
                )
                if "kendall_tau" in correlation_metrics:
                    all_kendall_tau.append(correlation_metrics["kendall_tau"])
                if "spearman" in correlation_metrics:
                    all_spearman.append(correlation_metrics["spearman"])

            except Exception as e:
                print(f"Warning: Could not calculate ranking metrics for dataset {dataset_id}: {e}")

        # Print average ranking metrics
        if all_ndcg_1:
            print(f"  - NDCG@1: {sum(all_ndcg_1) / len(all_ndcg_1):.4f}")
        if all_ndcg_3:
            print(f"  - NDCG@3: {sum(all_ndcg_3) / len(all_ndcg_3):.4f}")
        if all_ndcg_5:
            print(f"  - NDCG@5: {sum(all_ndcg_5) / len(all_ndcg_5):.4f}")
        if all_ndcg_10:
            print(f"  - NDCG@10: {sum(all_ndcg_10) / len(all_ndcg_10):.4f}")
        if all_ndcg_full:
            print(f"  - NDCG (full): {sum(all_ndcg_full) / len(all_ndcg_full):.4f}")
        if all_map_1:
            print(f"  - MAP@1: {sum(all_map_1) / len(all_map_1):.4f}")
        if all_map_3:
            print(f"  - MAP@3: {sum(all_map_3) / len(all_map_3):.4f}")
        if all_map_5:
            print(f"  - MAP@5: {sum(all_map_5) / len(all_map_5):.4f}")
        if all_map_10:
            print(f"  - MAP@10: {sum(all_map_10) / len(all_map_10):.4f}")
        if all_map_full:
            print(f"  - MAP (full): {sum(all_map_full) / len(all_map_full):.4f}")
        if all_kendall_tau:
            print(f"  - Kendall's Tau: {sum(all_kendall_tau) / len(all_kendall_tau):.4f}")
        if all_spearman:
            print(f"  - Spearman's Rho: {sum(all_spearman) / len(all_spearman):.4f}")

        print(f"  - Datasets with rankings: {len(dataset_predictions)}")
        print(
            f"  - Datasets with 2+ models: {len([d for d in dataset_predictions.values() if len(d) >= 2])}"
        )
        print("--------------------------")

    else:
        print("No valid predictions produced.")

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
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", default="output/artifact_graph_data")
    p.add_argument(
        "--model",
        default="openai/gpt-4o",
        choices=[
            "openai/gpt-4o",
            "openai/o3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
    )
    p.add_argument(
        "--hops",
        type=int,
        choices=[0, 1],
        default=1,
        help="Number of hops for neighborhood context (0 or 1)",
    )
    p.add_argument(
        "--no-info",
        action="store_false",
        dest="use_info",
        help="Disable using model/dataset info in the prompt",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Cap the number of pairs to predict (0 for no limit)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for LLM calls (default: 4)",
    )
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN attribute predictions data")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()

    run(
        graph_data_dir=Path(a.graph_data_dir),
        model_name=a.model,
        hops=a.hops,
        use_info=a.use_info,
        metric_name=None,
        max_pairs=a.max_pairs,
        max_workers=a.max_workers,
        use_gnn_data=a.use_gnn_data,
    )
