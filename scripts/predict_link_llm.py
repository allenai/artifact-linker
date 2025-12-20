#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from functools import partial
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from artifact_graph.models.llm_link_predictor import LLMLinkPredictor
from artifact_graph.utils.evaluation_utils import evaluate_binary_classification
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import (
    create_safe_filename,
    prepare_link_predictor_dataset,
)


def predict_single_link(
    model_id: int,
    dataset_id: int,
    true_label: int,
    link_predictor: LLMLinkPredictor,
    G,
    node_metadata: Dict,
) -> Dict:
    """Predict a single link and return the result."""
    obj = link_predictor.predict(
        model_id=model_id,
        dataset_id=dataset_id,
        G=G,
        node_metadata=node_metadata,
    )

    row = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_name": node_metadata.get(model_id, {}).get("name"),
        "dataset_name": node_metadata.get(dataset_id, {}).get("name"),
        "true_label": true_label,
        "predicted_label": None,
        "reason": "",
        "status": "Failed",
    }

    if obj and (obj.get("prediction") is not None):
        pred_label = 1 if bool(obj["prediction"]) else 0
        row.update(
            {
                "predicted_label": pred_label,
                "reason": obj.get("reason", ""),
                "status": "Success",
            }
        )

    return row


def run(
    graph_data_dir: Path,
    model_name: str,
    hops: int,
    use_info: bool,
    seed: int,
    max_pairs: int,
    max_workers: int = 4,
    use_gnn_data: bool = False,
):
    safe_model_name = create_safe_filename(model_name)
    output_file = Path(
        f"output/final_results/llm_link_predictions_{hops}hop_{safe_model_name}{'_gnn' if use_gnn_data else ''}.json"
    )

    G, node_metadata, _ = load_nx_graph(graph_data_dir=str(graph_data_dir))

    edges, labels = prepare_link_predictor_dataset(G, seed=seed, max_pairs=max_pairs)

    if use_gnn_data:
        # for the same with GNN link prediction
        with open("output/final_results/gnn_link_predictions.json", "r") as f:
            gnn_link_predictions = json.load(f)
        edges = gnn_link_predictions["test_predictions"]["edges"]
        labels = [edge["ground_truth"] for edge in edges]
        edges = [(edge["v_id"], edge["u_id"]) for edge in edges]
        # end for GNN

    print(f"Total pairs to predict: {len(edges)} (hops={hops}, use_info={use_info})")
    print(f"Using {max_workers} parallel workers for predictions")

    link_predictor = LLMLinkPredictor(model_name=model_name, hop_number=hops, use_info=use_info)

    # Create prediction function with fixed parameters
    predict_func = partial(
        predict_single_link, link_predictor=link_predictor, G=G, node_metadata=node_metadata
    )

    out_rows = []
    y_true, y_pred = [], []

    # Prepare tasks for parallel processing
    tasks = [(model_id, dataset_id, label) for (model_id, dataset_id), label in zip(edges, labels)]

    # Process predictions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(predict_func, model_id, dataset_id, label): (
                model_id,
                dataset_id,
                label,
            )
            for model_id, dataset_id, label in tasks
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Predicting links",
        ):
            try:
                row = future.result()
                out_rows.append(row)

                # Collect successful predictions for evaluation
                if row["status"] == "Success" and row["predicted_label"] is not None:
                    y_true.append(row["true_label"])
                    y_pred.append(row["predicted_label"])

            except Exception as e:
                model_id, dataset_id, label = future_to_task[future]
                print(f"Error predicting link ({model_id}, {dataset_id}): {e}")
                # Add failed result
                row = {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "model_name": node_metadata.get(model_id, {}).get("name"),
                    "dataset_name": node_metadata.get(dataset_id, {}).get("name"),
                    "true_label": label,
                    "predicted_label": None,
                    "reason": f"Error: {str(e)}",
                    "status": "Failed",
                }
                out_rows.append(row)

    if y_pred:
        metrics = evaluate_binary_classification(y_true, y_pred)
        print("\n--- Binary Classification Metrics ---")
        for k, v in metrics.items():
            print(f"  - {k.capitalize()}: {v:.4f}")
        print("------------------------------------")
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
        choices=["openai/gpt-4o", "openai/o3", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
        default="openai/gpt-4o",
    )
    p.add_argument(
        "--hops",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Number of hops for neighborhood context",
    )
    p.add_argument(
        "--no-info",
        action="store_false",
        dest="use_info",
        help="Disable using model/dataset info in the prompt",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs", type=int, default=5000)
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for LLM calls (default: 4)",
    )
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN link predictions data")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_data_dir=Path(a.graph_data_dir),
        model_name=a.model,
        hops=a.hops,
        use_info=a.use_info,
        seed=a.seed,
        max_pairs=a.max_pairs,
        max_workers=a.max_workers,
        use_gnn_data=a.use_gnn_data,
    )
