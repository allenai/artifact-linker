#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from artifact_graph.models.baseline_link_predictor import BaselineLinkPredictor
from artifact_graph.utils.evaluation_utils import evaluate_binary_classification
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import prepare_link_predictor_dataset

Edge = Tuple[int, int]


def run(graph_data_dir: Path, mode: str, max_pairs: int, seed: int, use_gnn_data: bool, **kwargs):
    output_file = Path(f"output/final_results/baseline_link_predictions_{mode}{'_gnn' if use_gnn_data else ''}.json")

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

    if not edges:
        print("No edges to predict.")
        return

    print(f"Total pairs to predict: {len(edges)} (mode={mode})")

    predictor = BaselineLinkPredictor(mode=mode, **kwargs)

    out_rows = []
    y_true, y_pred = [], []

    for (m, d), y in tqdm(zip(edges, labels), total=len(edges), desc="Predicting links (baseline)"):
        result = predictor.predict(
            model_id=m,
            dataset_id=d,
            G=G,
            node_metadata=node_metadata,
        )

        row = {
            "model_id": m,
            "dataset_id": d,
            "model_name": node_metadata.get(m, {}).get("name"),
            "dataset_name": node_metadata.get(d, {}).get("name"),
            "true_label": y,
            "predicted_label": None,
            "reason": "",
            "status": "Failed",
        }

        if result and (result.get("prediction") is not None):
            pred_label = 1 if bool(result["prediction"]) else 0
            y_true.append(y)
            y_pred.append(pred_label)
            row.update(
                {
                    "predicted_label": pred_label,
                    "reason": result.get("reason", ""),
                    "status": "Success",
                }
            )
        out_rows.append(row)

    if y_pred:
        metrics = evaluate_binary_classification(y_true, y_pred)
        print("\n--- Binary Classification Metrics (Baseline) ---")
        for k, v in metrics.items():
            print(f"  - {k.capitalize()}: {v:.4f}")
        print("---------------------------------------------")

        # --- Degree-Controlled Evaluation ---
        print("\n--- Degree-Controlled Performance (Baseline) ---")
        degrees = dict(G.degree())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        # Re-extract edges that correspond to y_true/y_pred
        # Note: We only added to y_true/y_pred if result["prediction"] was not None
        # So we need to filter edges the same way
        valid_edges = []
        for row in out_rows:
            if row["status"] == "Success":
                valid_edges.append((row["model_id"], row["dataset_id"]))

        valid_edges = np.array(valid_edges)
        if len(valid_edges) > 0:
            u_ids = valid_edges[:, 0]
            v_ids = valid_edges[:, 1]

            u_degs = np.array([degrees.get(n, 0) for n in u_ids])
            v_degs = np.array([degrees.get(n, 0) for n in v_ids])
            edge_min_deg = np.minimum(u_degs, v_degs)

            buckets = {
                "Tail (deg<=5)": edge_min_deg <= 5,
                "Medium (5<deg<=20)": (edge_min_deg > 5) & (edge_min_deg <= 20),
                "Head (deg>20)": edge_min_deg > 20
            }

            from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

            for name, mask in buckets.items():
                if mask.sum() > 0:
                    sub_true = y_true_np[mask]
                    sub_pred = y_pred_np[mask]

                    sub_f1 = f1_score(sub_true, sub_pred, zero_division=0)
                    sub_acc = accuracy_score(sub_true, sub_pred)

                    # AUC requires probs, here we only have binary labels usually, unless we extract scores
                    # For baseline, 'prediction' is often just boolean or score. 
                    # Let's just stick to F1/Acc for now as baseline scores vary wildly in scale
                    print(f"  [{name}] N={mask.sum()} | F1: {sub_f1:.4f} | Acc: {sub_acc:.4f}")
        print("------------------------------------------------")
        # ------------------------------------
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
    p = argparse.ArgumentParser(description="Run Baseline Link Predictor")
    p.add_argument("--graph-data-dir", default="output/artifact_graph_data")
    p.add_argument(
        "--mode",
        default="downloads",
        choices=[
            "downloads",
            "common_neighbors",
            "jaccard",
            "adamic_adar",
            "preferential_attachment",
            "resource_allocation",
            "katz",
        ],
        help="Prediction mode",
    )
    # Download-based parameters
    p.add_argument(
        "--model-download-threshold",
        type=int,
        default=500,
        help="Download threshold for models (downloads mode only)",
    )
    p.add_argument(
        "--dataset-download-threshold",
        type=int,
        default=1000,
        help="Download threshold for datasets (downloads mode only)",
    )
    # Graph-based parameters
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="General threshold for graph-based methods",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta parameter for Katz centrality",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=500000,
        help="Cap the number of pairs to predict (0 for no limit)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling")
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN link predictions data")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()

    # Prepare mode-specific kwargs
    kwargs = {}
    if a.mode == "downloads":
        kwargs["model_download_threshold"] = a.model_download_threshold
        kwargs["dataset_download_threshold"] = a.dataset_download_threshold
    elif a.threshold is not None:
        kwargs["threshold"] = a.threshold

    if a.mode == "katz":
        kwargs["beta"] = a.beta

    run(
        graph_data_dir=Path(a.graph_data_dir),
        mode=a.mode,
        max_pairs=a.max_pairs,
        seed=a.seed,
        use_gnn_data=a.use_gnn_data,
        **kwargs,
    )
