#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifact_graph.models.baseline_link_ranker import BaselineLinkRanker
from artifact_graph.utils.evaluation_utils import (
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import prepare_link_ranker_dataset


def run(
    graph_data_dir: Path,
    mode: str,
    max_pairs: int,
    seed: int,
    candidates_per_dataset: int,
    use_gnn_data: bool = False,
):
    output_file = Path(
        f"output/final_results/baseline_link_rankings_{mode}{'_gnn' if use_gnn_data else ''}.json"
    )
    G, node_metadata, _ = load_nx_graph(str(graph_data_dir))

    if use_gnn_data:
        with open("output/final_results/gnn_link_rankings.json", "r") as f:
            gnn_link_rankings = json.load(f)
        ranking_data = {}
        gnn_results = gnn_link_rankings["detailed_rankings_by_dataset"]
        for result in gnn_results:
            dataset_id = result["dataset_id"]
            all_candidates = result["ranked_candidates"]
            pos_candidates = []
            neg_candidates = []
            for candidate in all_candidates:
                if candidate["ground_truth_label"]:
                    pos_candidates.append(candidate["model_id"])
                else:
                    neg_candidates.append(candidate["model_id"])
            ranking_data[dataset_id] = (pos_candidates, neg_candidates)
    else:
        ranking_data = prepare_link_ranker_dataset(
            G, seed=seed, candidates_per_dataset=candidates_per_dataset
        )

    # Limit the number of datasets if max_pairs is specified, matching llm script behavior
    if max_pairs > 0:
        ranking_items = list(ranking_data.items())[:max_pairs]
        ranking_data = dict(ranking_items)

    if not ranking_data:
        print("No ranking data to process.")
        return

    print(f"Total datasets to rank: {len(ranking_data)} (mode={mode})")

    ranker = BaselineLinkRanker(mode=mode, seed=seed)
    all_results = []
    all_true_relevance = []
    all_pred_rankings = []

    for dataset_id, (positive_models, negative_candidates) in tqdm(
        ranking_data.items(), desc="Ranking links (baseline)"
    ):
        result = ranker.rank(
            dataset_id=dataset_id,
            positive_models=positive_models,
            negative_candidates=negative_candidates,
            G=G,
            node_metadata=node_metadata,
        )
        if result and result.get("ranked_model_ids"):
            all_results.append(result)
            all_pred_rankings.append(result["ranked_model_ids"])
            all_true_relevance.append(set(positive_models))

    if all_pred_rankings:
        k_values = [1, 3, 5, 10]
        recalls = {
            k: np.mean(
                [
                    calculate_recall_at_k(p, t, k)
                    for p, t in zip(all_pred_rankings, all_true_relevance)
                ]
            )
            for k in k_values
        }
        precisions = {
            k: np.mean(
                [
                    calculate_precision_at_k(p, t, k)
                    for p, t in zip(all_pred_rankings, all_true_relevance)
                ]
            )
            for k in k_values
        }

        print("\n--- Link Ranking Evaluation Metrics (Baseline) ---")
        for k in k_values:
            print(f"  - Recall@{k}: {recalls[k]:.4f}")
        for k in k_values:
            print(f"  - Precision@{k}: {precisions[k]:.4f}")
        print("---------------------------------------------")
    else:
        print("No valid rankings produced.")

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

    serializable_results = convert_numpy_types(all_results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nRanking results saved to {output_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Run Baseline Link Ranker")
    p.add_argument("--graph-data-dir", default="output/artifact_graph_data")
    p.add_argument(
        "--mode",
        default="downloads",
        choices=["downloads", "random", "connectivity"],
        help="Ranking mode: downloads (by popularity), random, or connectivity (by graph degree)",
    )
    p.add_argument("--max-pairs", type=int, default=0, help="Max datasets to rank (0 for all).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--candidates-per-dataset",
        type=int,
        default=10,
        help="Number of negative candidates per dataset",
    )
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN rankings as input")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_data_dir=Path(a.graph_data_dir),
        mode=a.mode,
        max_pairs=a.max_pairs,
        seed=a.seed,
        candidates_per_dataset=a.candidates_per_dataset,
        use_gnn_data=a.use_gnn_data,
    )
