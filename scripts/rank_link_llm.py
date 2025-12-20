#!/usr/bin/env python3
"""
Script for LLM-based link ranking.
"""

import argparse
import concurrent.futures
import json
from functools import partial
from pathlib import Path

from tqdm import tqdm

from artifact_graph.models.llm_link_ranker import LLMLinkRanker
from artifact_graph.utils.evaluation_utils import (
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import prepare_link_ranker_dataset


def rank_single_dataset(
    dataset_id: int,
    positive_models: list,
    negative_candidates: list,
    ranker: LLMLinkRanker,
    G,
    node_metadata: dict,
) -> dict:
    """Rank models for a single dataset."""
    result = ranker.rank(
        dataset_id=dataset_id,
        positive_models=positive_models,
        negative_candidates=negative_candidates,
        G=G,
        node_metadata=node_metadata,
    )

    return result


def run(
    graph_data_dir: Path,
    model_name: str,
    hops: int,
    use_info: bool,
    seed: int = 42,
    candidates_per_dataset: int = 10,
    max_pairs: int = 0,
    max_workers: int = 4,
    use_gnn_data: bool = False,
):
    safe_model_name = model_name.replace("/", "_")
    output_file = Path(
        f"output/final_results/llm_link_rankings_{hops}hop_{safe_model_name}{'_gnn' if use_gnn_data else ''}.json"
    )

    G, node_metadata, _ = load_nx_graph(graph_data_dir=str(graph_data_dir))
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

    if not ranking_data:
        print("No ranking data available")
        return

    # Limit the number of datasets if max_pairs is specified
    if max_pairs > 0:
        ranking_items = list(ranking_data.items())[:max_pairs]
        ranking_data = dict(ranking_items)

    print(f"Total datasets to rank: {len(ranking_data)} (hops={hops}, use_info={use_info})")
    print(f"Using {max_workers} parallel workers for ranking")

    ranker = LLMLinkRanker(model_name=model_name, hop_number=hops, use_info=use_info)

    # Create ranking function with fixed parameters
    rank_func = partial(rank_single_dataset, ranker=ranker, G=G, node_metadata=node_metadata)

    out_rows = []

    # Prepare tasks for parallel processing
    tasks = list(ranking_data.items())

    # Process rankings in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(rank_func, dataset_id, positive_models, negative_candidates): dataset_id
            for dataset_id, (positive_models, negative_candidates) in tasks
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Ranking datasets",
        ):
            try:
                result = future.result()
                out_rows.append(result)

            except Exception as e:
                dataset_id = future_to_task[future]
                print(f"Error ranking dataset {dataset_id}: {e}")
                # Add failed result
                out_rows.append(None)

    # Calculate evaluation metrics
    print("\n--- Link Ranking Metrics ---")

    valid_results = [r for r in out_rows if r and r.get("ranked_model_ids")]
    if valid_results:
        # Calculate metrics for different K values
        k_values = [1, 3, 5, 10]

        all_metrics = {f"recall@{k}": [] for k in k_values}
        all_metrics.update({f"precision@{k}": [] for k in k_values})

        for result in valid_results:
            # Use the actual field names from the result
            if "ranked_model_ids" in result:
                predicted_model_ids = result["ranked_model_ids"]
                positive_models = set(result["positive_models"])

                # Calculate metrics for different K values
                for k in k_values:
                    recall_k = calculate_recall_at_k(predicted_model_ids, positive_models, k)
                    precision_k = calculate_precision_at_k(predicted_model_ids, positive_models, k)

                    all_metrics[f"recall@{k}"].append(recall_k)
                    all_metrics[f"precision@{k}"].append(precision_k)

        # Print average metrics
        for metric_name, values in all_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                print(f"  - {metric_name.upper()}: {avg_value:.4f}")

        print(f"  - Valid rankings: {len(valid_results)}/{len(out_rows)}")
    else:
        print("No valid rankings produced.")

    print("------------------------------")

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
    print(f"\nRankings saved to {output_file}")


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
        help="Number of hops for neighborhood context",
    )
    p.add_argument(
        "--no-info", action="store_false", dest="use_info", help="Disable using model/dataset info"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--candidates-per-dataset",
        type=int,
        default=10,
        help="Number of negative candidates per dataset",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=500000000,
        help="Cap the number of datasets to rank (0 for no limit)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for LLM calls (default: 4)",
    )
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN rankings as input")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_data_dir=Path(a.graph_data_dir),
        model_name=a.model,
        hops=a.hops,
        use_info=a.use_info,
        seed=a.seed,
        candidates_per_dataset=a.candidates_per_dataset,
        max_pairs=a.max_pairs,
        max_workers=a.max_workers,
        use_gnn_data=a.use_gnn_data,
    )
