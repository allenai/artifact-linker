#!/usr/bin/env python3
"""
Script for LLM-based attribute ranking.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path

from tqdm import tqdm

from artifact_graph.models.llm_attribute_ranker import LLMAttributeRanker
from artifact_graph.utils.evaluation_utils import (
    calculate_map_continuous,
    calculate_ndcg_standard,
    calculate_ranking_correlation,
)
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_utils import prepare_attribute_ranker_dataset


def run(
    graph_data_dir: Path,
    model_name: str,
    hops: int,
    use_info: bool,
    metric_name: str | None,
    max_models_per_dataset: int = 20,
    max_workers: int = 4,
    use_gnn_data: bool = False,
):
    safe_model_name = model_name.replace("/", "_")
    metric_str = metric_name if metric_name else "all"
    output_file = Path(
        f"output/final_results/llm_attribute_rankings_{hops}hop_{safe_model_name}_{metric_str}{'_gnn' if use_gnn_data else ''}.json"
    )

    G, node_metadata, edge_metadata = load_nx_graph(graph_data_dir=str(graph_data_dir))
    if use_gnn_data:
        dataset_metrics = {}
        with open("output/final_results/gnn_attribute_rankings.json", "r") as f:
            gnn_attribute_rankings = json.load(f)
        ranking_data = gnn_attribute_rankings["results"]
        ranking_data_final = {}
        for result in ranking_data:
            dataset_id = result["dataset_id"]
            ranking_data_final[dataset_id] = []
            candidate_models = result["predicted_ranking"]
            for model in candidate_models:
                model_id = model["model_id"]
                predicted_value = model["predicted_score"]
                ground_truth = model["ground_truth_score"]
                ranking_data_final[dataset_id].append((model_id, predicted_value))
                dataset_metrics[dataset_id] = result["metric_used"]
        ranking_data = ranking_data_final
    else:
        ranking_data, dataset_metrics = prepare_attribute_ranker_dataset(G, metric_name)

    if not ranking_data:
        print(f"No ranking data available for metric '{metric_str}'")
        return

    if metric_name is None:
        print(
            f"Total datasets to rank: {len(ranking_data)} (hops={hops}, use_info={use_info}, auto-selected metrics per dataset)"
        )
    else:
        print(
            f"Total datasets to rank: {len(ranking_data)} (hops={hops}, use_info={use_info}, metric={metric_name})"
        )

    ranker = LLMAttributeRanker(model_name=model_name, hop_number=hops, use_info=use_info)

    print(f"Using {max_workers} parallel workers for ranking")

    out_rows = []

    # Prepare all ranking tasks
    ranking_tasks = []
    for dataset_id, models_to_rank in ranking_data.items():
        # Chunk models if there are too many for a single dataset
        chunks = [
            models_to_rank[i : i + max_models_per_dataset]
            for i in range(0, len(models_to_rank), max_models_per_dataset)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            # Get the metric name for this specific dataset
            dataset_metric = dataset_metrics.get(dataset_id, metric_name or "accuracy")

            ranking_tasks.append(
                {
                    "dataset_id": dataset_id,
                    "models_to_rank": chunk,
                    "metric_name": dataset_metric,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                }
            )

    def rank_single_task(task):
        """Rank models for a single task."""
        try:
            result = ranker.rank(
                dataset_id=task["dataset_id"],
                models_to_rank=task["models_to_rank"],
                G=G,
                node_metadata=node_metadata,
                edge_metadata=edge_metadata,
                metric_name=task["metric_name"],
            )

            if result:
                result["chunk_index"] = task["chunk_idx"]
                result["total_chunks"] = task["total_chunks"]

            return result
        except Exception as e:
            print(f"Error in ranking task for dataset {task['dataset_id']}: {e}")
            return None

    # Execute ranking tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(rank_single_task, task) for task in ranking_tasks]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Ranking models"
        ):
            result = future.result()
            out_rows.append(result)

    # Calculate evaluation metrics
    print("\n--- LLM Attribute Ranking Metrics ---")

    valid_results = [r for r in out_rows if r and r.get("ranked_models")]
    if valid_results:
        all_kendall_tau = []
        all_spearman = []
        all_pearson_r = []
        all_weighted_kendall_tau = []
        all_rank_biased_overlap = []
        all_hit_at_1 = []
        all_hit_at_3 = []
        all_hit_at_5 = []
        all_recall_at_1 = []
        all_recall_at_3 = []
        all_recall_at_5 = []
        all_top_1_overlap = []
        all_top_3_overlap = []
        all_top_5_overlap = []
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

        for result in valid_results:
            if "ranked_models" in result:
                ranked_models = result["ranked_models"]
                if not ranked_models:
                    continue

                # Extract predicted ranking and ground truth from the unified structure
                predicted_items_with_scores = []
                ground_truth = {}

                for item in ranked_models:
                    model_id = item["model_id"]
                    dataset_id = result["dataset_id"]
                    item_key = f"{model_id}_{dataset_id}"  # Create a unique key
                    predicted_items_with_scores.append((item_key, item.get("expected_score", 0)))
                    ground_truth[item_key] = item["true_value"]

                # Calculate continuous NDCG and MAP for attribute ranking
                try:
                    # NDCG@k metrics
                    ndcg_1 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=1)
                    ndcg_3 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=3)
                    ndcg_5 = calculate_ndcg_standard(predicted_items_with_scores, ground_truth, k=5)
                    ndcg_10 = calculate_ndcg_standard(
                        predicted_items_with_scores, ground_truth, k=10
                    )
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
                    map_10 = calculate_map_continuous(
                        predicted_items_with_scores, ground_truth, k=10
                    )
                    map_full = calculate_map_continuous(predicted_items_with_scores, ground_truth)

                    all_map_1.append(map_1)
                    all_map_3.append(map_3)
                    all_map_5.append(map_5)
                    all_map_10.append(map_10)
                    all_map_full.append(map_full)

                except Exception as e:
                    print(f"Warning: Could not calculate NDCG/MAP metrics: {e}")

                # Calculate ranking correlation and advanced metrics
                try:
                    correlation_metrics = calculate_ranking_correlation(
                        predicted_items_with_scores, ground_truth
                    )
                    # Collect all new metrics
                    if "kendall_tau" in correlation_metrics:
                        all_kendall_tau.append(correlation_metrics["kendall_tau"])
                    if "spearman_rho" in correlation_metrics:
                        all_spearman.append(correlation_metrics["spearman_rho"])
                    if "pearson_r" in correlation_metrics:
                        all_pearson_r.append(correlation_metrics["pearson_r"])
                    if "weighted_kendall_tau" in correlation_metrics:
                        all_weighted_kendall_tau.append(correlation_metrics["weighted_kendall_tau"])
                    if "rank_biased_overlap" in correlation_metrics:
                        all_rank_biased_overlap.append(correlation_metrics["rank_biased_overlap"])

                    # Hit@k metrics
                    if "hit_at_1" in correlation_metrics:
                        all_hit_at_1.append(correlation_metrics["hit_at_1"])
                    if "hit_at_3" in correlation_metrics:
                        all_hit_at_3.append(correlation_metrics["hit_at_3"])
                    if "hit_at_5" in correlation_metrics:
                        all_hit_at_5.append(correlation_metrics["hit_at_5"])

                    # Recall@k metrics
                    if "recall_at_1" in correlation_metrics:
                        all_recall_at_1.append(correlation_metrics["recall_at_1"])
                    if "recall_at_3" in correlation_metrics:
                        all_recall_at_3.append(correlation_metrics["recall_at_3"])
                    if "recall_at_5" in correlation_metrics:
                        all_recall_at_5.append(correlation_metrics["recall_at_5"])

                    # Top-k overlap metrics
                    if "top_1_overlap" in correlation_metrics:
                        all_top_1_overlap.append(correlation_metrics["top_1_overlap"])
                    if "top_3_overlap" in correlation_metrics:
                        all_top_3_overlap.append(correlation_metrics["top_3_overlap"])
                    if "top_5_overlap" in correlation_metrics:
                        all_top_5_overlap.append(correlation_metrics["top_5_overlap"])

                except Exception as e:
                    print(f"Warning: Could not calculate correlation metrics: {e}")

        # Print average metrics - prioritize correlation metrics
        print("  === Primary Ranking Correlation Metrics ===")
        if all_kendall_tau:
            print(f"  - Kendall's Tau: {sum(all_kendall_tau) / len(all_kendall_tau):.4f}")
        if all_spearman:
            print(f"  - Spearman's Rho: {sum(all_spearman) / len(all_spearman):.4f}")
        if all_pearson_r:
            print(f"  - Pearson R: {sum(all_pearson_r) / len(all_pearson_r):.4f}")
        if all_weighted_kendall_tau:
            print(
                f"  - Weighted Kendall Tau: {sum(all_weighted_kendall_tau) / len(all_weighted_kendall_tau):.4f}"
            )
        if all_rank_biased_overlap:
            print(
                f"  - Rank Biased Overlap: {sum(all_rank_biased_overlap) / len(all_rank_biased_overlap):.4f}"
            )

        print("  === Hit@k and Recall@k Metrics ===")
        if all_hit_at_1:
            print(f"  - Hit@1: {sum(all_hit_at_1) / len(all_hit_at_1):.4f}")
        if all_hit_at_3:
            print(f"  - Hit@3: {sum(all_hit_at_3) / len(all_hit_at_3):.4f}")
        if all_hit_at_5:
            print(f"  - Hit@5: {sum(all_hit_at_5) / len(all_hit_at_5):.4f}")
        if all_recall_at_1:
            print(f"  - Recall@1: {sum(all_recall_at_1) / len(all_recall_at_1):.4f}")
        if all_recall_at_3:
            print(f"  - Recall@3: {sum(all_recall_at_3) / len(all_recall_at_3):.4f}")
        if all_recall_at_5:
            print(f"  - Recall@5: {sum(all_recall_at_5) / len(all_recall_at_5):.4f}")

        print("  === Top-k Overlap Metrics ===")
        if all_top_1_overlap:
            print(f"  - Top-1 Overlap: {sum(all_top_1_overlap) / len(all_top_1_overlap):.4f}")
        if all_top_3_overlap:
            print(f"  - Top-3 Overlap: {sum(all_top_3_overlap) / len(all_top_3_overlap):.4f}")
        if all_top_5_overlap:
            print(f"  - Top-5 Overlap: {sum(all_top_5_overlap) / len(all_top_5_overlap):.4f}")

        print("  === NDCG and MAP (Reference) ===")
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

        print(f"  - Valid rankings: {len(valid_results)}/{len(out_rows)}")
    else:
        print("No valid rankings produced.")

    print("----------------------------------")

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
        default=0,
        help="Number of hops for neighborhood context (0 or 1)",
    )
    p.add_argument(
        "--no-info",
        action="store_false",
        dest="use_info",
        help="Disable using model/dataset info in the prompt",
    )
    p.add_argument(
        "--metric",
        default="accuracy",
        help="Metric name to rank by (use 'auto' for most frequent metric)",
    )
    p.add_argument(
        "--max-models-per-dataset",
        type=int,
        default=10,
        help="Maximum models per dataset ranking chunk",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads for parallel processing",
    )
    p.add_argument("--use-gnn-data", action="store_true", help="Use GNN data")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()

    run(
        graph_data_dir=Path(a.graph_data_dir),
        model_name=a.model,
        hops=a.hops,
        use_info=a.use_info,
        metric_name=None,
        max_models_per_dataset=a.max_models_per_dataset,
        max_workers=a.max_workers,
        use_gnn_data=a.use_gnn_data,
    )
