#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from artifact_graph.models.llm_link_predictor import LLMBinaryRanker
from artifact_graph.utils.graph_builder import load_artifact_graph_from_json


def _load_summaries(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
        return data if isinstance(data, dict) else {}


def _extract_datasets_with_neighbors(G, min_neighbors: int = 2) -> List[str]:
    """Extract datasets that have at least min_neighbors connected models."""
    datasets = []
    for node in G.nodes():
        if G.nodes[node].get("type") == "dataset":
            neighbor_models = [
                neighbor
                for neighbor in G.neighbors(node)
                if G.nodes[neighbor].get("type") == "model"
            ]
            if len(neighbor_models) >= min_neighbors:
                datasets.append(node)
    return datasets


def calculate_ranking_metrics(
    ranked_models: List[str], neighbor_models: List[str]
) -> Dict[str, float]:
    """Calculate ranking metrics: NDCG@k, MAP, MRR."""
    if not ranked_models or not neighbor_models:
        return {"ndcg@5": 0.0, "ndcg@10": 0.0, "map": 0.0, "mrr": 0.0}

    # Create relevance scores (1 for neighbors, 0 for others)
    relevance_scores = [1 if model in neighbor_models else 0 for model in ranked_models]

    def dcg_at_k(scores: List[int], k: int) -> float:
        """Calculate Discounted Cumulative Gain at k."""
        scores_k = scores[:k]
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores_k))

    def ndcg_at_k(scores: List[int], k: int) -> float:
        """Calculate Normalized DCG at k."""
        dcg = dcg_at_k(scores, k)
        ideal_scores = sorted(scores, reverse=True)
        idcg = dcg_at_k(ideal_scores, k)
        return dcg / idcg if idcg > 0 else 0.0

    def average_precision(scores: List[int]) -> float:
        """Calculate Average Precision."""
        if not any(scores):
            return 0.0

        precision_at_k = []
        relevant_count = 0
        for i, score in enumerate(scores):
            if score == 1:
                relevant_count += 1
                precision_at_k.append(relevant_count / (i + 1))

        return sum(precision_at_k) / len(neighbor_models) if neighbor_models else 0.0

    def mean_reciprocal_rank(scores: List[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, score in enumerate(scores):
            if score == 1:
                return 1.0 / (i + 1)
        return 0.0

    return {
        "ndcg@5": ndcg_at_k(relevance_scores, 5),
        "ndcg@10": ndcg_at_k(relevance_scores, 10),
        "map": average_precision(relevance_scores),
        "mrr": mean_reciprocal_rank(relevance_scores),
    }


def run(
    graph_file: Path,
    summaries_file: Path,
    model_name: str,
    seed: int,
    max_datasets: int,
    num_negative_samples: int,
    max_models_to_rank: int,
):
    rng = random.Random(seed)

    # Create output filename
    safe_model_name = model_name.replace("/", "_")
    output_file = Path(f"output/llm_ranking_results_{safe_model_name}.json")

    # Load data
    G = load_artifact_graph_from_json(json_file=str(graph_file), min_downloads=1)
    summaries = _load_summaries(summaries_file)

    # Find datasets with sufficient neighbors for ranking
    candidate_datasets = _extract_datasets_with_neighbors(G, min_neighbors=2)

    if max_datasets > 0:
        candidate_datasets = rng.sample(
            candidate_datasets, min(max_datasets, len(candidate_datasets))
        )

    print(f"Selected {len(candidate_datasets)} datasets for ranking evaluation")

    # Initialize ranker
    ranker = LLMBinaryRanker(model_name=model_name)

    # Collect results
    results = []
    all_metrics = []

    for i, dataset_name in enumerate(candidate_datasets):
        print(f"\nProcessing dataset {i+1}/{len(candidate_datasets)}: {dataset_name}")

        try:
            # Get ranking from LLM
            ranking_result = ranker.rank_models_for_dataset(
                dataset_name=dataset_name,
                G=G,
                summaries=summaries,
                num_negative_samples=num_negative_samples,
                max_models_to_rank=max_models_to_rank,
            )

            if ranking_result:
                # Calculate ranking metrics
                ranked_models = ranking_result["ranked_models"]
                neighbor_models = ranking_result["neighbor_models"]

                metrics = calculate_ranking_metrics(ranked_models, neighbor_models)
                all_metrics.append(metrics)

                result_item = {
                    "dataset_name": dataset_name,
                    "ranked_models": ranked_models,
                    "neighbor_models": neighbor_models,
                    "negative_models": ranking_result["negative_models"],
                    "reasoning": ranking_result["reasoning"],
                    "metrics": metrics,
                    "status": "Success",
                }

                print(
                    f"  - Neighbors: {len(neighbor_models)}, Negatives: {len(ranking_result['negative_models'])}"
                )
                print(f"  - NDCG@5: {metrics['ndcg@5']:.3f}, MAP: {metrics['map']:.3f}")

            else:
                result_item = {
                    "dataset_name": dataset_name,
                    "status": "Failed",
                    "error": "LLM ranking failed",
                }
                print("  - Ranking failed")

            results.append(result_item)

        except Exception as e:
            print(f"  - Error processing dataset: {e}")
            results.append({"dataset_name": dataset_name, "status": "Failed", "error": str(e)})

    # Calculate overall metrics
    if all_metrics:
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0].keys()
        }

        print("\n--- Overall Ranking Metrics ---")
        for metric, value in avg_metrics.items():
            print(f"  - {metric.upper()}: {value:.4f}")
        print("--------------------------------")

        # Add summary to results
        summary = {
            "total_datasets": len(candidate_datasets),
            "successful_rankings": len(all_metrics),
            "failed_rankings": len(candidate_datasets) - len(all_metrics),
            "average_metrics": avg_metrics,
            "model_name": model_name,
            "num_negative_samples": num_negative_samples,
            "max_models_to_rank": max_models_to_rank,
        }
    else:
        print("No successful rankings produced.")
        summary = {
            "total_datasets": len(candidate_datasets),
            "successful_rankings": 0,
            "failed_rankings": len(candidate_datasets),
            "model_name": model_name,
        }

    # Save results
    output_data = {"summary": summary, "results": results}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nRanking results saved to {output_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LLM model ranking for datasets")
    p.add_argument("--graph-file", default="output/perfect_model_dataset_metrics.json")
    p.add_argument("--summaries-file", default="output/readme_summaries.json")
    p.add_argument(
        "--model",
        choices=["openai/gpt-4o", "openai/o3", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
        default="openai/gpt-4o",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-datasets",
        type=int,
        default=5,
        help="Maximum number of datasets to evaluate (0 for all)",
    )
    p.add_argument(
        "--num-negative-samples",
        type=int,
        default=5,
        help="Number of negative models to sample for each dataset",
    )
    p.add_argument(
        "--max-models-to-rank",
        type=int,
        default=10,
        help="Maximum number of models to rank per dataset",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_file=Path(a.graph_file),
        summaries_file=Path(a.summaries_file),
        model_name=a.model,
        seed=a.seed,
        max_datasets=a.max_datasets,
        num_negative_samples=a.num_negative_samples,
        max_models_to_rank=a.max_models_to_rank,
    )
