#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

from artifact_graph.models.llm_link_predictor import LLMAttributeRanker
from artifact_graph.utils.graph_builder import load_artifact_graph_from_json


def _load_summaries(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
        return data if isinstance(data, dict) else {}


def _extract_positive_edges_with_attribute(G, attribute_name: str) -> List[Tuple[str, str, float]]:
    """Extract positive edges (model-dataset pairs) that have the specified attribute."""
    edges_with_scores = []

    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")

        # Ensure we have a model-dataset pair
        if u_type == "model" and v_type == "dataset":
            model, dataset = u, v
        elif v_type == "model" and u_type == "dataset":
            model, dataset = v, u
        else:
            continue

        # Check if the attribute exists
        if attribute_name in data:
            score = data[attribute_name]
            # Handle complex score structures
            if isinstance(score, dict):
                if "score" in score:
                    score = score["score"]
                elif "value" in score:
                    score = score["value"]
                else:
                    continue

            try:
                score_float = float(score)
                edges_with_scores.append((model, dataset, score_float))
            except (ValueError, TypeError):
                continue

    return edges_with_scores


def calculate_ranking_correlation(
    llm_ranking: List[dict], true_scores: Dict[Tuple[str, str], float]
) -> Dict[str, float]:
    """Calculate correlation between LLM ranking and true scores."""
    if not llm_ranking:
        return {"kendall_tau": 0.0, "spearman_rho": 0.0, "mse": float("inf")}

    # Extract LLM predicted scores and true scores in the same order
    llm_scores = []
    actual_scores = []

    for item in llm_ranking:
        model = item["model"]
        dataset = item["dataset"]
        expected_score = item.get("expected_score", 0.5)

        edge_key = (model, dataset)
        if edge_key in true_scores:
            llm_scores.append(expected_score)
            actual_scores.append(true_scores[edge_key])

    if len(llm_scores) < 2:
        return {"kendall_tau": 0.0, "spearman_rho": 0.0, "mse": float("inf")}

    # Calculate correlations
    kendall_tau, _ = kendalltau(actual_scores, llm_scores)
    spearman_rho, _ = spearmanr(actual_scores, llm_scores)

    # Calculate MSE
    mse = np.mean((np.array(llm_scores) - np.array(actual_scores)) ** 2)

    return {
        "kendall_tau": float(kendall_tau) if not np.isnan(kendall_tau) else 0.0,
        "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
        "mse": float(mse),
    }


def run(
    graph_file: Path,
    summaries_file: Path,
    model_name: str,
    attribute_name: str,
    seed: int,
    max_batches: int,
    batch_size: int,
):
    rng = random.Random(seed)

    # Create output filename
    safe_model_name = model_name.replace("/", "_")
    output_file = Path(f"output/llm_attribute_ranking_{attribute_name}_{safe_model_name}.json")

    # Load data
    G = load_artifact_graph_from_json(json_file=str(graph_file), min_downloads=1)
    summaries = _load_summaries(summaries_file)

    # Extract positive edges with the specified attribute
    edges_with_scores = _extract_positive_edges_with_attribute(G, attribute_name)

    if not edges_with_scores:
        print(f"No edges found with attribute '{attribute_name}'")
        return

    print(f"Found {len(edges_with_scores)} edges with attribute '{attribute_name}'")

    # Create true scores dictionary for correlation calculation
    true_scores = {(model, dataset): score for model, dataset, score in edges_with_scores}

    # Shuffle and create batches
    rng.shuffle(edges_with_scores)

    # Create batches of edges to rank
    batches = []
    for i in range(0, len(edges_with_scores), batch_size):
        batch = edges_with_scores[i : i + batch_size]
        if len(batch) >= 2:  # Need at least 2 items to rank
            batches.append(batch)

    if max_batches > 0:
        batches = batches[:max_batches]

    print(f"Processing {len(batches)} batches of size up to {batch_size}")

    # Initialize ranker
    ranker = LLMAttributeRanker(model_name=model_name)

    # Collect results
    results = []
    all_correlations = []

    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)} ({len(batch)} edges)")

        try:
            # Extract edges for this batch (without scores for ranking)
            batch_edges = [(model, dataset) for model, dataset, _ in batch]

            # Get ranking from LLM
            ranking_result = ranker.rank_edges_by_attribute(
                positive_edges=batch_edges,
                G=G,
                attribute_name=attribute_name,
                summaries=summaries,
                max_edges_to_rank=batch_size,
            )

            if ranking_result:
                # Calculate correlation with true scores
                correlations = calculate_ranking_correlation(
                    ranking_result["ranked_pairs"], true_scores
                )
                all_correlations.append(correlations)

                # Add true scores to the result for analysis
                for item in ranking_result["ranked_pairs"]:
                    model = item["model"]
                    dataset = item["dataset"]
                    item["true_score"] = true_scores.get((model, dataset), 0.0)

                result_item = {
                    "batch_id": i,
                    "batch_size": len(batch),
                    "ranking_result": ranking_result,
                    "correlations": correlations,
                    "status": "Success",
                }

                print(f"  - Kendall τ: {correlations['kendall_tau']:.3f}")
                print(f"  - Spearman ρ: {correlations['spearman_rho']:.3f}")
                print(f"  - MSE: {correlations['mse']:.4f}")

            else:
                result_item = {
                    "batch_id": i,
                    "batch_size": len(batch),
                    "status": "Failed",
                    "error": "LLM ranking failed",
                }
                print("  - Ranking failed")

            results.append(result_item)

        except Exception as e:
            print(f"  - Error processing batch: {e}")
            results.append(
                {"batch_id": i, "batch_size": len(batch), "status": "Failed", "error": str(e)}
            )

    # Calculate overall metrics
    if all_correlations:
        avg_correlations = {
            metric: np.mean([c[metric] for c in all_correlations])
            for metric in all_correlations[0].keys()
        }

        print("\n--- Overall Attribute Ranking Metrics ---")
        for metric, value in avg_correlations.items():
            print(f"  - {metric.upper()}: {value:.4f}")
        print("------------------------------------------")

        # Add summary to results
        summary = {
            "total_edges": len(edges_with_scores),
            "total_batches": len(batches),
            "successful_batches": len(all_correlations),
            "failed_batches": len(batches) - len(all_correlations),
            "average_correlations": avg_correlations,
            "model_name": model_name,
            "attribute_name": attribute_name,
            "batch_size": batch_size,
        }
    else:
        print("No successful rankings produced.")
        summary = {
            "total_edges": len(edges_with_scores),
            "total_batches": len(batches),
            "successful_batches": 0,
            "failed_batches": len(batches),
            "model_name": model_name,
            "attribute_name": attribute_name,
        }

    # Save results
    output_data = {"summary": summary, "results": results}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAttribute ranking results saved to {output_file}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLM attribute ranking for model-dataset pairs"
    )
    p.add_argument("--graph-file", default="output/perfect_model_dataset_metrics.json")
    p.add_argument("--summaries-file", default="output/readme_summaries.json")
    p.add_argument(
        "--model",
        choices=["openai/gpt-4o", "openai/o3", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
        default="openai/gpt-4o",
    )
    p.add_argument(
        "--attribute", default="accuracy", help="Attribute/metric to rank by (e.g., accuracy, f1)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Maximum number of batches to process (0 for all)",
    )
    p.add_argument(
        "--batch-size", type=int, default=8, help="Number of edges to rank in each batch"
    )
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_file=Path(a.graph_file),
        summaries_file=Path(a.summaries_file),
        model_name=a.model,
        attribute_name=a.attribute,
        seed=a.seed,
        max_batches=a.max_batches,
        batch_size=a.batch_size,
    )
