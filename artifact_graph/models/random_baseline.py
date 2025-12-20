#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


class RandomBaseline:
    """
    Random baseline for comparison. Makes random predictions/rankings.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def predict(
        self,
        edge_pairs: List[Tuple[str, str]],
        G=None,
        mode="simple",
        summaries: dict | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Random binary predictions."""
        results = []
        for model_name, dataset_name in edge_pairs:
            prediction = self.rng.choice([True, False])
            result = {
                "prediction": prediction,
                "reason": "Random baseline prediction",
                "confidence": self.rng.uniform(0.1, 0.9),
            }
            results.append(result)
        return results

    def rank_models_for_dataset(
        self,
        dataset_name: str,
        G,
        summaries: dict | None = None,
        num_negative_samples: int = 5,
        max_models_to_rank: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Random model ranking."""
        try:
            # Get models like the download-based ranker
            neighbor_models = []
            all_models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]

            for neighbor in G.neighbors(dataset_name):
                if G.nodes[neighbor].get("type") == "model":
                    neighbor_models.append(neighbor)

            connected_models = set(neighbor_models)
            unconnected_models = [m for m in all_models if m not in connected_models]

            actual_negative_samples = min(num_negative_samples, len(unconnected_models))
            negative_models = self.rng.sample(unconnected_models, actual_negative_samples)

            all_models_to_rank = neighbor_models + negative_models
            if len(all_models_to_rank) > max_models_to_rank:
                all_models_to_rank = self.rng.sample(all_models_to_rank, max_models_to_rank)

            # Random shuffle
            ranked_models = all_models_to_rank.copy()
            self.rng.shuffle(ranked_models)

            return {
                "ranked_models": ranked_models,
                "reasoning": "Random baseline ranking",
                "dataset_name": dataset_name,
                "neighbor_models": neighbor_models,
                "negative_models": negative_models,
                "total_models_ranked": len(all_models_to_rank),
            }

        except Exception as e:
            print(f"Error in random ranking for dataset {dataset_name}: {e}")
            return None

    def rank_edges_by_attribute(
        self,
        positive_edges: List[Tuple[str, str]],
        G,
        attribute_name: str,
        summaries: dict | None = None,
        max_edges_to_rank: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Random attribute ranking."""
        try:
            edges_to_rank = (
                positive_edges[:max_edges_to_rank]
                if len(positive_edges) > max_edges_to_rank
                else positive_edges
            )

            # Random shuffle
            shuffled_edges = edges_to_rank.copy()
            self.rng.shuffle(shuffled_edges)

            ranked_pairs = []
            for rank, (model, dataset) in enumerate(shuffled_edges, 1):
                ranked_pairs.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "rank": rank,
                        "expected_score": self.rng.uniform(0.1, 0.9),
                    }
                )

            return {
                "ranked_pairs": ranked_pairs,
                "reasoning": "Random baseline ranking",
                "attribute_name": attribute_name,
                "total_edges_ranked": len(edges_to_rank),
                "original_edges_count": len(positive_edges),
            }

        except Exception as e:
            print(f"Error in random attribute ranking: {e}")
            return None
