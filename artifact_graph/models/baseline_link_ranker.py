#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import networkx as nx


class BaselineLinkRanker:
    """Ranks candidate models for a dataset based on different baseline strategies."""

    def __init__(self, mode: str = "downloads", seed: int = 42):
        """
        Initialize baseline link ranker.

        Args:
            mode: Ranking strategy
                - "downloads": Rank by download counts (highest first)
                - "random": Random ranking
                - "connectivity": Rank by node degree/connectivity (highest first)
            seed: Random seed for reproducible random rankings
        """
        self.mode = mode
        self.seed = seed
        if mode not in ["downloads", "random", "connectivity"]:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'downloads', 'random', or 'connectivity'."
            )

    def rank(
        self,
        dataset_id: int,
        positive_models: List[int],
        negative_candidates: List[int],
        G: nx.Graph,
        node_metadata: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Ranks a combined list of positive and negative models.

        Args:
            dataset_id: The ID of the dataset.
            positive_models: A list of model IDs known to connect to the dataset.
            negative_candidates: A list of model IDs that are candidates for connection.
            G: The NetworkX graph (for interface compatibility).
            node_metadata: Dictionary with node metadata.

        Returns:
            A dictionary containing the ranked list of model IDs.
        """
        try:
            all_models_to_rank = list(set(positive_models + negative_candidates))

            if self.mode == "downloads":
                return self._rank_by_downloads(dataset_id, all_models_to_rank, node_metadata)
            elif self.mode == "random":
                return self._rank_randomly(dataset_id, all_models_to_rank)
            elif self.mode == "connectivity":
                return self._rank_by_connectivity(dataset_id, all_models_to_rank, G)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except Exception as e:
            print(f"Error ranking links with baseline: {e}")
            return None

    def _rank_by_downloads(
        self, dataset_id: int, all_models_to_rank: List[int], node_metadata: dict
    ) -> Dict[str, Any]:
        """Rank models by download counts (highest first)."""
        models_with_downloads = []
        for model_id in all_models_to_rank:
            node_data = node_metadata.get(model_id, {})
            # Try different possible paths to downloads field
            downloads = node_data.get("downloads", 0)
            if downloads == 0:
                downloads = node_data.get("info", {}).get("downloads", 0)
            models_with_downloads.append((model_id, downloads))

        # Group models by download count and shuffle within each group
        from collections import defaultdict

        download_groups = defaultdict(list)
        for model_id, downloads in models_with_downloads:
            download_groups[downloads].append(model_id)

        # Create a local random instance for reproducibility
        rng = random.Random(self.seed)

        # Shuffle within each download group and create final ranking
        ranked_model_ids = []
        for downloads in sorted(download_groups.keys(), reverse=True):  # Highest downloads first
            group = download_groups[downloads]
            rng.shuffle(group)  # Randomly shuffle models with same download count
            ranked_model_ids.extend(group)

        return {
            "ranked_model_ids": ranked_model_ids,
            "reasoning": f"Ranked {len(ranked_model_ids)} candidate models for dataset {dataset_id} by download counts.",
        }

    def _rank_randomly(self, dataset_id: int, all_models_to_rank: List[int]) -> Dict[str, Any]:
        """Rank models randomly."""
        # Create a local random instance to ensure reproducibility
        rng = random.Random(self.seed)
        ranked_model_ids = all_models_to_rank.copy()
        rng.shuffle(ranked_model_ids)

        return {
            "ranked_model_ids": ranked_model_ids,
            "reasoning": f"Randomly ranked {len(ranked_model_ids)} candidate models for dataset {dataset_id} (seed={self.seed}).",
        }

    def _rank_by_connectivity(
        self, dataset_id: int, all_models_to_rank: List[int], G: nx.Graph
    ) -> Dict[str, Any]:
        """Rank models by their connectivity/degree in the graph (highest first)."""
        models_with_connectivity = []
        for model_id in all_models_to_rank:
            # Get the degree (number of connections) for this model
            connectivity = G.degree(model_id) if model_id in G else 0
            models_with_connectivity.append((model_id, float(connectivity)))

        # Group models by connectivity level and shuffle within each group
        from collections import defaultdict

        connectivity_groups = defaultdict(list)
        for model_id, connectivity in models_with_connectivity:
            connectivity_groups[connectivity].append(model_id)

        # Create a local random instance for reproducibility
        rng = random.Random(self.seed)

        # Shuffle within each connectivity group and create final ranking
        ranked_model_ids = []
        for connectivity in sorted(
            connectivity_groups.keys(), reverse=True
        ):  # Highest connectivity first
            group = connectivity_groups[connectivity]
            rng.shuffle(group)  # Randomly shuffle models with same connectivity
            ranked_model_ids.extend(group)

        return {
            "ranked_model_ids": ranked_model_ids,
            "reasoning": f"Ranked {len(ranked_model_ids)} candidate models for dataset {dataset_id} by graph connectivity (degree).",
        }
