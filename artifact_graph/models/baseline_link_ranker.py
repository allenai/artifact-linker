#!/usr/bin/env python3
"""Baseline link ranker using various graph-based heuristics."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


class BaselineLinkRanker:
    """Ranks candidate models for a dataset based on different baseline strategies."""

    VALID_MODES = [
        "downloads",
        "random",
        "connectivity",
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
        "resource_allocation",
        "katz",
    ]

    def __init__(self, mode: str = "downloads", seed: int = 42, **kwargs):
        """
        Initialize baseline link ranker.

        Args:
            mode: Ranking strategy
                - "downloads": Rank by download counts (highest first)
                - "random": Random ranking
                - "connectivity": Rank by node degree (highest first)
                - "common_neighbors": Rank by common neighbors count
                - "jaccard": Rank by Jaccard coefficient
                - "adamic_adar": Rank by Adamic-Adar index
                - "preferential_attachment": Rank by degree product
                - "resource_allocation": Rank by resource allocation index
                - "katz": Rank by Katz-like path score
            seed: Random seed for reproducible rankings
        """
        self.mode = mode
        self.seed = seed
        self.beta = kwargs.get("beta", 0.1)  # Katz parameter

        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}. Must be one of {self.VALID_MODES}.")

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
            G: The NetworkX graph.
            node_metadata: Dictionary with node metadata.

        Returns:
            A dictionary containing the ranked list of model IDs and scores.
        """
        try:
            all_models = list(set(positive_models + negative_candidates))

            # Compute scores for all models
            scores = self._compute_scores(dataset_id, all_models, G, node_metadata)

            # Rank by score (descending), with tie-breaking shuffle
            ranked = self._rank_with_tiebreaking(scores)

            return {
                "dataset_id": int(dataset_id),
                "positive_models": [int(m) for m in positive_models],
                "ranked_model_ids": [int(m) for m, _ in ranked],
                "scores": {int(m): float(s) for m, s in ranked},
                "reasoning": f"Ranked {len(ranked)} models by {self.mode}.",
            }

        except Exception as e:
            print(f"Error ranking links with baseline ({self.mode}): {e}")
            return None

    def _compute_scores(
        self,
        dataset_id: int,
        models: List[int],
        G: nx.Graph,
        node_metadata: dict,
    ) -> List[tuple]:
        """Compute scores for all models based on the current mode."""
        score_fn = {
            "downloads": lambda m: self._get_downloads(m, G, node_metadata),
            "random": lambda m: random.Random(self.seed + m).random(),
            "connectivity": lambda m: float(G.degree(m)) if m in G else 0.0,
            "common_neighbors": lambda m: self._common_neighbors(m, dataset_id, G),
            "jaccard": lambda m: self._jaccard(m, dataset_id, G),
            "adamic_adar": lambda m: self._adamic_adar(m, dataset_id, G),
            "preferential_attachment": lambda m: self._pref_attach(m, dataset_id, G),
            "resource_allocation": lambda m: self._resource_alloc(m, dataset_id, G),
            "katz": lambda m: self._katz(m, dataset_id, G),
        }[self.mode]

        return [(m, score_fn(m)) for m in models]

    def _rank_with_tiebreaking(self, scores: List[tuple]) -> List[tuple]:
        """Rank scores descending with random tie-breaking."""
        groups = defaultdict(list)
        for model_id, score in scores:
            groups[score].append(model_id)

        rng = random.Random(self.seed)
        result = []
        for score in sorted(groups.keys(), reverse=True):
            group = groups[score]
            rng.shuffle(group)
            result.extend((m, score) for m in group)

        return result

    # =========================================================================
    # Score Functions
    # =========================================================================

    def _get_downloads(self, model_id: int, G: nx.Graph, node_metadata: dict) -> float:
        """Get download count for a model."""
        if model_id in G.nodes:
            downloads = G.nodes[model_id].get("downloads", 0)
            if downloads:
                return float(downloads)
        node_data = node_metadata.get(model_id, {})
        downloads = node_data.get("downloads", 0) or node_data.get("info", {}).get("downloads", 0)
        return float(downloads)

    def _common_neighbors(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Count common neighbors between model and dataset."""
        if model_id not in G or dataset_id not in G:
            return 0.0
        m_neigh = set(G.neighbors(model_id))
        d_neigh = set(G.neighbors(dataset_id))
        return float(len(m_neigh & d_neigh))

    def _jaccard(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate Jaccard coefficient."""
        if model_id not in G or dataset_id not in G:
            return 0.0
        m_neigh = set(G.neighbors(model_id))
        d_neigh = set(G.neighbors(dataset_id))
        union = m_neigh | d_neigh
        return len(m_neigh & d_neigh) / len(union) if union else 0.0

    def _adamic_adar(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate Adamic-Adar index."""
        if model_id not in G or dataset_id not in G:
            return 0.0
        m_neigh = set(G.neighbors(model_id))
        d_neigh = set(G.neighbors(dataset_id))
        common = m_neigh & d_neigh
        score = 0.0
        for n in common:
            deg = G.degree(n)
            if deg > 1:
                score += 1.0 / np.log(deg)
        return score

    def _pref_attach(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate preferential attachment score (product of degrees)."""
        m_deg = G.degree(model_id) if model_id in G else 0
        d_deg = G.degree(dataset_id) if dataset_id in G else 0
        return float(m_deg * d_deg)

    def _resource_alloc(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate resource allocation index."""
        if model_id not in G or dataset_id not in G:
            return 0.0
        m_neigh = set(G.neighbors(model_id))
        d_neigh = set(G.neighbors(dataset_id))
        common = m_neigh & d_neigh
        score = 0.0
        for n in common:
            deg = G.degree(n)
            if deg > 0:
                score += 1.0 / deg
        return score

    def _katz(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate simplified Katz-like score based on short paths."""
        if model_id not in G or dataset_id not in G:
            return 0.0

        # Remove direct edge if exists (we're predicting missing links)
        has_edge = G.has_edge(model_id, dataset_id)
        if has_edge:
            G.remove_edge(model_id, dataset_id)

        try:
            score = 0.0
            # Paths of length 2: through common neighbors
            m_neigh = set(G.neighbors(model_id))
            d_neigh = set(G.neighbors(dataset_id))
            path2_count = len(m_neigh & d_neigh)
            score += (self.beta ** 2) * path2_count

            # Paths of length 3 and 4 (simplified approximation)
            for path_len in [3, 4]:
                try:
                    if nx.has_path(G, model_id, dataset_id):
                        sp = nx.shortest_path_length(G, model_id, dataset_id)
                        if sp == path_len:
                            score += self.beta ** path_len
                except Exception:
                    pass

            return score
        finally:
            if has_edge:
                G.add_edge(model_id, dataset_id)
