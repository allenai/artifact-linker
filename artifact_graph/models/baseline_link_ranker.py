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
        "matrix_factorization",
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
            "matrix_factorization": lambda m: self._mf(m, dataset_id, G),
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
        downloads = node_data.get("downloads", 0)
        if not downloads:
            info = node_data.get("info", {})
            if isinstance(info, dict):
                downloads = info.get("downloads", 0)
        return float(downloads or 0)

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

    def _precompute_katz(self, G: nx.Graph):
        """Precompute Katz score matrix using sparse matrix powers (one-time cost)."""
        nodes = sorted(G.nodes())
        self._katz_node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float64)
        A.setdiag(0)
        A.eliminate_zeros()

        A2 = A.dot(A)
        A3 = A2.dot(A)
        A4 = A3.dot(A)

        self._katz_matrix = (
            (self.beta ** 2) * A2
            + (self.beta ** 3) * A3
            + (self.beta ** 4) * A4
        )
        self._katz_matrix.setdiag(0)
        self._katz_matrix.eliminate_zeros()
        print(f"Precomputed Katz matrix (ranker): {n} nodes, beta={self.beta}, nnz={self._katz_matrix.nnz}")

    def _katz(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Look up precomputed Katz score for a node pair."""
        if not hasattr(self, "_katz_matrix") or self._katz_matrix is None:
            self._precompute_katz(G)

        i = self._katz_node_to_idx.get(model_id)
        j = self._katz_node_to_idx.get(dataset_id)
        if i is None or j is None:
            return 0.0
        return float(self._katz_matrix[i, j])

    def _precompute_mf(self, G: nx.Graph):
        """Precompute Matrix Factorization via truncated SVD (one-time cost)."""
        from scipy.sparse.linalg import svds

        nodes = sorted(G.nodes())
        self._mf_node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float64)
        A.setdiag(0)
        A.eliminate_zeros()

        k = min(64, min(A.shape) - 1)
        U, S, Vt = svds(A.astype(np.float64), k=k)
        self._mf_U_S = U * S[np.newaxis, :]  # (n, k)
        self._mf_Vt = Vt  # (k, n)
        print(f"Precomputed MF (ranker): {n} nodes, rank={k}")

    def _mf(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Look up precomputed MF score for a node pair."""
        if not hasattr(self, "_mf_U_S") or self._mf_U_S is None:
            self._precompute_mf(G)

        i = self._mf_node_to_idx.get(model_id)
        j = self._mf_node_to_idx.get(dataset_id)
        if i is None or j is None:
            return 0.0
        return float(self._mf_U_S[i] @ self._mf_Vt[:, j])
