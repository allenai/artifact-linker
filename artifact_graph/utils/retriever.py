#!/usr/bin/env python3
"""
Retrieval utilities for RAG-based prediction and ranking.

Supports multiple retrieval strategies:
- embedding: Cosine similarity on node embeddings
- bm25: BM25 on node descriptions/info
- heuristic: Graph-based heuristics (common neighbors, etc.)
- hybrid: Combination of multiple strategies
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class CandidateRetriever:
    """Retrieves top-k candidates using various strategies."""

    def __init__(
        self,
        strategy: str = "embedding",
        embeddings: Optional[np.ndarray] = None,
        node_metadata: Optional[Dict] = None,
        top_k: int = 100,
    ):
        """
        Initialize retriever.

        Args:
            strategy: Retrieval strategy ("embedding", "bm25", "heuristic", "hybrid")
            embeddings: Node embeddings array (shape: [num_nodes, embed_dim])
            node_metadata: Node metadata dictionary
            top_k: Number of candidates to retrieve
        """
        self.strategy = strategy
        self.embeddings = embeddings
        self.node_metadata = node_metadata or {}
        self.top_k = top_k

        # Normalize embeddings for cosine similarity
        if embeddings is not None:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            self.normalized_embeddings = embeddings / norms
        else:
            self.normalized_embeddings = None

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        strategy: str = "embedding",
        top_k: int = 100,
    ) -> "CandidateRetriever":
        """Load retriever from data directory."""
        data_dir = Path(data_dir)

        # Load embeddings
        embeddings = None
        emb_path = data_dir / "node_embeddings.npy"
        if emb_path.exists():
            embeddings = np.load(emb_path)

        # Load metadata
        node_metadata = {}
        meta_path = data_dir / "node_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                node_metadata = {int(k): v for k, v in json.load(f).items()}

        return cls(
            strategy=strategy,
            embeddings=embeddings,
            node_metadata=node_metadata,
            top_k=top_k,
        )

    def retrieve(
        self,
        query_id: int,
        candidate_ids: List[int],
        G=None,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k candidates for a query.

        Args:
            query_id: Query node ID (e.g., dataset ID)
            candidate_ids: List of candidate node IDs (e.g., model IDs)
            G: NetworkX graph (optional, for heuristic strategies)

        Returns:
            List of (candidate_id, score) tuples, sorted by score descending.
        """
        if self.strategy == "embedding":
            return self._retrieve_by_embedding(query_id, candidate_ids)
        elif self.strategy == "bm25":
            return self._retrieve_by_bm25(query_id, candidate_ids)
        elif self.strategy == "heuristic":
            return self._retrieve_by_heuristic(query_id, candidate_ids, G)
        elif self.strategy == "hybrid":
            return self._retrieve_hybrid(query_id, candidate_ids, G)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _retrieve_by_embedding(
        self,
        query_id: int,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Retrieve using cosine similarity on embeddings."""
        if self.normalized_embeddings is None:
            # Fallback: return all candidates with random scores
            import random
            scores = [(cid, random.random()) for cid in candidate_ids]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[: self.top_k]

        # Get query embedding
        if query_id >= len(self.normalized_embeddings):
            return [(cid, 0.0) for cid in candidate_ids[: self.top_k]]

        query_emb = self.normalized_embeddings[query_id]

        # Compute similarities for all candidates
        scores = []
        for cid in candidate_ids:
            if cid < len(self.normalized_embeddings):
                cand_emb = self.normalized_embeddings[cid]
                sim = float(np.dot(query_emb, cand_emb))
            else:
                sim = 0.0
            scores.append((cid, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.top_k]

    def _retrieve_by_bm25(
        self,
        query_id: int,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Retrieve using BM25 on text descriptions."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # Fallback to embedding
            return self._retrieve_by_embedding(query_id, candidate_ids)

        # Get query text
        query_info = self.node_metadata.get(query_id, {})
        query_text = f"{query_info.get('name', '')} {query_info.get('info', '')}"
        query_tokens = query_text.lower().split()

        # Build corpus from candidates
        corpus = []
        for cid in candidate_ids:
            cand_info = self.node_metadata.get(cid, {})
            cand_text = f"{cand_info.get('name', '')} {cand_info.get('info', '')}"
            corpus.append(cand_text.lower().split())

        if not corpus or not query_tokens:
            return [(cid, 0.0) for cid in candidate_ids[: self.top_k]]

        # Compute BM25 scores
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens)

        scores = list(zip(candidate_ids, bm25_scores))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.top_k]

    def _retrieve_by_heuristic(
        self,
        query_id: int,
        candidate_ids: List[int],
        G,
    ) -> List[Tuple[int, float]]:
        """Retrieve using graph heuristics (common neighbors + downloads)."""
        if G is None:
            return self._retrieve_by_embedding(query_id, candidate_ids)

        query_neighbors = set(G.neighbors(query_id)) if query_id in G else set()

        scores = []
        for cid in candidate_ids:
            score = 0.0

            # Common neighbors
            if cid in G:
                cand_neighbors = set(G.neighbors(cid))
                common = len(query_neighbors & cand_neighbors)
                score += common * 10  # Weight common neighbors

            # Downloads (popularity)
            downloads = self.node_metadata.get(cid, {}).get("downloads", 0)
            if downloads:
                score += np.log1p(downloads)

            scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.top_k]

    def _retrieve_hybrid(
        self,
        query_id: int,
        candidate_ids: List[int],
        G,
    ) -> List[Tuple[int, float]]:
        """Hybrid retrieval combining multiple strategies."""
        # Get scores from each strategy
        emb_scores = dict(self._retrieve_by_embedding(query_id, candidate_ids))
        heur_scores = dict(self._retrieve_by_heuristic(query_id, candidate_ids, G))

        # Normalize scores to [0, 1]
        def normalize(score_dict):
            if not score_dict:
                return score_dict
            max_s = max(score_dict.values())
            min_s = min(score_dict.values())
            if max_s == min_s:
                return {k: 0.5 for k in score_dict}
            return {k: (v - min_s) / (max_s - min_s) for k, v in score_dict.items()}

        emb_norm = normalize(emb_scores)
        heur_norm = normalize(heur_scores)

        # Combine with weights
        combined = []
        for cid in candidate_ids:
            score = 0.7 * emb_norm.get(cid, 0) + 0.3 * heur_norm.get(cid, 0)
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[: self.top_k]


def filter_candidates_by_retrieval(
    query_id: int,
    candidate_ids: List[int],
    retriever: CandidateRetriever,
    G=None,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Filter candidates using retrieval.

    Returns:
        Tuple of (filtered_candidate_ids, retrieval_scores)
    """
    retrieved = retriever.retrieve(query_id, candidate_ids, G)
    filtered_ids = [cid for cid, _ in retrieved]
    scores = {cid: score for cid, score in retrieved}
    return filtered_ids, scores
