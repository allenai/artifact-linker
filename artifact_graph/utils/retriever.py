#!/usr/bin/env python3
"""
Embedding-based retrieval utilities for RAG-based prediction and ranking.

Uses cosine similarity on node embeddings to score and retrieve candidates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class CandidateRetriever:
    """Retrieves top-k candidates using embedding cosine similarity."""

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        top_k: int = 100,
    ):
        """
        Initialize retriever.

        Args:
            embeddings: Node embeddings array (shape: [num_nodes, embed_dim])
            top_k: Number of candidates to retrieve
        """
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
        top_k: int = 100,
        **kwargs,
    ) -> "CandidateRetriever":
        """Load retriever from data directory.

        Extra **kwargs (e.g. strategy) are accepted for backward compatibility
        but ignored — only embedding-based retrieval is used.
        """
        data_dir = Path(data_dir)

        embeddings = None
        emb_path = data_dir / "node_embeddings.npy"
        if emb_path.exists():
            arr = np.load(emb_path, allow_pickle=False)
            # Handle structured arrays with (node_id, embedding) fields
            if hasattr(arr.dtype, "names") and arr.dtype.names and "embedding" in arr.dtype.names:
                embeddings = np.array(arr["embedding"], dtype=np.float32)
            else:
                embeddings = arr

        return cls(embeddings=embeddings, top_k=top_k)

    def score_all(
        self,
        query_id: int,
        candidate_ids: List[int],
        G=None,
    ) -> List[Tuple[int, float]]:
        """
        Compute cosine similarity scores for ALL candidates (no top-k truncation).

        Args:
            query_id: Query node ID (e.g., dataset ID)
            candidate_ids: List of candidate node IDs (e.g., model IDs)
            G: Unused, kept for interface compatibility.

        Returns:
            List of (candidate_id, score) tuples, sorted by score descending.
        """
        if self.normalized_embeddings is None:
            import random
            scores = [(cid, random.random()) for cid in candidate_ids]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores

        if query_id >= len(self.normalized_embeddings):
            return [(cid, 0.0) for cid in candidate_ids]

        query_emb = self.normalized_embeddings[query_id]

        scores = []
        for cid in candidate_ids:
            if cid < len(self.normalized_embeddings):
                sim = float(np.dot(query_emb, self.normalized_embeddings[cid]))
            else:
                sim = 0.0
            scores.append((cid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def retrieve(
        self,
        query_id: int,
        candidate_ids: List[int],
        G=None,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k candidates for a query.

        Returns:
            List of (candidate_id, score) tuples, sorted by score descending,
            truncated to top_k.
        """
        return self.score_all(query_id, candidate_ids, G)[: self.top_k]


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
