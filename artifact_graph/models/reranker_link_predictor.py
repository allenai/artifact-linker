"""Link prediction using a reranker / embedding-similarity model.

For each (model, dataset) pair the predictor:
1. Builds a *query* text from the dataset (+ optional 1-hop neighbor context).
2. Builds a *document* text from the model (+ optional 1-hop neighbor context).
3. Calls ``RerankerClient.score_single`` to obtain a continuous relevance score.
4. Derives a boolean prediction via ``score >= threshold``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from artifact_graph.utils.reranker_client import RerankerClient

# Neighbor tuple: (name, info)
_NeighborInfo = Tuple[str, Optional[str]]


class RerankerLinkPredictor:
    """Link predictor backed by a reranker / embedding-similarity model."""

    def __init__(
        self,
        reranker_model: str = "jina/jinaai/jina-reranker-v2-base-multilingual",
        hop_number: int = 0,
        use_info: bool = True,
        threshold: float = 0.5,
        max_neighbors: int = 10,
    ):
        self.reranker = RerankerClient.create(reranker_model)
        self.hop_number = hop_number
        self.use_info = use_info
        self.threshold = threshold
        self.max_neighbors = max_neighbors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pair(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
    ) -> Tuple[str, str]:
        """Build (query, document) text pair for a (model, dataset) edge.

        This is separated from ``predict`` so that the runner can collect all
        pairs first and then call ``reranker.score_pairs`` in a single batch.
        """
        meta = node_metadata or {}
        query = self._build_dataset_text(dataset_id, model_id, G, meta)
        document = self._build_model_text(model_id, dataset_id, G, meta)
        return query, document

    def predict(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
    ) -> Optional[dict]:
        """Predict a single (model, dataset) pair (fallback path).

        Returns dict with ``prediction`` (bool), ``score`` (float), ``reason``.
        Prefer ``build_pair`` + ``reranker.score_pairs`` for batch usage.
        """
        try:
            query, document = self.build_pair(model_id, dataset_id, G, node_metadata)
            score = self.reranker.score_single(query, document)
            prediction = score >= self.threshold

            return {
                "prediction": prediction,
                "score": float(score),
                "reason": f"reranker score={score:.4f}",
            }
        except Exception as e:
            m_name = node_metadata.get(model_id, {}).get("name", str(model_id))
            d_name = node_metadata.get(dataset_id, {}).get("name", str(dataset_id))
            print(f"Error in reranker prediction for ({m_name}, {d_name}): {e}")
            return None

    # ------------------------------------------------------------------
    # Text construction
    # ------------------------------------------------------------------

    def _build_dataset_text(
        self, dataset_id: int, exclude_model_id: int, G: nx.Graph, meta: dict,
    ) -> str:
        """Build query text from a dataset node (+ optional 1-hop model neighbours)."""
        name = meta.get(dataset_id, {}).get("name", str(dataset_id))
        parts = [f"Which model should be evaluated on this dataset ({name})?"]
        if self.use_info:
            info = meta.get(dataset_id, {}).get("info")
            if info:
                parts.append(info)

        if self.hop_number > 0 and G:
            neighbors = self._collect_neighbors(
                G, meta, dataset_id, target_type="model",
                exclude_ids={exclude_model_id}, max_n=self.max_neighbors,
            )
            if neighbors:
                nbr_strs = []
                for n_name, n_info in neighbors:
                    s = n_name
                    if self.use_info and n_info:
                        s += f" ({n_info})"
                    nbr_strs.append(s)
                parts.append("Models evaluated on this dataset: " + ", ".join(nbr_strs))

        return ". ".join(parts)

    def _build_model_text(
        self, model_id: int, exclude_dataset_id: int, G: nx.Graph, meta: dict,
    ) -> str:
        """Build document text from a model node (+ optional 1-hop dataset neighbours)."""
        name = meta.get(model_id, {}).get("name", str(model_id))
        parts = [f"Model: {name}"]
        if self.use_info:
            info = meta.get(model_id, {}).get("info")
            if info:
                parts.append(info)

        if self.hop_number > 0 and G:
            neighbors = self._collect_neighbors(
                G, meta, model_id, target_type="dataset",
                exclude_ids={exclude_dataset_id}, max_n=self.max_neighbors,
            )
            if neighbors:
                nbr_strs = []
                for n_name, n_info in neighbors:
                    s = n_name
                    if self.use_info and n_info:
                        s += f" ({n_info})"
                    nbr_strs.append(s)
                parts.append("Evaluated on datasets: " + ", ".join(nbr_strs))

        return ". ".join(parts)

    @staticmethod
    def _collect_neighbors(
        G: nx.Graph,
        meta: dict,
        node_id: int,
        target_type: str,
        exclude_ids: Set[int] | None = None,
        max_n: int = 10,
    ) -> List[_NeighborInfo]:
        exclude = exclude_ids or set()
        neighbors: List[_NeighborInfo] = []
        for nbr_id in G.neighbors(node_id):
            if len(neighbors) >= max_n:
                break
            if nbr_id in exclude:
                continue
            if G.nodes[nbr_id].get("type") != target_type:
                continue
            neighbors.append((
                meta.get(nbr_id, {}).get("name", str(nbr_id)),
                meta.get(nbr_id, {}).get("info"),
            ))
        return neighbors
