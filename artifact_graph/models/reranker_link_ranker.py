"""Link ranking using a reranker / embedding-similarity model.

For each dataset the ranker:
1. Builds a *query* text from the dataset (+ optional 1-hop context).
2. Builds *document* texts for every candidate model.
3. Calls ``RerankerClient.score`` to batch-score all documents at once.
4. Returns models sorted by descending relevance score.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from artifact_graph.utils.reranker_client import RerankerClient

_NeighborInfo = Tuple[str, Optional[str]]


class RerankerLinkRanker:
    """Link ranker backed by a reranker / embedding-similarity model."""

    def __init__(
        self,
        reranker_model: str = "jina/jinaai/jina-reranker-v2-base-multilingual",
        hop_number: int = 0,
        use_info: bool = True,
        max_neighbors: int = 10,
    ):
        self.reranker = RerankerClient.create(reranker_model)
        self.hop_number = hop_number
        self.use_info = use_info
        self.max_neighbors = max_neighbors

    # ------------------------------------------------------------------
    # Public API (matches LLMLinkRanker interface)
    # ------------------------------------------------------------------

    def rank(
        self,
        dataset_id: int,
        positive_models: List[int],
        negative_candidates: List[int],
        G: nx.Graph,
        node_metadata: dict,
    ):
        """Rank models for a single dataset.

        Returns ranking result dict or *None* on failure.
        """
        try:
            meta = node_metadata or {}
            all_models = positive_models + negative_candidates
            models_set = set(all_models)
            dataset_name = meta.get(dataset_id, {}).get("name", str(dataset_id))

            # Build query from dataset
            query = self._build_dataset_text(dataset_id, models_set, G, meta)

            # Build one document per candidate model
            documents: List[str] = []
            for model_id in all_models:
                documents.append(
                    self._build_model_text(model_id, {dataset_id}, G, meta)
                )

            # Score all documents against query
            scores = self.reranker.score(query, documents)

            # Sort by score descending
            indexed_scores = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True,
            )
            ranked_model_ids = [all_models[i] for i, _ in indexed_scores]
            ranked_model_names = [
                meta.get(mid, {}).get("name", str(mid)) for mid in ranked_model_ids
            ]

            return {
                "ranked_model_ids": ranked_model_ids,
                "ranked_model_names": ranked_model_names,
                "reasoning": "reranker-based relevance scoring",
                "scores": {all_models[i]: s for i, s in indexed_scores},
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "positive_models": positive_models,
                "negative_candidates": negative_candidates,
                "total_models_ranked": len(all_models),
            }

        except Exception as e:
            d_name = node_metadata.get(dataset_id, {}).get("name", str(dataset_id))
            print(f"Error in reranker ranking for dataset {d_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Text construction (same helpers as predictor)
    # ------------------------------------------------------------------

    def _build_dataset_text(
        self, dataset_id: int, exclude_model_ids: Set[int], G: nx.Graph, meta: dict,
    ) -> str:
        name = meta.get(dataset_id, {}).get("name", str(dataset_id))
        parts = [f"Dataset: {name}"]
        if self.use_info:
            info = meta.get(dataset_id, {}).get("info")
            if info:
                parts.append(info)

        if self.hop_number > 0 and G:
            neighbors = self._collect_neighbors(
                G, meta, dataset_id, target_type="model",
                exclude_ids=exclude_model_ids, max_n=self.max_neighbors,
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
        self, model_id: int, exclude_dataset_ids: Set[int], G: nx.Graph, meta: dict,
    ) -> str:
        name = meta.get(model_id, {}).get("name", str(model_id))
        parts = [f"Model: {name}"]
        if self.use_info:
            info = meta.get(model_id, {}).get("info")
            if info:
                parts.append(info)

        if self.hop_number > 0 and G:
            neighbors = self._collect_neighbors(
                G, meta, model_id, target_type="dataset",
                exclude_ids=exclude_dataset_ids, max_n=self.max_neighbors,
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
