from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json

# Neighbor tuple: (name, info)
LinkNeighborInfo = Tuple[str, Optional[str]]


class LLMLinkRanker:
    """Model ranker for given datasets using LLM with optional RAG retrieval."""

    def __init__(
        self,
        model_name="openai/gpt-3.5-turbo",
        hop_number: int = 1,
        use_info: bool = True,
        use_rag: bool = False,
        rag_top_k: int = 100,
        data_dir: Optional[str] = None,
        retriever=None,
    ):
        """
        Initialize LLM link ranker.

        Args:
            model_name: LLM model to use
            hop_number: Number of hops for neighbor context
            use_info: Whether to include node info in prompts
            use_rag: Whether to use RAG for candidate filtering
            rag_top_k: Number of candidates to retrieve before LLM ranking
            data_dir: Data directory for loading embeddings (for RAG)
            retriever: CandidateRetriever instance (optional, created on demand)
        """
        self.model_name = model_name
        self.hop_number = hop_number
        self.use_info = use_info
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.data_dir = data_dir
        self.retriever = retriever

    # ------------------------------------------------------------------
    # Neighbor collection helpers (symmetric for model / dataset sides)
    # ------------------------------------------------------------------

    def _get_neighbors_by_type(
        self,
        G: nx.Graph,
        node_metadata: dict,
        node_id: int,
        target_type: str,
        exclude_ids: Set[int] | None = None,
        max_neighbors: int = 5,
    ) -> List[LinkNeighborInfo]:
        """Collect neighbors of a specific type connected to *node_id*.

        Returns list of ``(name, info)``.
        """
        if self.hop_number <= 0 or not G:
            return []

        exclude = exclude_ids or set()
        neighbors: List[LinkNeighborInfo] = []

        for nbr_id in G.neighbors(node_id):
            if len(neighbors) >= max_neighbors:
                break
            if nbr_id in exclude or G.nodes[nbr_id].get("type") != target_type:
                continue
            neighbors.append((
                node_metadata.get(nbr_id, {}).get("name", str(nbr_id)),
                node_metadata.get(nbr_id, {}).get("info"),
            ))

        return neighbors

    def _get_model_dataset_neighbors(
        self, G: nx.Graph, node_metadata: dict, model_id: int,
        exclude_ids: Set[int] | None = None, max_neighbors: int = 5,
    ) -> List[LinkNeighborInfo]:
        return self._get_neighbors_by_type(G, node_metadata, model_id, "dataset", exclude_ids, max_neighbors)

    def _get_dataset_model_neighbors(
        self, G: nx.Graph, node_metadata: dict, dataset_id: int,
        exclude_ids: Set[int] | None = None, max_neighbors: int = 10,
    ) -> List[LinkNeighborInfo]:
        return self._get_neighbors_by_type(G, node_metadata, dataset_id, "model", exclude_ids, max_neighbors)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rank(
        self,
        dataset_id: int,
        positive_models: List[int],
        negative_candidates: List[int],
        G: nx.Graph,
        node_metadata: dict,
    ):
        """
        Rank models for a single dataset.

        Args:
            dataset_id: The dataset ID to rank models for.
            positive_models: List of positive model IDs.
            negative_candidates: List of negative candidate model IDs.
            G: NetworkX graph with integer node IDs.
            node_metadata: Dictionary with node metadata.

        Returns:
            Ranking result for this dataset.
        """
        try:
            dataset_name = node_metadata.get(dataset_id, {}).get("name")
            dataset_info = node_metadata.get(dataset_id, {}).get("info")

            # Combine positive and negative models
            all_models_to_rank = positive_models + negative_candidates

            # Apply RAG filtering if enabled
            retrieval_scores = {}
            if self.use_rag and len(all_models_to_rank) > self.rag_top_k:
                from artifact_graph.utils.retriever import filter_candidates_by_retrieval

                # Ensure retriever is available
                if self.retriever is None:
                    from artifact_graph.utils.retriever import CandidateRetriever

                    self.retriever = CandidateRetriever.from_data_dir(
                        self.data_dir, top_k=self.rag_top_k,
                    ) if self.data_dir else CandidateRetriever(top_k=self.rag_top_k)

                # Retrieve top-k candidates purely by retrieval score — no label peeking
                filtered_candidates, retrieval_scores = filter_candidates_by_retrieval(
                    dataset_id, all_models_to_rank, self.retriever, G
                )

                original_count = len(all_models_to_rank)
                all_models_to_rank = filtered_candidates
                print(f"  RAG: {original_count} -> {len(all_models_to_rank)} candidates")

            models_set = set(all_models_to_rank)

            # --- Model-side neighbours (per model) ---
            model_info = {}
            for model_id in all_models_to_rank:
                ds_neighbors = self._get_model_dataset_neighbors(
                    G, node_metadata, model_id,
                    exclude_ids={dataset_id},
                    max_neighbors=5,
                )
                paper_neighbors = self._get_neighbors_by_type(
                    G, node_metadata, model_id, "paper", max_neighbors=3,
                )
                code_neighbors = self._get_neighbors_by_type(
                    G, node_metadata, model_id, "codebase", max_neighbors=3,
                )
                model_info[model_id] = (
                    node_metadata.get(model_id, {}).get("name"),
                    node_metadata.get(model_id, {}).get("info"),
                    ds_neighbors,
                    paper_neighbors,
                    code_neighbors,
                )

            # --- Dataset-side neighbours ---
            dataset_model_neighbors = self._get_dataset_model_neighbors(
                G, node_metadata, dataset_id,
                exclude_ids=models_set,
                max_neighbors=10,
            )
            dataset_paper_neighbors = self._get_neighbors_by_type(
                G, node_metadata, dataset_id, "paper", max_neighbors=3,
            )
            dataset_code_neighbors = self._get_neighbors_by_type(
                G, node_metadata, dataset_id, "codebase", max_neighbors=3,
            )

            # Shuffle to avoid bias
            random.shuffle(all_models_to_rank)

            prompt = self._build_ranking_prompt(
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                models_to_rank=all_models_to_rank,
                model_info=model_info,
                dataset_model_neighbors=dataset_model_neighbors,
                dataset_paper_neighbors=dataset_paper_neighbors,
                dataset_code_neighbors=dataset_code_neighbors,
            )

            messages = [{"role": "user", "content": prompt}]
            response = call_llm(messages, model=self.model_name, agent_name="link_ranker")

            if not response["success"]:
                print(
                    f"Warning: LLM ranking call failed for dataset {dataset_name}. Error: {response.get('error')}"
                )
                ranking_result = None
            else:
                answer = response["content"].strip()
                ranking_result = self._parse_ranking_answer(
                    answer, dataset_name, all_models_to_rank, model_info
                )

            # Add metadata
            if ranking_result:
                ranking_result.update(
                    {
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "positive_models": positive_models,
                        "negative_candidates": negative_candidates,
                        "total_models_ranked": len(all_models_to_rank),
                        "rag_enabled": self.use_rag,
                        "rag_top_k": self.rag_top_k if self.use_rag else None,
                        "retrieval_scores": retrieval_scores if self.use_rag else None,
                    }
                )

            return ranking_result

        except Exception as e:
            print(f"Error ranking for dataset {dataset_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_ranking_prompt(
        self,
        dataset_name: str,
        dataset_info: str | None,
        models_to_rank: List[int],
        model_info: Dict[int, Tuple[str, str, List[LinkNeighborInfo], List[LinkNeighborInfo], List[LinkNeighborInfo]]],
        dataset_model_neighbors: List[LinkNeighborInfo],
        dataset_paper_neighbors: List[LinkNeighborInfo] | None = None,
        dataset_code_neighbors: List[LinkNeighborInfo] | None = None,
    ):
        prompt = f"Given a dataset named '{dataset_name}'"

        if self.use_info and dataset_info:
            prompt += f"\n\nMore information about this dataset: {dataset_info}"

        # Dataset-side 1-hop: other models evaluated on this dataset
        if self.hop_number > 0 and dataset_model_neighbors:
            prompt += f"\n\nOther models that have been evaluated on {dataset_name}:\n"
            for neighbor_name, neighbor_info in dataset_model_neighbors:
                prompt += f"- {neighbor_name}"
                if self.use_info and neighbor_info:
                    prompt += f": {neighbor_info}"
                prompt += "\n"

        if self.hop_number > 0 and dataset_paper_neighbors:
            prompt += f"\nRelated papers for {dataset_name}:\n"
            for name, info in dataset_paper_neighbors:
                prompt += f"- {name}"
                if self.use_info and info:
                    prompt += f": {info}"
                prompt += "\n"

        if self.hop_number > 0 and dataset_code_neighbors:
            prompt += f"\nRelated code repositories for {dataset_name}:\n"
            for name, info in dataset_code_neighbors:
                prompt += f"- {name}"
                if self.use_info and info:
                    prompt += f": {info}"
                prompt += "\n"

        prompt += f"\n\nPlease rank the following {len(models_to_rank)} machine learning models by how likely they are to be evaluated on this dataset (most relevant first):\n\n"

        for i, model_id in enumerate(models_to_rank, 1):
            m_name, m_info, ds_neighbors, paper_nbrs, code_nbrs = model_info[model_id]
            prompt += f"\n\n{i}. {m_name}"
            if self.use_info and m_info:
                prompt += f" - {m_info}"

            # Model-side 1-hop: datasets this model was evaluated on
            if self.hop_number > 0 and ds_neighbors:
                prompt += f"\n {m_name} was also evaluated on:"
                for ds_name, ds_info in ds_neighbors[:5]:
                    prompt += f"\n     * {ds_name}"
                    if self.use_info and ds_info:
                        prompt += f": {ds_info}"
                if len(ds_neighbors) > 5:
                    prompt += f"\n     * and {len(ds_neighbors) - 5} others"

            if self.hop_number > 0 and paper_nbrs:
                prompt += f"\n Related papers:"
                for p_name, p_info in paper_nbrs:
                    prompt += f"\n     * {p_name}"
                    if self.use_info and p_info:
                        prompt += f": {p_info}"

            if self.hop_number > 0 and code_nbrs:
                prompt += f"\n Related code repositories:"
                for c_name, c_info in code_nbrs:
                    prompt += f"\n     * {c_name}"
                    if self.use_info and c_info:
                        prompt += f": {c_info}"

            prompt += "\n"

        prompt += "\nProvide your answer as a JSON object with the following structure:"
        prompt += """
{
  "ranked_models": ["model1", "model2", "model3", ...],
  "reasoning": "Brief explanation of your ranking criteria and decisions"
}

        The 'ranked_models' list should contain all model names in order from most to least likely to be evaluated on the dataset."""
        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_ranking_answer(
        self,
        answer: str,
        dataset_name: str,
        original_model_ids: List[int],
        model_info: Dict[int, Tuple[str, str, List[LinkNeighborInfo]]],
    ):
        result_json = parse_llm_response_to_json(answer)
        if not result_json:
            print(
                f"Warning: Could not parse LLM ranking output for dataset {dataset_name}. Output was: {answer}"
            )
            return None

        try:
            ranked_model_names = result_json.get("ranked_models", [])
            reasoning = result_json.get("reasoning", "")

            # Create name to ID mapping
            name_to_id = {info[0]: mid for mid, info in model_info.items()}
            original_names = [model_info[mid][0] for mid in original_model_ids]

            # Validate that all original models are present in ranking
            if not isinstance(ranked_model_names, list):
                return None

            original_name_set = set(original_names)
            ranked_name_set = set(ranked_model_names)

            if ranked_name_set != original_name_set:
                print(
                    f"Warning: Ranked models don't match original models for dataset {dataset_name}"
                )
                # Try to fix by filtering and adding missing models
                valid_ranked = [m for m in ranked_model_names if m in original_name_set]
                missing = [m for m in original_names if m not in ranked_name_set]
                ranked_model_names = valid_ranked + missing

            # Convert back to IDs
            ranked_model_ids = [
                name_to_id.get(name)
                for name in ranked_model_names
                if name_to_id.get(name) is not None
            ]

            return {
                "ranked_model_ids": ranked_model_ids,
                "ranked_model_names": ranked_model_names,
                "reasoning": reasoning,
            }

        except (ValueError, TypeError) as e:
            print(f"Warning: Could not process ranking JSON for dataset {dataset_name}. Error: {e}")
            return None
