from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMLinkRanker:
    """Model ranker for given datasets using LLM."""

    def __init__(
        self, model_name="openai/gpt-3.5-turbo", hop_number: int = 1, use_info: bool = True
    ):
        self.model_name = model_name
        self.hop_number = hop_number
        self.use_info = use_info

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

            # Get model names, info, and neighbors
            model_info = {}
            for model_id in all_models_to_rank:
                model_name = node_metadata.get(model_id, {}).get("name")
                info = node_metadata.get(model_id, {}).get("info")

                # Get model neighbors if hop_number > 0
                dataset_neighbors = []
                if self.hop_number > 0 and G:
                    # Get model's connected datasets (excluding current dataset)
                    for neighbor_id in G.neighbors(model_id):
                        if (
                            neighbor_id != dataset_id
                            and G.nodes[neighbor_id].get("type") == "dataset"
                        ):
                            neighbor_name = node_metadata.get(neighbor_id, {}).get("name")
                            neighbor_info = node_metadata.get(neighbor_id, {}).get("info")
                            dataset_neighbors.append((neighbor_name, neighbor_info))

                model_info[model_id] = (model_name, info, dataset_neighbors)

            # Get dataset's connected models (excluding models we're ranking) for context
            dataset_model_neighbors = []
            if self.hop_number > 0 and G:
                for neighbor_id in G.neighbors(dataset_id):
                    if (
                        neighbor_id not in all_models_to_rank
                        and G.nodes[neighbor_id].get("type") == "model"
                    ):
                        neighbor_name = node_metadata.get(neighbor_id, {}).get("name")
                        neighbor_info = node_metadata.get(neighbor_id, {}).get("info")
                        dataset_model_neighbors.append((neighbor_name, neighbor_info))

            # Shuffle to avoid bias
            import random

            random.shuffle(all_models_to_rank)

            prompt = self._build_ranking_prompt(
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                models_to_rank=all_models_to_rank,
                model_info=model_info,
                dataset_model_neighbors=dataset_model_neighbors,
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
                    }
                )

            return ranking_result

        except Exception as e:
            print(f"Error ranking for dataset {dataset_id}: {e}")
            return None

    def _build_ranking_prompt(
        self,
        dataset_name: str,
        dataset_info: str | None,
        models_to_rank: List[int],
        model_info: Dict[int, Tuple[str, str, List[Tuple[str, str]]]],
        dataset_model_neighbors: List[Tuple[str, str]],
    ):
        prompt = f"Given a dataset named '{dataset_name}'"

        if self.use_info and dataset_info:
            prompt += f"\n\nMore information about this dataset: {dataset_info}"

        # Add neighborhood context if available
        if self.hop_number > 0 and dataset_model_neighbors:
            prompt += f"\n\nOther models that have been evaluated on {dataset_name}:\n"
            for neighbor_name, neighbor_info in dataset_model_neighbors:
                prompt += f"- {neighbor_name}"
                if self.use_info and neighbor_info:
                    prompt += f": {neighbor_info}"
                prompt += "\n"

        prompt += f"\n\nPlease rank the following {len(models_to_rank)} machine learning models by how likely they are to be evaluated on this dataset (most relevant first):\n\n"

        for i, model_id in enumerate(models_to_rank, 1):
            model_name, info, dataset_neighbors = model_info[model_id]
            prompt += f"\n\n{i}. {model_name}"
            if self.use_info and info:
                prompt += f" - {info}"

            # Add model's dataset history if available
            if self.hop_number > 0 and dataset_neighbors:
                prompt += f"\n {model_name} was also evaluated on:"
                for ds_name, ds_info in dataset_neighbors[:5]:  # Limit to first 5
                    prompt += f"\n     * {ds_name}"
                    if self.use_info and ds_info:
                        prompt += f": {ds_info}"
                if len(dataset_neighbors) > 5:
                    prompt += f"\n     * and {len(dataset_neighbors) - 5} others"

            prompt += "\n"

        prompt += "\nProvide your answer as a JSON object with the following structure:"
        prompt += """
{
  "ranked_models": ["model1", "model2", "model3", ...],
  "reasoning": "Brief explanation of your ranking criteria and decisions"
}

        The 'ranked_models' list should contain all model names in order from most to least likely to be evaluated on the dataset."""
        return prompt

    def _parse_ranking_answer(
        self,
        answer: str,
        dataset_name: str,
        original_model_ids: List[int],
        model_info: Dict[int, Tuple[str, str, List[Tuple[str, str]]]],
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
            name_to_id = {name: mid for mid, (name, _, _) in model_info.items()}
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
