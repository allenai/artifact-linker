from __future__ import annotations

import random
from typing import Dict, List, Tuple

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMAttributeRanker:
    """Attribute ranker for model-dataset pairs using LLM."""

    def __init__(
        self, model_name="openai/gpt-3.5-turbo", hop_number: int = 1, use_info: bool = True
    ):
        self.model_name = model_name
        self.hop_number = hop_number
        self.use_info = use_info

    def rank(
        self,
        dataset_id: int,
        models_to_rank: List[Tuple[int, float]],  # List of (model_id, true_metric_value)
        G: nx.Graph,
        node_metadata: dict,
        edge_metadata: dict,
        metric_name: str,
    ):
        """
        Rank models for a single dataset by their expected attribute scores.

        Args:
            dataset_id: The ID of the dataset to rank models for.
            models_to_rank: List of (model_id, true_metric_value) tuples.
            G: NetworkX graph with integer node IDs.
            node_metadata: Dictionary with node metadata.
            edge_metadata: Dictionary with edge metadata.
            metric_name: The name of the attribute to rank by.

        Returns:
            Ranking result for this dataset.
        """
        try:
            dataset_name = node_metadata.get(dataset_id, {}).get("name")
            dataset_info = node_metadata.get(dataset_id, {}).get("info")

            # Shuffle to avoid ordering bias
            models_shuffled = models_to_rank.copy()
            random.shuffle(models_shuffled)

            # Extract model information and their neighbors
            model_info_map = {}
            for model_id, true_value in models_shuffled:
                model_name = node_metadata.get(model_id, {}).get("name")
                model_info_text = node_metadata.get(model_id, {}).get("info")

                # Get model neighbors if hop_number > 0 (limit to 3 neighbors max)
                model_neighbors = None
                if self.hop_number > 0:
                    model_neighbors = []
                    for neighbor_id in G.neighbors(model_id):
                        if len(model_neighbors) >= 3:  # Limit to 3 neighbors
                            break

                        edge_key = tuple(sorted((model_id, neighbor_id)))
                        if (
                            neighbor_id != dataset_id
                            and G.nodes[neighbor_id].get("type") == "dataset"
                        ):
                            neighbor_name = node_metadata.get(neighbor_id, {}).get("name")
                            edge_attrs = G.edges[model_id, neighbor_id]
                            edge_meta = edge_metadata.get(edge_key, {})

                            all_metrics = {}
                            if metric_name and metric_name in edge_attrs:
                                all_metrics[metric_name] = edge_attrs[metric_name]

                            for metric_key, metric_value in edge_attrs.items():
                                if (
                                    isinstance(metric_value, (int, float))
                                    and metric_key != metric_name
                                ):
                                    all_metrics[metric_key] = metric_value

                            if all_metrics:
                                neighbor_info = node_metadata.get(neighbor_id, {}).get("info", "")
                                model_neighbors.append(
                                    (neighbor_name, all_metrics, edge_meta, neighbor_info)
                                )

                model_info_map[model_id] = {
                    "model_name": model_name,
                    "model_info": model_info_text,
                    "true_value": true_value,
                    "model_neighbors": model_neighbors,
                }

            prompt = self._build_attribute_ranking_prompt(
                dataset_name, dataset_info, models_shuffled, model_info_map, metric_name
            )

            messages = [{"role": "user", "content": prompt}]
            response = call_llm(messages, model=self.model_name, agent_name="attribute_ranker")

            if not response["success"]:
                print(f"Warning: LLM attribute ranking call failed. Error: {response.get('error')}")
                ranking_result = None
            else:
                answer = response["content"].strip()
                ranking_result = self._parse_attribute_ranking_answer(
                    answer, models_shuffled, model_info_map, metric_name
                )

            # Add metadata
            if ranking_result:
                ranking_result.update(
                    {
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "metric_name": metric_name,
                        "models_ranked": models_to_rank,
                        "total_models_ranked": len(models_to_rank),
                    }
                )

            return ranking_result

        except Exception as e:
            print(f"Error ranking models for dataset {dataset_id}: {e}")
            return None

    def _build_attribute_ranking_prompt(
        self,
        dataset_name: str,
        dataset_info: str | None,
        models_to_rank: List[Tuple[int, float]],
        model_info_map: Dict,
        metric_name: str,
    ):
        prompt = f"Dataset: {dataset_name}\n"

        if self.use_info and dataset_info:
            prompt += f"\nDataset Information:\n{dataset_info}\n"

        prompt += "\nModels to rank:\n"
        for model_id, _ in models_to_rank:
            info = model_info_map[model_id]
            model_name = info["model_name"]
            model_info = info["model_info"]
            model_neighbors = info["model_neighbors"]

            prompt += f"\n- {model_name}"
            if self.use_info and model_info:
                prompt += f": {model_info}"

            # Add model's performance on other datasets if available
            if self.hop_number > 0 and model_neighbors:
                prompt += f"\n  {model_name}'s performance on other datasets:"
                for ds_name, all_metrics, edge_meta, ds_info in model_neighbors:
                    metrics_strs = []
                    if metric_name in all_metrics:
                        metrics_strs.append(f"{metric_name}: {all_metrics[metric_name]:.3f}")

                    for metric_key, metric_value in all_metrics.items():
                        if metric_key != metric_name:
                            metrics_strs.append(f"{metric_key}: {metric_value:.3f}")

                    metrics_display = ", ".join(metrics_strs)
                    if self.use_info and ds_info:
                        prompt += f"\n    * {ds_name}: {metrics_display} (info: {ds_info})"
                    else:
                        prompt += f"\n    * {ds_name}: {metrics_display}"

            prompt += "\n"

        # Move task description to the end
        prompt += f"\nTask: For the dataset '{dataset_name}', please rank the above models based on their expected '{metric_name}' performance. Order them from best (highest score) to worst (lowest score).\n"
        prompt += f"\nConsider the models' capabilities and the dataset's characteristics to predict their {metric_name} performance.\n"
        prompt += "\nProvide your answer as a JSON object with this structure:"
        prompt += """
{
  "ranked_models": [
    {"model": "model_name_1", "rank": 1, "expected_score": 0.95},
    {"model": "model_name_2", "rank": 2, "expected_score": 0.87},
    ...
  ],
  "reasoning": "Your brief explanation of the ranking criteria."
}

The 'ranked_models' list should contain all models, ordered from highest to lowest expected score.
'expected_score' should be your predicted float value for the metric."""
        return prompt

    def _parse_attribute_ranking_answer(
        self,
        answer: str,
        original_models: List[Tuple[int, float]],
        model_info_map: Dict,
        metric_name: str,
    ):
        result_json = parse_llm_response_to_json(answer)
        if not result_json:
            print(f"Warning: Could not parse LLM attribute ranking output. Output was: {answer}")
            return None

        try:
            ranked_models_from_llm = result_json.get("ranked_models", [])
            reasoning = result_json.get("reasoning", "")

            if not isinstance(ranked_models_from_llm, list):
                return None

            # Create name to ID mapping
            name_to_id = {info["model_name"]: model_id for model_id, info in model_info_map.items()}
            original_model_names = set(name_to_id.keys())

            valid_ranked_models = []
            for item in ranked_models_from_llm:
                if isinstance(item, dict):
                    model_name = item.get("model")
                    if model_name in name_to_id:
                        model_id = name_to_id[model_name]
                        true_value = model_info_map[model_id]["true_value"]

                        valid_ranked_models.append(
                            {
                                "model_id": model_id,
                                "model_name": model_name,
                                "rank": item.get("rank"),
                                "expected_score": item.get("expected_score"),
                                "true_value": true_value,
                            }
                        )

            # Add missing models to the end
            ranked_model_names = {item["model_name"] for item in valid_ranked_models}
            missing_names = original_model_names - ranked_model_names
            if missing_names:
                print(f"Warning: Some models missing from ranking: {missing_names}")
                for model_name in missing_names:
                    model_id = name_to_id[model_name]
                    true_value = model_info_map[model_id]["true_value"]
                    valid_ranked_models.append(
                        {
                            "model_id": model_id,
                            "model_name": model_name,
                            "rank": len(valid_ranked_models) + 1,
                            "expected_score": 0.5,
                            "true_value": true_value,
                        }
                    )

            return {"ranked_models": valid_ranked_models, "reasoning": reasoning}

        except (ValueError, TypeError) as e:
            print(f"Warning: Could not process attribute ranking JSON. Error: {e}")
            return None
