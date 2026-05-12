from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json

# Neighbor tuple: (name, metrics_dict, edge_meta, info)
NeighborInfo = Tuple[str, Dict[str, float], dict, str]


class LLMAttributeRanker:
    """Attribute ranker for model-dataset pairs using LLM."""

    def __init__(
        self, model_name="openai/gpt-3.5-turbo", hop_number: int = 1, use_info: bool = True
    ):
        self.model_name = model_name
        self.hop_number = hop_number
        self.use_info = use_info

    # ------------------------------------------------------------------
    # Neighbor collection helpers (symmetric for model / dataset sides)
    # ------------------------------------------------------------------

    # Simple neighbor info: (name, info)
    SimpleNeighborInfo = Tuple[str, str]

    def _get_model_dataset_neighbors(
        self,
        G: nx.Graph,
        node_metadata: dict,
        edge_metadata: dict,
        model_id: int,
        metric_name: str,
        exclude_ids: Set[int] | None = None,
        max_neighbors: int = 3,
    ) -> List[NeighborInfo]:
        """Collect datasets connected to *model_id* (model-side 1-hop).

        Returns list of ``(dataset_name, metrics_dict, edge_meta, dataset_info)``
        for each neighbouring dataset that carries at least one numeric metric.
        """
        if self.hop_number <= 0:
            return []

        exclude = exclude_ids or set()
        neighbors: List[NeighborInfo] = []

        for nbr_id in G.neighbors(model_id):
            if len(neighbors) >= max_neighbors:
                break
            if nbr_id in exclude or G.nodes[nbr_id].get("type") != "dataset":
                continue

            metrics = self._collect_edge_metrics(G, model_id, nbr_id, metric_name)
            if not metrics:
                continue

            edge_key = tuple(sorted((model_id, nbr_id)))
            neighbors.append((
                node_metadata.get(nbr_id, {}).get("name", str(nbr_id)),
                metrics,
                edge_metadata.get(edge_key, {}),
                node_metadata.get(nbr_id, {}).get("info", ""),
            ))

        return neighbors

    def _get_dataset_model_neighbors(
        self,
        G: nx.Graph,
        node_metadata: dict,
        edge_metadata: dict,
        dataset_id: int,
        metric_name: str,
        exclude_ids: Set[int] | None = None,
        max_neighbors: int = 10,
    ) -> List[NeighborInfo]:
        """Collect models connected to *dataset_id* (dataset-side 1-hop).

        Returns list of ``(model_name, metrics_dict, edge_meta, model_info)``
        for each neighbouring model that carries at least one numeric metric.
        """
        if self.hop_number <= 0:
            return []

        exclude = exclude_ids or set()
        neighbors: List[NeighborInfo] = []

        for nbr_id in G.neighbors(dataset_id):
            if len(neighbors) >= max_neighbors:
                break
            if nbr_id in exclude or G.nodes[nbr_id].get("type") != "model":
                continue

            metrics = self._collect_edge_metrics(G, nbr_id, dataset_id, metric_name)
            if not metrics:
                continue

            edge_key = tuple(sorted((nbr_id, dataset_id)))
            neighbors.append((
                node_metadata.get(nbr_id, {}).get("name", str(nbr_id)),
                metrics,
                edge_metadata.get(edge_key, {}),
                node_metadata.get(nbr_id, {}).get("info", ""),
            ))

        return neighbors

    def _get_simple_neighbors(
        self,
        G: nx.Graph,
        node_metadata: dict,
        node_id: int,
        target_type: str,
        max_neighbors: int = 3,
    ) -> List[Tuple[str, str]]:
        """Collect neighbors of a given type as (name, info) pairs."""
        if self.hop_number <= 0:
            return []
        neighbors = []
        for nbr_id in G.neighbors(node_id):
            if len(neighbors) >= max_neighbors:
                break
            if G.nodes[nbr_id].get("type") != target_type:
                continue
            neighbors.append((
                node_metadata.get(nbr_id, {}).get("name", str(nbr_id)),
                node_metadata.get(nbr_id, {}).get("info", ""),
            ))
        return neighbors

    @staticmethod
    def _collect_edge_metrics(
        G: nx.Graph, u: int, v: int, target_metric: str,
    ) -> Dict[str, float]:
        """Extract numeric metrics from the edge (u, v), target metric first."""
        edge_attrs = G.edges[u, v]
        metrics: Dict[str, float] = {}

        # Target metric goes first
        if target_metric and target_metric in edge_attrs:
            metrics[target_metric] = edge_attrs[target_metric]

        for key, val in edge_attrs.items():
            if isinstance(val, (int, float)) and key != target_metric:
                metrics[key] = val

        return metrics

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

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

            model_ids_ranked = {mid for mid, _ in models_shuffled}

            # --- Model-side neighbours (per model) ---
            model_info_map = {}
            for model_id, true_value in models_shuffled:
                model_neighbors = self._get_model_dataset_neighbors(
                    G, node_metadata, edge_metadata, model_id, metric_name,
                    exclude_ids={dataset_id},
                    max_neighbors=3,
                )
                model_paper_nbrs = self._get_simple_neighbors(
                    G, node_metadata, model_id, "paper", max_neighbors=3,
                )
                model_code_nbrs = self._get_simple_neighbors(
                    G, node_metadata, model_id, "codebase", max_neighbors=3,
                )
                model_info_map[model_id] = {
                    "model_name": node_metadata.get(model_id, {}).get("name"),
                    "model_info": node_metadata.get(model_id, {}).get("info"),
                    "true_value": true_value,
                    "model_neighbors": model_neighbors,
                    "paper_neighbors": model_paper_nbrs,
                    "code_neighbors": model_code_nbrs,
                }

            # --- Dataset-side neighbours ---
            dataset_model_neighbors = self._get_dataset_model_neighbors(
                G, node_metadata, edge_metadata, dataset_id, metric_name,
                exclude_ids=model_ids_ranked,
                max_neighbors=10,
            )
            dataset_paper_nbrs = self._get_simple_neighbors(
                G, node_metadata, dataset_id, "paper", max_neighbors=3,
            )
            dataset_code_nbrs = self._get_simple_neighbors(
                G, node_metadata, dataset_id, "codebase", max_neighbors=3,
            )

            prompt = self._build_attribute_ranking_prompt(
                dataset_name, dataset_info, models_shuffled, model_info_map, metric_name,
                dataset_model_neighbors=dataset_model_neighbors,
                dataset_paper_neighbors=dataset_paper_nbrs,
                dataset_code_neighbors=dataset_code_nbrs,
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

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metrics(metrics: Dict[str, float], target_metric: str) -> str:
        """Format a metrics dict as a comma-separated string, target metric first."""
        parts: List[str] = []
        if target_metric in metrics:
            parts.append(f"{target_metric}: {metrics[target_metric]:.3f}")
        for mk, mv in metrics.items():
            if mk != target_metric:
                parts.append(f"{mk}: {mv:.3f}")
        return ", ".join(parts)

    def _build_attribute_ranking_prompt(
        self,
        dataset_name: str,
        dataset_info: str | None,
        models_to_rank: List[Tuple[int, float]],
        model_info_map: Dict,
        metric_name: str,
        dataset_model_neighbors: List[NeighborInfo] | None = None,
        dataset_paper_neighbors: List[Tuple[str, str]] | None = None,
        dataset_code_neighbors: List[Tuple[str, str]] | None = None,
    ):
        prompt = f"Dataset: {dataset_name}\n"

        if self.use_info and dataset_info:
            prompt += f"\nDataset Information:\n{dataset_info}\n"

        # Dataset-side 1-hop: known performance of other models on this dataset
        if self.hop_number > 0 and dataset_model_neighbors:
            prompt += f"\nKnown performance of other models on {dataset_name}:\n"
            for nbr_name, nbr_metrics, _edge_meta, nbr_info in dataset_model_neighbors:
                metrics_str = self._format_metrics(nbr_metrics, metric_name)
                if self.use_info and nbr_info:
                    prompt += f"  - {nbr_name}: {metrics_str} (info: {nbr_info})\n"
                else:
                    prompt += f"  - {nbr_name}: {metrics_str}\n"

        if self.hop_number > 0 and dataset_paper_neighbors:
            prompt += f"\nRelated papers for {dataset_name}:\n"
            for name, info in dataset_paper_neighbors:
                if self.use_info and info:
                    prompt += f"  - {name}: {info}\n"
                else:
                    prompt += f"  - {name}\n"

        if self.hop_number > 0 and dataset_code_neighbors:
            prompt += f"\nRelated code repositories for {dataset_name}:\n"
            for name, info in dataset_code_neighbors:
                if self.use_info and info:
                    prompt += f"  - {name}: {info}\n"
                else:
                    prompt += f"  - {name}\n"

        prompt += "\nModels to rank:\n"
        for model_id, _ in models_to_rank:
            info = model_info_map[model_id]
            m_name = info["model_name"]
            m_info = info["model_info"]
            model_neighbors = info["model_neighbors"]
            paper_nbrs = info.get("paper_neighbors", [])
            code_nbrs = info.get("code_neighbors", [])

            prompt += f"\n- {m_name}"
            if self.use_info and m_info:
                prompt += f": {m_info}"

            # Model-side 1-hop: this model's performance on other datasets
            if self.hop_number > 0 and model_neighbors:
                prompt += f"\n  {m_name}'s performance on other datasets:"
                for ds_name, ds_metrics, _edge_meta, ds_info in model_neighbors:
                    metrics_str = self._format_metrics(ds_metrics, metric_name)
                    if self.use_info and ds_info:
                        prompt += f"\n    * {ds_name}: {metrics_str} (info: {ds_info})"
                    else:
                        prompt += f"\n    * {ds_name}: {metrics_str}"

            if self.hop_number > 0 and paper_nbrs:
                prompt += f"\n  Related papers:"
                for p_name, p_info in paper_nbrs:
                    if self.use_info and p_info:
                        prompt += f"\n    * {p_name}: {p_info}"
                    else:
                        prompt += f"\n    * {p_name}"

            if self.hop_number > 0 and code_nbrs:
                prompt += f"\n  Related code repositories:"
                for c_name, c_info in code_nbrs:
                    if self.use_info and c_info:
                        prompt += f"\n    * {c_name}: {c_info}"
                    else:
                        prompt += f"\n    * {c_name}"

            prompt += "\n"

        # Task description
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

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

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
