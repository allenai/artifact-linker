# predictors.py
from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm
from artifact_graph.utils.llm_client import call_llm
import re


def _parse_llm_response_to_json(content: str) -> Optional[Dict[str, Any]]:
    try:
        # Remove <think>...</think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        else:
            content = content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            json_str = content[start:end]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return None


def _run_prediction_loop(
    predictor,
    edge_pairs,
    G=None,
    mode="simple",
    metric_name=None,
    summaries: dict | None = None,
):
    is_binary = isinstance(predictor, LLMBinaryLinkPredictor)
    if mode == "neighborhood":
        if not is_binary and not metric_name:
            raise ValueError(
                "A specific metric_name must be provided for 'neighborhood' mode for the non-binary predictor."
            )
        if not G:
            raise ValueError("A graph G must be provided for 'neighborhood' mode.")

    summaries = summaries or {}
    if not summaries:
        print("Warning: No summaries provided. Proceeding without README summaries.")

    results = []
    for model_name, dataset_name in tqdm(edge_pairs, desc="Predicting Links"):
        try:
            model_card = summaries['models'].get(model_name)['model_info']
            if not model_card:
                print(f"Warning: Could not find summary for model {model_name}")

            dataset_card = summaries['datasets'].get(dataset_name)['model_info']
            if not dataset_card:
                print(f"Warning: Could not find summary for dataset {dataset_name}")

            model_neighbors = None
            dataset_neighbors = None
            if mode == "neighborhood" and G:
                model_neighbors = []
                for neighbor in G.neighbors(model_name):
                    if (
                        neighbor != dataset_name
                        and G.nodes[neighbor].get("type") == "dataset"
                    ):
                        if is_binary:
                            model_neighbors.append(neighbor)
                        elif metric_name and metric_name in G[model_name][neighbor]:
                            model_neighbors.append(
                                (neighbor, G[model_name][neighbor][metric_name])
                            )

                dataset_neighbors = []
                for neighbor in G.neighbors(dataset_name):
                    if (
                        neighbor != model_name
                        and G.nodes[neighbor].get("type") == "model"
                    ):
                        if is_binary:
                            dataset_neighbors.append(neighbor)
                        elif metric_name and metric_name in G[neighbor][dataset_name]:
                            dataset_neighbors.append(
                                (neighbor, G[neighbor][dataset_name][metric_name])
                            )

            prompt = predictor._build_prompt(
                model_name=model_name,
                dataset_name=dataset_name,
                model_card=model_card,
                dataset_card=dataset_card,
                model_neighbors=model_neighbors,
                dataset_neighbors=dataset_neighbors,
                mode=mode,
                metric_name=metric_name,
            )

            messages = [{"role": "user", "content": prompt}]
            agent_name = "binary_link_predictor" if is_binary else "link_predictor"
            response = call_llm(
                messages, model=predictor.model_name, agent_name=agent_name
            )

            if not response["success"]:
                print(
                    f"Warning: LLM call failed for ({model_name}, {dataset_name}). Error: {response.get('error')}"
                )
                prediction_result = None
            else:
                answer = response["content"].strip()
                prediction_result = predictor._parse_llm_answer(
                    answer, model_name, dataset_name
                )

            results.append(prediction_result)

        except Exception as e:
            print(f"Error predicting for ({model_name}, {dataset_name}): {e}")
            results.append(None)

    return results


class LLMLinkPredictor:
    def __init__(self, model_name="openai/gpt-3.5-turbo"):
        self.model_name = model_name

    def predict(
        self,
        edge_pairs,
        G=None,
        mode="simple",
        metric_name=None,
        summaries: dict | None = None,
    ):
        return _run_prediction_loop(
            predictor=self,
            edge_pairs=edge_pairs,
            G=G,
            mode=mode,
            metric_name=metric_name,
            summaries=summaries,
        )

    def _build_prompt(
        self,
        model_name,
        dataset_name,
        model_card=None,
        dataset_card=None,
        model_neighbors=None,
        dataset_neighbors=None,
        mode="simple",
        metric_name=None,
    ):
        metric_str = metric_name if metric_name else "performance"
        prediction_instruction = f"Please predict the expected {metric_str} that this model would achieve on this dataset. Provide your answer as a JSON object with two keys: 'prediction' (a float between 0 and 1) and 'reason' (a brief explanation of your reasoning)."

        prompt = f"Given a machine learning model named '{model_name}' and a dataset named '{dataset_name}'"
        if mode != "zero-shot":
            if model_card:
                prompt += f"\nModel card: {model_card}"
            if dataset_card:
                prompt += f"\nDataset card: {dataset_card}"

        if mode == "neighborhood":
            prompt += f"\nThe model's performance on other datasets (metric: {metric_str}):\n"
            if model_neighbors:
                for ds, acc in model_neighbors:
                    prompt += f"- {ds}: {acc:.2f}\n"
            else:
                prompt += "- (no other datasets)\n"
            prompt += "The dataset's performance with other models:\n"
            if dataset_neighbors:
                for mdl, acc in dataset_neighbors:
                    prompt += f"- {mdl}: {acc:.2f}\n"
            else:
                prompt += "- (no other models)\n"

        prompt += f"\n{prediction_instruction}"
        return prompt

    def _parse_llm_answer(self, answer, model_name, dataset_name):
        result_json = _parse_llm_response_to_json(answer)
        if not result_json:
            print(
                f"Warning: Could not parse LLM JSON output for ({model_name}, {dataset_name}). Output was: {answer}"
            )
            return None
        try:
            prediction = result_json.get("prediction")
            reason = result_json.get("reason", "")

            if prediction is None:
                return None

            prob = float(prediction)
            final_prediction = max(0.0, min(1.0, prob))
            return {"prediction": final_prediction, "reason": reason}

        except (ValueError, TypeError):
            print(
                f"Warning: Could not process parsed JSON for ({model_name}, {dataset_name}). JSON was: {result_json}"
            )
            return None


class LLMBinaryRanker:
    def __init__(self, model_name="openai/gpt-3.5-turbo"):
        self.model_name = model_name

    def rank_models_for_dataset(
        self,
        dataset_name: str,
        G,
        summaries: dict | None = None,
        num_negative_samples: int = 5,
        max_models_to_rank: int = 10,
    ):
        """
        Given a dataset, find its neighbor models and negative samples,
        then ask LLM to rank all models by relevance to the dataset.
        
        Args:
            dataset_name: The target dataset
            G: The graph containing model-dataset connections
            summaries: Dictionary containing model and dataset summaries
            num_negative_samples: Number of negative models to sample
            max_models_to_rank: Maximum total models to rank (neighbors + negatives)
        
        Returns:
            Dictionary with ranked model list and reasoning
        """
        import random
        
        summaries = summaries or {}
        
        # Find neighbor models (positive samples)
        neighbor_models = []
        all_models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
        
        for neighbor in G.neighbors(dataset_name):
            if G.nodes[neighbor].get("type") == "model":
                neighbor_models.append(neighbor)
        
        # Sample negative models (not connected to the dataset)
        connected_models = set(neighbor_models)
        unconnected_models = [m for m in all_models if m not in connected_models]
        
        # Limit negative samples
        actual_negative_samples = min(num_negative_samples, len(unconnected_models))
        negative_models = random.sample(unconnected_models, actual_negative_samples)
        
        # Combine and limit total models to rank
        all_models_to_rank = neighbor_models + negative_models
        if len(all_models_to_rank) > max_models_to_rank:
            # Keep all neighbors, sample from negatives if needed
            if len(neighbor_models) >= max_models_to_rank:
                all_models_to_rank = neighbor_models[:max_models_to_rank]
            else:
                remaining_slots = max_models_to_rank - len(neighbor_models)
                negative_models = negative_models[:remaining_slots]
                all_models_to_rank = neighbor_models + negative_models
        
        # Shuffle the list so LLM doesn't see any ordering bias
        random.shuffle(all_models_to_rank)
        
        # Build ranking prompt
        prompt = self._build_ranking_prompt(
            dataset_name=dataset_name,
            models_to_rank=all_models_to_rank,
            dataset_card=summaries.get('datasets', {}).get(dataset_name, {}).get('model_info'),
            model_summaries={m: summaries.get('models', {}).get(m, {}).get('model_info') for m in all_models_to_rank}
        )
        
        # Get ranking from LLM
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(
            messages, model=self.model_name, agent_name="binary_ranker"
        )
        
        if not response["success"]:
            print(f"Warning: LLM ranking call failed for dataset {dataset_name}. Error: {response.get('error')}")
            return None
        
        answer = response["content"].strip()
        ranking_result = self._parse_ranking_answer(answer, dataset_name, all_models_to_rank)
        
        # Add metadata about the ranking task
        if ranking_result:
            ranking_result.update({
                "dataset_name": dataset_name,
                "neighbor_models": neighbor_models,
                "negative_models": negative_models,
                "total_models_ranked": len(all_models_to_rank)
            })
        
        return ranking_result

    def _build_ranking_prompt(
        self,
        dataset_name: str,
        models_to_rank: list,
        dataset_card: str | None = None,
        model_summaries: dict | None = None,
    ):
        model_summaries = model_summaries or {}
        
        prompt = f"Given a dataset named '{dataset_name}'"
        if dataset_card:
            prompt += f"\nDataset description: {dataset_card}"
        
        prompt += f"\nPlease rank the following {len(models_to_rank)} machine learning models by how likely they are to be evaluated on this dataset (most relevant first):\n\n"
        
        for i, model in enumerate(models_to_rank, 1):
            prompt += f"{i}. {model}"
            if model_summaries.get(model):
                prompt += f" - {model_summaries[model]}"
            prompt += "\n"
        
        prompt += "\nProvide your answer as a JSON object with the following structure:"
        prompt += """
{
  "ranked_models": ["model1", "model2", "model3", ...],
  "reasoning": "Brief explanation of your ranking criteria and decisions"
}

The 'ranked_models' list should contain all model names in order from most to least likely to be evaluated on the dataset."""
        
        return prompt

    def _parse_ranking_answer(self, answer: str, dataset_name: str, original_models: list):
        result_json = _parse_llm_response_to_json(answer)
        if not result_json:
            print(f"Warning: Could not parse LLM ranking output for dataset {dataset_name}. Output was: {answer}")
            return None
        
        try:
            ranked_models = result_json.get("ranked_models", [])
            reasoning = result_json.get("reasoning", "")
            
            # Validate that all original models are present in ranking
            if not isinstance(ranked_models, list):
                return None
            
            original_set = set(original_models)
            ranked_set = set(ranked_models)
            
            if ranked_set != original_set:
                print(f"Warning: Ranked models don't match original models for dataset {dataset_name}")
                # Try to fix by filtering and adding missing models
                valid_ranked = [m for m in ranked_models if m in original_set]
                missing = [m for m in original_models if m not in ranked_set]
                ranked_models = valid_ranked + missing
            
            return {
                "ranked_models": ranked_models,
                "reasoning": reasoning
            }
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not process ranking JSON for dataset {dataset_name}. Error: {e}")
            return None


class LLMAttributeRanker:
    def __init__(self, model_name="openai/gpt-3.5-turbo"):
        self.model_name = model_name

    def rank_edges_by_attribute(
        self,
        positive_edges: list,
        G,
        attribute_name: str,
        summaries: dict | None = None,
        max_edges_to_rank: int = 10,
    ):
        """
        Given a list of positive edges (model-dataset pairs), ask LLM to rank them
        by their expected attribute scores (e.g., accuracy, f1, etc.).
        
        Args:
            positive_edges: List of (model, dataset) tuples
            G: The graph containing model-dataset connections
            attribute_name: The attribute/metric to rank by (e.g., 'accuracy', 'f1')
            summaries: Dictionary containing model and dataset summaries
            max_edges_to_rank: Maximum number of edges to rank at once
        
        Returns:
            Dictionary with ranked edges and reasoning
        """
        import random
        
        summaries = summaries or {}
        
        # Limit the number of edges to rank if needed
        edges_to_rank = positive_edges[:max_edges_to_rank] if len(positive_edges) > max_edges_to_rank else positive_edges
        
        # Shuffle to avoid any ordering bias
        edges_to_rank = edges_to_rank.copy()
        random.shuffle(edges_to_rank)
        
        # Build ranking prompt
        prompt = self._build_attribute_ranking_prompt(
            edges_to_rank=edges_to_rank,
            attribute_name=attribute_name,
            summaries=summaries
        )
        
        # Get ranking from LLM
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(
            messages, model=self.model_name, agent_name="attribute_ranker"
        )
        
        if not response["success"]:
            print(f"Warning: LLM attribute ranking call failed. Error: {response.get('error')}")
            return None
        
        answer = response["content"].strip()
        ranking_result = self._parse_attribute_ranking_answer(answer, edges_to_rank, attribute_name)
        
        # Add metadata about the ranking task
        if ranking_result:
            ranking_result.update({
                "attribute_name": attribute_name,
                "total_edges_ranked": len(edges_to_rank),
                "original_edges_count": len(positive_edges)
            })
        
        return ranking_result

    def _build_attribute_ranking_prompt(
        self,
        edges_to_rank: list,
        attribute_name: str,
        summaries: dict | None = None,
    ):
        summaries = summaries or {}
        
        prompt = f"Please rank the following {len(edges_to_rank)} model-dataset pairs by their expected {attribute_name} performance (highest {attribute_name} first):\n\n"
        
        for i, (model, dataset) in enumerate(edges_to_rank, 1):
            prompt += f"{i}. Model: {model}, Dataset: {dataset}\n"
            
            # Add model summary if available
            model_info = summaries.get('models', {}).get(model, {}).get('model_info')
            if model_info:
                prompt += f"   Model description: {model_info[:200]}{'...' if len(model_info) > 200 else ''}\n"
            
            # Add dataset summary if available
            dataset_info = summaries.get('datasets', {}).get(dataset, {}).get('model_info')
            if dataset_info:
                prompt += f"   Dataset description: {dataset_info[:200]}{'...' if len(dataset_info) > 200 else ''}\n"
            
            prompt += "\n"
        
        prompt += f"Consider the compatibility between each model and dataset, as well as the model's capabilities and the dataset's characteristics when predicting {attribute_name} performance.\n\n"
        
        prompt += "Provide your answer as a JSON object with the following structure:"
        prompt += """
{
  "ranked_pairs": [
    {"model": "model1", "dataset": "dataset1", "rank": 1, "expected_score": 0.95},
    {"model": "model2", "dataset": "dataset2", "rank": 2, "expected_score": 0.87},
    ...
  ],
  "reasoning": "Brief explanation of your ranking criteria and key decisions"
}

The 'ranked_pairs' list should contain all model-dataset pairs ordered from highest to lowest expected """ + attribute_name + """ score.
Include an 'expected_score' (float between 0 and 1) for each pair representing your predicted """ + attribute_name + """ value."""
        
        return prompt

    def _parse_attribute_ranking_answer(self, answer: str, original_edges: list, attribute_name: str):
        result_json = _parse_llm_response_to_json(answer)
        if not result_json:
            print(f"Warning: Could not parse LLM attribute ranking output. Output was: {answer}")
            return None
        
        try:
            ranked_pairs = result_json.get("ranked_pairs", [])
            reasoning = result_json.get("reasoning", "")
            
            if not isinstance(ranked_pairs, list):
                return None
            
            # Validate and normalize the ranking
            original_edges_set = set(original_edges)
            valid_ranked_pairs = []
            
            for item in ranked_pairs:
                if isinstance(item, dict):
                    model = item.get("model")
                    dataset = item.get("dataset")
                    rank = item.get("rank")
                    expected_score = item.get("expected_score")
                    
                    if model and dataset and (model, dataset) in original_edges_set:
                        valid_ranked_pairs.append({
                            "model": model,
                            "dataset": dataset,
                            "rank": rank,
                            "expected_score": expected_score
                        })
            
            # Check if we have all original edges
            ranked_edges_set = {(item["model"], item["dataset"]) for item in valid_ranked_pairs}
            missing_edges = original_edges_set - ranked_edges_set
            
            if missing_edges:
                print(f"Warning: Some edges missing from ranking: {missing_edges}")
                # Add missing edges at the end with lowest scores
                for model, dataset in missing_edges:
                    valid_ranked_pairs.append({
                        "model": model,
                        "dataset": dataset,
                        "rank": len(valid_ranked_pairs) + 1,
                        "expected_score": 0.5  # Default middle score
                    })
            
            return {
                "ranked_pairs": valid_ranked_pairs,
                "reasoning": reasoning
            }
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not process attribute ranking JSON. Error: {e}")
            return None


class LLMBinaryLinkPredictor:
    def __init__(self, model_name="openai/gpt-3.5-turbo"):
        self.model_name = model_name

    def predict(
        self,
        edge_pairs,
        G=None,
        mode="simple",
        summaries: dict | None = None,
    ):
        return _run_prediction_loop(
            predictor=self,
            edge_pairs=edge_pairs,
            G=G,
            mode=mode,
            metric_name=None,
            summaries=summaries,
        )

    def _build_prompt(
        self,
        model_name,
        dataset_name,
        model_card=None,
        dataset_card=None,
        model_neighbors=None,
        dataset_neighbors=None,
        mode="simple",
        metric_name=None,
    ):
        prediction_instruction = "Please predict whether this model and dataset are connected (i.e., the model is evaluated on the dataset). Provide your answer as a JSON object with two keys: 'prediction' (a boolean, true or false) and 'reason' (a brief explanation of your reasoning)."

        prompt = f"Given a machine learning model named '{model_name}' and a dataset named '{dataset_name}'"

        if mode != "zero-shot":
            if model_card:
                prompt += f"\nModel card: {model_card}"
            if dataset_card:
                prompt += f"\nDataset card: {dataset_card}"

        if mode == "neighborhood":
            prompt += "\nThe model is also connected to the following datasets:\n"
            if model_neighbors:
                for ds in model_neighbors:
                    prompt += f"- {ds}\n"
            else:
                prompt += "- (no other datasets)\n"
            prompt += "The dataset is also connected to the following models:\n"
            if dataset_neighbors:
                for mdl in dataset_neighbors:
                    prompt += f"- {mdl}\n"
            else:
                prompt += "- (no other models)\n"

        prompt += f"\n{prediction_instruction}"
        return prompt

    def _parse_llm_answer(self, answer, model_name, dataset_name):
        result_json = _parse_llm_response_to_json(answer)
        if not result_json:
            print(
                f"Warning: Could not parse LLM JSON output for ({model_name}, {dataset_name}). Output was: {answer}"
            )
            return None
        try:
            prediction = result_json.get("prediction")
            reason = result_json.get("reason", "")

            if not isinstance(prediction, bool):
                return None

            return {"prediction": prediction, "reason": reason}

        except (ValueError, TypeError):
            print(
                f"Warning: Could not process parsed JSON for ({model_name}, {dataset_name}). JSON was: {result_json}"
            )
            return None
