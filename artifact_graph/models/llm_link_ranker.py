from __future__ import annotations
import json
import random
from typing import Any, Dict, List, Optional
from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMLinkRanker:
    """Model ranker for given datasets using LLM."""
    
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
        """
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
            messages, model=self.model_name, agent_name="link_ranker"
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
        result_json = parse_llm_response_to_json(answer)
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
