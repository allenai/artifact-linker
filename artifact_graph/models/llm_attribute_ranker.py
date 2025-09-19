from __future__ import annotations
import json
import random
from typing import Any, Dict, List, Optional, Tuple
from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMAttributeRanker:
    """Attribute ranker for model-dataset pairs using LLM."""
    
    def __init__(self, model_name="openai/gpt-3.5-turbo"):
        self.model_name = model_name

    def rank_edges_by_attribute(
        self,
        positive_edges: List[Tuple[str, str]],
        G,
        attribute_name: str,
        summaries: dict | None = None,
        max_edges_to_rank: int = 10,
    ):
        """
        Given a list of positive edges (model-dataset pairs), ask LLM to rank them
        by their expected attribute scores (e.g., accuracy, f1, etc.).
        """
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
        edges_to_rank: List[Tuple[str, str]],
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

    def _parse_attribute_ranking_answer(self, answer: str, original_edges: List[Tuple[str, str]], attribute_name: str):
        result_json = parse_llm_response_to_json(answer)
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
