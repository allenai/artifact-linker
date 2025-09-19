from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMAttributePredictor:
    """Attribute value predictor using LLM."""
    
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
        """Predict attribute values for model-dataset pairs."""
        if mode == "neighborhood":
            if not metric_name:
                raise ValueError(
                    "A specific metric_name must be provided for 'neighborhood' mode."
                )
            if not G:
                raise ValueError("A graph G must be provided for 'neighborhood' mode.")

        summaries = summaries or {}
        if not summaries:
            print("Warning: No summaries provided. Proceeding without README summaries.")

        results = []
        for model_name, dataset_name in tqdm(edge_pairs, desc="Predicting Attributes"):
            try:
                model_card = summaries.get('models', {}).get(model_name, {}).get('model_info')
                if not model_card:
                    print(f"Warning: Could not find summary for model {model_name}")

                dataset_card = summaries.get('datasets', {}).get(dataset_name, {}).get('model_info')
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
                            and metric_name in G[model_name][neighbor]
                        ):
                            model_neighbors.append(
                                (neighbor, G[model_name][neighbor][metric_name])
                            )

                    dataset_neighbors = []
                    for neighbor in G.neighbors(dataset_name):
                        if (
                            neighbor != model_name
                            and G.nodes[neighbor].get("type") == "model"
                            and metric_name in G[neighbor][dataset_name]
                        ):
                            dataset_neighbors.append(
                                (neighbor, G[neighbor][dataset_name][metric_name])
                            )

                prompt = self._build_prompt(
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
                response = call_llm(
                    messages, model=self.model_name, agent_name="attribute_predictor"
                )

                if not response["success"]:
                    print(
                        f"Warning: LLM call failed for ({model_name}, {dataset_name}). Error: {response.get('error')}"
                    )
                    prediction_result = None
                else:
                    answer = response["content"].strip()
                    prediction_result = self._parse_llm_answer(
                        answer, model_name, dataset_name
                    )

                results.append(prediction_result)

            except Exception as e:
                print(f"Error predicting for ({model_name}, {dataset_name}): {e}")
                results.append(None)

        return results

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
        result_json = parse_llm_response_to_json(answer)
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
