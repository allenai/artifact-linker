from __future__ import annotations

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMLinkPredictor:
    def __init__(
        self, model_name="openai/gpt-3.5-turbo", hop_number: int = 1, use_info: bool = True
    ):
        self.model_name = model_name
        self.hop_number = hop_number
        self.use_info = use_info

    def predict(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
    ):
        """Predict whether a single model should be evaluated on a single dataset."""
        node_metadata = node_metadata or {}

        try:
            model_name = node_metadata.get(model_id, {}).get("name")
            dataset_name = node_metadata.get(dataset_id, {}).get("name")

            model_info = node_metadata.get(model_id, {}).get("info")
            dataset_info = node_metadata.get(dataset_id, {}).get("info")

            model_neighbor_names = []
            dataset_neighbor_names = []
            model_neighbor_infos = []
            dataset_neighbor_infos = []
            if self.hop_number > 0 and G:
                # Get model's k-hop neighbors
                UG = G.to_undirected()
                neighbor_ids_1 = list(
                    nx.single_source_shortest_path_length(
                        UG, model_id, cutoff=self.hop_number
                    ).keys()
                )
                neighbor_ids_2 = list(
                    nx.single_source_shortest_path_length(
                        UG, dataset_id, cutoff=self.hop_number
                    ).keys()
                )
                neighbor_ids = set(neighbor_ids_1 + neighbor_ids_2)
                for neighbor_id in neighbor_ids:
                    if neighbor_id == model_id or neighbor_id == dataset_id:
                        continue
                    if node_metadata.get(neighbor_id, {}).get("type") == "dataset":
                        dataset_neighbor_names.append(
                            node_metadata.get(neighbor_id, {}).get("name")
                        )
                        dataset_neighbor_infos.append(
                            node_metadata.get(neighbor_id, {}).get("info")
                        )
                    if node_metadata.get(neighbor_id, {}).get("type") == "model":
                        model_neighbor_names.append(node_metadata.get(neighbor_id, {}).get("name"))
                        model_neighbor_infos.append(node_metadata.get(neighbor_id, {}).get("info"))

            prompt = self._build_prompt(
                model_name=model_name,
                dataset_name=dataset_name,
                model_info=model_info,
                dataset_info=dataset_info,
                model_neighbor_names=model_neighbor_names,
                dataset_neighbor_names=dataset_neighbor_names,
                model_neighbor_infos=model_neighbor_infos,
                dataset_neighbor_infos=dataset_neighbor_infos,
            )

            breakpoint()
            messages = [{"role": "user", "content": prompt}]
            response = call_llm(messages, model=self.model_name, agent_name="link_predictor")

            if not response["success"]:
                print(
                    f"Warning: LLM call failed for ({model_name}, {dataset_name}). Error: {response.get('error')}"
                )
                return None
            else:
                answer = response["content"].strip()
                return self._parse_llm_answer(answer, model_name, dataset_name)

        except Exception as e:
            print(f"Error predicting for ({model_id}, {dataset_id}): {e}")
            return None

    def _build_prompt(
        self,
        model_name,
        dataset_name,
        model_info=None,
        dataset_info=None,
        model_neighbor_names=None,
        dataset_neighbor_names=None,
        model_neighbor_infos=None,
        dataset_neighbor_infos=None,
    ):
        prediction_instruction = "Please predict whether this model should be evaluated on this dataset. Provide your answer as a JSON object with two keys: 'prediction' (a boolean, true or false) and 'reason' (a brief explanation of your reasoning)."

        prompt = f"Given a machine learning model named '{model_name}' and a dataset named '{dataset_name}'."

        if self.use_info:
            if model_info:
                prompt += f"\n\nMore information about this model: {model_info}"
            if dataset_info:
                prompt += f"\n\nMore information about this dataset: {dataset_info}"

        if model_neighbor_names and model_neighbor_infos:
            prompt += "\n\n\nThere are other models that are evaluated on the dataset to judge whether the model and dataset are connected:\n"
            for mdl, info in zip(model_neighbor_names, model_neighbor_infos):
                prompt += f"- {mdl}: {info}\n"

        if dataset_neighbor_names and dataset_neighbor_infos:
            prompt += "\n\n\nThere are other datasets that are evaluated on the model to judge whether the model and dataset are connected:\n"
            for ds, info in zip(dataset_neighbor_names, dataset_neighbor_infos):
                prompt += f"- {ds}: {info}\n"

        prompt += f"\n\n\n{prediction_instruction}"
        breakpoint()
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

            if not isinstance(prediction, bool):
                return None

            return {"prediction": prediction, "reason": reason}

        except (ValueError, TypeError):
            print(
                f"Warning: Could not process parsed JSON for ({model_name}, {dataset_name}). JSON was: {result_json}"
            )
            return None
