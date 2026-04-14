from __future__ import annotations

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json


class LLMAttributePredictor:
    """Attribute value predictor using LLM."""

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
        edge_metadata: dict,
        metric_name: str,
    ):
        """Predict attribute value for a single model-dataset pair."""
        try:
            model_name = node_metadata.get(model_id, {}).get("name")
            dataset_name = node_metadata.get(dataset_id, {}).get("name")
            model_info = node_metadata.get(model_id, {}).get("info")
            dataset_info = node_metadata.get(dataset_id, {}).get("info")

            model_neighbors = None
            dataset_neighbors = None
            model_paper_neighbors = None
            model_code_neighbors = None
            dataset_paper_neighbors = None
            dataset_code_neighbors = None
            max_model_neighbors = 10   # Cap neighbors to avoid overly long prompts
            max_dataset_neighbors = 10
            if self.hop_number > 0:
                model_neighbors = []
                model_paper_neighbors = []
                model_code_neighbors = []
                for neighbor_id in G.neighbors(model_id):
                    if neighbor_id == dataset_id:
                        continue
                    ntype = G.nodes[neighbor_id].get("type")
                    if ntype == "dataset" and len(model_neighbors) < max_model_neighbors:
                        edge_key = tuple(sorted((model_id, neighbor_id)))
                        neighbor_name = node_metadata.get(neighbor_id, {}).get("name")
                        edge_attrs = G.edges[model_id, neighbor_id]
                        edge_meta = edge_metadata.get(edge_key, {})

                        all_metrics = {}
                        if metric_name and metric_name in edge_attrs:
                            all_metrics[metric_name] = edge_attrs[metric_name]

                        for metric_key, metric_value in edge_attrs.items():
                            if isinstance(metric_value, (int, float)) and metric_key != metric_name:
                                all_metrics[metric_key] = metric_value

                        if all_metrics:
                            neighbor_info = node_metadata.get(neighbor_id, {}).get("info", "")
                            model_neighbors.append(
                                (neighbor_name, all_metrics, edge_meta, neighbor_info)
                            )
                    elif ntype == "paper" and len(model_paper_neighbors) < 3:
                        model_paper_neighbors.append((
                            node_metadata.get(neighbor_id, {}).get("name", str(neighbor_id)),
                            node_metadata.get(neighbor_id, {}).get("info", ""),
                        ))
                    elif ntype == "codebase" and len(model_code_neighbors) < 3:
                        model_code_neighbors.append((
                            node_metadata.get(neighbor_id, {}).get("name", str(neighbor_id)),
                            node_metadata.get(neighbor_id, {}).get("info", ""),
                        ))

                dataset_neighbors = []
                dataset_paper_neighbors = []
                dataset_code_neighbors = []
                for neighbor_id in G.neighbors(dataset_id):
                    if neighbor_id == model_id:
                        continue
                    ntype = G.nodes[neighbor_id].get("type")
                    if ntype == "model" and len(dataset_neighbors) < max_dataset_neighbors:
                        edge_key = tuple(sorted((neighbor_id, dataset_id)))
                        neighbor_name = node_metadata.get(neighbor_id, {}).get("name")
                        edge_attrs = G.edges[neighbor_id, dataset_id]
                        edge_meta = edge_metadata.get(edge_key, {})

                        all_metrics = {}
                        if metric_name and metric_name in edge_attrs:
                            all_metrics[metric_name] = edge_attrs[metric_name]

                        for metric_key, metric_value in edge_attrs.items():
                            if isinstance(metric_value, (int, float)) and metric_key != metric_name:
                                all_metrics[metric_key] = metric_value

                        if all_metrics:
                            neighbor_info = node_metadata.get(neighbor_id, {}).get("info", "")
                            dataset_neighbors.append(
                                (neighbor_name, all_metrics, edge_meta, neighbor_info)
                            )
                    elif ntype == "paper" and len(dataset_paper_neighbors) < 3:
                        dataset_paper_neighbors.append((
                            node_metadata.get(neighbor_id, {}).get("name", str(neighbor_id)),
                            node_metadata.get(neighbor_id, {}).get("info", ""),
                        ))
                    elif ntype == "codebase" and len(dataset_code_neighbors) < 3:
                        dataset_code_neighbors.append((
                            node_metadata.get(neighbor_id, {}).get("name", str(neighbor_id)),
                            node_metadata.get(neighbor_id, {}).get("info", ""),
                        ))

            prompt = self._build_prompt(
                model_name=model_name,
                dataset_name=dataset_name,
                model_info=model_info,
                dataset_info=dataset_info,
                model_neighbors=model_neighbors,
                dataset_neighbors=dataset_neighbors,
                metric_name=metric_name,
                model_paper_neighbors=model_paper_neighbors,
                model_code_neighbors=model_code_neighbors,
                dataset_paper_neighbors=dataset_paper_neighbors,
                dataset_code_neighbors=dataset_code_neighbors,
            )

            messages = [{"role": "user", "content": prompt}]
            response = call_llm(messages, model=self.model_name, agent_name="attribute_predictor")

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
        model_neighbors=None,
        dataset_neighbors=None,
        metric_name=None,
        model_paper_neighbors=None,
        model_code_neighbors=None,
        dataset_paper_neighbors=None,
        dataset_code_neighbors=None,
    ):
        metric_str = metric_name if metric_name else "performance"
        prediction_instruction = f"Please predict the {metric_str} that {model_name} would achieve on {dataset_name}. Provide your answer as a JSON object with two keys: 'prediction' (a float between 0 and 1) and 'reason' (a brief explanation of your reasoning)."

        prompt = f"Given a machine learning model named '{model_name}' and a dataset named '{dataset_name}'."

        if self.use_info:
            if model_info:
                prompt += f"\n\nMore information about this model: {model_info}"
            if dataset_info:
                prompt += f"\n\nMore information about this dataset: {dataset_info}"

        if self.hop_number > 0:
            prompt += f"\n{model_name}'s performance on other datasets:\n"
            if model_neighbors:
                for ds, all_metrics, meta, ds_info in model_neighbors:
                    metrics_strs = []
                    if metric_name in all_metrics:
                        metrics_strs.append(f"{metric_name}: {all_metrics[metric_name]:.3f}")

                    for metric_key, metric_value in all_metrics.items():
                        if metric_key != metric_name:
                            metrics_strs.append(f"{metric_key}: {metric_value:.3f}")

                    metrics_display = ", ".join(metrics_strs)
                    if self.use_info and ds_info:
                        prompt += f"- {ds}: {metrics_display} (info: {ds_info})\n"
                    else:
                        prompt += f"- {ds}: {metrics_display}\n"
            else:
                prompt += "- (no other datasets)\n"

            if model_paper_neighbors:
                prompt += f"\nRelated papers for {model_name}:\n"
                for name, info in model_paper_neighbors:
                    if self.use_info and info:
                        prompt += f"- {name}: {info}\n"
                    else:
                        prompt += f"- {name}\n"

            if model_code_neighbors:
                prompt += f"\nRelated code repositories for {model_name}:\n"
                for name, info in model_code_neighbors:
                    if self.use_info and info:
                        prompt += f"- {name}: {info}\n"
                    else:
                        prompt += f"- {name}\n"

            prompt += f"\n{dataset_name}'s performance with other models:\n"
            if dataset_neighbors:
                for mdl, all_metrics, meta, mdl_info in dataset_neighbors:
                    metrics_strs = []
                    if metric_name in all_metrics:
                        metrics_strs.append(f"{metric_name}: {all_metrics[metric_name]:.3f}")

                    for metric_key, metric_value in all_metrics.items():
                        if metric_key != metric_name:
                            metrics_strs.append(f"{metric_key}: {metric_value:.3f}")

                    metrics_display = ", ".join(metrics_strs)
                    if self.use_info and mdl_info:
                        prompt += f"- {mdl}: {metrics_display} (info: {mdl_info})\n"
                    else:
                        prompt += f"- {mdl}: {metrics_display}\n"
            else:
                prompt += "- (no other models)\n"

            if dataset_paper_neighbors:
                prompt += f"\nRelated papers for {dataset_name}:\n"
                for name, info in dataset_paper_neighbors:
                    if self.use_info and info:
                        prompt += f"- {name}: {info}\n"
                    else:
                        prompt += f"- {name}\n"

            if dataset_code_neighbors:
                prompt += f"\nRelated code repositories for {dataset_name}:\n"
                for name, info in dataset_code_neighbors:
                    if self.use_info and info:
                        prompt += f"- {name}: {info}\n"
                    else:
                        prompt += f"- {name}\n"

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
