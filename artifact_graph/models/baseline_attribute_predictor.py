#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx
import numpy as np


class BaselineAttributePredictor:
    def __init__(self, mode: str = "dataset_average"):
        if mode not in {"global_average", "dataset_average"}:
            raise ValueError("mode must be 'global_average' or 'dataset_average'")
        self.mode = mode
        self._global_average_cache: Dict[str, float] = {}

    def _calculate_global_average(
        self, G: nx.Graph, model_id: int, dataset_id: int, metric_name: str, edge_metadata: dict
    ) -> tuple[float, int]:
        cache_key = f"{metric_name}_{model_id}_{dataset_id}"
        if cache_key in self._global_average_cache:
            return self._global_average_cache[cache_key], -1

        scores = []
        for u, v in G.edges():
            # Skip the target edge itself
            if {u, v} == {model_id, dataset_id}:
                continue
            
            # Get edge metadata
            edge_key = (u, v)
            if edge_key not in edge_metadata:
                edge_key = (v, u)  # Try reverse order
            
            if edge_key in edge_metadata:
                edge_data = edge_metadata[edge_key]
                if "metrics" in edge_data and metric_name in edge_data["metrics"]:
                    scores.append(float(edge_data["metrics"][metric_name]))
        
        avg = float(np.mean(scores)) if scores else 0.5
        self._global_average_cache[cache_key] = avg
        return avg, len(scores)

    def _calculate_dataset_average(
        self, G: nx.Graph, dataset_id: int, exclude_model_id: int, metric_name: str, edge_metadata: dict
    ) -> tuple[float, int]:
        scores = []
        for model_id in G.neighbors(dataset_id):
            # Skip the target model
            if model_id == exclude_model_id:
                continue
            
            # Only consider model nodes
            if G.nodes[model_id].get("type") == "model":
                # Get edge metadata for this model-dataset pair
                edge_key = (model_id, dataset_id)
                if edge_key not in edge_metadata:
                    edge_key = (dataset_id, model_id)  # Try reverse order
                
                if edge_key in edge_metadata:
                    edge_data = edge_metadata[edge_key]
                    if "metrics" in edge_data and metric_name in edge_data["metrics"]:
                        scores.append(float(edge_data["metrics"][metric_name]))
        
        avg = float(np.mean(scores)) if scores else 0.5
        return avg, len(scores)

    def predict(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
        edge_metadata: dict,
        metric_name: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            if self.mode == "global_average":
                value, n = self._calculate_global_average(G, model_id, dataset_id, metric_name, edge_metadata)
                reason = (
                    f"Global average of '{metric_name}' from {n if n >= 0 else 'cached'} edges."
                )
            else:
                value, n = self._calculate_dataset_average(G, dataset_id, model_id, metric_name, edge_metadata)
                dataset_name = node_metadata.get(dataset_id, {}).get("name", f"ID_{dataset_id}")
                reason = f"Average of '{metric_name}' from {n} other models on '{dataset_name}'."

            return {"prediction": value, "reason": reason}
        except Exception as e:
            model_name = node_metadata.get(model_id, {}).get("name", f"ID_{model_id}")
            dataset_name = node_metadata.get(dataset_id, {}).get("name", f"ID_{dataset_id}")
            print(f"Error predicting for ({model_name}, {dataset_name}): {e}")
            return None
