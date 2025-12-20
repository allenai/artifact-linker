#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional
import networkx as nx
import numpy as np


class BaselineLinkPredictor:
    def __init__(self, mode: str = "downloads", **kwargs):
        """
        Initialize baseline link predictor with different modes.

        Args:
            mode: Prediction strategy
                - "downloads": Based on download thresholds (original)
                - "common_neighbors": Based on common neighbors count
                - "jaccard": Based on Jaccard coefficient
                - "adamic_adar": Based on Adamic-Adar index
                - "preferential_attachment": Based on degree product
                - "resource_allocation": Based on resource allocation index
                - "katz": Based on Katz centrality (simplified)
        """
        self.mode = mode
        
        # Mode-specific parameters
        if mode == "downloads":
            self.model_download_threshold = kwargs.get("model_download_threshold", 1000)
            self.dataset_download_threshold = kwargs.get("dataset_download_threshold", 100)
        elif mode == "common_neighbors":
            self.threshold = kwargs.get("threshold", 1)
        elif mode == "jaccard":
            self.threshold = kwargs.get("threshold", 0.1)
        elif mode == "adamic_adar":
            self.threshold = kwargs.get("threshold", 1.0)
        elif mode == "preferential_attachment":
            self.threshold = kwargs.get("threshold", 50)
        elif mode == "resource_allocation":
            self.threshold = kwargs.get("threshold", 0.1)
        elif mode == "katz":
            self.threshold = kwargs.get("threshold", 0.01)  # Lower default threshold
            self.beta = kwargs.get("beta", 0.1)  # Katz parameter
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_common_neighbors_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Count common neighbors between model and dataset."""
        model_neighbors = set(G.neighbors(model_id))
        dataset_neighbors = set(G.neighbors(dataset_id))
        return float(len(model_neighbors.intersection(dataset_neighbors)))

    def _get_jaccard_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate Jaccard coefficient."""
        model_neighbors = set(G.neighbors(model_id))
        dataset_neighbors = set(G.neighbors(dataset_id))
        intersection = model_neighbors.intersection(dataset_neighbors)
        union = model_neighbors.union(dataset_neighbors)
        return len(intersection) / len(union) if union else 0

    def _get_adamic_adar_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate Adamic-Adar index."""
        model_neighbors = set(G.neighbors(model_id))
        dataset_neighbors = set(G.neighbors(dataset_id))
        common_neighbors = model_neighbors.intersection(dataset_neighbors)
        
        score = 0
        for neighbor in common_neighbors:
            degree = G.degree(neighbor)
            if degree > 1:
                score += 1 / np.log(degree)
        return score

    def _get_preferential_attachment_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate preferential attachment score (product of degrees)."""
        return float(G.degree(model_id) * G.degree(dataset_id))

    def _get_resource_allocation_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate resource allocation index."""
        model_neighbors = set(G.neighbors(model_id))
        dataset_neighbors = set(G.neighbors(dataset_id))
        common_neighbors = model_neighbors.intersection(dataset_neighbors)
        
        score = 0
        for neighbor in common_neighbors:
            degree = G.degree(neighbor)
            if degree > 0:
                score += 1 / degree
        return score

    def _get_katz_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Calculate simplified Katz centrality score for link prediction."""
        try:
            # For link prediction, we should NOT consider direct paths (since we're predicting missing links)
            # Instead, we calculate Katz-like score based on paths through intermediary nodes
            
            # Create a temporary graph without the potential edge
            G_temp = G.copy()
            if G_temp.has_edge(model_id, dataset_id):
                G_temp.remove_edge(model_id, dataset_id)
            
            score = 0.0
            # Calculate score based on paths of length 2, 3, 4 (limited for efficiency)
            for path_length in range(2, 5):  # paths of length 2, 3, 4
                try:
                    # Count all paths of this length
                    if path_length == 2:
                        # Paths of length 2: model -> intermediate -> dataset
                        model_neighbors = set(G_temp.neighbors(model_id))
                        dataset_neighbors = set(G_temp.neighbors(dataset_id))
                        common_neighbors = model_neighbors.intersection(dataset_neighbors)
                        paths_count = len(common_neighbors)
                    else:
                        # For longer paths, use a simplified approximation
                        # This is computationally expensive, so we use a heuristic
                        paths_count = 0
                        if nx.has_path(G_temp, model_id, dataset_id):
                            try:
                                shortest_path = nx.shortest_path_length(G_temp, model_id, dataset_id)
                                if shortest_path == path_length:
                                    paths_count = 1  # Simplified: assume 1 path of this length
                            except:
                                paths_count = 0
                    
                    score += (self.beta ** path_length) * paths_count
                except:
                    continue
            
            return float(score)
        except:
            return 0.0

    def predict(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
    ) -> Optional[Dict[str, Any]]:
        try:
            if self.mode == "downloads":
                return self._predict_downloads(model_id, dataset_id, G, node_metadata)
            elif self.mode == "common_neighbors":
                return self._predict_common_neighbors(model_id, dataset_id, G, node_metadata)
            elif self.mode == "jaccard":
                return self._predict_jaccard(model_id, dataset_id, G, node_metadata)
            elif self.mode == "adamic_adar":
                return self._predict_adamic_adar(model_id, dataset_id, G, node_metadata)
            elif self.mode == "preferential_attachment":
                return self._predict_preferential_attachment(model_id, dataset_id, G, node_metadata)
            elif self.mode == "resource_allocation":
                return self._predict_resource_allocation(model_id, dataset_id, G, node_metadata)
            elif self.mode == "katz":
                return self._predict_katz(model_id, dataset_id, G, node_metadata)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as e:
            print(f"Error predicting for ({model_id}, {dataset_id}): {e}")
            return None

    def _predict_downloads(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        node_metadata = node_metadata or {}
        
        model_downloads = 0
        if G and model_id in G.nodes:
            model_downloads = G.nodes[model_id].get("downloads", 0)
        elif model_id in node_metadata:
            model_downloads = node_metadata[model_id].get("downloads", 0)

        dataset_downloads = 0
        if G and dataset_id in G.nodes:
            dataset_downloads = G.nodes[dataset_id].get("downloads", 0)
        elif dataset_id in node_metadata:
            dataset_downloads = node_metadata[dataset_id].get("downloads", 0)

        model_name = node_metadata.get(model_id, {}).get("name", f"ID_{model_id}")
        dataset_name = node_metadata.get(dataset_id, {}).get("name", f"ID_{dataset_id}")

        prediction = (model_downloads >= self.model_download_threshold and
                      dataset_downloads >= self.dataset_download_threshold)

        reason = (
            f"Model '{model_name}' ({model_downloads} downloads, threshold: {self.model_download_threshold}) & "
            f"Dataset '{dataset_name}' ({dataset_downloads} downloads, threshold: {self.dataset_download_threshold}). "
            f"Link predicted: {prediction}."
        )

        return {
            "prediction": prediction,
            "reason": reason,
            "model_downloads": model_downloads,
            "dataset_downloads": dataset_downloads,
        }

    def _predict_common_neighbors(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_common_neighbors_score(model_id, dataset_id, G)
        prediction = score >= self.threshold
        reason = f"Common neighbors: {score} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

    def _predict_jaccard(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_jaccard_score(model_id, dataset_id, G)
        prediction = score >= self.threshold
        reason = f"Jaccard coefficient: {score:.4f} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

    def _predict_adamic_adar(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_adamic_adar_score(model_id, dataset_id, G)
        prediction = score >= self.threshold
        reason = f"Adamic-Adar index: {score:.4f} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

    def _predict_preferential_attachment(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_preferential_attachment_score(model_id, dataset_id, G)
        prediction = score >= self.threshold
        reason = f"Preferential attachment: {score} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

    def _predict_resource_allocation(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_resource_allocation_score(model_id, dataset_id, G)
        prediction = score >= self.threshold
        reason = f"Resource allocation: {score:.4f} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

    def _predict_katz(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_katz_score(model_id, dataset_id, G)
        prediction = score >= self.threshold
        reason = f"Katz score: {score:.4f} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}