#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import random


class BaselineAttributeRanker:
    """Ranks models for a dataset based on different baseline strategies."""

    def __init__(self, mode: str = "downloads", seed: int = 42):
        """
        Initialize baseline attribute ranker.
        
        Args:
            mode: Ranking strategy
                - "downloads": Rank by download counts (highest first)
                - "random": Random ranking
                - "connectivity": Rank by node degree/connectivity (highest first)
                - "predicted": Rank by baseline predicted values (highest first)
            seed: Random seed for reproducible random rankings
        """
        self.mode = mode
        self.seed = seed
        if mode not in ["downloads", "random", "connectivity", "predicted"]:
            raise ValueError(f"Unknown mode: {mode}. Must be 'downloads', 'random', 'connectivity', or 'predicted'.")

    def rank(
        self,
        dataset_id: int,
        models_to_rank: List[Tuple[int, float]],
        G: nx.Graph,
        node_metadata: dict,
        edge_metadata: dict,
        metric_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Ranks a list of models based on the selected strategy.

        Args:
            dataset_id: The ID of the dataset.
            models_to_rank: A list of (model_id, true_metric_value) tuples.
            G: The NetworkX graph.
            node_metadata: Dictionary with node metadata.
            edge_metadata: Dictionary with edge metadata (for interface compatibility).
            metric_name: The name of the attribute (for interface compatibility).

        Returns:
            A dictionary containing the ranked list of models.
        """
        try:
            if self.mode == "downloads":
                return self._rank_by_downloads(dataset_id, models_to_rank, node_metadata)
            elif self.mode == "random":
                return self._rank_randomly(dataset_id, models_to_rank)
            elif self.mode == "connectivity":
                return self._rank_by_connectivity(dataset_id, models_to_rank, G)
            elif self.mode == "predicted":
                return self._rank_by_predicted_values(dataset_id, models_to_rank, G, node_metadata, edge_metadata, metric_name)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except Exception as e:
            print(f"Error ranking attributes with baseline: {e}")
            return None

    def _rank_by_downloads(self, dataset_id: int, models_to_rank: List[Tuple[int, float]], node_metadata: dict) -> Dict[str, Any]:
        """Rank models by download counts (highest first)."""
        model_ids = [m_id for m_id, _ in models_to_rank]

        models_with_downloads = []
        for model_id in model_ids:
            node_data = node_metadata.get(model_id, {})
            # Try different possible paths to downloads field
            downloads = node_data.get("downloads", 0)
            if downloads == 0:
                downloads = node_data.get("info", {}).get("downloads", 0)
            models_with_downloads.append((model_id, downloads))

        # Group models by download count and shuffle within each group
        from collections import defaultdict
        download_groups = defaultdict(list)
        for model_id, downloads in models_with_downloads:
            download_groups[downloads].append(model_id)
        
        # Create a local random instance for reproducibility
        rng = random.Random(self.seed)
        
        # Shuffle within each download group and create final ranking
        ranked_model_ids = []
        for downloads in sorted(download_groups.keys(), reverse=True):  # Highest downloads first
            group = download_groups[downloads]
            rng.shuffle(group)  # Randomly shuffle models with same download count
            ranked_model_ids.extend(group)
        
        ranked_pairs = []
        true_metrics_map = {m_id: metric for m_id, metric in models_to_rank}
        downloads_map = {m_id: dls for m_id, dls in models_with_downloads}

        for rank, model_id in enumerate(ranked_model_ids, 1):
            ranked_pairs.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "true_value": true_metrics_map[model_id],
                "expected_score": downloads_map[model_id]
            })

        return {
            "ranked_models": ranked_pairs,  # Changed from "ranked_pairs" to "ranked_models" for consistency
            "reasoning": f"Ranked {len(ranked_model_ids)} models for dataset {dataset_id} by download counts.",
        }

    def _rank_randomly(self, dataset_id: int, models_to_rank: List[Tuple[int, float]]) -> Dict[str, Any]:
        """Rank models randomly."""
        model_ids = [m_id for m_id, _ in models_to_rank]
        
        # Create a local random instance to ensure reproducibility
        rng = random.Random(self.seed)
        ranked_model_ids = model_ids.copy()
        rng.shuffle(ranked_model_ids)
        
        ranked_pairs = []
        true_metrics_map = {m_id: metric for m_id, metric in models_to_rank}

        for rank, model_id in enumerate(ranked_model_ids, 1):
            ranked_pairs.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "true_value": true_metrics_map[model_id],
                "expected_score": rng.random()  # Random score for ranking
            })

        return {
            "ranked_models": ranked_pairs,  # Changed from "ranked_pairs" to "ranked_models" for consistency
            "reasoning": f"Randomly ranked {len(ranked_model_ids)} models for dataset {dataset_id} (seed={self.seed}).",
        }

    def _rank_by_connectivity(self, dataset_id: int, models_to_rank: List[Tuple[int, float]], G: nx.Graph) -> Dict[str, Any]:
        """Rank models by their connectivity/degree in the graph (highest first)."""
        model_ids = [m_id for m_id, _ in models_to_rank]

        models_with_connectivity = []
        for model_id in model_ids:
            # Get the degree (number of connections) for this model
            connectivity = G.degree(model_id) if model_id in G else 0
            models_with_connectivity.append((model_id, float(connectivity)))

        # Group models by connectivity level and shuffle within each group
        from collections import defaultdict
        connectivity_groups = defaultdict(list)
        for model_id, connectivity in models_with_connectivity:
            connectivity_groups[connectivity].append(model_id)
        
        # Create a local random instance for reproducibility
        rng = random.Random(self.seed)
        
        # Shuffle within each connectivity group and create final ranking
        ranked_model_ids = []
        for connectivity in sorted(connectivity_groups.keys(), reverse=True):  # Highest connectivity first
            group = connectivity_groups[connectivity]
            rng.shuffle(group)  # Randomly shuffle models with same connectivity
            ranked_model_ids.extend(group)
        
        ranked_pairs = []
        true_metrics_map = {m_id: metric for m_id, metric in models_to_rank}
        connectivity_map = {m_id: conn for m_id, conn in models_with_connectivity}

        for rank, model_id in enumerate(ranked_model_ids, 1):
            ranked_pairs.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "true_value": true_metrics_map[model_id],
                "expected_score": connectivity_map[model_id]
            })

        return {
            "ranked_models": ranked_pairs,  # Changed from "ranked_pairs" to "ranked_models" for consistency
            "reasoning": f"Ranked {len(ranked_model_ids)} models for dataset {dataset_id} by graph connectivity (degree).",
        }

    def _rank_by_predicted_values(
        self, 
        dataset_id: int, 
        models_to_rank: List[Tuple[int, float]], 
        G: nx.Graph, 
        node_metadata: dict, 
        edge_metadata: dict, 
        metric_name: str
    ) -> Dict[str, Any]:
        """Rank models by their predicted values using baseline attribute predictor."""
        from artifact_graph.models.baseline_attribute_predictor import BaselineAttributePredictor
        
        # Use dataset_average mode for more meaningful predictions
        predictor = BaselineAttributePredictor(mode="dataset_average")
        
        model_ids = [m_id for m_id, _ in models_to_rank]
        models_with_predictions = []
        
        for model_id in model_ids:
            # Use baseline predictor to get predicted value for this model-dataset-metric combination
            prediction_result = predictor.predict(
                model_id=model_id,
                dataset_id=dataset_id,
                G=G,
                node_metadata=node_metadata,
                edge_metadata=edge_metadata,
                metric_name=metric_name
            )
            
            if prediction_result and "prediction" in prediction_result:
                predicted_value = prediction_result["prediction"]
            else:
                predicted_value = 0.5  # Default fallback value
            
            models_with_predictions.append((model_id, float(predicted_value)))
        
        # Group models by predicted value and shuffle within each group
        from collections import defaultdict
        prediction_groups = defaultdict(list)
        for model_id, predicted_value in models_with_predictions:
            # Round to avoid floating point precision issues
            rounded_prediction = round(predicted_value, 6)
            prediction_groups[rounded_prediction].append(model_id)
        
        # Create a local random instance for reproducibility
        rng = random.Random(self.seed)
        
        # Shuffle within each prediction group and create final ranking
        ranked_model_ids = []
        for predicted_value in sorted(prediction_groups.keys(), reverse=True):  # Highest predictions first
            group = prediction_groups[predicted_value]
            rng.shuffle(group)  # Randomly shuffle models with same predicted value
            ranked_model_ids.extend(group)
        
        ranked_pairs = []
        true_metrics_map = {m_id: metric for m_id, metric in models_to_rank}
        predictions_map = {m_id: pred for m_id, pred in models_with_predictions}

        for rank, model_id in enumerate(ranked_model_ids, 1):
            ranked_pairs.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "true_value": true_metrics_map[model_id],
                "expected_score": predictions_map[model_id]
            })

        return {
            "ranked_models": ranked_pairs,
            "reasoning": f"Ranked {len(ranked_model_ids)} models for dataset {dataset_id} by baseline predicted values for '{metric_name}'.",
        }
