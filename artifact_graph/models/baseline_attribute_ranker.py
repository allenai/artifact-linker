#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import random


class BaselineAttributeRanker:
    """Ranks models for a dataset using baseline attribute prediction strategies."""

    VALID_MODES = {"global_average", "dataset_average", "model_average"}

    def __init__(self, mode: str = "dataset_average", seed: int = 42):
        """
        Initialize baseline attribute ranker.

        Args:
            mode: Prediction strategy used to produce scores for ranking.
                - "dataset_average": Average metric across other models on the same dataset.
                - "model_average": Average metric across other datasets for the same model.
                - "global_average": Average metric across all edges in the graph.
            seed: Random seed for breaking ties reproducibly.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode: {mode}. Must be one of {sorted(self.VALID_MODES)}."
            )
        self.mode = mode
        self.seed = seed

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
        Ranks a list of models based on predicted attribute values.

        Args:
            dataset_id: The ID of the dataset.
            models_to_rank: A list of (model_id, true_metric_value) tuples.
            G: The NetworkX graph.
            node_metadata: Dictionary with node metadata.
            edge_metadata: Dictionary with edge metadata.
            metric_name: The name of the metric to rank on.

        Returns:
            A dictionary containing the ranked list of models, or None on error.
        """
        try:
            return self._rank_by_predicted_values(
                dataset_id, models_to_rank, G, node_metadata, edge_metadata, metric_name
            )
        except Exception as e:
            print(f"Error ranking attributes with baseline ({self.mode}): {e}")
            return None

    def _rank_by_predicted_values(
        self,
        dataset_id: int,
        models_to_rank: List[Tuple[int, float]],
        G: nx.Graph,
        node_metadata: dict,
        edge_metadata: dict,
        metric_name: str,
    ) -> Dict[str, Any]:
        """Rank models by their predicted values using BaselineAttributePredictor."""
        from artifact_graph.models.baseline_attribute_predictor import BaselineAttributePredictor

        predictor = BaselineAttributePredictor(mode=self.mode)

        model_ids = [m_id for m_id, _ in models_to_rank]
        models_with_predictions = []

        for model_id in model_ids:
            prediction_result = predictor.predict(
                model_id=model_id,
                dataset_id=dataset_id,
                G=G,
                node_metadata=node_metadata,
                edge_metadata=edge_metadata,
                metric_name=metric_name,
            )

            if prediction_result and "prediction" in prediction_result:
                predicted_value = prediction_result["prediction"]
            else:
                predicted_value = 0.5  # Default fallback value

            models_with_predictions.append((model_id, float(predicted_value)))

        # Group models by predicted value and shuffle within each group (tie-breaking)
        prediction_groups: Dict[float, List[int]] = defaultdict(list)
        for model_id, predicted_value in models_with_predictions:
            rounded_prediction = round(predicted_value, 6)
            prediction_groups[rounded_prediction].append(model_id)

        rng = random.Random(self.seed)

        ranked_model_ids = []
        for predicted_value in sorted(prediction_groups.keys(), reverse=True):
            group = prediction_groups[predicted_value]
            rng.shuffle(group)
            ranked_model_ids.extend(group)

        true_metrics_map = {m_id: metric for m_id, metric in models_to_rank}
        predictions_map = {m_id: pred for m_id, pred in models_with_predictions}

        ranked_pairs = []
        for rank, model_id in enumerate(ranked_model_ids, 1):
            ranked_pairs.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                "rank": rank,
                "true_value": true_metrics_map[model_id],
                "expected_score": predictions_map[model_id],
            })

        return {
            "dataset_id": dataset_id,
            "metric_used": metric_name,
            "ranked_models": ranked_pairs,
            "reasoning": (
                f"Ranked {len(ranked_model_ids)} models for dataset {dataset_id} "
                f"by baseline predicted values ({self.mode}) for '{metric_name}'."
            ),
        }
