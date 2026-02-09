#!/usr/bin/env python3
"""GNN evaluation utilities for link prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    metrics: Dict[str, float]
    predictions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"metrics": self.metrics}
        if self.predictions is not None:
            result["predictions"] = self.predictions
        return result


class GNNLinkEvaluator:
    """Evaluator for GNN link prediction models."""

    # Degree bucket definitions
    DEGREE_BUCKETS = {
        "Tail (deg<=5)": lambda d: d <= 5,
        "Medium (5<deg<=20)": lambda d: (d > 5) & (d <= 20),
        "Head (deg>20)": lambda d: d > 20,
    }

    def __init__(self, threshold: float = 0.5):
        """
        Initialize evaluator.

        Args:
            threshold: Probability threshold for binary classification.
        """
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(
        self,
        model,
        z: torch.Tensor,
        split,
        node_degrees: Optional[torch.Tensor] = None,
        return_predictions: bool = False,
    ) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Evaluate model on a data split.

        Args:
            model: GNN model with decode method.
            z: Node embeddings from encoder.
            split: Data split with pos/neg edge indices.
            node_degrees: Optional node degrees for stratified evaluation.
            return_predictions: Whether to return detailed predictions.

        Returns:
            Metrics dictionary, optionally with predictions data.
        """
        model.eval()

        # Compute predictions
        pos_logits = model.decode(z, split.pos_edge_label_index)
        neg_logits = model.decode(z, split.neg_edge_label_index)

        y_true = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0
        ).cpu().numpy()

        y_prob = torch.sigmoid(
            torch.cat([pos_logits, neg_logits], dim=0)
        ).cpu().numpy()

        y_pred = (y_prob > self.threshold).astype(np.int32)

        # Compute base metrics
        metrics = self._compute_metrics(y_true, y_pred, y_prob)

        # Compute degree-controlled metrics if degrees provided
        if node_degrees is not None:
            all_edges = torch.cat(
                [split.pos_edge_label_index, split.neg_edge_label_index], dim=1
            )
            degree_metrics = self._compute_degree_metrics(
                y_true, y_pred, y_prob, all_edges, node_degrees
            )
            metrics.update(degree_metrics)

        if return_predictions:
            predictions = self._create_predictions_data(
                split, y_true, y_prob, y_pred
            )
            return metrics, predictions

        return metrics

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
        }

    def _compute_degree_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        all_edges: torch.Tensor,
        node_degrees: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute metrics stratified by node degree."""
        u_deg = node_degrees[all_edges[0]].cpu().numpy()
        v_deg = node_degrees[all_edges[1]].cpu().numpy()
        edge_min_deg = np.minimum(u_deg, v_deg)

        metrics = {}

        for bucket_name, condition in self.DEGREE_BUCKETS.items():
            mask = condition(edge_min_deg)

            if mask.sum() > 0:
                sub_true = y_true[mask]
                sub_pred = y_pred[mask]
                sub_prob = y_prob[mask]

                metrics[f"f1_{bucket_name}"] = float(
                    f1_score(sub_true, sub_pred, zero_division=0)
                )
                metrics[f"acc_{bucket_name}"] = float(accuracy_score(sub_true, sub_pred))

                try:
                    metrics[f"auc_{bucket_name}"] = float(
                        roc_auc_score(sub_true, sub_prob)
                    )
                except ValueError:
                    # Only one class present
                    metrics[f"auc_{bucket_name}"] = 0.0

        return metrics

    def _create_predictions_data(
        self,
        split,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Create detailed predictions data for saving."""
        edge_index = torch.cat(
            [split.pos_edge_label_index, split.neg_edge_label_index], dim=1
        )
        edge_indices = edge_index.cpu().numpy()

        edges = [
            {
                "v_id": int(edge_indices[0, i]),
                "u_id": int(edge_indices[1, i]),
                "ground_truth": float(y_true[i]),
                "prediction_prob": float(y_prob[i]),
                "prediction_binary": int(y_pred[i]),
            }
            for i in range(len(y_true))
        ]

        return {
            "edges": edges,
            "summary": {
                "total_edges": len(y_true),
                "positive_edges": int(y_true.sum()),
                "negative_edges": int(len(y_true) - y_true.sum()),
                "correct_predictions": int((y_true == y_pred).sum()),
                "accuracy": float((y_true == y_pred).mean()),
            },
        }

    def print_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "test",
    ):
        """Print metrics in a formatted way."""
        print(f"\n{prefix}_auc {metrics['auc']:.4f} | "
              f"{prefix}_f1 {metrics['f1']:.4f} | "
              f"{prefix}_prec {metrics['precision']:.4f} | "
              f"{prefix}_rec {metrics['recall']:.4f} | "
              f"{prefix}_acc {metrics['accuracy']:.4f}")

        # Print degree-controlled breakdown
        degree_keys = [k for k in metrics if any(b in k for b in ["Tail", "Medium", "Head"])]
        if degree_keys:
            print("\n--- Degree-Controlled Performance ---")
            for k in sorted(degree_keys):
                print(f"  {k}: {metrics[k]:.4f}")
            print("-" * 40)
