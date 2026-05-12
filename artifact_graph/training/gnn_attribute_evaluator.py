#!/usr/bin/env python3
"""GNN evaluation utilities for attribute prediction (regression)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from ..utils.evaluation_utils import calculate_r_squared, evaluate_regression


class GNNAttributeEvaluator:
    """Evaluator for GNN attribute prediction (regression) models."""

    @torch.no_grad()
    def evaluate(
        self,
        model,
        G: Data,
        split: Data,
        return_preds: bool = False,
    ) -> Dict[str, float] | Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Evaluate model on a data split.

        Args:
            model: GNN model with encode/decode methods.
            G: Graph data with node features and edge_index.
            split: Data split with edge_label_index and edge_label.
            return_preds: Whether to return detailed predictions.

        Returns:
            Metrics dictionary, optionally with prediction records.
        """
        model.eval()
        z = model.encode(G.x, G.edge_index)
        logits = model.decode(z, split.edge_label_index).squeeze(-1)
        logits_clipped = torch.clamp(logits, min=-10.0, max=10.0)
        y_pred = torch.sigmoid(logits_clipped).cpu().numpy()
        y_true = split.edge_label.cpu().numpy()

        # Base regression metrics
        metrics = self._compute_metrics(y_true, y_pred)

        # Degree-controlled metrics
        degree_metrics = self._compute_degree_metrics(G, split, y_true, y_pred)
        metrics.update(degree_metrics)

        if not return_preds:
            return metrics

        records = self._create_prediction_records(split, y_pred, y_true)
        return metrics, records

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard regression metrics."""
        return evaluate_regression(y_pred.tolist(), y_true.tolist())

    def _compute_degree_metrics(
        self,
        G: Data,
        split: Data,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics stratified by node degree."""
        node_degrees = degree(G.edge_index[0], G.num_nodes)
        edges = split.edge_label_index
        edge_min_deg = np.minimum(
            node_degrees[edges[0]].cpu().numpy(),
            node_degrees[edges[1]].cpu().numpy(),
        )

        metrics = {}
        for name, mask in [
            ("Tail", edge_min_deg <= 5),
            ("Medium", (edge_min_deg > 5) & (edge_min_deg <= 20)),
            ("Head", edge_min_deg > 20),
        ]:
            if mask.sum() > 0:
                metrics[f"r_squared_{name}"] = calculate_r_squared(
                    y_pred[mask].tolist(), y_true[mask].tolist()
                )

        return metrics

    def _create_prediction_records(
        self,
        split: Data,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create detailed prediction records for saving."""
        edges_list = split.edge_label_index.t().cpu().numpy().tolist()
        return [
            {
                "input": {"edge": [int(u), int(v)]},
                "prediction": float(p),
                "ground_truth": float(t),
            }
            for (u, v), p, t in zip(edges_list, y_pred.tolist(), y_true.tolist())
        ]

    def print_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "test",
    ):
        """Print metrics in a formatted way."""
        print(f"\n{prefix}_mse {metrics['mse']:.6f} | "
              f"{prefix}_r_squared {metrics['r_squared']:.4f}")

        degree_keys = [k for k in metrics if any(b in k for b in ["Tail", "Medium", "Head"])]
        if degree_keys:
            print("\n--- Degree-Controlled Performance ---")
            for k in sorted(degree_keys):
                print(f"  {k}: {metrics[k]:.4f}")
            print("-" * 40)
