#!/usr/bin/env python3
"""
SOTAFinder evaluator – Uncertainty-aware evaluation for SOTA discovery.

Supports three scoring modes:
  * **mean-only**  (β = 0):  standard regression evaluation on μ.
  * **UCB**        (β > 0):  exploration-aware score  μ + β·σ.
  * **per-edge**:  returns full (μ, σ) for downstream analysis.

Metrics reported:
  * Standard regression: MSE, MAE, RMSE, R², MAPE.
  * Uncertainty quality:  avg_sigma, NLL (calibration proxy).
  * Degree-stratified R² (Tail / Medium / Head).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from ..utils.evaluation_utils import calculate_r_squared, evaluate_regression


class SOTAFinderEvaluator:
    """Evaluator for SOTAFinder (dual-head μ / log_var model).

    Args:
        beta: UCB exploration coefficient used when computing ranking scores.
              Set to 0 for pure mean-based evaluation.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(
        self,
        model,
        G: Data,
        split: Data,
        return_preds: bool = False,
    ) -> Dict[str, float] | Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Evaluate on a data split.

        Args:
            model:        SOTAFinder model with encode / decode.
            G:            Graph data (node features + edge_index).
            split:        Data split with edge_label_index and edge_label.
            return_preds: Also return per-edge prediction records.

        Returns:
            Metrics dict, optionally with prediction records.
        """
        model.eval()
        z = model.encode(G.x, G.edge_index)
        mu, log_var = model.decode(z, split.edge_label_index)

        # Clamp logits and convert to probability space
        mu_clipped = torch.clamp(mu, min=-10.0, max=10.0)
        y_pred_mean = torch.sigmoid(mu_clipped).cpu().numpy()

        # Sigma in probability space (delta-method approximation)
        sigma_logit = torch.exp(0.5 * log_var)
        sigma_prob = (sigma_logit * torch.sigmoid(mu_clipped) * (1 - torch.sigmoid(mu_clipped))).cpu().numpy()

        y_true = split.edge_label.cpu().numpy()

        # --- Standard regression metrics (on μ) ---
        metrics = self._compute_metrics(y_true, y_pred_mean)

        # --- Uncertainty-specific metrics ---
        metrics["avg_sigma"] = float(np.mean(sigma_prob))
        metrics["max_sigma"] = float(np.max(sigma_prob)) if len(sigma_prob) > 0 else 0.0

        # Gaussian NLL in logit space (calibration quality proxy)
        y_true_t = torch.tensor(y_true, dtype=torch.float32)
        y_clip = torch.clamp(y_true_t, min=1e-7, max=1 - 1e-7)
        y_logits = torch.log(y_clip / (1 - y_clip))
        var = torch.exp(log_var).clamp(min=1e-6).cpu()
        nll = 0.5 * (torch.log(var) + (mu.cpu() - y_logits).pow(2) / var).mean()
        metrics["nll"] = float(nll)

        # --- UCB exploration score (in probability space) ---
        ucb_score = y_pred_mean + self.beta * sigma_prob
        metrics["ucb_mean"] = float(np.mean(ucb_score))

        # --- Degree-stratified metrics ---
        degree_metrics = self._compute_degree_metrics(G, split, y_true, y_pred_mean)
        metrics.update(degree_metrics)

        if not return_preds:
            return metrics

        records = self._create_prediction_records(
            split, y_pred_mean, y_true, sigma_prob, ucb_score
        )
        return metrics, records

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Dict[str, float]:
        return evaluate_regression(y_pred.tolist(), y_true.tolist())

    def _compute_degree_metrics(
        self,
        G: Data,
        split: Data,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
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
        sigma: np.ndarray,
        ucb: np.ndarray,
    ) -> List[Dict[str, Any]]:
        edges_list = split.edge_label_index.t().cpu().numpy().tolist()
        return [
            {
                "input": {"edge": [int(u), int(v)]},
                "prediction": float(p),
                "ground_truth": float(t),
                "sigma": float(s),
                "ucb_score": float(sc),
            }
            for (u, v), p, t, s, sc in zip(
                edges_list, y_pred.tolist(), y_true.tolist(),
                sigma.tolist(), ucb.tolist()
            )
        ]

    def print_metrics(self, metrics: Dict[str, float], prefix: str = "test"):
        print(
            f"\n{prefix}_mse {metrics['mse']:.6f} | "
            f"{prefix}_r² {metrics['r_squared']:.4f} | "
            f"avg_σ {metrics.get('avg_sigma', 0):.4f} | "
            f"NLL {metrics.get('nll', 0):.4f}"
        )
        degree_keys = [k for k in metrics if any(b in k for b in ["Tail", "Medium", "Head"])]
        if degree_keys:
            print("\n--- Degree-Controlled Performance ---")
            for k in sorted(degree_keys):
                print(f"  {k}: {metrics[k]:.4f}")
            print("-" * 40)
