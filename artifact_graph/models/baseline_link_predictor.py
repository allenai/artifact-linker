#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional
import networkx as nx
import numpy as np


class BaselineLinkPredictor:
    VALID_MODES = [
        "downloads", "random", "connectivity", "common_neighbors",
        "jaccard", "adamic_adar", "preferential_attachment",
        "resource_allocation", "katz", "matrix_factorization",
    ]

    def __init__(self, mode: str = "downloads", **kwargs):
        """
        Initialize baseline link predictor with different modes.

        Args:
            mode: Prediction strategy
                - "downloads": Based on download thresholds
                - "random": Random prediction with probability threshold
                - "connectivity": Based on combined node degrees
                - "common_neighbors": Based on common neighbors count
                - "jaccard": Based on Jaccard coefficient
                - "adamic_adar": Based on Adamic-Adar index
                - "preferential_attachment": Based on degree product
                - "resource_allocation": Based on resource allocation index
                - "katz": Based on Katz centrality (simplified)
        """
        self.mode = mode
        self.seed = kwargs.get("seed", 42)

        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}. Must be one of {self.VALID_MODES}.")

        # Mode-specific parameters
        if mode == "downloads":
            self.model_download_threshold = kwargs.get("model_download_threshold", 1000)
            self.dataset_download_threshold = kwargs.get("dataset_download_threshold", 100)
        elif mode == "random":
            self.threshold = kwargs.get("threshold", 0.5)
        elif mode == "connectivity":
            self.threshold = kwargs.get("threshold", 10)  # Combined degree threshold
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
            self.threshold = kwargs.get("threshold", 0.01)
            self.beta = kwargs.get("beta", 0.1)
        elif mode == "matrix_factorization":
            self.threshold = kwargs.get("threshold", 0.5)
            self.mf_rank = kwargs.get("mf_rank", 64)
            self._mf_scores = None

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

    def _precompute_katz(self, G: nx.Graph):
        """Precompute Katz score matrix using sparse matrix powers (one-time cost)."""
        import scipy.sparse as sp

        nodes = sorted(G.nodes())
        self._katz_node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float64)
        A.setdiag(0)
        A.eliminate_zeros()

        # Katz score = sum_{l=2}^{4} beta^l * A^l  (skip l=1 since direct edges are not paths)
        A2 = A.dot(A)
        A3 = A2.dot(A)
        A4 = A3.dot(A)

        self._katz_matrix = (
            (self.beta ** 2) * A2
            + (self.beta ** 3) * A3
            + (self.beta ** 4) * A4
        )
        self._katz_matrix.setdiag(0)
        self._katz_matrix.eliminate_zeros()
        print(f"Precomputed Katz matrix: {n} nodes, beta={self.beta}, nnz={self._katz_matrix.nnz}")

    def _get_katz_score(self, model_id: int, dataset_id: int, G: nx.Graph) -> float:
        """Look up precomputed Katz score for a node pair."""
        if not hasattr(self, "_katz_matrix") or self._katz_matrix is None:
            self._precompute_katz(G)

        i = self._katz_node_to_idx.get(model_id)
        j = self._katz_node_to_idx.get(dataset_id)
        if i is None or j is None:
            return 0.0
        return float(self._katz_matrix[i, j])

    def _precompute_mf(self, G: nx.Graph, node_metadata: dict):
        """Precompute Matrix Factorization scores via truncated SVD on the adjacency matrix."""
        from scipy.sparse.linalg import svds
        import scipy.sparse as sp

        nodes = sorted(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float64)
        A.setdiag(0)
        A.eliminate_zeros()

        k = min(self.mf_rank, min(A.shape) - 1)
        U, S, Vt = svds(A.astype(np.float64), k=k)

        # Reconstruct: score(i, j) = (U * S) @ Vt  -> row i dot col j
        U_S = U * S[np.newaxis, :]  # (n, k)
        self._mf_U_S = U_S
        self._mf_Vt = Vt  # (k, n)
        self._mf_node_to_idx = node_to_idx
        print(f"Precomputed MF: {n} nodes, rank={k}")

    def _get_mf_score(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> float:
        """Look up precomputed MF score for a node pair."""
        if self._mf_scores is None and not hasattr(self, "_mf_U_S"):
            self._precompute_mf(G, node_metadata)

        i = self._mf_node_to_idx.get(model_id)
        j = self._mf_node_to_idx.get(dataset_id)
        if i is None or j is None:
            return 0.0
        return float(self._mf_U_S[i] @ self._mf_Vt[:, j])

    def predict(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
    ) -> Optional[Dict[str, Any]]:
        try:
            predict_fn = {
                "downloads": self._predict_downloads,
                "random": self._predict_random,
                "connectivity": self._predict_connectivity,
                "common_neighbors": self._predict_common_neighbors,
                "jaccard": self._predict_jaccard,
                "adamic_adar": self._predict_adamic_adar,
                "preferential_attachment": self._predict_preferential_attachment,
                "resource_allocation": self._predict_resource_allocation,
                "katz": self._predict_katz,
                "matrix_factorization": self._predict_matrix_factorization,
            }.get(self.mode)

            if predict_fn is None:
                raise ValueError(f"Unknown mode: {self.mode}")
            return predict_fn(model_id, dataset_id, G, node_metadata)
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
            "score": float(model_downloads + dataset_downloads),  # Combined score for ranking
        }

    def _predict_random(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        """Random prediction based on a seeded random value."""
        import random
        rng = random.Random(self.seed + hash((model_id, dataset_id)) % 10000)
        score = rng.random()
        prediction = score >= self.threshold
        reason = f"Random score: {score:.4f} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

    def _predict_connectivity(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        """Predict based on combined node degrees."""
        m_deg = G.degree(model_id) if model_id in G else 0
        d_deg = G.degree(dataset_id) if dataset_id in G else 0
        score = float(m_deg + d_deg)
        prediction = score >= self.threshold
        reason = f"Combined degree: {score} (model={m_deg}, dataset={d_deg}) ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}

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

    def _predict_matrix_factorization(self, model_id: int, dataset_id: int, G: nx.Graph, node_metadata: dict) -> Dict[str, Any]:
        score = self._get_mf_score(model_id, dataset_id, G, node_metadata)
        prediction = score >= self.threshold
        reason = f"MF score: {score:.4f} ({'✓' if prediction else '✗'} >= {self.threshold})"
        return {"prediction": prediction, "reason": reason, "score": score}