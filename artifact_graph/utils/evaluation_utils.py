#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from scipy.stats import kendalltau, spearmanr

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    kendalltau = None
    spearmanr = None

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


def calculate_ndcg(ranked_items: List[Any], relevant_items: set, k: Optional[int] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    Args:
        ranked_items: List of items in ranked order
        relevant_items: Set of relevant items
        k: Consider only top k items (None for all items)

    Returns:
        NDCG score
    """
    if k is not None:
        ranked_items = ranked_items[:k]

    if not ranked_items or not relevant_items:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Calculate IDCG (ideal DCG)
    num_relevant = min(len(relevant_items), len(ranked_items))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_map(ranked_items: List[Any], relevant_items: set, k: Optional[int] = None) -> float:
    """
    Calculate Mean Average Precision (MAP).

    Args:
        ranked_items: List of items in ranked order
        relevant_items: Set of relevant items
        k: Consider only top k items (None for all items)

    Returns:
        MAP score
    """
    if k is not None:
        ranked_items = ranked_items[:k]

    if not ranked_items or not relevant_items:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0

    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / len(relevant_items) if relevant_items else 0.0


def calculate_recall_at_k(ranked_items: List[Any], relevant_items: set, k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        ranked_items: List of items in ranked order
        relevant_items: Set of relevant items
        k: Number of top items to consider

    Returns:
        Recall@K score
    """
    if not relevant_items:
        return 0.0

    top_k = ranked_items[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    return relevant_in_top_k / len(relevant_items)


def calculate_precision_at_k(ranked_items: List[Any], relevant_items: set, k: int) -> float:
    """
    Calculate Precision@K.

    Args:
        ranked_items: List of items in ranked order
        relevant_items: Set of relevant items
        k: Number of top items to consider

    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0

    top_k = ranked_items[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    return relevant_in_top_k / k


def calculate_mrr(ranked_items: List[Any], relevant_items: set) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        ranked_items: List of items in ranked order
        relevant_items: Set of relevant items

    Returns:
        MRR score (reciprocal rank of first relevant item)
    """
    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_standard(
    predicted_ranking: List[Tuple[Any, float]],
    ground_truth: Dict[Any, float],
    k: Optional[int] = None,
    num_levels: int = 5,
) -> float:
    """
    Calculate standard NDCG with discrete relevance levels (0, 1, 2, 3, 4).

    Args:
        predicted_ranking: List of (item, predicted_score) tuples in ranked order
        ground_truth: Dictionary mapping items to ground truth relevance scores
        k: Consider only top k items (None for all items)
        num_levels: Number of discrete relevance levels (default: 5, i.e., 0-4)

    Returns:
        NDCG score using standard formula: DCG@k = Σ(2^rel_i - 1) / log2(i + 1)
    """
    if k is not None:
        predicted_ranking = predicted_ranking[:k]

    if not predicted_ranking:
        return 0.0

    # Convert continuous scores to discrete relevance levels
    all_scores = list(ground_truth.values())
    if not all_scores:
        return 0.0

    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score

    def score_to_relevance(score):
        if score_range == 0:
            return num_levels - 1  # All items have same score, assign highest relevance
        # Map continuous score to discrete level [0, num_levels-1]
        normalized = (score - min_score) / score_range
        return min(int(normalized * num_levels), num_levels - 1)

    # Calculate DCG using standard formula
    dcg = 0.0
    for i, (item, _) in enumerate(predicted_ranking):
        if item in ground_truth:
            rel_i = score_to_relevance(ground_truth[item])
            dcg += (2**rel_i - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Calculate IDCG (ideal DCG)
    all_items_with_relevance = [
        (item, score_to_relevance(score)) for item, score in ground_truth.items()
    ]
    # Sort by relevance in descending order
    ideal_ranking = sorted(all_items_with_relevance, key=lambda x: x[1], reverse=True)

    if k is not None:
        ideal_ranking = ideal_ranking[:k]

    idcg = 0.0
    for i, (_, relevance) in enumerate(ideal_ranking):
        idcg += (2**relevance - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_ndcg_continuous(
    predicted_ranking: List[Tuple[Any, float]],
    ground_truth: Dict[Any, float],
    k: Optional[int] = None,
) -> float:
    """
    Calculate NDCG for continuous relevance scores (simplified for attribute ranking).

    This is a simplified version that directly uses continuous values as relevance scores.
    While not standard NDCG, it's useful for ranking tasks with continuous ground truth.

    Args:
        predicted_ranking: List of (item, predicted_score) tuples in ranked order
        ground_truth: Dictionary mapping items to ground truth relevance scores
        k: Consider only top k items (None for all items)

    Returns:
        NDCG score using continuous relevance: DCG@k = Σ(rel_i) / log2(i + 1)
    """
    if k is not None:
        predicted_ranking = predicted_ranking[:k]

    if not predicted_ranking:
        return 0.0

    # Calculate DCG for predicted ranking (simplified formula for continuous values)
    dcg = 0.0
    for i, (item, _) in enumerate(predicted_ranking):
        if item in ground_truth:
            relevance = ground_truth[item]
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Calculate IDCG (ideal DCG) - sort by ground truth scores
    all_items_with_scores = [(item, score) for item, score in ground_truth.items()]
    # Sort by ground truth scores in descending order
    ideal_ranking = sorted(all_items_with_scores, key=lambda x: x[1], reverse=True)

    if k is not None:
        ideal_ranking = ideal_ranking[:k]

    idcg = 0.0
    for i, (_, relevance) in enumerate(ideal_ranking):
        idcg += relevance / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_map_continuous(
    predicted_ranking: List[Tuple[Any, float]],
    ground_truth: Dict[Any, float],
    k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> float:
    """
    Calculate MAP for continuous relevance scores (attribute ranking).

    For continuous scores, we define "relevant" items as those above a threshold.
    If no threshold is provided, we use the median of ground truth scores.

    Args:
        predicted_ranking: List of (item, predicted_score) tuples in ranked order
        ground_truth: Dictionary mapping items to ground truth relevance scores
        k: Consider only top k items (None for all items)
        threshold: Relevance threshold (None to use median)

    Returns:
        MAP score
    """
    if k is not None:
        predicted_ranking = predicted_ranking[:k]

    if not predicted_ranking:
        return 0.0

    # Determine relevance threshold
    if threshold is None:
        all_scores = list(ground_truth.values())
        if not all_scores:
            return 0.0
        threshold = sorted(all_scores)[len(all_scores) // 2]  # median

    # Create relevant items set based on threshold
    relevant_items = {item for item, score in ground_truth.items() if score > threshold}

    if not relevant_items:
        return 0.0

    # Calculate MAP using binary relevance
    relevant_count = 0
    precision_sum = 0.0

    for i, (item, _) in enumerate(predicted_ranking):
        if item in relevant_items:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / len(relevant_items) if relevant_items else 0.0


def calculate_ranking_correlation(
    predicted_ranking: List[Tuple[Any, float]], ground_truth: Dict[Any, float]
) -> Dict[str, float]:
    """
    Calculate comprehensive ranking correlation and global ranking metrics.

    Args:
        predicted_ranking: List of (item, predicted_score) tuples
        ground_truth: Dictionary mapping items to ground truth scores

    Returns:
        Dictionary with various correlation and ranking metrics
    """
    results = {}

    # Filter to items that exist in both rankings
    common_items = []
    pred_scores = []
    true_scores = []

    for item, pred_score in predicted_ranking:
        if item in ground_truth:
            common_items.append(item)
            pred_scores.append(pred_score)
            true_scores.append(ground_truth[item])

    if len(common_items) < 2:
        return {
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "pearson_r": 0.0,
            "weighted_kendall_tau": 0.0,
            "rank_biased_overlap": 0.0,
            "normalized_kendall_distance": 1.0,
            "footrule_distance": 1.0,
            "weighted_footrule_distance": 1.0,
            "top_1_overlap": 0.0,
            "top_3_overlap": 0.0,
            "top_5_overlap": 0.0,
            "top_10_overlap": 0.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "hit_at_10": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "common_items": len(common_items),
        }

    # Convert scores to ranks
    pred_ranks = _scores_to_ranks(pred_scores)
    true_ranks = _scores_to_ranks(true_scores)

    # Standard correlations
    if SCIPY_AVAILABLE:
        try:
            tau, tau_p = kendalltau(pred_scores, true_scores)
            rho, rho_p = spearmanr(pred_scores, true_scores)
            results["kendall_tau"] = float(tau) if not math.isnan(tau) else 0.0
            results["spearman_rho"] = float(rho) if not math.isnan(rho) else 0.0
            results["kendall_tau_pvalue"] = float(tau_p) if not math.isnan(tau_p) else 1.0
            results["spearman_rho_pvalue"] = float(rho_p) if not math.isnan(rho_p) else 1.0

            # Pearson correlation on scores
            if NUMPY_AVAILABLE:
                pearson_r = float(np.corrcoef(pred_scores, true_scores)[0, 1])
                results["pearson_r"] = pearson_r if not math.isnan(pearson_r) else 0.0
            else:
                results["pearson_r"] = _calculate_pearson_manual(pred_scores, true_scores)

        except Exception as e:
            print(f"Warning: Could not calculate correlations: {e}")
            results["kendall_tau"] = 0.0
            results["spearman_rho"] = 0.0
            results["pearson_r"] = 0.0
    else:
        results["kendall_tau"] = 0.0
        results["spearman_rho"] = 0.0
        results["pearson_r"] = 0.0
        print("Warning: scipy not available, correlation metrics set to 0")

    # Advanced ranking metrics
    results["weighted_kendall_tau"] = _calculate_weighted_kendall_tau(pred_ranks, true_ranks)
    results["rank_biased_overlap"] = _calculate_rank_biased_overlap(pred_ranks, true_ranks, p=0.9)
    results["normalized_kendall_distance"] = _calculate_normalized_kendall_distance(
        pred_ranks, true_ranks
    )
    results["footrule_distance"] = _calculate_footrule_distance(pred_ranks, true_ranks)
    results["weighted_footrule_distance"] = _calculate_weighted_footrule_distance(
        pred_ranks, true_ranks
    )

    # Top-k overlap and Hit@k metrics
    n = len(common_items)
    k_values = [1, 3, 5, min(10, n)]
    for k in k_values:
        if k <= n:
            overlap = _calculate_top_k_overlap(pred_ranks, true_ranks, k)
            results[f"top_{k}_overlap"] = overlap

            # Hit@k: whether the top-1 true item is in predicted top-k
            hit_at_k = _calculate_hit_at_k(pred_ranks, true_ranks, k)
            results[f"hit_at_{k}"] = hit_at_k

            # Recall@k: fraction of top-3 true items in predicted top-k
            recall_at_k = _calculate_recall_at_k(pred_ranks, true_ranks, k, top_true_k=3)
            results[f"recall_at_{k}"] = recall_at_k

    results["common_items"] = len(common_items)
    return results


# Helper functions for advanced ranking metrics
def _scores_to_ranks(scores: List[float]) -> List[int]:
    """Convert scores to ranks (1-based, higher score = lower rank number)."""
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1
    return ranks


def _calculate_pearson_manual(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation manually when numpy is not available."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y[i] - mean_y) ** 2 for i in range(n))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / math.sqrt(den_x * den_y)


def _calculate_weighted_kendall_tau(pred_ranks: List[int], true_ranks: List[int]) -> float:
    """
    Calculate weighted Kendall Tau that gives more weight to disagreements at top positions.
    """
    n = len(pred_ranks)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    total_weight = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Weight decreases with position (top positions more important)
            weight = 1.0 / ((i + 1) + (j + 1))
            total_weight += weight

            pred_order = pred_ranks[i] < pred_ranks[j]
            true_order = true_ranks[i] < true_ranks[j]

            if pred_order == true_order:
                concordant += weight
            else:
                discordant += weight

    if total_weight == 0:
        return 0.0

    return (concordant - discordant) / total_weight


def _calculate_rank_biased_overlap(
    pred_ranks: List[int], true_ranks: List[int], p: float = 0.9
) -> float:
    """
    Calculate Rank-Biased Overlap (RBO) with parameter p.
    Higher p gives more weight to top positions.
    """
    n = len(pred_ranks)
    if n < 2:
        return 0.0

    # Convert ranks to top-k lists
    pred_order = sorted(range(n), key=lambda i: pred_ranks[i])
    true_order = sorted(range(n), key=lambda i: true_ranks[i])

    overlap_sum = 0.0
    for k in range(1, n + 1):
        pred_top_k = set(pred_order[:k])
        true_top_k = set(true_order[:k])
        overlap_k = len(pred_top_k.intersection(true_top_k)) / k
        overlap_sum += overlap_k * (p ** (k - 1))

    return (1 - p) * overlap_sum


def _calculate_normalized_kendall_distance(pred_ranks: List[int], true_ranks: List[int]) -> float:
    """
    Calculate normalized Kendall distance (0 = identical, 1 = completely reversed).
    """
    n = len(pred_ranks)
    if n < 2:
        return 0.0

    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_order = pred_ranks[i] < pred_ranks[j]
            true_order = true_ranks[i] < true_ranks[j]
            if pred_order != true_order:
                inversions += 1

    max_inversions = n * (n - 1) // 2
    return inversions / max_inversions if max_inversions > 0 else 0.0


def _calculate_footrule_distance(pred_ranks: List[int], true_ranks: List[int]) -> float:
    """
    Calculate normalized Spearman footrule distance.
    """
    n = len(pred_ranks)
    if n < 2:
        return 0.0

    distance = sum(abs(pred_ranks[i] - true_ranks[i]) for i in range(n))
    max_distance = n * (n - 1) // 2 if n % 2 == 0 else n * n // 2
    return distance / max_distance if max_distance > 0 else 0.0


def _calculate_weighted_footrule_distance(pred_ranks: List[int], true_ranks: List[int]) -> float:
    """
    Calculate weighted footrule distance that penalizes top position errors more.
    """
    n = len(pred_ranks)
    if n < 2:
        return 0.0

    weighted_distance = 0.0
    max_weighted_distance = 0.0

    for i in range(n):
        # Weight decreases with true rank (top positions more important)
        weight = 1.0 / true_ranks[i]
        weighted_distance += weight * abs(pred_ranks[i] - true_ranks[i])
        max_weighted_distance += weight * (n - 1)  # Maximum possible displacement

    return weighted_distance / max_weighted_distance if max_weighted_distance > 0 else 0.0


def _calculate_top_k_overlap(pred_ranks: List[int], true_ranks: List[int], k: int) -> float:
    """
    Calculate overlap in top-k positions.
    """
    n = len(pred_ranks)
    k = min(k, n)

    if k == 0:
        return 0.0

    # Get top-k items according to each ranking
    pred_top_k = set(i for i in range(n) if pred_ranks[i] <= k)
    true_top_k = set(i for i in range(n) if true_ranks[i] <= k)

    overlap = len(pred_top_k.intersection(true_top_k))
    return overlap / k


def _calculate_hit_at_k(pred_ranks: List[int], true_ranks: List[int], k: int) -> float:
    """
    Calculate Hit@k: whether the best true item (rank 1) appears in predicted top-k.

    Args:
        pred_ranks: Predicted ranks (1-based)
        true_ranks: True ranks (1-based)
        k: Top-k threshold

    Returns:
        1.0 if the true rank-1 item is in predicted top-k, 0.0 otherwise
    """
    n = len(pred_ranks)
    k = min(k, n)

    if k == 0:
        return 0.0

    # Find the item that should be ranked #1 (true rank = 1)
    best_true_item = None
    for i in range(n):
        if true_ranks[i] == 1:
            best_true_item = i
            break

    if best_true_item is None:
        return 0.0

    # Check if this item is in predicted top-k
    return 1.0 if pred_ranks[best_true_item] <= k else 0.0


def _calculate_recall_at_k(
    pred_ranks: List[int], true_ranks: List[int], k: int, top_true_k: int = 3
) -> float:
    """
    Calculate Recall@k: fraction of top true items that appear in predicted top-k.

    Args:
        pred_ranks: Predicted ranks (1-based)
        true_ranks: True ranks (1-based)
        k: Predicted top-k threshold
        top_true_k: Consider top this many true items as relevant

    Returns:
        Recall@k score
    """
    n = len(pred_ranks)
    k = min(k, n)
    top_true_k = min(top_true_k, n)

    if k == 0 or top_true_k == 0:
        return 0.0

    # Get top true items (true rank <= top_true_k)
    true_relevant_items = set(i for i in range(n) if true_ranks[i] <= top_true_k)

    # Get predicted top-k items
    pred_top_k_items = set(i for i in range(n) if pred_ranks[i] <= k)

    # Calculate recall
    hits = len(true_relevant_items.intersection(pred_top_k_items))
    return hits / len(true_relevant_items) if true_relevant_items else 0.0


def calculate_mse(predicted_scores: List[float], true_scores: List[float]) -> float:
    """Calculate Mean Squared Error."""
    if len(predicted_scores) != len(true_scores):
        raise ValueError("Predicted and true scores must have the same length")

    if not predicted_scores:
        return 0.0

    mse = sum((p - t) ** 2 for p, t in zip(predicted_scores, true_scores)) / len(predicted_scores)
    return float(mse)


def calculate_mae(predicted_scores: List[float], true_scores: List[float]) -> float:
    """Calculate Mean Absolute Error."""
    if len(predicted_scores) != len(true_scores):
        raise ValueError("Predicted and true scores must have the same length")

    if not predicted_scores:
        return 0.0

    mae = sum(abs(p - t) for p, t in zip(predicted_scores, true_scores)) / len(predicted_scores)
    return float(mae)


def calculate_rmse(predicted_scores: List[float], true_scores: List[float]) -> float:
    """Calculate Root Mean Squared Error."""
    mse = calculate_mse(predicted_scores, true_scores)
    return math.sqrt(mse)


def calculate_mape(predicted_scores: List[float], true_scores: List[float]) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Note: This metric is undefined when true_scores contains zeros.
    Such values are excluded from the calculation.
    """
    if len(predicted_scores) != len(true_scores):
        raise ValueError("Predicted and true scores must have the same length")

    valid_pairs = [(p, t) for p, t in zip(predicted_scores, true_scores) if t != 0]

    if not valid_pairs:
        return float("inf")  # All true values are zero

    mape = sum(abs((t - p) / t) for p, t in valid_pairs) / len(valid_pairs) * 100
    return mape


def calculate_r_squared(predicted_scores: List[float], true_scores: List[float]) -> float:
    """Calculate R-squared (coefficient of determination)."""
    if len(predicted_scores) != len(true_scores):
        raise ValueError("Predicted and true scores must have the same length")

    if not predicted_scores:
        return 0.0

    # Calculate mean of true values
    mean_true = sum(true_scores) / len(true_scores)

    # Total sum of squares
    ss_tot = sum((t - mean_true) ** 2 for t in true_scores)

    # Residual sum of squares
    ss_res = sum((t - p) ** 2 for p, t in zip(predicted_scores, true_scores))

    # R-squared
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def calculate_mean_absolute_difference(
    predicted_scores: List[float], true_scores: List[float]
) -> float:
    """Calculate Mean Absolute Difference (same as MAE, included for clarity)."""
    return calculate_mae(predicted_scores, true_scores)


def evaluate_ranking_performance(
    ranked_models: List[str],
    positive_models: set,
    negative_models: set,
    k_values: List[int] = [5, 10],
) -> Dict[str, float]:
    """
    Evaluate ranking performance with multiple metrics.

    Args:
        ranked_models: List of models in ranked order
        positive_models: Set of positive (relevant) models
        negative_models: Set of negative models
        k_values: List of k values for top-k evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    results = {}

    # Calculate metrics for different k values
    for k in k_values:
        results[f"ndcg@{k}"] = calculate_ndcg(ranked_models, positive_models, k)
        results[f"map@{k}"] = calculate_map(ranked_models, positive_models, k)

    # Overall metrics
    results["ndcg"] = calculate_ndcg(ranked_models, positive_models)
    results["map"] = calculate_map(ranked_models, positive_models)
    results["mrr"] = calculate_mrr(ranked_models, positive_models)

    # Count-based metrics
    total_models = len(ranked_models)
    total_positive = len(positive_models)
    total_negative = len(negative_models)

    results.update(
        {
            "total_models_ranked": total_models,
            "total_positive_models": total_positive,
            "total_negative_models": total_negative,
        }
    )

    return results


def evaluate_binary_classification(
    true_labels: List[int], pred_labels: List[int]
) -> Dict[str, float]:
    """Evaluate binary classification performance."""
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
    }
