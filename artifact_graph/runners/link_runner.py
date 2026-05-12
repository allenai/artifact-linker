#!/usr/bin/env python3
"""Unified runner for link prediction and ranking tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from .runner_utils import run_parallel, run_sequential

from ..utils.link_prediction_utils import (
    collect_link_predictions,
    create_link_prediction_row,
    load_link_prediction_data,
    print_link_prediction_metrics,
    print_degree_metrics,
    save_link_predictions,
)
from ..utils.link_ranking_utils import (
    load_link_ranking_data,
    print_link_ranking_metrics,
    save_link_rankings,
)

MethodType = Literal["gnn", "llm", "baseline", "reranker"]
EmbeddingMode = Literal["random", "embedding"]


@dataclass
class LinkConfig:
    """Configuration for link prediction/ranking."""
    data_dir: str = "output/artifact_graph_data"
    split_dir: str = "output/artifact_graph_splits"
    output_dir: str = "output/final_results"
    seed: int = 42
    workers: int = 4
    # Method-specific
    method: MethodType = "baseline"
    # LLM options
    llm_model: str = "openai/gpt-4o"
    hops: int = 1
    use_info: bool = True
    # RAG options (for LLM)
    use_rag: bool = False
    rag_top_k: int = 100
    rag_strategy: str = "embedding"  # compatibility arg; retriever uses embedding only
    # Reranker options
    reranker_model: str = "jina/jinaai/jina-reranker-v2-base-multilingual"
    # Baseline options
    baseline_mode: str = "downloads"
    # GNN options
    model_path: str = ""
    gnn_model: str = "gatv2"  # "gatv2" | "gcn" | "ncn" | "ncnc" | "neognn" | "buddy"
    epochs: int = 300
    patience: int = 40
    lr: float = 5e-3
    hidden: int = 64
    num_layers: int = 3
    heads: int = 3
    dropout: float = 0.2
    embedding_mode: EmbeddingMode = "random"  # "random" for ablation, "embedding" for real
    threshold: Optional[float] = 0.5  # probability threshold for F1/precision/recall
    neg_ratio: Optional[int] = None  # negative:positive ratio for training (e.g. 5 means 1:5); None = use all


def _get_output_path(config: LinkConfig, task: str) -> Path:
    """Generate output path based on configuration.

    Includes a split-type prefix (trans_/induc_) to avoid overwriting
    results when running both transductive and inductive experiments.
    """
    from .runner_utils import detect_split_type
    split_prefix = detect_split_type(config.split_dir)

    if config.method == "gnn":
        emb_tag = "random" if config.embedding_mode == "random" else "emb"
        model_tag = config.gnn_model
        return Path(config.output_dir) / f"{split_prefix}_gnn_{model_tag}_link_{task}_{emb_tag}.json"
    elif config.method == "llm":
        safe_name = config.llm_model.replace("/", "_")
        return Path(config.output_dir) / f"{split_prefix}_llm_link_{task}_{config.hops}hop_{safe_name}.json"
    elif config.method == "reranker":
        safe_name = config.reranker_model.replace("/", "_")
        return Path(config.output_dir) / f"{split_prefix}_reranker_link_{task}_{config.hops}hop_{safe_name}.json"
    else:
        return Path(config.output_dir) / f"{split_prefix}_baseline_link_{task}_{config.baseline_mode}.json"


def _build_link_ranking_tasks(
    ranking_data: Dict[int, tuple[List[int], List[int]]]
) -> List[tuple[int, List[int], List[int]]]:
    """Build ranking tasks shared by link ranking runners.

    Datasets are sorted by ID for deterministic ordering across runs.
    """
    return [(did, pos, neg) for did, (pos, neg) in sorted(ranking_data.items())]


def _load_link_prediction_inputs(config: LinkConfig) -> tuple[Any, Dict, List[tuple[int, int]], List[int]]:
    """Load link prediction inputs from split data."""
    return load_link_prediction_data(config.split_dir, config.seed)


def _load_link_ranking_inputs(config: LinkConfig) -> tuple[Any, Dict, Dict[int, tuple[List[int], List[int]]]]:
    """Load link ranking inputs from split data."""
    G, node_meta, ranking_data = load_link_ranking_data(config.split_dir)
    return G, node_meta, ranking_data


def _run_link_ranking_with_ranker(
    config: LinkConfig,
    output: Path,
    ranker: Any,
    method_name: str,
    use_parallel: bool,
    include_failed_item: bool = True,
) -> Dict[str, Any]:
    """Shared execution path for LLM/Baseline link ranking."""
    G, node_meta, ranking_data = _load_link_ranking_inputs(config)
    print(f"Ranking {len(ranking_data)} datasets [{method_name}]")

    def rank_one(did, pos, neg):
        return ranker.rank(dataset_id=did, positive_models=pos, negative_candidates=neg, G=G, node_metadata=node_meta)

    items = _build_link_ranking_tasks(ranking_data)
    if use_parallel:
        results = run_parallel(rank_one, items, config.workers, include_failed_item=include_failed_item)
    else:
        results = run_sequential(rank_one, items)
    print_link_ranking_metrics(results, method_name)
    save_link_rankings(results, output)
    return {"rankings": results, "output": str(output)}


def _run_link_prediction_with_predictor(
    config: LinkConfig,
    output: Path,
    predictor: Any,
    method_name: str,
    use_parallel: bool,
    include_failed_item: bool = True,
    degree_metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared execution path for LLM/Baseline link prediction."""
    G, meta, edges, labels = _load_link_prediction_inputs(config)
    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    # --- RAG pre-filtering (if supported) ---
    rag_selected: set = set()
    rag_scores: dict = {}
    if hasattr(predictor, "prepare_prediction_batch"):
        prep = predictor.prepare_prediction_batch(edges=edges, G=G)
        scored = prep.get("scored_pairs", 0) if isinstance(prep, dict) else 0
        selected = prep.get("selected_pairs", 0) if isinstance(prep, dict) else 0
        if scored > 0:
            rag_selected = getattr(predictor, "_rag_selected", set())
            rag_scores = getattr(predictor, "_rag_scores", {})
            print(
                f"RAG filtered {len(edges)} pairs -> {len(rag_selected)} selected for LLM "
                f"({len(edges) - len(rag_selected)} will be auto-negative)"
            )

    # --- Split edges into selected (need LLM) vs filtered (instant negative) ---
    selected_items: List[tuple] = []
    filtered_predictions: List[Dict[str, Any]] = []

    for (m, d), label in zip(edges, labels):
        if rag_selected and (m, d) not in rag_selected:
            # Instant negative — no LLM call needed
            row = create_link_prediction_row(m, d, label, meta)
            score = rag_scores.get((m, d), 0.0)
            row.update(
                predicted_label=0,
                reason="Filtered out by RAG top-k",
                status="Success",
                score=score,
                retrieval_score=score,
            )
            filtered_predictions.append(row)
        else:
            selected_items.append((m, d, label))

    # --- Run LLM only on selected pairs ---
    def predict_one(m, d, label):
        row = create_link_prediction_row(m, d, label, meta)
        result = predictor.predict(model_id=m, dataset_id=d, G=G, node_metadata=meta)
        if result and result.get("prediction") is not None:
            if result.get("retrieval_score") is not None:
                row["retrieval_score"] = result.get("retrieval_score")
            row.update(
                predicted_label=1 if result["prediction"] else 0,
                reason=result.get("reason", ""),
                status="Success",
                score=result.get("score"),
            )
        return row

    print(f"Predicting {len(selected_items)} pairs [{method_name}]"
          + (f" (+ {len(filtered_predictions)} RAG-filtered)" if filtered_predictions else ""))

    if selected_items:
        if use_parallel:
            llm_predictions = run_parallel(
                predict_one,
                selected_items,
                config.workers,
                include_failed_item=include_failed_item,
            )
        else:
            llm_predictions = run_sequential(predict_one, selected_items)
    else:
        llm_predictions = []

    predictions = llm_predictions + filtered_predictions

    y_true, y_pred, y_score = collect_link_predictions(predictions)
    print_link_prediction_metrics(y_true, y_pred, y_score, method_name)
    if degree_metric_name:
        print_degree_metrics(predictions, G, degree_metric_name)

    save_link_predictions(predictions, output)
    return {"predictions": predictions, "output": str(output)}


# =============================================================================
# Link Prediction
# =============================================================================

def run_link_prediction(config: LinkConfig) -> Dict[str, Any]:
    """
    Run link prediction with any method (GNN/LLM/Baseline).

    Returns:
        Dictionary with predictions and metrics.
    """
    output = _get_output_path(config, "predictions")

    if config.method == "gnn":
        return _run_gnn_link_prediction(config, output)
    elif config.method == "llm":
        return _run_llm_link_prediction(config, output)
    elif config.method == "reranker":
        return _run_reranker_link_prediction(config, output)
    else:
        return _run_baseline_link_prediction(config, output)


def _run_gnn_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_gnn_prediction_runner import run as run_impl
    return run_impl(config, output)


def _run_llm_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_llm_prediction_runner import run as run_impl
    return run_impl(config, output)


def _run_reranker_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_reranker_prediction_runner import run as run_impl
    return run_impl(config, output)


def _run_baseline_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_baseline_prediction_runner import run as run_impl
    return run_impl(config, output)


# =============================================================================
# Link Ranking
# =============================================================================

def run_link_ranking(config: LinkConfig) -> Dict[str, Any]:
    """
    Run link ranking with any method (GNN/LLM/Baseline).

    Returns:
        Dictionary with rankings and metrics.
    """
    output = _get_output_path(config, "rankings")

    if config.method == "gnn":
        return _run_gnn_link_ranking(config, output)
    elif config.method == "llm":
        return _run_llm_link_ranking(config, output)
    elif config.method == "reranker":
        return _run_reranker_link_ranking(config, output)
    else:
        return _run_baseline_link_ranking(config, output)


def _run_gnn_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_gnn_ranking_runner import run as run_impl
    return run_impl(config, output)


def _run_llm_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_llm_ranking_runner import run as run_impl
    return run_impl(config, output)


def _run_reranker_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_reranker_ranking_runner import run as run_impl
    return run_impl(config, output)


def _run_baseline_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from .link_baseline_ranking_runner import run as run_impl
    return run_impl(config, output)
