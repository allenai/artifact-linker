#!/usr/bin/env python3
"""Unified runner for attribute prediction and ranking tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from .runner_utils import run_parallel, run_sequential

from ..utils.attribute_prediction_utils import (
    collect_attribute_predictions,
    create_attribute_prediction_row,
    load_attribute_prediction_data,
    print_attribute_prediction_metrics,
    save_attribute_predictions,
)
from ..utils.attribute_ranking_utils import (
    load_attribute_ranking_data,
    print_attribute_ranking_metrics,
    save_attribute_rankings,
)

MethodType = Literal["gnn", "llm", "baseline"]
EmbeddingMode = Literal["random", "embedding"]


@dataclass
class AttributeConfig:
    """Configuration for attribute prediction/ranking."""
    data_dir: str = "output/artifact_graph_data"
    split_dir: str = "output/artifact_graph_splits"
    output_dir: str = "output/final_results"
    seed: int = 42
    workers: int = 4
    metric_name: Optional[str] = None
    metric_file: str = "edge_metadata_normalized.json"  # use "edge_metadata_normalized_attr.json" for filtered metrics
    # Method-specific
    method: MethodType = "baseline"
    # LLM options
    llm_model: str = "openai/gpt-4o"
    hops: int = 1
    use_info: bool = True
    # Baseline options
    baseline_mode: str = "dataset_average"
    # GNN options
    model_path: str = ""
    epochs: int = 500
    lr: float = 0.005
    hidden: int = 128
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.2
    embedding_mode: EmbeddingMode = "random"  # "random" for ablation, "embedding" for real
    gnn_model: str = "gatv2"  # model architecture: "gatv2" | "gcn" | "ncn" | "ncnc" | "neognn" | "buddy"
    neg_ratio: int = 0  # negative:positive ratio for attribute training (0 = disabled)
    neg_target: float = 0.1  # margin threshold for unobserved pairs
    link_model_path: str = ""  # path to link prediction model for two-stage ranking
    rank_all_models: bool = False  # rank ALL models per dataset (not just observed)


def _get_output_path(config: AttributeConfig, task: str) -> Path:
    """Generate output path based on configuration.

    Includes a split-type prefix (trans_/induc_) to avoid overwriting
    results when running both transductive and inductive experiments.
    """
    from .runner_utils import detect_split_type
    split_prefix = detect_split_type(config.split_dir)

    if config.method == "gnn":
        emb_tag = "random" if config.embedding_mode == "random" else "emb"
        model_tag = config.gnn_model
        return Path(config.output_dir) / f"{split_prefix}_gnn_{model_tag}_attr_{task}_{emb_tag}.json"
    elif config.method == "llm":
        safe_name = config.llm_model.replace("/", "_")
        return Path(config.output_dir) / f"{split_prefix}_llm_attr_{task}_{config.hops}hop_{safe_name}.json"
    else:
        return Path(config.output_dir) / f"{split_prefix}_baseline_attr_{task}_{config.baseline_mode}.json"


def _build_attr_ranking_tasks(
    ranking_data: Dict[int, List],
    dataset_metrics: Dict[int, str],
    metric_name: Optional[str],
) -> List[tuple[int, List, str]]:
    """Build one ranking task per dataset shared by all rankers.

    Datasets are sorted by ID for deterministic ordering across runs.
    Datasets with fewer than 2 models are skipped (cannot rank).
    """
    tasks: List[tuple[int, List, str]] = []
    for did in sorted(ranking_data.keys()):
        models = ranking_data[did]
        if len(models) < 2:
            continue
        metric = dataset_metrics.get(did, metric_name or "accuracy")
        tasks.append((did, models, metric))
    return tasks


def _load_attribute_prediction_inputs(
    config: AttributeConfig,
) -> tuple[Any, Dict, Dict, List, List[float], List[str]]:
    """Load attribute prediction inputs from split data."""
    G, node_meta, edge_meta, edges, true_metrics, metric_names = load_attribute_prediction_data(
        config.split_dir, config.metric_name, metric_file=config.metric_file
    )
    return G, node_meta, edge_meta, edges, true_metrics, metric_names


def _run_attribute_prediction_with_predictor(
    config: AttributeConfig,
    output: Path,
    predictor: Any,
    method_name: str,
    use_parallel: bool,
    include_failed_item: bool = True,
) -> Dict[str, Any]:
    """Shared execution path for LLM/Baseline attribute prediction."""
    G, node_meta, edge_meta, edges, true_metrics, metric_names = _load_attribute_prediction_inputs(config)
    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    print(f"Predicting {len(edges)} attributes [{method_name}]")

    def predict_one(edge, true_val, metric):
        m, d = edge
        row = create_attribute_prediction_row(m, d, metric, true_val, node_meta)
        result = predictor.predict(m, d, G=G, node_metadata=node_meta, edge_metadata=edge_meta, metric_name=metric)
        if result and result.get("prediction") is not None:
            row.update(predicted_value=result["prediction"], reason=result.get("reason", ""), status="Success")
        return row

    items = list(zip(edges, true_metrics, metric_names))
    if use_parallel:
        predictions = run_parallel(
            predict_one,
            items,
            config.workers,
            include_failed_item=include_failed_item,
        )
    else:
        predictions = run_sequential(predict_one, items)

    pred_vals, true_vals = collect_attribute_predictions(predictions)
    print_attribute_prediction_metrics(pred_vals, true_vals, method_name, len(predictions))
    save_attribute_predictions(predictions, output)
    return {"predictions": predictions, "output": str(output)}


def _load_attribute_ranking_inputs(
    config: AttributeConfig,
) -> tuple[Any, Dict, Dict, Dict, Dict]:
    """Load attribute ranking inputs from split data."""
    return load_attribute_ranking_data(config.split_dir, config.metric_name, metric_file=config.metric_file)


def _run_attribute_ranking_with_ranker(
    config: AttributeConfig,
    output: Path,
    ranker: Any,
    method_name: str,
    use_parallel: bool,
    include_failed_item: bool = True,
) -> Dict[str, Any]:
    """Shared execution path for LLM/Baseline attribute ranking."""
    G, node_meta, edge_meta, ranking_data, dataset_metrics = _load_attribute_ranking_inputs(config)
    skipped = sum(1 for models in ranking_data.values() if len(models) < 2)
    print(f"Ranking {len(ranking_data)} datasets [{method_name}] ({skipped} single-model datasets skipped)")

    tasks = _build_attr_ranking_tasks(
        ranking_data,
        dataset_metrics,
        config.metric_name,
    )

    def rank_one(did, models, metric):
        return ranker.rank(
            dataset_id=did, models_to_rank=models,
            G=G, node_metadata=node_meta, edge_metadata=edge_meta, metric_name=metric
        )

    if use_parallel:
        results = run_parallel(
            rank_one,
            tasks,
            config.workers,
            include_failed_item=include_failed_item,
        )
    else:
        results = run_sequential(rank_one, tasks)
    print_attribute_ranking_metrics(results, method_name)
    save_attribute_rankings(results, output)
    return {"rankings": results, "output": str(output)}


# =============================================================================
# Attribute Prediction
# =============================================================================

def run_attribute_prediction(config: AttributeConfig) -> Dict[str, Any]:
    """
    Run attribute prediction with any method (GNN/LLM/Baseline).

    Returns:
        Dictionary with predictions and metrics.
    """
    output = _get_output_path(config, "predictions")

    if config.method == "gnn":
        return _run_gnn_attribute_prediction(config, output)
    elif config.method == "llm":
        return _run_llm_attribute_prediction(config, output)
    else:
        return _run_baseline_attribute_prediction(config, output)


def _run_gnn_attribute_prediction(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from .attribute_gnn_prediction_runner import run as run_impl
    return run_impl(config, output)


def _run_llm_attribute_prediction(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from .attribute_llm_prediction_runner import run as run_impl
    return run_impl(config, output)


def _run_baseline_attribute_prediction(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from .attribute_baseline_prediction_runner import run as run_impl
    return run_impl(config, output)


# =============================================================================
# Attribute Ranking
# =============================================================================

def run_attribute_ranking(config: AttributeConfig) -> Dict[str, Any]:
    """
    Run attribute ranking with any method (GNN/LLM/Baseline).

    Returns:
        Dictionary with rankings and metrics.
    """
    output = _get_output_path(config, "rankings")

    if config.method == "gnn":
        return _run_gnn_attribute_ranking(config, output)
    elif config.method == "llm":
        return _run_llm_attribute_ranking(config, output)
    else:
        return _run_baseline_attribute_ranking(config, output)


def _run_gnn_attribute_ranking(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from .attribute_gnn_ranking_runner import run as run_impl
    return run_impl(config, output)


def _run_llm_attribute_ranking(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from .attribute_llm_ranking_runner import run as run_impl
    return run_impl(config, output)


def _run_baseline_attribute_ranking(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from .attribute_baseline_ranking_runner import run as run_impl
    return run_impl(config, output)
