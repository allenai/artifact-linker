#!/usr/bin/env python3
"""Unified runner for attribute prediction and ranking tasks."""
from __future__ import annotations

import concurrent.futures
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from tqdm import tqdm

from ..utils.attribute_prediction_utils import (
    collect_attribute_predictions,
    create_attribute_prediction_row,
    load_attribute_prediction_data,
    print_attribute_prediction_metrics,
    save_attribute_predictions,
)
from ..utils.attribute_ranking_utils import (
    create_attribute_ranking_row,
    load_attribute_ranking_data,
    print_attribute_ranking_metrics,
    save_attribute_rankings,
)
from ..utils.link_prediction_utils import convert_numpy_types

MethodType = Literal["gnn", "llm", "baseline"]
EmbeddingMode = Literal["random", "embedding"]


@dataclass
class AttributeConfig:
    """Configuration for attribute prediction/ranking."""
    data_dir: str = "output/artifact_graph_data"
    split_dir: str = "output/artifact_graph_splits"
    output_dir: str = "output/final_results"
    seed: int = 42
    max_pairs: int = 10
    max_datasets: int = 0
    max_models_per_dataset: int = 20
    workers: int = 4
    use_gnn_data: bool = False
    metric_name: Optional[str] = None
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


def _get_output_path(config: AttributeConfig, task: str) -> Path:
    """Generate output path based on configuration."""
    suffix = "_gnn" if config.use_gnn_data else ""
    if config.method == "gnn":
        emb_tag = "random" if config.embedding_mode == "random" else "emb"
        return Path(config.output_dir) / f"gnn_attr_{task}_{emb_tag}{suffix}.json"
    elif config.method == "llm":
        safe_name = config.llm_model.replace("/", "_")
        return Path(config.output_dir) / f"llm_attr_{task}_{config.hops}hop_{safe_name}{suffix}.json"
    else:
        return Path(config.output_dir) / f"baseline_attr_{task}_{config.baseline_mode}{suffix}.json"


def _load_node_embeddings(split_dir: str, mode: EmbeddingMode):
    """Load node embeddings from split directory.

    Both real and random embeddings are saved to the split root during
    graph splitting, so splits are self-contained.

    Args:
        split_dir: Path to split root directory (contains node_embeddings*.npy).
        mode: "random" for ablation random embeddings, "embedding" for real embeddings.

    Returns:
        Node feature tensor.
    """
    import torch

    sd = Path(split_dir)
    if mode == "embedding":
        emb_path = sd / "node_embeddings_voyage.npy"
        arr = np.load(emb_path, allow_pickle=False)
        if hasattr(arr.dtype, "names") and arr.dtype.names and "embedding" in arr.dtype.names:
            x = torch.from_numpy(arr["embedding"]).float()
        else:
            x = torch.from_numpy(arr).float()
        print(f"Using real embeddings from {emb_path} (dim={x.size(1)})")
    else:
        emb_path = sd / "node_embeddings_random.npy"
        arr = np.load(emb_path, allow_pickle=False)
        x = torch.from_numpy(arr).float()
        print(f"[Ablation] Using random embeddings from {emb_path} (dim={x.size(1)})")

    return x


def _run_parallel(fn: Callable, items: List, workers: int) -> List[Dict]:
    """Run function in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, *item): item for item in items}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": str(e)})
    return results


def _run_sequential(fn: Callable, items: List) -> List[Dict]:
    """Run function sequentially."""
    return [fn(*item) for item in tqdm(items, total=len(items))]


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
    """Run GNN attribute prediction (training + evaluation)."""
    import torch
    from ..training import (
        AttributeModelConfig, AttributeTrainingConfig,
        GNNAttributeTrainer, GNNAttributeEvaluator,
        build_attribute_model, load_attribute_split,
    )
    from ..training.gnn_attribute_trainer import set_seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forced_x = _load_node_embeddings(config.split_dir, config.embedding_mode)

    G_tr, S_tr = load_attribute_split(f"{config.split_dir}/train_split", forced_x)
    G_va, S_va = load_attribute_split(f"{config.split_dir}/val_split", forced_x)
    G_te, S_te = load_attribute_split(f"{config.split_dir}/test_split", forced_x)

    for G in (G_tr, G_va, G_te):
        G.x, G.edge_index = G.x.to(device), G.edge_index.to(device)
    for S in (S_tr, S_va, S_te):
        S.edge_label_index, S.edge_label = S.edge_label_index.to(device), S.edge_label.to(device)

    model_cfg = AttributeModelConfig(G_tr.x.size(1), config.hidden, config.num_layers, config.heads, config.dropout)
    train_cfg = AttributeTrainingConfig(config.epochs, config.lr, seed=config.seed)

    model = build_attribute_model(model_cfg, device)
    trainer = GNNAttributeTrainer(model, device, train_cfg)
    trainer.train(G_tr, S_tr, G_tr, S_va)

    evaluator = GNNAttributeEvaluator()
    test_metrics, test_records = evaluator.evaluate(model, G_tr, S_te, return_preds=True)
    evaluator.print_metrics(test_metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"split": "test", "num_records": len(test_records), "records": test_records}, f, indent=2)

    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    model_path = Path(config.model_path) if config.model_path else (
        Path(config.output_dir) / f"gnn_attribute_prediction_model_{emb_tag}.pth"
    )
    trainer.save_model(model_path, model_cfg)
    print(f"💾 Saved: {output}, {model_path}")

    return {"metrics": test_metrics, "output": str(output)}


def _run_llm_attribute_prediction(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    """Run LLM attribute prediction."""
    from ..models import LLMAttributePredictor

    G, node_meta, edge_meta, edges, true_metrics, metric_names = load_attribute_prediction_data(
        config.data_dir, config.metric_name, config.use_gnn_data
    )

    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    if config.max_pairs > 0:
        edges = edges[:config.max_pairs]
        true_metrics = true_metrics[:config.max_pairs]
        metric_names = metric_names[:config.max_pairs]

    predictor = LLMAttributePredictor(
        model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info
    )
    method_name = f"LLM ({config.llm_model})"
    print(f"Predicting {len(edges)} attributes [{method_name}]")

    def predict_one(edge, true_val, metric):
        m, d = edge
        row = create_attribute_prediction_row(m, d, metric, true_val, node_meta)
        result = predictor.predict(m, d, G=G, node_metadata=node_meta, edge_metadata=edge_meta, metric_name=metric)
        if result and result.get("prediction") is not None:
            row.update(predicted_value=result["prediction"], reason=result.get("reason", ""), status="Success")
        return row

    items = list(zip(edges, true_metrics, metric_names))
    predictions = _run_parallel(predict_one, items, config.workers)

    pred_vals, true_vals = collect_attribute_predictions(predictions)
    print_attribute_prediction_metrics(pred_vals, true_vals, method_name, len(predictions))
    save_attribute_predictions(predictions, output)

    return {"predictions": predictions, "output": str(output)}


def _run_baseline_attribute_prediction(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    """Run baseline attribute prediction."""
    from ..models import BaselineAttributePredictor

    G, node_meta, edge_meta, edges, true_metrics, metric_names = load_attribute_prediction_data(
        config.data_dir, config.metric_name, config.use_gnn_data
    )

    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    if config.max_pairs > 0:
        edges = edges[:config.max_pairs]
        true_metrics = true_metrics[:config.max_pairs]
        metric_names = metric_names[:config.max_pairs]

    predictor = BaselineAttributePredictor(mode=config.baseline_mode)
    method_name = f"Baseline ({config.baseline_mode})"
    print(f"Predicting {len(edges)} attributes [{method_name}]")

    def predict_one(edge, true_val, metric):
        m, d = edge
        row = create_attribute_prediction_row(m, d, metric, true_val, node_meta)
        result = predictor.predict(m, d, G=G, node_metadata=node_meta, edge_metadata=edge_meta, metric_name=metric)
        if result and result.get("prediction") is not None:
            row.update(predicted_value=result["prediction"], reason=result.get("reason", ""), status="Success")
        return row

    items = list(zip(edges, true_metrics, metric_names))
    predictions = _run_sequential(predict_one, items)

    pred_vals, true_vals = collect_attribute_predictions(predictions)
    print_attribute_prediction_metrics(pred_vals, true_vals, method_name, len(predictions))
    save_attribute_predictions(predictions, output)

    return {"predictions": predictions, "output": str(output)}


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
    """Run GNN attribute ranking (inference only)."""
    import torch
    from ..training.gnn_attribute_trainer import load_split_edge_metadata

    random.seed(config.seed)
    np.random.seed(config.seed)

    test_split_dir = Path(config.split_dir) / "test_split"
    test_pos = np.load(test_split_dir / "pos_edges.npz")["edges"]

    # Load per-split edge metadata (already normalized, saved during splitting)
    edge_meta = load_split_edge_metadata(test_split_dir)

    # Load node metadata for type lookup
    with open(test_split_dir / "node_metadata.json") as f:
        node_meta = {int(k): v for k, v in json.load(f).items()}

    # Build tasks
    ds_models = {}
    for i in range(test_pos.shape[1]):
        u, v = int(test_pos[0, i]), int(test_pos[1, i])
        metrics = edge_meta.get((u, v), edge_meta.get((v, u), {}))
        if not metrics:
            continue

        ut = node_meta.get(u, {}).get("type")
        did, mid = (u, v) if ut == "dataset" else (v, u)

        for metric, value in metrics.items():
            try:
                score = float(value)
                if score > 1.0:
                    score /= 100.0
                ds_models.setdefault((did, metric), []).append((mid, score))
            except (ValueError, TypeError):
                pass

    # Select best metric per dataset
    ds_best = {}
    for (did, metric), models in ds_models.items():
        if len(models) >= 3:
            if did not in ds_best or metric == "accuracy" or len(models) > len(ds_best[did][1]):
                ds_best[did] = (metric, models)

    valid = list(ds_best.items())
    if config.max_datasets > 0 and len(valid) > config.max_datasets:
        valid = random.sample(valid, config.max_datasets)

    tasks = []
    for did, (metric, models) in valid:
        unique = dict(models)
        models = sorted(unique.items(), key=lambda x: x[1], reverse=True)[:config.max_models_per_dataset]
        tasks.append({
            "dataset_id": did, "models": [m for m, _ in models],
            "ground_truth": dict(models), "metric": metric
        })

    print(f"Built {len(tasks)} ranking tasks [GNN]")

    # Load model and embeddings
    try:
        from ..models.gnn_link_predictor import GNNLinkPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(config.model_path, map_location=device)
        model = GNNLinkPredictor(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        x = _load_node_embeddings(config.split_dir, config.embedding_mode).to(device)

        edges_path = Path(config.split_dir) / "train_split" / "edges.npz"
        if not edges_path.exists():
            edges_path = Path(config.data_dir) / "edges.npz"
        edges = np.load(edges_path)["edges"]
        # Ensure shape is (2, num_edges)
        if edges.shape[0] != 2:
            edges = edges.T
        edge_index = torch.from_numpy(edges).long().to(device)

        with torch.no_grad():
            z = model.encode(x, edge_index)
        use_gnn = True
    except Exception as e:
        print(f"⚠️ Model load failed: {e}, using random")
        use_gnn = False

    results = []
    for t in tasks:
        if use_gnn:
            # Edge format must match training: (model_id, dataset_id)
            pairs = torch.tensor([[m, t["dataset_id"]] for m in t["models"]], dtype=torch.long, device=device).t()
            with torch.no_grad():
                scores = torch.sigmoid(model.decode(z, pairs)).cpu().numpy().flatten()
            pred = {m: float(s) for m, s in zip(t["models"], scores)}
        else:
            pred = {m: random.random() for m in t["models"]}

        ranked = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        results.append(create_attribute_ranking_row(
            dataset_id=t["dataset_id"],
            metric_used=t["metric"],
            ranked_models=[
                {"model_id": m, "expected_score": pred[m], "true_value": t["ground_truth"].get(m, 0)}
                for m, _ in ranked
            ],
        ))

    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    print_attribute_ranking_metrics(results, f"GNN Attribute Ranking ({emb_tag})")
    save_attribute_rankings({"results": results, "model_used": use_gnn, "embedding_mode": config.embedding_mode}, output)

    return {"rankings": results, "output": str(output)}


def _run_llm_attribute_ranking(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    """Run LLM attribute ranking."""
    from ..models import LLMAttributeRanker

    G, node_meta, edge_meta, ranking_data, dataset_metrics = load_attribute_ranking_data(
        config.data_dir, config.metric_name, config.use_gnn_data
    )

    if config.max_datasets > 0:
        ranking_data = dict(list(ranking_data.items())[:config.max_datasets])

    ranker = LLMAttributeRanker(
        model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info
    )
    method_name = f"LLM ({config.llm_model})"
    print(f"Ranking {len(ranking_data)} datasets [{method_name}]")

    # Build tasks (chunk large datasets)
    tasks = []
    for did, models in ranking_data.items():
        metric = dataset_metrics.get(did, config.metric_name or "accuracy")
        for i in range(0, len(models), config.max_models_per_dataset):
            chunk = models[i:i + config.max_models_per_dataset]
            tasks.append((did, chunk, metric))

    def rank_one(did, models, metric):
        return ranker.rank(
            dataset_id=did, models_to_rank=models,
            G=G, node_metadata=node_meta, edge_metadata=edge_meta, metric_name=metric
        )

    results = _run_parallel(rank_one, tasks, config.workers)

    print_attribute_ranking_metrics(results, method_name)
    save_attribute_rankings(results, output)

    return {"rankings": results, "output": str(output)}


def _run_baseline_attribute_ranking(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    """Run baseline attribute ranking."""
    from ..models import BaselineAttributeRanker

    G, node_meta, edge_meta, ranking_data, dataset_metrics = load_attribute_ranking_data(
        config.data_dir, config.metric_name, config.use_gnn_data
    )

    if config.max_datasets > 0:
        ranking_data = dict(list(ranking_data.items())[:config.max_datasets])

    ranker = BaselineAttributeRanker(mode=config.baseline_mode)
    method_name = f"Baseline ({config.baseline_mode})"
    print(f"Ranking {len(ranking_data)} datasets [{method_name}]")

    # Build tasks (chunk large datasets)
    tasks = []
    for did, models in ranking_data.items():
        metric = dataset_metrics.get(did, config.metric_name or "accuracy")
        for i in range(0, len(models), config.max_models_per_dataset):
            chunk = models[i:i + config.max_models_per_dataset]
            tasks.append((did, chunk, metric))

    def rank_one(did, models, metric):
        return ranker.rank(
            dataset_id=did, models_to_rank=models,
            G=G, node_metadata=node_meta, edge_metadata=edge_meta, metric_name=metric
        )

    results = _run_sequential(rank_one, tasks)

    print_attribute_ranking_metrics(results, method_name)
    save_attribute_rankings(results, output)

    return {"rankings": results, "output": str(output)}
