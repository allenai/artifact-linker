#!/usr/bin/env python3
"""Unified runner for link prediction and ranking tasks."""
from __future__ import annotations

import concurrent.futures
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..utils.link_prediction_utils import (
    collect_valid_predictions,
    convert_numpy_types,
    create_prediction_row,
    load_prediction_data,
    print_classification_metrics,
    print_degree_controlled_metrics,
    save_predictions,
)
from ..utils.ranking_utils import (
    load_link_ranking_data,
    print_link_ranking_metrics,
    save_rankings,
)

MethodType = Literal["gnn", "llm", "baseline"]


@dataclass
class LinkConfig:
    """Configuration for link prediction/ranking."""
    data_dir: str = "output/artifact_graph_data"
    split_dir: str = "output/artifact_graph_splits"
    output_dir: str = "output/final_results"
    seed: int = 42
    max_pairs: int = 5000
    max_datasets: int = 0
    workers: int = 4
    use_gnn_data: bool = False
    # Method-specific
    method: MethodType = "baseline"
    # LLM options
    llm_model: str = "openai/gpt-4o"
    hops: int = 1
    use_info: bool = True
    # Baseline options
    baseline_mode: str = "downloads"
    threshold: Optional[float] = None
    # GNN options
    model_path: str = ""
    epochs: int = 300
    patience: int = 40
    lr: float = 5e-3
    hidden: int = 64
    num_layers: int = 3
    heads: int = 3
    dropout: float = 0.2
    # Ranking options
    candidates_per_dataset: int = 10


def _get_output_path(config: LinkConfig, task: str) -> Path:
    """Generate output path based on configuration."""
    suffix = "_gnn" if config.use_gnn_data else ""
    if config.method == "gnn":
        return Path(config.output_dir) / f"gnn_link_{task}{suffix}.json"
    elif config.method == "llm":
        safe_name = config.llm_model.replace("/", "_")
        return Path(config.output_dir) / f"llm_link_{task}_{config.hops}hop_{safe_name}{suffix}.json"
    else:
        return Path(config.output_dir) / f"baseline_link_{task}_{config.baseline_mode}{suffix}.json"


def _create_predictor(config: LinkConfig):
    """Create appropriate predictor based on method."""
    if config.method == "llm":
        from ..models import LLMLinkPredictor
        return LLMLinkPredictor(model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info)
    elif config.method == "baseline":
        from ..models import BaselineLinkPredictor
        kwargs = {}
        if config.baseline_mode == "katz":
            kwargs["beta"] = 0.1
        if config.threshold is not None:
            kwargs["threshold"] = config.threshold
        return BaselineLinkPredictor(mode=config.baseline_mode, **kwargs)
    else:
        raise ValueError("GNN prediction uses a different flow")


def _create_ranker(config: LinkConfig):
    """Create appropriate ranker based on method."""
    if config.method == "llm":
        from ..models import LLMLinkRanker
        return LLMLinkRanker(model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info)
    elif config.method == "baseline":
        from ..models import BaselineLinkRanker
        return BaselineLinkRanker(mode=config.baseline_mode)
    else:
        raise ValueError("GNN ranking uses a different flow")


def _run_predictions_parallel(predict_fn: Callable, items: List, workers: int) -> List[Dict]:
    """Run predictions in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(predict_fn, *item): item for item in items}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                results.append(f.result())
            except Exception as e:
                item = futures[f]
                results.append({"error": str(e), "item": item})
    return results


def _run_predictions_sequential(predict_fn: Callable, items: List) -> List[Dict]:
    """Run predictions sequentially."""
    return [predict_fn(*item) for item in tqdm(items, total=len(items))]


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
    
    # LLM or Baseline
    G, meta, edges, labels = load_prediction_data(
        config.data_dir, config.seed, config.max_pairs, config.use_gnn_data
    )
    
    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}
    
    predictor = _create_predictor(config)
    method_name = f"LLM ({config.llm_model})" if config.method == "llm" else f"Baseline ({config.baseline_mode})"
    print(f"Predicting {len(edges)} pairs [{method_name}]")
    
    def predict_one(m, d, label):
        row = create_prediction_row(m, d, label, meta)
        result = predictor.predict(model_id=m, dataset_id=d, G=G, node_metadata=meta)
        if result and result.get("prediction") is not None:
            row.update(predicted_label=1 if result["prediction"] else 0, reason=result.get("reason", ""), status="Success")
        return row
    
    items = [(m, d, l) for (m, d), l in zip(edges, labels)]
    
    if config.method == "llm":
        predictions = _run_predictions_parallel(predict_one, items, config.workers)
    else:
        predictions = _run_predictions_sequential(predict_one, items)
    
    y_true, y_pred = collect_valid_predictions(predictions)
    print_classification_metrics(y_true, y_pred, method_name)
    
    if config.method == "baseline":
        print_degree_controlled_metrics(predictions, G, config.baseline_mode)
    
    save_predictions(predictions, output)
    return {"predictions": predictions, "output": str(output)}


def _run_gnn_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run GNN link prediction (training + evaluation)."""
    import torch
    from torch_geometric.utils import degree
    from ..data import load_all_splits
    from ..models import GNNEvaluator, GNNTrainer, ModelConfig, TrainingConfig, build_model
    from ..models.gnn_trainer import set_seed
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, train_split, val_data, val_split, test_data, test_split = load_all_splits(config.split_dir, device)
    
    model_cfg = ModelConfig(train_data.x.size(1), config.hidden, config.num_layers, config.heads, config.dropout)
    train_cfg = TrainingConfig(config.epochs, patience=config.patience, lr=config.lr, seed=config.seed)
    
    model = build_model(model_cfg, device)
    trainer = GNNTrainer(model, device, train_cfg)
    trainer.train(train_data, train_split, val_data, val_split)
    
    evaluator = GNNEvaluator()
    node_degrees = degree(train_data.edge_index[0], train_data.num_nodes)
    
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
    
    test_metrics, test_preds = evaluator.evaluate(model, z, test_split, node_degrees, return_predictions=True)
    evaluator.print_metrics(test_metrics)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(convert_numpy_types({"test_metrics": test_metrics, "test_predictions": test_preds}), f, indent=2)
    
    model_path = Path(config.output_dir) / "gnn_link_prediction_model.pth"
    trainer.save_model(model_path, model_cfg)
    print(f"💾 Saved: {output}, {model_path}")
    
    return {"metrics": test_metrics, "output": str(output)}


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
    
    # LLM or Baseline
    G, node_meta, ranking_data = load_link_ranking_data(
        config.data_dir, config.seed, config.candidates_per_dataset, config.use_gnn_data
    )
    
    if config.max_datasets > 0:
        ranking_data = dict(list(ranking_data.items())[:config.max_datasets])
    
    ranker = _create_ranker(config)
    method_name = f"LLM ({config.llm_model})" if config.method == "llm" else f"Baseline ({config.baseline_mode})"
    print(f"Ranking {len(ranking_data)} datasets [{method_name}]")
    
    def rank_one(did, pos, neg):
        return ranker.rank(dataset_id=did, positive_models=pos, negative_candidates=neg, G=G, node_metadata=node_meta)
    
    items = [(did, pos, neg) for did, (pos, neg) in ranking_data.items()]
    
    if config.method == "llm":
        results = _run_predictions_parallel(rank_one, items, config.workers)
    else:
        results = _run_predictions_sequential(rank_one, items)
    
    print_link_ranking_metrics(results, method_name)
    save_rankings(results, output)
    
    return {"rankings": results, "output": str(output)}


def _run_gnn_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run GNN link ranking (inference only)."""
    import torch
    from ..utils.graph_builder import load_nx_graph
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    G, _, _ = load_nx_graph(config.data_dir)
    test_pos = np.load(Path(config.split_dir) / "test_split" / "pos_edges.npz")["edges"]
    model_ids = {n for n, d in G.nodes(data=True) if d.get("type") == "model"}
    
    # Build tasks
    all_pos_by_ds = {}
    for u, v in G.edges():
        ut, vt = G.nodes[u].get("type"), G.nodes[v].get("type")
        if ut == "dataset" and vt == "model":
            all_pos_by_ds.setdefault(u, set()).add(v)
        elif ut == "model" and vt == "dataset":
            all_pos_by_ds.setdefault(v, set()).add(u)
    
    test_pos_by_ds = {}
    for i in range(test_pos.shape[1]):
        u, v = int(test_pos[0, i]), int(test_pos[1, i])
        ut, vt = G.nodes.get(u, {}).get("type"), G.nodes.get(v, {}).get("type")
        if ut == "dataset" and vt == "model":
            test_pos_by_ds.setdefault(u, set()).add(v)
        elif ut == "model" and vt == "dataset":
            test_pos_by_ds.setdefault(v, set()).add(u)
    
    valid_dids = list(test_pos_by_ds.keys())
    if config.max_datasets > 0 and len(valid_dids) > config.max_datasets:
        valid_dids = random.sample(valid_dids, config.max_datasets)
    
    tasks = []
    for did in valid_dids:
        pos = list(test_pos_by_ds[did])
        neg_pool = list(model_ids - all_pos_by_ds.get(did, set()))
        if neg_pool:
            neg = random.sample(neg_pool, min(config.candidates_per_dataset, len(neg_pool)))
            tasks.append({"dataset_id": did, "positive_models": pos, "negative_candidates": neg})
    
    print(f"Built {len(tasks)} ranking tasks [GNN]")
    
    # Load model
    try:
        from ..models.gnn_link_predictor import GNNLinkPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(config.model_path, map_location=device)
        model = GNNLinkPredictor(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        emb = np.load(Path(config.data_dir) / "node_embeddings.npy", allow_pickle=False)
        x = torch.randn(emb.shape[0], emb.shape[1] if emb.ndim > 1 else 768).to(device)
        edge_index = torch.from_numpy(np.load(Path(config.split_dir) / "train_split" / "edges.npz")["edges"]).long().to(device)
        
        with torch.no_grad():
            z = model.encode(x, edge_index)
        use_gnn = True
    except Exception as e:
        print(f"⚠️ Model load failed: {e}, using random")
        use_gnn = False
    
    results = []
    for t in tasks:
        candidates = t["positive_models"] + t["negative_candidates"]
        if use_gnn:
            pairs = torch.tensor([[t["dataset_id"], m] for m in candidates], dtype=torch.long, device=device).t()
            probs = torch.sigmoid(model.decode(z, pairs)).cpu().tolist()
            ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted([(m, random.random()) for m in candidates], key=lambda x: x[1], reverse=True)
        
        results.append({
            "dataset_id": t["dataset_id"],
            "positive_models": t["positive_models"],
            "ranked_model_ids": [m for m, _ in ranked],
        })
    
    print_link_ranking_metrics(results, "GNN Link Ranking")
    save_rankings({"results": results, "model_used": use_gnn}, output)
    
    return {"rankings": results, "output": str(output)}
