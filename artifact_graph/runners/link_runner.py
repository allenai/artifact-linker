#!/usr/bin/env python3
"""Unified runner for link prediction and ranking tasks."""
from __future__ import annotations

import concurrent.futures
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from tqdm import tqdm

from ..utils.link_prediction_utils import (
    collect_link_predictions,
    convert_numpy_types,
    create_link_prediction_row,
    load_link_prediction_data,
    print_link_prediction_metrics,
    print_degree_metrics,
    save_link_predictions,
)
from ..utils.link_ranking_utils import (
    create_link_ranking_row,
    load_link_ranking_data,
    print_link_ranking_metrics,
    save_link_rankings,
)

MethodType = Literal["gnn", "llm", "baseline"]
EmbeddingMode = Literal["random", "embedding"]


@dataclass
class LinkConfig:
    """Configuration for link prediction/ranking."""
    data_dir: str = "output/artifact_graph_data"
    split_dir: str = "output/artifact_graph_splits"
    output_dir: str = "output/final_results"
    seed: int = 42
    max_pairs: int = 0  # 0 = use all pairs
    max_datasets: int = 0  # 0 = use all datasets
    workers: int = 4
    use_gnn_data: bool = False
    # Method-specific
    method: MethodType = "baseline"
    # LLM options
    llm_model: str = "openai/gpt-4o"
    hops: int = 1
    use_info: bool = True
    # RAG options (for LLM)
    use_rag: bool = False
    rag_top_k: int = 100
    rag_strategy: str = "hybrid"  # embedding, bm25, heuristic, hybrid
    # Baseline options
    baseline_mode: str = "downloads"
    threshold: Optional[float] = None
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
    threshold: float = 0.5  # probability threshold for F1/precision/recall


def _get_output_path(config: LinkConfig, task: str) -> Path:
    """Generate output path based on configuration."""
    suffix = "_gnn" if config.use_gnn_data else ""
    if config.method == "gnn":
        emb_tag = "random" if config.embedding_mode == "random" else "emb"
        model_tag = config.gnn_model
        return Path(config.output_dir) / f"gnn_{model_tag}_link_{task}_{emb_tag}{suffix}.json"
    elif config.method == "llm":
        safe_name = config.llm_model.replace("/", "_")
        return Path(config.output_dir) / f"llm_link_{task}_{config.hops}hop_{safe_name}{suffix}.json"
    else:
        return Path(config.output_dir) / f"baseline_link_{task}_{config.baseline_mode}{suffix}.json"


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
                item = futures[f]
                results.append({"error": str(e), "item": item})
    return results


def _run_sequential(fn: Callable, items: List) -> List[Dict]:
    """Run function sequentially."""
    return [fn(*item) for item in tqdm(items, total=len(items))]


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
    else:
        return _run_baseline_link_prediction(config, output)


def _run_gnn_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run GNN link prediction (training + evaluation)."""
    import torch
    from torch_geometric.utils import degree
    from ..data import load_all_splits
    from ..training import GNNLinkEvaluator, GNNLinkTrainer, LinkModelConfig, LinkTrainingConfig, build_link_model
    from ..training.gnn_link_trainer import set_seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_random = config.embedding_mode == "random"
    train_data, train_split, val_data, val_split, test_data, test_split = load_all_splits(
        config.split_dir, device,
        use_random_embeddings=use_random,
    )

    model_cfg = LinkModelConfig(
        in_channels=train_data.x.size(1),
        hidden_channels=config.hidden,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
        model_type=config.gnn_model,
    )
    train_cfg = LinkTrainingConfig(
        config.epochs, 
        patience=config.patience, 
        lr=config.lr, 
        seed=config.seed,
        threshold=config.threshold,
    )

    model = build_link_model(model_cfg, device)
    print(f"🏗️ Model: {config.gnn_model} | params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = GNNLinkTrainer(model, device, train_cfg)
    trainer.train(train_data, train_split, val_data, val_split)

    evaluator = GNNLinkEvaluator(threshold=config.threshold)
    node_degrees = degree(train_data.edge_index[0], train_data.num_nodes)

    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)

    test_metrics, test_preds = evaluator.evaluate(model, z, test_split, node_degrees, return_predictions=True)
    evaluator.print_metrics(test_metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(convert_numpy_types({"test_metrics": test_metrics, "test_predictions": test_preds}), f, indent=2)

    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    model_path = Path(config.model_path) if config.model_path else (
        Path(config.output_dir) / f"gnn_{config.gnn_model}_link_prediction_model_{emb_tag}.pth"
    )
    trainer.save_model(model_path, model_cfg)
    print(f"💾 Saved: {output}, {model_path}")

    return {"metrics": test_metrics, "output": str(output)}


def _run_llm_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run LLM link prediction."""
    from ..models import LLMLinkPredictor

    G, meta, edges, labels = load_link_prediction_data(
        config.data_dir, config.seed, config.max_pairs, config.use_gnn_data,
        split_dir=config.split_dir if config.split_dir else None,
    )

    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    predictor = LLMLinkPredictor(
        model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info
    )
    method_name = f"LLM ({config.llm_model})"

    # Apply RAG filtering: use retrieval scores to pre-filter likely positive pairs
    retrieval_scores = {}
    if config.use_rag:
        from ..utils.retriever import CandidateRetriever
        retriever = CandidateRetriever.from_data_dir(
            config.data_dir, strategy=config.rag_strategy, top_k=config.rag_top_k,
        )
        print(f"RAG pre-filtering {len(edges)} pairs...")
        edges_by_dataset = {}
        for (m, d), label in zip(edges, labels):
            edges_by_dataset.setdefault(d, []).append((m, label))

        for dataset_id, model_labels in edges_by_dataset.items():
            model_ids = [m for m, _ in model_labels]
            retrieved = retriever.retrieve(dataset_id, model_ids, G)
            for mid, score in retrieved:
                retrieval_scores[(mid, dataset_id)] = score

        print(f"RAG computed scores for {len(retrieval_scores)} pairs")

    print(f"Predicting {len(edges)} pairs [{method_name}]")

    def predict_one(m, d, label):
        row = create_link_prediction_row(m, d, label, meta)
        if (m, d) in retrieval_scores:
            row["retrieval_score"] = retrieval_scores[(m, d)]
        result = predictor.predict(model_id=m, dataset_id=d, G=G, node_metadata=meta)
        if result and result.get("prediction") is not None:
            row.update(
                predicted_label=1 if result["prediction"] else 0,
                reason=result.get("reason", ""),
                status="Success",
                score=result.get("score"),
            )
        return row

    items = [(m, d, l) for (m, d), l in zip(edges, labels)]
    predictions = _run_parallel(predict_one, items, config.workers)

    y_true, y_pred, y_score = collect_link_predictions(predictions)
    print_link_prediction_metrics(y_true, y_pred, y_score, method_name)

    save_link_predictions(predictions, output)
    return {"predictions": predictions, "output": str(output)}


def _run_baseline_link_prediction(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run baseline link prediction."""
    from ..models import BaselineLinkPredictor

    G, meta, edges, labels = load_link_prediction_data(
        config.data_dir, config.seed, config.max_pairs, config.use_gnn_data,
        split_dir=config.split_dir if config.split_dir else None,
    )

    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    kwargs = {}
    if config.baseline_mode == "katz":
        kwargs["beta"] = 0.1
    if config.threshold is not None:
        kwargs["threshold"] = config.threshold
    predictor = BaselineLinkPredictor(mode=config.baseline_mode, **kwargs)
    method_name = f"Baseline ({config.baseline_mode})"
    print(f"Predicting {len(edges)} pairs [{method_name}]")

    def predict_one(m, d, label):
        row = create_link_prediction_row(m, d, label, meta)
        result = predictor.predict(model_id=m, dataset_id=d, G=G, node_metadata=meta)
        if result and result.get("prediction") is not None:
            row.update(
                predicted_label=1 if result["prediction"] else 0,
                reason=result.get("reason", ""),
                status="Success",
                score=result.get("score"),
            )
        return row

    items = [(m, d, l) for (m, d), l in zip(edges, labels)]
    predictions = _run_sequential(predict_one, items)

    y_true, y_pred, y_score = collect_link_predictions(predictions)
    print_link_prediction_metrics(y_true, y_pred, y_score, method_name)
    print_degree_metrics(predictions, G, config.baseline_mode)

    save_link_predictions(predictions, output)
    return {"predictions": predictions, "output": str(output)}


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
    else:
        return _run_baseline_link_ranking(config, output)


def _run_gnn_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run GNN link ranking (inference only).

    Uses only split-level data (no data_dir needed).
    """
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)

    split_root = Path(config.split_dir)

    # Load node metadata (same across splits)
    with open(split_root / "train_split" / "node_metadata.json") as f:
        node_meta = {int(k): v for k, v in json.load(f).items()}

    model_ids = {nid for nid, info in node_meta.items() if info.get("type") == "model"}

    # Collect ALL positive edges across all splits to build negative candidates
    all_pos_by_ds: Dict[int, set] = {}
    for split_name in ["train_split", "val_split", "test_split"]:
        pos_path = split_root / split_name / "pos_edges.npz"
        if not pos_path.exists():
            continue
        pos = np.load(pos_path)["edges"]
        for i in range(pos.shape[1]):
            u, v = int(pos[0, i]), int(pos[1, i])
            ut = node_meta.get(u, {}).get("type")
            vt = node_meta.get(v, {}).get("type")
            if ut == "dataset" and vt == "model":
                all_pos_by_ds.setdefault(u, set()).add(v)
            elif ut == "model" and vt == "dataset":
                all_pos_by_ds.setdefault(v, set()).add(u)

    # Build test positive edges by dataset
    test_pos = np.load(split_root / "test_split" / "pos_edges.npz")["edges"]
    test_pos_by_ds: Dict[int, set] = {}
    for i in range(test_pos.shape[1]):
        u, v = int(test_pos[0, i]), int(test_pos[1, i])
        ut = node_meta.get(u, {}).get("type")
        vt = node_meta.get(v, {}).get("type")
        if ut == "dataset" and vt == "model":
            test_pos_by_ds.setdefault(u, set()).add(v)
        elif ut == "model" and vt == "dataset":
            test_pos_by_ds.setdefault(v, set()).add(u)

    valid_dids = list(test_pos_by_ds.keys())
    if config.max_datasets > 0 and len(valid_dids) > config.max_datasets:
        valid_dids = random.sample(valid_dids, config.max_datasets)

    # Build tasks with ALL models as candidates (full negative)
    tasks = []
    for did in valid_dids:
        pos = list(test_pos_by_ds[did])
        neg = list(model_ids - all_pos_by_ds.get(did, set()))  # ALL other models
        if neg:
            tasks.append({"dataset_id": did, "positive_models": pos, "negative_candidates": neg})

    print(f"Built {len(tasks)} ranking tasks [GNN]")

    # Load model and embeddings
    try:
        from ..training.gnn_link_trainer import load_link_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, best_metrics = load_link_model(config.model_path, device)
        print(f"Loaded model from {config.model_path} (best val metrics: {best_metrics})")

        x = _load_node_embeddings(config.split_dir, config.embedding_mode).to(device)

        edges = np.load(Path(config.split_dir) / "train_split" / "edges.npz")["edges"]
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
        candidates = t["positive_models"] + t["negative_candidates"]
        if use_gnn:
            # Edge format must match training: (model_id, dataset_id)
            pairs = torch.tensor([[m, t["dataset_id"]] for m in candidates], dtype=torch.long, device=device).t()
            probs = torch.sigmoid(model.decode(z, pairs)).cpu().tolist()
            ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted([(m, random.random()) for m in candidates], key=lambda x: x[1], reverse=True)

        results.append(create_link_ranking_row(
            dataset_id=t["dataset_id"],
            positive_models=t["positive_models"],
            ranked_model_ids=[m for m, _ in ranked],
        ))

    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    method_label = f"GNN ({config.gnn_model}) Link Ranking ({emb_tag})"
    print_link_ranking_metrics(results, method_label)
    save_link_rankings({
        "results": results, "model_used": use_gnn,
        "embedding_mode": config.embedding_mode, "gnn_model": config.gnn_model,
    }, output)

    return {"rankings": results, "output": str(output)}


def _run_llm_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run LLM link ranking."""
    from ..models import LLMLinkRanker

    G, node_meta, ranking_data = load_link_ranking_data(
        config.data_dir, config.use_gnn_data, split_dir=config.split_dir
    )

    if config.max_datasets > 0:
        ranking_data = dict(list(ranking_data.items())[:config.max_datasets])

    # Create retriever if RAG is enabled
    retriever = None
    if config.use_rag:
        from ..utils.retriever import CandidateRetriever
        retriever = CandidateRetriever.from_data_dir(
            config.data_dir, strategy=config.rag_strategy, top_k=config.rag_top_k,
        )

    ranker = LLMLinkRanker(
        model_name=config.llm_model,
        hop_number=config.hops,
        use_info=config.use_info,
        use_rag=config.use_rag,
        rag_top_k=config.rag_top_k,
        retriever=retriever,
    )
    method_name = f"LLM ({config.llm_model})"
    print(f"Ranking {len(ranking_data)} datasets [{method_name}]")

    def rank_one(did, pos, neg):
        return ranker.rank(dataset_id=did, positive_models=pos, negative_candidates=neg, G=G, node_metadata=node_meta)

    items = [(did, pos, neg) for did, (pos, neg) in ranking_data.items()]
    results = _run_parallel(rank_one, items, config.workers)

    print_link_ranking_metrics(results, method_name)
    save_link_rankings(results, output)

    return {"rankings": results, "output": str(output)}


def _run_baseline_link_ranking(config: LinkConfig, output: Path) -> Dict[str, Any]:
    """Run baseline link ranking."""
    from ..models import BaselineLinkRanker

    G, node_meta, ranking_data = load_link_ranking_data(
        config.data_dir, config.use_gnn_data, split_dir=config.split_dir
    )

    if config.max_datasets > 0:
        ranking_data = dict(list(ranking_data.items())[:config.max_datasets])

    ranker = BaselineLinkRanker(mode=config.baseline_mode)
    method_name = f"Baseline ({config.baseline_mode})"
    print(f"Ranking {len(ranking_data)} datasets [{method_name}]")

    def rank_one(did, pos, neg):
        return ranker.rank(dataset_id=did, positive_models=pos, negative_candidates=neg, G=G, node_metadata=node_meta)

    items = [(did, pos, neg) for did, (pos, neg) in ranking_data.items()]
    results = _run_sequential(rank_one, items)

    print_link_ranking_metrics(results, method_name)
    save_link_rankings(results, output)

    return {"rankings": results, "output": str(output)}
