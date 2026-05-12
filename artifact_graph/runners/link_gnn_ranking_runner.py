#!/usr/bin/env python3
"""GNN runner for link ranking."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .link_runner import LinkConfig, _load_link_ranking_inputs
from .runner_utils import load_node_embeddings
from ..utils.link_ranking_utils import create_link_ranking_row, print_link_ranking_metrics, save_link_rankings


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)

    _, node_meta, ranking_data = _load_link_ranking_inputs(config)
    tasks = []
    for did, (pos, neg) in ranking_data.items():
        tasks.append({"dataset_id": did, "positive_models": pos, "negative_candidates": neg})

    print(f"Built {len(tasks)} ranking tasks [GNN]")

    try:
        from ..training.gnn_link_trainer import load_link_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, best_metrics = load_link_model(config.model_path, device)
        print(f"Loaded model from {config.model_path} (best val metrics: {best_metrics})")

        x = load_node_embeddings(config.split_dir, config.embedding_mode).to(device)
        # Use test_split msg-passing graph for ranking (includes support edges
        # in the inductive support setting for GNN message passing on new nodes)
        edges = np.load(Path(config.split_dir) / "test_split" / "edges.npz")["edges"]
        if edges.shape[0] != 2:
            edges = edges.T
        edge_index = torch.from_numpy(edges).long().to(device)

        with torch.no_grad():
            z = model.encode(x, edge_index)
        use_gnn = True
    except Exception as e:
        print(f"Model load failed: {e}, using random")
        use_gnn = False

    results = []
    for t in tasks:
        candidates = t["positive_models"] + t["negative_candidates"]
        if use_gnn:
            pairs = torch.tensor([[m, t["dataset_id"]] for m in candidates], dtype=torch.long, device=device).t()
            probs = torch.sigmoid(model.decode(z, pairs)).cpu().tolist()
            ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted([(m, random.random()) for m in candidates], key=lambda x: x[1], reverse=True)

        results.append(
            create_link_ranking_row(
                dataset_id=t["dataset_id"],
                positive_models=t["positive_models"],
                ranked_model_ids=[m for m, _ in ranked],
            )
        )

    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    method_label = f"GNN ({config.gnn_model}) Link Ranking ({emb_tag})"
    print_link_ranking_metrics(results, method_label)
    save_link_rankings(
        {
            "results": results,
            "model_used": use_gnn,
            "embedding_mode": config.embedding_mode,
            "gnn_model": config.gnn_model,
        },
        output,
    )
    return {"rankings": results, "output": str(output)}
