#!/usr/bin/env python3
"""GNN runner for attribute ranking.

Supports two-stage ranking: link_score × attr_score when --link-model-path
is provided. This naturally pushes unobserved (irrelevant) pairs to low
scores while preserving fine-grained regression for plausible pairs.

When --rank-all-models is set, ranks ALL model nodes per dataset (not just
those with observed evaluation edges), testing the practical use case.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .attribute_runner import (
    AttributeConfig,
    _build_attr_ranking_tasks,
    _load_attribute_ranking_inputs,
)
from .runner_utils import load_node_embeddings
from ..utils.attribute_ranking_utils import create_attribute_ranking_row, print_attribute_ranking_metrics, save_attribute_rankings


def _load_all_model_ids(split_dir: str) -> List[int]:
    """Get all model node IDs from node metadata."""
    with open(Path(split_dir) / "test_split" / "node_metadata.json") as f:
        meta = json.load(f)
    return [int(nid) for nid, info in meta.items() if info.get("type") == "model"]


def run(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)

    G, node_meta, _, ranking_data, dataset_metrics = _load_attribute_ranking_inputs(config)
    ranking_tasks = _build_attr_ranking_tasks(
        ranking_data,
        dataset_metrics,
        config.metric_name,
    )

    # Build tasks: each dataset has models to rank + ground truth
    tasks = []
    for did, models, metric in ranking_tasks:
        truth = {m: float(v) for m, v in models}
        observed_models = [m for m, _ in models]
        tasks.append({
            "dataset_id": did,
            "observed_models": observed_models,
            "ground_truth": truth,
            "metric": metric,
        })

    # If ranking all models, get the full list
    all_model_ids: Optional[List[int]] = None
    if config.rank_all_models:
        all_model_ids = _load_all_model_ids(config.split_dir)
        print(f"Ranking ALL {len(all_model_ids)} models per dataset")

    print(f"Built {len(tasks)} ranking tasks [GNN]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load attribute model ---
    attr_model = None
    attr_z = None
    try:
        from ..training.gnn_attribute_trainer import AttributeModelConfig, build_attribute_model
        ckpt = torch.load(config.model_path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["model_config"])
        cfg_dict.pop("backbone", None)
        if "model_type" not in cfg_dict:
            cfg_dict["model_type"] = "gatv2"
        model_cfg = AttributeModelConfig(**cfg_dict)
        attr_model = build_attribute_model(model_cfg, device)
        attr_model.load_state_dict(ckpt["model_state_dict"])
        attr_model.eval()

        x = load_node_embeddings(config.split_dir, config.embedding_mode).to(device)
        edges = np.load(Path(config.split_dir) / "test_split" / "edges.npz")["edges"]
        if edges.shape[0] != 2:
            edges = edges.T
        edge_index = torch.from_numpy(edges).long().to(device)

        with torch.no_grad():
            attr_z = attr_model.encode(x, edge_index)
        print(f"Loaded attribute model from {config.model_path}")
    except Exception as e:
        print(f"Attribute model load failed: {e}, using random")

    # --- Load link model (optional, for two-stage scoring) ---
    link_model = None
    link_z = None
    if config.link_model_path:
        try:
            from ..training.gnn_link_trainer import load_link_model
            link_model, best_metrics = load_link_model(config.link_model_path, device)
            link_model.eval()

            x_link = load_node_embeddings(config.split_dir, config.embedding_mode).to(device)
            edges_link = np.load(Path(config.split_dir) / "test_split" / "edges.npz")["edges"]
            if edges_link.shape[0] != 2:
                edges_link = edges_link.T
            edge_index_link = torch.from_numpy(edges_link).long().to(device)

            with torch.no_grad():
                link_z = link_model.encode(x_link, edge_index_link)
            print(f"Loaded link model from {config.link_model_path} (two-stage scoring enabled)")
        except Exception as e:
            print(f"Link model load failed: {e}, two-stage disabled")

    use_two_stage = link_model is not None and link_z is not None

    # --- Run ranking ---
    results = []
    for t in tasks:
        did = t["dataset_id"]
        # Determine which models to rank
        if all_model_ids is not None:
            candidates = all_model_ids
        else:
            candidates = t["observed_models"]

        if attr_model is not None and attr_z is not None:
            pairs = torch.tensor([[m, did] for m in candidates], dtype=torch.long, device=device).t()

            with torch.no_grad():
                attr_scores = torch.sigmoid(
                    torch.clamp(attr_model.decode(attr_z, pairs), -10, 10)
                ).cpu().numpy().flatten()

            if use_two_stage:
                with torch.no_grad():
                    link_scores = torch.sigmoid(
                        link_model.decode(link_z, pairs)
                    ).cpu().numpy().flatten()
                # Combine: link_score * attr_score
                final_scores = link_scores * attr_scores
            else:
                final_scores = attr_scores

            pred = {m: float(s) for m, s in zip(candidates, final_scores)}
        else:
            pred = {m: random.random() for m in candidates}

        ranked = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        results.append(
            create_attribute_ranking_row(
                dataset_id=did,
                metric_used=t["metric"],
                ranked_models=[
                    {"model_id": m, "expected_score": pred[m], "true_value": t["ground_truth"].get(m, 0.0)}
                    for m, _ in ranked
                ],
            )
        )

    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    model_tag = config.gnn_model
    label = f"GNN Attribute Ranking ({model_tag}, {emb_tag})"
    if use_two_stage:
        label += " [two-stage]"
    if all_model_ids:
        label += " [all-models]"
    print_attribute_ranking_metrics(results, label)
    save_attribute_rankings({"results": results, "model_used": attr_model is not None, "embedding_mode": config.embedding_mode}, output)
    return {"rankings": results, "output": str(output)}
