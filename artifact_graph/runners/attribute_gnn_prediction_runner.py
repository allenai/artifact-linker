#!/usr/bin/env python3
"""GNN runner for attribute prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .attribute_runner import AttributeConfig
from .runner_utils import load_node_embeddings
from ..utils.attribute_prediction_utils import print_attribute_prediction_metrics


def run(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    import torch
    from ..training import (
        AttributeModelConfig,
        AttributeTrainingConfig,
        GNNAttributeEvaluator,
        GNNAttributeTrainer,
        build_attribute_model,
        load_attribute_split,
    )
    from ..training.gnn_attribute_trainer import set_seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forced_x = load_node_embeddings(config.split_dir, config.embedding_mode)
    G_tr, S_tr = load_attribute_split(f"{config.split_dir}/train_split", forced_x, config.metric_name, metric_file=config.metric_file)
    # Use test_split for validation (val merged into train)
    G_va, S_va = load_attribute_split(f"{config.split_dir}/test_split", forced_x, config.metric_name, metric_file=config.metric_file)
    G_te, S_te = load_attribute_split(f"{config.split_dir}/test_split", forced_x, config.metric_name, metric_file=config.metric_file)

    for G in (G_tr, G_va, G_te):
        G.x, G.edge_index = G.x.to(device), G.edge_index.to(device)
    for S in (S_tr, S_va, S_te):
        S.edge_label_index, S.edge_label = S.edge_label_index.to(device), S.edge_label.to(device)

    model_cfg = AttributeModelConfig(
        G_tr.x.size(1), config.hidden, config.num_layers, config.heads,
        config.dropout, config.gnn_model,
    )
    train_cfg = AttributeTrainingConfig(
        config.epochs, config.lr, seed=config.seed,
        neg_ratio=config.neg_ratio, neg_target=config.neg_target,
    )

    model = build_attribute_model(model_cfg, device)
    print(f"Model: {config.gnn_model} | "
          f"params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = GNNAttributeTrainer(model, device, train_cfg)
    # Use G_tr for validation encoding during training (support edges only used at test time)
    trainer.train(G_tr, S_tr, G_tr, S_va)

    model.eval()
    evaluator = GNNAttributeEvaluator()
    # Use G_te for test encoding (may include support edges in inductive setting)
    test_metrics, test_records = evaluator.evaluate(model, G_te, S_te, return_preds=True)
    pred_vals = [float(r["prediction"]) for r in test_records]
    true_vals = [float(r["ground_truth"]) for r in test_records]
    method_name = f"GNN ({config.embedding_mode})"
    print_attribute_prediction_metrics(pred_vals, true_vals, method_name, len(test_records))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"split": "test", "num_records": len(test_records), "records": test_records}, f, indent=2)

    from .runner_utils import detect_split_type
    split_prefix = detect_split_type(config.split_dir)
    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    model_tag = config.gnn_model
    model_path = Path(config.model_path) if config.model_path else (
        Path(config.output_dir) / f"{split_prefix}_gnn_{model_tag}_attribute_prediction_model_{emb_tag}.pth"
    )
    trainer.save_model(model_path, model_cfg)
    print(f"Saved: {output}, {model_path}")

    return {"metrics": test_metrics, "output": str(output)}
