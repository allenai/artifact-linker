#!/usr/bin/env python3
"""GNN runner for link prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .link_runner import LinkConfig
from ..utils.graph_utils import convert_numpy_types


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    import torch
    from torch_geometric.utils import degree
    from ..data import load_all_splits
    from ..training import GNNLinkEvaluator, GNNLinkTrainer, LinkModelConfig, LinkTrainingConfig, build_link_model
    from ..training.gnn_link_trainer import set_seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_random = config.embedding_mode == "random"
    train_data, train_split, val_data, val_split, test_data, test_split = load_all_splits(
        config.split_dir, device, use_random_embeddings=use_random
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
        neg_ratio=config.neg_ratio,
    )

    model = build_link_model(model_cfg, device)
    print(f"Model: {config.gnn_model} | params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = GNNLinkTrainer(model, device, train_cfg)
    trainer.train(train_data, train_split, val_data, val_split)

    evaluator = GNNLinkEvaluator(threshold=config.threshold)
    node_degrees = degree(train_data.edge_index[0], train_data.num_nodes)

    # Use test_data.edge_index for encoding at test time.
    # In the support-edge inductive setting, test_data.edge_index includes
    # support edges so GNN can do message passing for new test nodes.
    model.eval()
    with torch.no_grad():
        z = model.encode(test_data.x, test_data.edge_index)
    test_metrics, test_preds = evaluator.evaluate(model, z, test_split, node_degrees, return_predictions=True)
    evaluator.print_metrics(test_metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    result_data = {"test_metrics": test_metrics}
    if len(test_preds) <= 100_000:
        result_data["test_predictions"] = test_preds
    with open(output, "w") as f:
        json.dump(convert_numpy_types(result_data), f, indent=2)

    from .runner_utils import detect_split_type
    split_prefix = detect_split_type(config.split_dir)
    emb_tag = "random" if config.embedding_mode == "random" else "emb"
    model_path = Path(config.model_path) if config.model_path else (
        Path(config.output_dir) / f"{split_prefix}_gnn_{config.gnn_model}_link_prediction_model_{emb_tag}.pth"
    )
    trainer.save_model(model_path, model_cfg)
    print(f"Saved: {output}, {model_path}")
    return {"metrics": test_metrics, "output": str(output)}
