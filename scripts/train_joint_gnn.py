#!/usr/bin/env python3
"""Train the joint GNN (shared encoder + link & attr heads) and save a checkpoint.

This is the single expensive step shared by all 4 GNN eval tasks. The four
{predict,rank}_{link,attribute}_gnn.py scripts load the checkpoint this writes
and each run exactly one task's evaluation.

Usage:
    python scripts/train_joint_gnn.py \
        --split-dir data/artifact_graph_splits_v3_0314_transductive \
        --output-dir data/joint_sweep --backbone gatv2
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.training.gnn_joint_trainer import (
    JointModelConfig, JointTrainingConfig, GNNJointTrainer, build_joint_model, set_seed,
)
from artifact_graph.training.gnn_attribute_trainer import load_attribute_split
from artifact_graph.runners.runner_utils import load_node_embeddings, detect_split_type


def main():
    p = argparse.ArgumentParser(description="Joint GNN training (saves checkpoint only)")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", default="gatv2",
                   choices=["gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy"])
    p.add_argument("--embedding-mode", default="embedding", choices=["random", "embedding"])
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--attr-weight", type=float, default=5.0)
    p.add_argument("--neg-ratio", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="",
                   help="Checkpoint output path (default: auto in output-dir)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)
    split_prefix = detect_split_type(str(split_dir))
    emb_tag = "random" if args.embedding_mode == "random" else "emb"

    set_seed(args.seed)

    forced_x = load_node_embeddings(str(split_dir), args.embedding_mode)
    G_tr, S_tr = load_attribute_split(str(split_dir / "train_split"), forced_x)
    G_te, S_te = load_attribute_split(str(split_dir / "test_split"), forced_x)
    for G in (G_tr, G_te):
        G.x, G.edge_index = G.x.to(device), G.edge_index.to(device)
    for S in (S_tr, S_te):
        S.edge_label_index = S.edge_label_index.to(device)
        S.edge_label = S.edge_label.to(device)

    cfg = JointModelConfig(
        in_channels=G_tr.x.size(1),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        backbone=args.backbone,
        model_type=args.backbone,
    )
    tcfg = JointTrainingConfig(
        epochs=args.epochs, lr=args.lr,
        neg_ratio=args.neg_ratio, attr_weight=args.attr_weight,
    )
    model = build_joint_model(cfg, device)
    print(f"Joint Model: {args.backbone} | layers={args.num_layers} | "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    trainer = GNNJointTrainer(model, device, tcfg)
    trainer.train(G_tr, S_tr, G_tr, S_te, verbose=False)

    model_path = Path(args.model_path) if args.model_path else (
        out_dir / f"{split_prefix}_joint_{args.backbone}_model_{emb_tag}.pth"
    )
    trainer.save_model(model_path, cfg)
    print(f"Saved checkpoint: {model_path}")


if __name__ == "__main__":
    main()
