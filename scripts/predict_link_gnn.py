#!/usr/bin/env python3
"""GNN-based link prediction."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_prediction
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="GNN Link Prediction")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--embedding-mode", choices=["random", "embedding"], default="embedding",
                    help="Embedding mode: 'random' for ablation, 'embedding' for real node embeddings")
    p.add_argument("--gnn-model", default="gatv2",
                    choices=["gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy"],
                    help="GNN model architecture (default: gatv2)")
    p.add_argument("--save-model-path", default="",
                    help="Path to save trained model (default: auto-generated in output-dir)")
    p.add_argument("--threshold", type=float, default=0.9,
                    help="Probability threshold for F1/precision/recall (default: 0.5)")
    args = p.parse_args()

    config = LinkConfig(
        method="gnn",
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        hidden=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        embedding_mode=args.embedding_mode,
        gnn_model=args.gnn_model,
        model_path=args.save_model_path,
        threshold=args.threshold,
    )
    run_link_prediction(config)


if __name__ == "__main__":
    main()
