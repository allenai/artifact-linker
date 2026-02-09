#!/usr/bin/env python3
"""GNN-based attribute prediction."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_attribute_prediction
from artifact_graph.runners.attribute_runner import AttributeConfig


def main():
    p = argparse.ArgumentParser(description="GNN Attribute Prediction")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--embedding-mode", choices=["random", "embedding"], default="random",
                    help="Embedding mode: 'random' for ablation, 'embedding' for real node embeddings")
    p.add_argument("--save-model-path", default="",
                    help="Path to save trained model (default: auto-generated in output-dir)")
    args = p.parse_args()

    config = AttributeConfig(
        method="gnn",
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        embedding_mode=args.embedding_mode,
        model_path=args.save_model_path,
    )
    run_attribute_prediction(config)


if __name__ == "__main__":
    main()
