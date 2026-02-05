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
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--split-dir", default="output/artifact_graph_splits")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    config = LinkConfig(
        method="gnn",
        data_dir=args.data_dir,
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
    )
    run_link_prediction(config)


if __name__ == "__main__":
    main()
