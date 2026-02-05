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
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--split-dir", default="output/artifact_graph_splits")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    config = AttributeConfig(
        method="gnn",
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        metric_name=args.metric,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    run_attribute_prediction(config)


if __name__ == "__main__":
    main()
