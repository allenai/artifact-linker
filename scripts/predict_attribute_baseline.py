#!/usr/bin/env python3
"""Baseline attribute prediction."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_attribute_prediction
from artifact_graph.runners.attribute_runner import AttributeConfig


def main():
    p = argparse.ArgumentParser(description="Baseline Attribute Prediction")
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="dataset_average")
    p.add_argument("--metric", default=None)
    p.add_argument("--use-gnn-data", action="store_true")
    args = p.parse_args()

    config = AttributeConfig(
        method="baseline",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline_mode=args.mode,
        metric_name=args.metric,
        use_gnn_data=args.use_gnn_data,
    )
    run_attribute_prediction(config)


if __name__ == "__main__":
    main()
