#!/usr/bin/env python3
"""Baseline link prediction using graph heuristics."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_prediction
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="Baseline Link Prediction")
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="downloads",
                   choices=["downloads", "common_neighbors", "jaccard", "adamic_adar",
                            "preferential_attachment", "resource_allocation", "katz"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--max-pairs", type=int, default=500000)
    p.add_argument("--use-gnn-data", action="store_true")
    args = p.parse_args()

    config = LinkConfig(
        method="baseline",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline_mode=args.mode,
        threshold=args.threshold,
        max_pairs=args.max_pairs,
        use_gnn_data=args.use_gnn_data,
    )
    run_link_prediction(config)


if __name__ == "__main__":
    main()
