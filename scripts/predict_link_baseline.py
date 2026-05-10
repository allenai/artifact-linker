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
    p.add_argument("--data-dir", default="../data/artifact_graph_data_v2_1125")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive",
                   help="Split directory (uses same test set as GNN)")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="downloads",
                   choices=["downloads", "random", "connectivity", "common_neighbors",
                            "jaccard", "adamic_adar", "preferential_attachment",
                            "resource_allocation", "katz", "matrix_factorization"],
                   help="Prediction heuristic")
    p.add_argument("--threshold", type=float, default=0.9)
    args = p.parse_args()

    config = LinkConfig(
        method="baseline",
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline_mode=args.mode,
        threshold=args.threshold,
    )
    run_link_prediction(config)


if __name__ == "__main__":
    main()
