#!/usr/bin/env python3
"""Baseline link ranking."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_ranking
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="Baseline Link Ranking")
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="downloads")
    p.add_argument("--max-datasets", type=int, default=0)
    p.add_argument("--candidates-per-dataset", type=int, default=10)
    p.add_argument("--use-gnn-data", action="store_true")
    args = p.parse_args()

    config = LinkConfig(
        method="baseline",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline_mode=args.mode,
        max_datasets=args.max_datasets,
        candidates_per_dataset=args.candidates_per_dataset,
        use_gnn_data=args.use_gnn_data,
    )
    run_link_ranking(config)


if __name__ == "__main__":
    main()
