#!/usr/bin/env python3
"""GNN-based link ranking."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_ranking
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="GNN Link Ranking")
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--split-dir", default="output/artifact_graph_splits")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--model-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-datasets", type=int, default=0)
    p.add_argument("--candidates-per-dataset", type=int, default=10)
    args = p.parse_args()

    config = LinkConfig(
        method="gnn",
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        seed=args.seed,
        max_datasets=args.max_datasets,
        candidates_per_dataset=args.candidates_per_dataset,
    )
    run_link_ranking(config)


if __name__ == "__main__":
    main()
