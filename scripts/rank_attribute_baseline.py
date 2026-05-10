#!/usr/bin/env python3
"""Baseline attribute ranking."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_attribute_ranking
from artifact_graph.runners.attribute_runner import AttributeConfig


def main():
    p = argparse.ArgumentParser(description="Baseline Attribute Ranking")
    p.add_argument("--data-dir", default="../data/artifact_graph_data_v2_1125")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive",
                   help="Split directory (uses same test set as GNN)")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="dataset_average")
    p.add_argument("--metric", default=None)
    p.add_argument("--metric-file", default="edge_metadata_normalized.json",
                   help="Edge metadata file name (e.g. edge_metadata_normalized_attr.json for filtered metrics)")
    args = p.parse_args()

    config = AttributeConfig(
        method="baseline",
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        baseline_mode=args.mode,
        metric_name=args.metric,
        metric_file=args.metric_file,
    )
    run_attribute_ranking(config)


if __name__ == "__main__":
    main()
