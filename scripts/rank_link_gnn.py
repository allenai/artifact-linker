#!/usr/bin/env python3
"""GNN-based link ranking (uses ALL models as candidates)."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_ranking
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="GNN Link Ranking")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--model-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-datasets", type=int, default=0, help="0 = all datasets")
    p.add_argument("--embedding-mode", choices=["random", "embedding"], default="random",
                    help="Embedding mode: 'random' for ablation, 'embedding' for real node embeddings")
    p.add_argument("--gnn-model", default="gatv2",
                    choices=["gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy"],
                    help="GNN model architecture (default: gatv2). "
                         "Only affects output naming; model type is auto-detected from checkpoint.")
    args = p.parse_args()

    config = LinkConfig(
        method="gnn",
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        seed=args.seed,
        max_datasets=args.max_datasets,
        embedding_mode=args.embedding_mode,
        gnn_model=args.gnn_model,
    )
    run_link_ranking(config)


if __name__ == "__main__":
    main()
