#!/usr/bin/env python3
"""GNN-based attribute ranking."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_attribute_ranking
from artifact_graph.runners.attribute_runner import AttributeConfig


def main():
    p = argparse.ArgumentParser(description="GNN Attribute Ranking")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--model-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-datasets", type=int, default=0)
    p.add_argument("--max-models-per-dataset", type=int, default=20)
    p.add_argument("--embedding-mode", choices=["random", "embedding"], default="random",
                    help="Embedding mode: 'random' for ablation, 'embedding' for real node embeddings")
    args = p.parse_args()

    config = AttributeConfig(
        method="gnn",
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        seed=args.seed,
        max_datasets=args.max_datasets,
        max_models_per_dataset=args.max_models_per_dataset,
        embedding_mode=args.embedding_mode,
    )
    run_attribute_ranking(config)


if __name__ == "__main__":
    main()
