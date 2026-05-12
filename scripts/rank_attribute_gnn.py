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
    p.add_argument("--metric", default=None)
    p.add_argument("--embedding-mode", choices=["random", "embedding"], default="random",
                    help="Embedding mode: 'random' for ablation, 'embedding' for real node embeddings")
    p.add_argument("--gnn-model", default="gatv2",
                    choices=["gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy"],
                    help="GNN model architecture (default: gatv2)")
    p.add_argument("--metric-file", default="edge_metadata_normalized.json",
                    help="Edge metadata file name (e.g. edge_metadata_normalized_attr.json for filtered metrics)")
    p.add_argument("--link-model-path", default="",
                    help="Path to link prediction model for two-stage scoring (link_score * attr_score)")
    p.add_argument("--rank-all-models", action="store_true",
                    help="Rank ALL models per dataset (not just observed ones)")
    args = p.parse_args()

    config = AttributeConfig(
        method="gnn",
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        seed=args.seed,
        metric_name=args.metric,
        embedding_mode=args.embedding_mode,
        gnn_model=args.gnn_model,
        metric_file=args.metric_file,
        link_model_path=args.link_model_path,
        rank_all_models=args.rank_all_models,
    )
    run_attribute_ranking(config)


if __name__ == "__main__":
    main()
