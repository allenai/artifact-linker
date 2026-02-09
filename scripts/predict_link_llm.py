#!/usr/bin/env python3
"""
LLM-based link prediction with optional RAG retrieval.

With RAG enabled, retrieval scores are computed for each pair to help
prioritize and augment predictions.

Examples:
    python predict_link_llm.py --max-pairs 1000
    python predict_link_llm.py --use-rag --rag-strategy hybrid
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_prediction
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="LLM Link Prediction")
    # Data
    p.add_argument("--data-dir", default="../data/artifact_graph_data_v2_1125")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive",
                   help="Split directory (uses same test set as GNN)")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--seed", type=int, default=42)

    # LLM options
    p.add_argument("--model", default="openai/gpt-4o")
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--no-info", action="store_false", dest="use_info")
    p.add_argument("--max-pairs", type=int, default=5000)
    p.add_argument("--workers", type=int, default=4)

    # RAG options
    p.add_argument("--use-rag", action="store_true",
                   help="Enable RAG retrieval scoring")
    p.add_argument("--rag-top-k", type=int, default=100,
                   help="Top-k for retrieval scoring")
    p.add_argument("--rag-strategy", default="hybrid",
                   choices=["embedding", "bm25", "heuristic", "hybrid"],
                   help="Retrieval strategy")
    args = p.parse_args()

    config = LinkConfig(
        method="llm",
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        llm_model=args.model,
        hops=args.hops,
        use_info=args.use_info,
        max_pairs=args.max_pairs,
        workers=args.workers,
        # RAG
        use_rag=args.use_rag,
        rag_top_k=args.rag_top_k,
        rag_strategy=args.rag_strategy,
    )
    run_link_prediction(config)


if __name__ == "__main__":
    main()
