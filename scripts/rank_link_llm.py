#!/usr/bin/env python3
"""LLM-based link ranking (uses ALL models as candidates)."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_link_ranking
from artifact_graph.runners.link_runner import LinkConfig


def main():
    p = argparse.ArgumentParser(description="LLM Link Ranking")
    p.add_argument("--data-dir", default="../data/artifact_graph_data_v2_1125")
    p.add_argument("--split-dir", default="../data/artifact_graph_splits_v2_1125_transductive")
    p.add_argument("--output-dir", default="../data/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model", default="openai/gpt-4o",
                   help="LLM model id (litellm format, e.g. openai/gpt-5.2)")
    p.add_argument("--hops", type=int, default=1,
                   help="Graph context hops (0 = no graph context)")
    p.add_argument("--no-info", action="store_true",
                   help="Disable node info text in prompts")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    config = LinkConfig(
        method="llm",
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        llm_model=args.llm_model,
        hops=args.hops,
        use_info=not args.no_info,
        workers=args.workers,
    )
    run_link_ranking(config)


if __name__ == "__main__":
    main()
