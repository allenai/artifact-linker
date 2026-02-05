#!/usr/bin/env python3
"""LLM-based attribute ranking."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners import run_attribute_ranking
from artifact_graph.runners.attribute_runner import AttributeConfig


def main():
    p = argparse.ArgumentParser(description="LLM Attribute Ranking")
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="openai/gpt-4o")
    p.add_argument("--hops", type=int, default=0)
    p.add_argument("--no-info", action="store_false", dest="use_info")
    p.add_argument("--metric", default="accuracy")
    p.add_argument("--max-datasets", type=int, default=0)
    p.add_argument("--max-models-per-dataset", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--use-gnn-data", action="store_true")
    args = p.parse_args()

    config = AttributeConfig(
        method="llm",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        llm_model=args.model,
        hops=args.hops,
        use_info=args.use_info,
        metric_name=args.metric,
        max_datasets=args.max_datasets,
        max_models_per_dataset=args.max_models_per_dataset,
        workers=args.workers,
        use_gnn_data=args.use_gnn_data,
    )
    run_attribute_ranking(config)


if __name__ == "__main__":
    main()
