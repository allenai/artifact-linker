#!/usr/bin/env python3
"""
Batch‑evaluate all model/dataset/metric triples from perfect_model_dataset_metrics.json

Usage:
    python hf_auto_eval.py --llm-model gpt-4o --runs 1 --dataset-max-fixes 3 --model-max-fixes 3 --metric-max-fixes 10 --limit 20 --output-dir results --memory-limit 8g
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing artifact_graph
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifact_graph.coder_docker import DockerCoder


def load_combinations(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("results", [])


def iter_triples(combos):
    """
    Yield (model_id, dataset_id, metric_name) for every simple metric.
    Skips metrics whose value is a dict (nested/complex).
    """
    for combo in combos:
        model = combo.get("model_id")
        dataset = combo.get("dataset_id")
        metrics = combo.get("metrics", {}) or {}
        for metric_name, value in metrics.items():
            if value is None or isinstance(value, dict):
                continue
            yield model, dataset, metric_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-file", default=str(Path(__file__).parent / "perfect_model_dataset_metrics.json")
    )
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--dataset-max-fixes", type=int, default=3, help="Maximum fixes for dataset processing"
    )
    parser.add_argument(
        "--model-max-fixes", type=int, default=3, help="Maximum fixes for model processing"
    )
    parser.add_argument(
        "--metric-max-fixes", type=int, default=10, help="Maximum fixes for metric evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Evaluate only first N triples (0 = all)"
    )
    parser.add_argument(
        "--output-dir",
        default="simple_results_0801",
        help="Output directory for results (default: simple_results)",
    )
    parser.add_argument(
        "--memory-limit", default="32g", help="Docker container memory limit (default: 8g)"
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--dataset-name", default="mnli", help="Dataset name to filter")
    args = parser.parse_args()

    combos = load_combinations(Path(args.json_file))
    triples = list(iter_triples(combos))
    # if args.limit and args.limit > 0:
    #    triples = triples[args.start_index: args.start_index + args.limit]

    filtered_triples = []
    for triple in triples:
        if args.dataset_name in triple[1]:
            filtered_triples.append(triple)

    triples = filtered_triples

    if not triples:
        print("No evaluable triples found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    summary = []
    success_count = 0

    print(f"Starting evaluation of {len(triples)} triples...\n")
    for i, (model, dataset, metric) in enumerate(triples, 1):
        print("=" * 80)
        print(f"[{i}/{len(triples)}] {model} | {dataset} | {metric}")
        print("=" * 80)
        # Create a safe directory name by replacing invalid characters
        safe_dir = f"{model}_{dataset}_{metric}"
        # Replace all invalid characters for Docker volume names
        import re

        safe_dir = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_dir)
        # Remove consecutive underscores
        safe_dir = re.sub(r"_+", "_", safe_dir)
        out_dir = f"{args.output_dir}/{safe_dir}"

        try:
            # Create a new DockerCoder instance for each evaluation with its own output directory
            coder = DockerCoder(
                model=args.llm_model, output_dir=out_dir, memory_limit=args.memory_limit
            )
            result = coder.evaluate(
                model_name=model,
                dataset_name=dataset,
                metric=metric,
                dataset_max_fixes=args.dataset_max_fixes,
                model_max_fixes=args.model_max_fixes,
                metric_max_fixes=args.metric_max_fixes,
            )
            success = result.get("success", False)
            if success:
                message = f"Completed {result.get('successful_runs', 0)}/{result.get('total_runs', 0)} runs"
            else:
                message = f"All {result.get('total_runs', 0)} runs failed"
        except Exception as e:
            success, message = False, f"Exception: {e}"

        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "success": success,
                "message": message,
                "output_dir": out_dir if success else None,
            }
        )
        if success:
            success_count += 1

    # Save summary
    summary_path = f"{args.output_dir}/batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total": len(triples),
                "success": success_count,
                "failed": len(triples) - success_count,
                "success_rate": success_count / len(triples),
                "results": summary,
            },
            f,
            indent=2,
        )

    print("\nDone.")
    print(
        f"Success: {success_count} / {len(triples)} " f"({success_count / len(triples) * 100:.1f}%)"
    )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
