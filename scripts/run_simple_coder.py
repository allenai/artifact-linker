#!/usr/bin/env python3
"""
Batch evaluate all model/dataset/metric triples using SimpleDockerCoder

Usage:
    python run_simple_coder.py --llm-model gpt-4o --max-fixes 10 --limit 20 --output-dir results --memory-limit 8g
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

# Allow importing artifact_graph
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifact_graph.simple_coder_docker import SimpleDockerCoder


def load_combinations(json_path: Path):
    """Load model-dataset-metric combinations from the JSON file."""
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
    parser.add_argument(
        "--max-fixes", type=int, default=5, help="Maximum fixes for the evaluation script"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Evaluate only first N triples (0 = all)"
    )
    parser.add_argument(
        "--output-dir",
        default="simple_results_0801_simple",
        help="Output directory for results (default: simple_results)",
    )
    parser.add_argument(
        "--memory-limit", default="32g", help="Docker container memory limit (default: 32g)"
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--dataset-name", default="mnli", help="Dataset name to filter")
    parser.add_argument("--eval-timeout", type=int, default=900, help="Evaluation timeout in seconds")
    parser.add_argument("--no-gpu", action="store_false", dest="enable_gpu", help="Disable GPU support")
    parser.add_argument("--gpu-device-ids", type=int, nargs='+', default=[3], help="GPU device IDs to use")
    args = parser.parse_args()

    combos = load_combinations(Path(args.json_file))
    triples = list(iter_triples(combos))

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
        safe_dir = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_dir)
        # Remove consecutive underscores
        safe_dir = re.sub(r"_+", "_", safe_dir)
        out_dir = f"{args.output_dir}/{safe_dir}"

        try:
            # Create a new SimpleDockerCoder instance for each evaluation
            coder = SimpleDockerCoder(
                model=args.llm_model, 
                output_dir=out_dir, 
                memory_limit=args.memory_limit, 
                enable_gpu=args.enable_gpu,
                eval_timeout=args.eval_timeout,
                gpu_device_ids=args.gpu_device_ids
            )
            result = coder.evaluate(
                model_name=model,
                dataset_name=dataset,
                metric=metric,
                max_fixes=args.max_fixes,
            )
            success = result.get("success", False)
            if success:
                results_data = result.get("experiment_results", {})
                message = f"Success. Results: {json.dumps(results_data)}"
            else:
                message = f"Failed. Error: {result.get('error', 'Unknown error')}"
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
                "success_rate": success_count / len(triples) if len(triples) > 0 else 0,
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
