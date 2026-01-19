#!/usr/bin/env python3
"""
Re-run failed NLI SOTA experiments that don't have results.json.
"""

import os
import json
import subprocess
import argparse
from pathlib import Path


def parse_dir_name(dir_name: str) -> tuple:
    """Parse directory name to extract model, dataset, and metric."""
    # Format: MODEL_DATASET_METRIC
    # Example: google_gemma-2-2b_facebook_anli_accuracy
    parts = dir_name.rsplit("_", 1)
    if len(parts) != 2:
        return None, None, None
    
    metric = parts[1]  # e.g., "accuracy"
    remaining = parts[0]
    
    # Known dataset patterns (order matters - more specific first)
    dataset_patterns = [
        #("nyu-mll_multi_nli", "nyu-mll/multi_nli"),
        #("stanfordnlp_snli", "stanfordnlp/snli"),
        #("facebook_anli", "facebook/anli"),
        #("facebook_xnli", "facebook/xnli"),
        #("marzieh-saeidi_SICK", "marzieh-saeidi/SICK"),
        #("SetFit_wnli", "SetFit/wnli"),
        #("SetFit_qnli", "SetFit/qnli"),
        #("SetFit_rte", "SetFit/rte"),
        #("tasksource_defeasible-nli", "tasksource/defeasible-nli"),
        #("kiddothe2b_contract-nli", "kiddothe2b/contract-nli"),
        #("pietrolesci_nli_fever", "pietrolesci/nli_fever"),
        #("tasksource_babi_nli", "tasksource/babi_nli"),
        ("allenai_scitail", "allenai/scitail"),
    ]
    
    model = None
    dataset = None
    
    for dir_pattern, dataset_id in dataset_patterns:
        if dir_pattern in remaining:
            idx = remaining.find(dir_pattern)
            model_part = remaining[:idx].rstrip("_")
            # Convert underscores back to slashes for model names
            model = model_part.replace("_", "/", 1)  # Only first underscore
            dataset = dataset_id
            break
    
    return model, dataset, metric


def find_failed_experiments(base_dir: Path, threshold: float = 0.05, only_low_results: bool = False) -> list:
    """Find all experiment directories without results.json or with results < threshold.
    
    Args:
        base_dir: Directory containing experiment results
        threshold: Re-run experiments with results below this threshold
        only_low_results: If True, only re-run experiments with low results (skip missing results.json)
    """
    failed = []
    
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / "results.json"
        model, dataset, metric = parse_dir_name(exp_dir.name)
        
        if not model or not dataset or not metric:
            continue
        
        should_rerun = False
        reason = ""
        
        if not results_file.exists():
            if not only_low_results:
                should_rerun = True
                reason = "no results.json"
        else:
            # Check if result value is below threshold
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    # results.json format: {"accuracy": 0.85} or similar
                    for key, value in results.items():
                        if isinstance(value, (int, float)) and value < threshold:
                            should_rerun = True
                            reason = f"{key}={value:.4f} < {threshold}"
                            break
            except (json.JSONDecodeError, IOError) as e:
                if not only_low_results:
                    should_rerun = True
                    reason = f"error reading results: {e}"
        
        if should_rerun:
            failed.append({
                "dir_name": exp_dir.name,
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "path": str(exp_dir),
                "reason": reason
            })
    
    return failed


def create_rerun_json(failed_experiments: list, output_file: Path) -> None:
    """Create JSON file for re-running failed experiments."""
    results = []
    for exp in failed_experiments:
        results.append({
            "model_id": exp["model"],
            "model_downloads": 0,
            "dataset_id": exp["dataset"],
            "dataset_downloads": 0,
            "metrics": {
                exp["metric"]: 0
            }
        })
    
    with open(output_file, 'w') as f:
        json.dump({"results": results}, f, indent=2)
    
    print(f"📝 Created re-run JSON file: {output_file}")
    print(f"   Contains {len(results)} experiments to re-run")


def remove_agent_response_files(failed_experiments: list) -> None:
    """Remove agent_response.json files from failed experiment directories."""
    removed = 0
    for exp in failed_experiments:
        agent_response_file = Path(exp["path"]) / "agent_response.json"
        if agent_response_file.exists():
            agent_response_file.unlink()
            removed += 1
    
    print(f"🗑️  Removed {removed} agent_response.json files to allow re-running")


def main():
    parser = argparse.ArgumentParser(description="Re-run failed NLI SOTA experiments")
    parser.add_argument(
        "--dir",
        type=str,
        default="smolagent_results_coding_agent_nli_sota_0112_full_shared_loader",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Max CodeAct iteration steps (default: 15, increased from 10)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.2",
        help="LLM model to use"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=5,
        help="GPU device ID to use"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be re-run, don't execute"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Re-run experiments with results below this threshold (default: 0.05)"
    )
    parser.add_argument(
        "--only-low-results",
        action="store_true",
        help="Only re-run experiments with low results (skip missing results.json)"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    base_dir = script_dir / args.dir
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return
    
    # Find failed experiments
    failed = find_failed_experiments(base_dir, threshold=args.threshold, only_low_results=args.only_low_results)
    
    if not failed:
        print("✅ All experiments have valid results - nothing to re-run!")
        return
    
    print(f"\n🔍 Found {len(failed)} experiments to re-run:\n")
    for i, exp in enumerate(failed, 1):
        print(f"  {i:2d}. {exp['model']}")
        print(f"      Dataset: {exp['dataset']}")
        print(f"      Metric: {exp['metric']}")
        print(f"      Reason: {exp['reason']}")
        print()
    
    if args.dry_run:
        print("🔸 Dry run mode - not executing")
        return
    
    # Create temporary JSON file for re-running
    rerun_json = script_dir / "rerun_failed_experiments.json"
    create_rerun_json(failed, rerun_json)
    
    # Remove agent_response.json files so they can be re-run
    remove_agent_response_files(failed)
    
    # Group by dataset for parallel execution
    by_dataset = {}
    for exp in failed:
        dataset = exp["dataset"]
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(exp)
    
    print(f"\n📊 Experiments by dataset:")
    for dataset, exps in by_dataset.items():
        print(f"  - {dataset}: {len(exps)} experiments")
    
    # Run the experiments
    print(f"\n🚀 Starting re-run with max_steps={args.max_steps}...")
    print(f"   GPU: {args.gpu_id}")
    print(f"   LLM: {args.llm_model}")
    print()
    
    cmd = [
        "python", "run_smolagent_advanced_coder.py",
        "--json-file", str(rerun_json),
        "--llm-model", args.llm_model,
        "--output-dir", args.dir,
        "--max-steps", str(args.max_steps),
        "--gpu-id", str(args.gpu_id)
    ]
    
    print(f"Running command:")
    print(f"  {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, cwd=str(script_dir))
    
    print("\n✅ Re-run complete!")


if __name__ == "__main__":
    main()

