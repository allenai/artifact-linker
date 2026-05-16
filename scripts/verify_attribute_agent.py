#!/usr/bin/env python3
"""
Run skills-multiagent evaluation coder with configurable modes.

Modes:
- oneturn_onetool: Single turn with only run_code_in_docker (max_steps=1)
- multiturn_onetool: Multi-turn with only run_code_in_docker
- multiturn_metadatatool: Multi-turn with metadata tools + base_tools
- multiturn_cachefiletool: Multi-turn with all tools including cached loaders
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get('HF_TOKEN'):
    raise RuntimeError("HF_TOKEN env var is required. Set it before running.")

from artifact_graph.coder import SkillsMultiAgentEvaluationCoder


class TeeOutput:
    def __init__(self, log_file, stream):
        self.terminal = stream
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ============== Batch Evaluation Helpers ==============

def load_evaluation_triples(
    json_file: str,
    dataset_filter: Optional[str] = None,
    limit: int = 0
) -> List[Tuple[str, str, str]]:
    with open(json_file, "r") as f:
        data = json.load(f)
    
    combos = data.get("results", [])
    
    triples = []
    for combo in combos:
        model = combo.get("model_id")
        dataset = combo.get("dataset_id")
        metrics = combo.get("metrics", {}) or {}
        
        for metric_name, value in metrics.items():
            if value is None or isinstance(value, dict):
                continue
            if dataset_filter and dataset_filter not in dataset:
                continue
            triples.append((model, dataset, metric_name))
    
    total_found = len(triples)
    if limit:
        triples = triples[:limit]
    
    print(f"📊 Found {total_found} matching triples" + (f", limiting to {limit}" if limit else ""))
    return triples


def make_safe_dirname(model: str, dataset: str, metric: str) -> str:
    safe_dir = f"{model}_{dataset}_{metric}"
    safe_dir = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_dir)
    safe_dir = re.sub(r"_+", "_", safe_dir)
    return safe_dir


def check_existing_results(output_dir: str) -> Optional[Dict[str, Any]]:
    response_file = os.path.join(output_dir, "agent_response.json")
    if os.path.exists(response_file):
        try:
            with open(response_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read existing agent response: {e}")

    # Fallback: results.json
    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            if data and any(isinstance(v, (int, float)) for v in data.values()):
                return {"skipped": True, "results": data}
        except Exception as e:
            print(f"⚠️ Failed to read existing results.json: {e}")

    return None


class EvaluationLogger:
    """Context manager to capture output for a single evaluation."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "run.log")
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None
    
    def __enter__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TeeOutput(self.log_file, self.original_stdout)
        sys.stderr = TeeOutput(self.log_file, self.original_stderr)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_file:
            self.log_file.close()
        print(f"📝 Evaluation log saved: {self.log_path}")
        return False


def save_batch_summary(
    output_dir: str,
    summary: List[Dict],
    total: int,
    success_count: int
) -> str:
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "success_rate": success_count / total if total else 0,
            "results": summary
        }, f, indent=2)
    return summary_path


# ============== Main Batch Evaluation ==============

def batch_evaluate(
    json_file: str,
    mode: str,
    llm_model: str = "gpt-4o",
    output_dir: str = "skills_multiagent_results",
    gpu_id: int = 0,
    limit: int = 0,
    dataset_filter: str = None,
    max_steps: int = None,
    max_samples: int = 200,
    num_splits: int = 1,
    split_id: int = 0,
):
    """Batch evaluate multiple model/dataset/metric combinations.

    num_splits / split_id: shard the triple list into num_splits parts and
    only process the split_id-th shard (0-indexed). Run multiple processes
    in parallel each with a different split_id to parallelise the full run.
    """

    triples = load_evaluation_triples(json_file, dataset_filter, limit)

    if not triples:
        print("❌ No evaluable triples found.")
        if dataset_filter:
            print(f"   Dataset filter '{dataset_filter}' may not match any entries.")
        return

    if num_splits > 1:
        triples = [t for i, t in enumerate(triples) if i % num_splits == split_id]
        print(f"🔀 Shard {split_id}/{num_splits}: processing {len(triples)} triples")

    os.makedirs(output_dir, exist_ok=True)

    script_dir = Path(__file__).parent
    dataset_loaders_dir = str(script_dir / "dataset_loaders")
    model_loaders_dir = str(script_dir / "model_loaders")

    coder = SkillsMultiAgentEvaluationCoder.from_mode_string(
        mode_str=mode,
        llm_model=llm_model,
        gpu_id=gpu_id,
        max_steps=max_steps,
        dataset_loaders_dir=dataset_loaders_dir,
        model_loaders_dir=model_loaders_dir,
    )
    
    summary = []
    success_count = 0
    
    print(f"Starting evaluation of {len(triples)} triples...\n")
    
    for i, (model, dataset, metric) in enumerate(triples, 1):
        print("=" * 80)
        print(f"[{i}/{len(triples)}] {model} | {dataset} | {metric}")
        print("=" * 80)
        
        # Create output directory
        safe_dir = make_safe_dirname(model, dataset, metric)
        out_dir = os.path.join(output_dir, safe_dir)
        
        # Check for existing results
        existing_results = check_existing_results(out_dir)
        if existing_results is not None:
            print(f"⏭️  Skipping - agent_response.json already exists")
            summary.append({
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "success": True,
                "message": f"Skipped. Existing results.",
                "output_dir": out_dir
            })
            success_count += 1
            continue
        
        # Run evaluation with logging
        with EvaluationLogger(out_dir):
            try:
                result = coder.evaluate(
                    model_name=model,
                    dataset_name=dataset,
                    metric=metric,
                    output_dir=out_dir,
                    max_samples=max_samples,
                )
                success = isinstance(result, dict) and result.get("success", False)
                message = f"Success. Results: {json.dumps(result)}" if success else f"Failed: {result}"
            except Exception as e:
                success = False
                message = f"Exception: {e}"
        
        summary.append({
            "model": model,
            "dataset": dataset,
            "metric": metric,
            "success": success,
            "message": message,
            "output_dir": out_dir if success else None
        })
        
        if success:
            success_count += 1
    
    # Save summary
    summary_path = save_batch_summary(output_dir, summary, len(triples), success_count)
    
    print(f"\nDone. Success: {success_count}/{len(triples)}")
    print(f"Summary: {summary_path}")


# ============== Entry Point ==============

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation coder with configurable modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  oneturn_onetool       Single turn, only run_code_in_docker (max_steps=1)
  multiturn_onetool     Multi-turn, only run_code_in_docker
  multiturn_metadatatool Multi-turn, with metadata tools + base_tools
  multiturn_cachefiletool Multi-turn, with all tools including cached loaders

Examples:
  # Run oneshot mode on GPU 5
  python verify_attribute_agent.py --mode oneturn_onetool --gpu-id 5

  # Run with cached loaders
  python verify_attribute_agent.py --mode multiturn_cachefiletool --gpu-id 7
"""
    )
    
    parser.add_argument("--json-file", default="perfect_model_dataset_metrics_v3_0120_coding_agent_filtered_hard_both_successful.json",
                        help="Input JSON file with model/dataset list")
    parser.add_argument("--mode", "-m", required=True,
                        choices=["oneturn_onetool", "multiturn_onetool", 
                                "multiturn_metadatatool", "multiturn_cachefiletool"],
                        help="Coder mode")
    parser.add_argument("--llm-model", default="openai/codex-5.2",
                        help="LLM model to use")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: auto-generated based on mode)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of triples to evaluate (0 = no limit)")
    parser.add_argument("--dataset-name", default=None,
                        help="Filter by dataset name")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps for the mode")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples to evaluate per dataset (-1 for no limit)")
    parser.add_argument("--num-splits", type=int, default=1,
                        help="Split dataset into N shards and run all in parallel (default: 1 = no split)")
    parser.add_argument("--split-id", type=int, default=None,
                        help="Which shard to process (0-indexed). If omitted, spawns num-splits subprocesses automatically.")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel shards, e.g. '0,1,2,3'. Cycles if fewer than num-splits.")

    args = parser.parse_args()

    if args.num_splits > 1 and args.split_id is None:
        if args.gpu_ids:
            gpu_list = [int(g) for g in args.gpu_ids.split(",")]
        else:
            gpu_list = [args.gpu_id]
        args.gpu_id = gpu_list
        _launch_parallel(args)
        return

    if args.split_id is None:
        args.split_id = 0

    if args.output_dir is None:
        llm_safe = args.llm_model.replace("/", "-")
        args.output_dir = f"skills_multiagent_results_v3_hard_{llm_safe}_{args.mode}"

    print(f"🚀 Running EvaluationCoder")
    print(f"   Mode: {args.mode}")
    print(f"   GPU: {args.gpu_id}")
    print(f"   JSON: {args.json_file}")
    print(f"   Output: {args.output_dir}")

    batch_evaluate(
        json_file=args.json_file,
        mode=args.mode,
        llm_model=args.llm_model,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        limit=args.limit,
        dataset_filter=args.dataset_name,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        num_splits=args.num_splits,
        split_id=args.split_id,
    )


def _launch_parallel(args):
    """When --num-splits > 1 and --split-id is not set by user, spawn N subprocesses."""
    import subprocess
    base_cmd = [sys.executable] + sys.argv[:]
    # Remove --num-splits from base_cmd and re-add it; add --split-id per process
    procs = []
    gpu_ids = args.gpu_id if isinstance(args.gpu_id, list) else [args.gpu_id]
    for i in range(args.num_splits):
        gpu = gpu_ids[i % len(gpu_ids)]
        cmd = base_cmd + ["--split-id", str(i), "--gpu-id", str(gpu)]
        print(f"  Spawning shard {i}/{args.num_splits} on GPU {gpu}: {' '.join(cmd[-6:])}")
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    print(f"All {args.num_splits} shards completed.")


if __name__ == "__main__":
    main()
