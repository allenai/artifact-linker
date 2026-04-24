#!/usr/bin/env python3
"""
Run evaluation coder with configurable modes.

Supports four modes:
- oneturn_onetool: Single turn with only run_code_in_docker (max_steps=1)
- multiturn_onetool: Multi-turn with only run_code_in_docker
- multiturn_metadatatool: Multi-turn with metadata tools + base_tools
- multiturn_cachefiletool: Multi-turn with all tools including cached loaders
"""

import os
import sys
import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup HF token before importing anything else
os.environ['HF_TOKEN'] = "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"

from artifact_graph.evaluation_coder_smolagent import EvaluationCoder, CoderMode
from artifact_graph.evaluation_coder_openai import OpenAIEvaluationCoder
from artifact_graph.evaluation_coder_multiagent import MultiAgentEvaluationCoder
from artifact_graph.evaluation_coder_skills_multiagent import SkillsMultiAgentEvaluationCoder


# ============== Logging Setup ==============

LOG_FILE_PATH = None


def setup_logging(output_dir: str, log_to_file: bool = True):
    """Setup logging to both console and file."""
    global LOG_FILE_PATH
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_to_file and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_FILE_PATH = os.path.join(output_dir, f"run_log_{timestamp}.log")
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"📝 Logging to: {LOG_FILE_PATH}")


class TeeOutput:
    """Tee stdout/stderr to both console and log file."""
    def __init__(self, log_file_path: str, stream):
        self.terminal = stream
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


# ============== Batch Evaluation Helpers ==============

def load_evaluation_triples(
    json_file: str, 
    dataset_filter: Optional[str] = None, 
    limit: int = 0
) -> List[Tuple[str, str, str]]:
    """Load model/dataset/metric triples from a JSON file."""
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
    """Create a safe directory name from model/dataset/metric."""
    safe_dir = f"{model}_{dataset}_{metric}"
    safe_dir = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_dir)
    safe_dir = re.sub(r"_+", "_", safe_dir)
    return safe_dir


def check_existing_results(output_dir: str) -> Optional[Dict[str, Any]]:
    """Check if agent_response.json or results.json already exists (supports both smolagent and multiagent backends)."""
    # Primary: agent_response.json (smolagent backend)
    response_file = os.path.join(output_dir, "agent_response.json")
    if os.path.exists(response_file):
        try:
            with open(response_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read existing agent response: {e}")

    # Fallback: results.json (multiagent backend)
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
        
        class DualWriter:
            def __init__(self, terminal, log_file):
                self.terminal = terminal
                self.log_file = log_file
            
            def write(self, message):
                self.terminal.write(message)
                self.log_file.write(message)
                self.log_file.flush()
            
            def flush(self):
                self.terminal.flush()
                self.log_file.flush()
        
        sys.stdout = DualWriter(self.original_stdout, self.log_file)
        sys.stderr = DualWriter(self.original_stderr, self.log_file)
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
    """Save batch evaluation summary to JSON file."""
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
    output_dir: str = "smolagent_results",
    gpu_id: int = 0,
    limit: int = 0,
    dataset_filter: str = None,
    max_steps: int = None,
    max_samples: int = 200,
    backend: str = "smolagents",
    num_splits: int = 1,
    split_id: int = 0,
):
    """Batch evaluate multiple model/dataset/metric combinations.
    
    num_splits / split_id: shard the triple list into num_splits parts and
    only process the split_id-th shard (0-indexed). Run multiple processes
    in parallel each with a different split_id to parallelise the full run.
    """
    
    # Load triples
    triples = load_evaluation_triples(json_file, dataset_filter, limit)
    
    if not triples:
        print("❌ No evaluable triples found.")
        if dataset_filter:
            print(f"   Dataset filter '{dataset_filter}' may not match any entries.")
        return

    # ── Shard the triple list ──────────────────────────────────────────────────
    if num_splits > 1:
        triples = [t for i, t in enumerate(triples) if i % num_splits == split_id]
        print(f"🔀 Shard {split_id}/{num_splits}: processing {len(triples)} triples")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get loaders directories
    script_dir = Path(__file__).parent
    dataset_loaders_dir = str(script_dir / "dataset_loaders")
    model_loaders_dir = str(script_dir / "model_loaders")
    
    # Create coder based on backend
    if backend == "openai":
        coder_cls = OpenAIEvaluationCoder
    elif backend == "multiagent":
        coder_cls = MultiAgentEvaluationCoder
    elif backend == "skills_multiagent":
        coder_cls = SkillsMultiAgentEvaluationCoder
    else:
        coder_cls = EvaluationCoder
    print(f"🔧 Backend: {backend} ({coder_cls.__name__})")
    coder = coder_cls.from_mode_string(
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
  python run_evaluation_coder.py --mode oneturn_onetool --gpu-id 5

  # Run with cached loaders
  python run_evaluation_coder.py --mode multiturn_cachefiletool --gpu-id 7
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
    parser.add_argument("--backend", default="smolagents",
                        choices=["smolagents", "openai", "multiagent", "skills_multiagent"],
                        help="Agent backend: smolagents (default), openai (OpenAI Agents SDK), "
                             "multiagent (Planning+Execution+Validation agents), "
                             "or skills_multiagent (multiagent with HF skills via ShellTool)")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Disable logging to file")
    parser.add_argument("--num-splits", type=int, default=1,
                        help="Split dataset into N shards and run all in parallel (default: 1 = no split)")
    parser.add_argument("--split-id", type=int, default=None,
                        help="Which shard to process (0-indexed). If omitted, spawns num-splits subprocesses automatically.")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel shards, e.g. '0,1,2,3'. Cycles if fewer than num-splits.")

    args = parser.parse_args()

    # ── Auto-parallel: spawn subprocesses when num_splits > 1 and no split_id given ──
    if args.num_splits > 1 and args.split_id is None:
        # Parse gpu-ids list
        if args.gpu_ids:
            gpu_list = [int(g) for g in args.gpu_ids.split(",")]
        else:
            gpu_list = [args.gpu_id]
        args.gpu_id = gpu_list  # pass list to _launch_parallel
        _launch_parallel(args)
        return

    # Default split_id to 0 if not set
    if args.split_id is None:
        args.split_id = 0
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        backend_tag = {
            "openai": "openai",
            "multiagent": "multiagent",
            "skills_multiagent": "skills_multiagent",
        }.get(args.backend, "smolagent")
        llm_safe = args.llm_model.replace("/", "-")
        args.output_dir = f"{backend_tag}_results_v3_hard_{llm_safe}_{args.mode}"
    
    # Setup logging
    setup_logging(args.output_dir, log_to_file=not args.no_log_file)
    
    # Tee stdout/stderr to log file
    if not args.no_log_file and LOG_FILE_PATH:
        sys.stdout = TeeOutput(LOG_FILE_PATH, sys.stdout)
        sys.stderr = TeeOutput(LOG_FILE_PATH, sys.stderr)
    
    print(f"🚀 Running EvaluationCoder")
    print(f"   Backend: {args.backend}")
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
        backend=args.backend,
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
