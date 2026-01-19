#!/usr/bin/env python3
"""
使用 smolagents 实现类似 SimpleDockerCoder 的功能
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path

import litellm
litellm.drop_params = True  # Automatically drop unsupported params like 'stop'
os.environ["LITELLM_DROP_PARAMS"] = "true"  # Also set via environment variable

# Monkey-patch litellm.completion to drop 'stop' parameter for models that don't support it
_original_completion = litellm.completion
def _patched_completion(*args, **kwargs):
    # Remove 'stop' parameter if present (for models like gpt-5.2 that don't support it)
    if 'stop' in kwargs:
        del kwargs['stop']
    return _original_completion(*args, **kwargs)
litellm.completion = _patched_completion
os.environ['HF_TOKEN'] = "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"

import logging
import sys
from datetime import datetime

# Will be configured later with file handler
logger = logging.getLogger(__name__)

from smolagents import CodeAgent, LiteLLMModel, tool


# ============== Global Configuration ==============
# These can be set before running the agent
GLOBAL_GPU_ID = 0  # Default GPU device ID
LOG_FILE_PATH = None  # Will be set in main()


def setup_logging(output_dir: str, log_to_file: bool = True):
    """Setup logging to both console and file."""
    global LOG_FILE_PATH
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
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


# ============== Tool Definitions ==============

@tool
def run_code_in_docker(
    code: str, 
    output_dir: str = "/tmp/eval_workspace"
) -> dict:
    """
    Execute Python code inside a Docker container with GPU support.
    The code will be saved to 'evaluate.py' and executed with 'python evaluate.py'.
    
    IMPORTANT: The 'code' parameter must be actual Python source code, NOT a shell command.
    
    Example of CORRECT usage:
        run_code_in_docker(
            code='''
import torch
print("Hello from Docker!")
print(f"CUDA available: {torch.cuda.is_available()}")
''',
            output_dir="/tmp/test"
        )
    
    Example of WRONG usage (do NOT do this):
        run_code_in_docker(code="python script.py")  # WRONG! This is a shell command, not Python code
    
    Args:
        code: The actual Python source code to execute (NOT a shell command)
        output_dir: Directory to store scripts and results
    
    Returns:
        A dict with 'success', 'exit_code', 'output', and 'results' keys
    """
    import time
    import threading
    
    # Always use global GPU ID (configured via --gpu-id CLI argument)
    gpu_id = GLOBAL_GPU_ID
    
    # Fixed timeout and memory limit (not configurable by Agent)
    timeout = 900  # 15 minutes
    memory_limit = "32g"
    
    # Convert to absolute path (Docker requires absolute paths for volume mounts)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "run_eval.py")  # Avoid conflict with 'evaluate' library
    
    # Save the script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"\n{'='*60}")
    print(f"🐳 Docker execution started")
    print(f"📁 Output dir: {output_dir}")
    print(f"⏱️  Timeout: {timeout}s")
    print(f"{'='*60}\n")
    
    # Build docker command using the same image as DockerManager
    # Get environment variables for HuggingFace and API access
    hf_token = os.getenv("HF_TOKEN", "")
    
    # Use --gpus device=X format to specify specific GPU (same as DockerManager)
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{output_dir}:/workspace",
        "-w", "/workspace",
        "-m", memory_limit,
        "--gpus", f"device={gpu_id}",  # Specify GPU device directly
        "-e", f"HF_TOKEN={hf_token}",  # HuggingFace token for gated models
        "-e", "PYTHONPATH=/workspace",
        "simple-coder:latest",  # Custom image with dependencies pre-installed
        "bash", "-c",
        "python run_eval.py"  # Dependencies already installed in image
    ]
    
    output_lines = []
    
    def stream_output(pipe, prefix=""):
        """Stream output from pipe in real-time."""
        for line in iter(pipe.readline, ''):
            if line:
                print(f"  {prefix}{line.rstrip()}")
                output_lines.append(line)
        pipe.close()
    
    try:
        # Use Popen for real-time streaming
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        start_time = time.time()
        for line in iter(process.stdout.readline, ''):
            if line:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] {line.rstrip()}")
                output_lines.append(line)
            
            # Check timeout
            if time.time() - start_time > timeout:
                process.kill()
                print(f"\n❌ Docker execution timed out after {timeout}s")
                return {
                    "success": False,
                    "exit_code": -1,
                    "output": "Execution timed out after " + str(timeout) + "s",
                    "results": {}
                }
        
        process.stdout.close()
        exit_code = process.wait()
        
        output = "".join(output_lines)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"🐳 Docker finished in {elapsed:.1f}s with exit code {exit_code}")
        print(f"{'='*60}\n")
        
        # Check for results.json
        results_path = os.path.join(output_dir, "results.json")
        results = {}
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            print(f"✅ Results: {results}")
        
        return {
            "success": exit_code == 0 and bool(results),
            "exit_code": exit_code,
            "output": output[-2000:],  # Truncate long outputs
            "results": results
        }
    except Exception as e:
        print(f"\n❌ Docker execution failed: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "output": str(e),
            "results": {}
        }


@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        The file contents as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def save_file(file_path: str, content: str) -> str:
    """
    Save content to a file.
    
    Args:
        file_path: Path to save the file
        content: Content to write
    
    Returns:
        Success or error message
    """
    try:
        # Convert to absolute path
        file_path = os.path.abspath(file_path)
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved to {file_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


# ============== Main Agent Setup ==============

def create_evaluation_agent(
    model_id: str = "gpt-4o",
    api_key: str = None,
    api_base: str = None,
    temperature: float = 0,
    max_steps: int = 10
):
    """Create a smolagents CodeAgent for ML evaluation tasks.
    
    Args:
        model_id: LLM model identifier
        api_key: API key for the LLM
        api_base: Base URL for the API
        temperature: Sampling temperature
        max_steps: Maximum CodeAct iteration steps (think -> act -> observe cycles)
    """
    
    model = LiteLLMModel(
        model_id=model_id,
        temperature=temperature,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        api_base=api_base,
    )
    
    # Core tools only - web_search is provided by add_base_tools=True
    tools = [
        run_code_in_docker,
        read_file,
        save_file,
    ]
    print(f"🔧 Tools: run_code_in_docker, read_file, save_file")
    
    agent = CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=False,
        max_steps=max_steps,
        verbosity_level=2,  # 0=silent, 1=basic, 2=detailed (shows each step)
    )
    
    return agent


def evaluate_model(
    agent,
    model_name: str,
    dataset_name: str,
    metric: str = "accuracy",
    output_dir: str = "results"
):
    """Run evaluation using the smolagents agent."""
    
    prompt = f"""
You are an expert ML engineer. Your task is to evaluate the model `{model_name}` 
on the dataset `{dataset_name}` using the metric `{metric}`.

AVAILABLE TOOLS:
- `run_code_in_docker(code, output_dir)` - Execute Python code in Docker with GPU
- `read_file(path)` / `save_file(path, content)` - File I/O

TASK:
1. Write a Python evaluation script that loads the model, evaluates it on the dataset, 
   and computes the {metric} metric.
2. Use `run_code_in_docker` to execute the script with output_dir="{output_dir}"
3. The script MUST save results to 'results.json' in the format: {{"{metric}": <value>}}
4. If it fails, analyze the error, fix the code and retry.

REQUIREMENTS:
- Use GPU: model.to("cuda"), inputs to GPU
- Use batched inference for speed
- For large datasets (hellaswag, winogrande, etc.) with >200 examples, randomly sample up to 200 examples
- Save results to results.json

Return the final evaluation results.
"""
    
    print("----- Agent 开始执行 -----")
    result = agent.run(prompt)
    print("\n----- 最终结果 -----")
    print(result)
    
    # Save agent's final response to file
    agent_response_path = os.path.join(output_dir, "agent_response.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Convert result to serializable format
        if isinstance(result, dict):
            response_data = result
        else:
            response_data = {"response": str(result)}
        
        with open(agent_response_path, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Agent response saved to: {agent_response_path}")
    except Exception as e:
        print(f"⚠️ Failed to save agent response: {e}")
    
    return result


# ============== Batch Evaluation Helper Functions ==============

import re
from typing import List, Tuple, Optional, Dict, Any


def load_evaluation_triples(
    json_file: str, 
    dataset_filter: Optional[str] = None, 
    limit: int = 0
) -> List[Tuple[str, str, str]]:
    """
    Load model/dataset/metric triples from a JSON file.
    
    Args:
        json_file: Path to JSON file with evaluation configurations
        dataset_filter: Filter to match dataset names (substring match)
        limit: Maximum number of triples to return (0 = no limit)
    
    Returns:
        List of (model_id, dataset_id, metric_name) tuples
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    combos = data.get("results", [])
    
    triples = []
    for combo in combos:
        model = combo.get("model_id")
        dataset = combo.get("dataset_id")
        metrics = combo.get("metrics", {}) or {}
        
        for metric_name, value in metrics.items():
            # Skip invalid metrics
            if value is None or isinstance(value, dict):
                continue
            # Apply dataset filter
            if dataset_filter and dataset_filter not in dataset:
                continue
            triples.append((model, dataset, metric_name))
    
    total_found = len(triples)
    if limit:
        triples = triples[:limit]
    
    print(f"📊 Found {total_found} matching triples" + (f", limiting to {limit}" if limit else ""))
    
    return triples


def make_safe_dirname(model: str, dataset: str, metric: str) -> str:
    """
    Create a safe directory name from model/dataset/metric.
    
    Replaces invalid characters with underscores and removes consecutive underscores.
    """
    safe_dir = f"{model}_{dataset}_{metric}"
    safe_dir = re.sub(r"[^a-zA-Z0-9._-]", "_", safe_dir)
    safe_dir = re.sub(r"_+", "_", safe_dir)
    return safe_dir


def check_existing_results(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Check if agent_response.json already exists in the output directory.
    This indicates the agent has already attempted this task (success or failure).
    
    Args:
        output_dir: Directory to check for agent_response.json
    
    Returns:
        Existing agent response dict if found and valid, None otherwise
    """
    response_file = os.path.join(output_dir, "agent_response.json")
    if os.path.exists(response_file):
        try:
            with open(response_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read existing agent response: {e}")
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
        
        # Create a custom writer that writes to both console and log file
        class DualWriter:
            def __init__(self, terminal, log_file):
                self.terminal = terminal
                self.log_file = log_file
            
            def write(self, message):
                # Write to terminal (which may already be a TeeOutput)
                self.terminal.write(message)
                # Also write to per-evaluation log
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


def run_single_evaluation(
    agent,
    model: str,
    dataset: str,
    metric: str,
    output_dir: str
) -> Tuple[bool, str]:
    """
    Run a single model evaluation.
    
    Args:
        agent: The evaluation agent
        model: Model ID
        dataset: Dataset ID
        metric: Metric name
        output_dir: Output directory
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Use context manager to capture all output to per-evaluation log
    with EvaluationLogger(output_dir):
        try:
            result = evaluate_model(
                agent=agent,
                model_name=model,
                dataset_name=dataset,
                metric=metric,
                output_dir=output_dir
            )
            success = isinstance(result, dict) and result.get("success", False)
            message = f"Success. Results: {json.dumps(result)}" if success else f"Failed: {result}"
            return success, message
        except Exception as e:
            return False, f"Exception: {e}"


def save_batch_summary(
    output_dir: str,
    summary: List[Dict],
    total: int,
    success_count: int
) -> str:
    """
    Save batch evaluation summary to JSON file.
    
    Returns:
        Path to the saved summary file
    """
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
    llm_model: str = "gpt-4o",
    output_dir: str = "smolagents_results",
    limit: int = 0,
    dataset_filter: str = None,
    max_steps: int = 10
):
    """Batch evaluate multiple model/dataset/metric combinations."""
    
    # Load and filter triples
    triples = load_evaluation_triples(json_file, dataset_filter, limit)
    
    if not triples:
        print("❌ No evaluable triples found.")
        if dataset_filter:
            print(f"   Dataset filter '{dataset_filter}' may not match any entries in the JSON file.")
            print(f"   Check that the filter is a substring of dataset IDs (e.g., 'sst2' matches 'SetFit/sst2').")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create agent
    agent = create_evaluation_agent(model_id=llm_model, max_steps=max_steps)
    
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
                "message": f"Skipped. Existing results: {json.dumps(existing_results)}",
                "output_dir": out_dir
            })
            success_count += 1
            continue
        
        # Run evaluation
        success, message = run_single_evaluation(agent, model, dataset, metric, out_dir)
        
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
    global GLOBAL_GPU_ID
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", default="perfect_model_dataset_metrics_v2_1125.json")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--output-dir", default="smolagents_results")
    parser.add_argument("--limit", type=int, default=0, 
                        help="Max number of triples to evaluate (0 = no limit)")
    parser.add_argument("--dataset-name", default=None, 
                        help="Filter by dataset name (e.g., 'sst2' matches 'SetFit/sst2')")
    parser.add_argument("--max-steps", type=int, default=10, 
                        help="Max CodeAct iteration steps (think -> act -> observe cycles)")
    parser.add_argument("--gpu-id", type=int, default=9,
                        help="GPU device ID to use")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Disable logging to file")
    args = parser.parse_args()
    
    # Setup logging (to both console and file)
    setup_logging(args.output_dir, log_to_file=not args.no_log_file)
    
    # Tee stdout/stderr to log file
    if not args.no_log_file and LOG_FILE_PATH:
        sys.stdout = TeeOutput(LOG_FILE_PATH, sys.stdout)
        sys.stderr = TeeOutput(LOG_FILE_PATH, sys.stderr)
    
    # Set global GPU ID
    GLOBAL_GPU_ID = args.gpu_id
    print(f"🎮 Using GPU device: {GLOBAL_GPU_ID}")
    
    # Show filter info
    if args.dataset_name:
        print(f"🔍 Filtering by dataset: '{args.dataset_name}'")
    
    batch_evaluate(
        json_file=args.json_file,
        llm_model=args.llm_model,
        output_dir=args.output_dir,
        limit=args.limit,
        dataset_filter=args.dataset_name,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()