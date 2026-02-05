#!/usr/bin/env python3
"""
预先为每个目标数据集生成正确的加载脚本。
在 Docker 中验证可以下载和加载，然后保存到文件夹中。
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

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


from smolagents import CodeAgent, LiteLLMModel, tool

# ============== Global Configuration ==============
GLOBAL_GPU_ID = 0
DATASET_LOADERS_DIR = "dataset_loaders"  # 保存数据集加载脚本的目录
CURRENT_OUTPUT_DIR = "/tmp/dataset_test"  # 当前输出目录（由脚本控制，不由 agent 控制）
DEFAULT_INPUT_JSON = "perfect_model_dataset_metrics_v3_0120_coding_agent.json"


def load_datasets_from_json(json_path: str, sort_by_popularity: bool = True) -> list:
    """Load unique dataset IDs from a JSON file, sorted by number of associated models.
    
    Expected JSON format:
    {
        "results": [
            {"model_id": "...", "dataset_id": "...", "metrics": {...}},
            ...
        ]
    }
    
    Args:
        json_path: Path to JSON file
        sort_by_popularity: If True, sort datasets by number of associated models (descending)
    
    Returns:
        List of dataset IDs, sorted by popularity (most models first) if sort_by_popularity=True
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Count models per dataset
    dataset_model_count = {}
    for item in data.get('results', []):
        dataset_id = item.get('dataset_id', '')
        if dataset_id:
            dataset_model_count[dataset_id] = dataset_model_count.get(dataset_id, 0) + 1
    
    if sort_by_popularity:
        # Sort by model count (descending), then alphabetically
        sorted_datasets = sorted(
            dataset_model_count.keys(),
            key=lambda d: (-dataset_model_count[d], d)
        )
        print(f"Datasets sorted by popularity (top 10):")
        for d in sorted_datasets[:10]:
            print(f"  {dataset_model_count[d]:4d} models - {d}")
        return sorted_datasets
    else:
        return sorted(list(dataset_model_count.keys()))


@tool
def run_code_in_docker(code: str) -> dict:
    """
    Execute Python code inside a Docker container to test dataset loading.
    The output directory is automatically set by the system.
    
    Args:
        code: The Python source code to execute
    
    Returns:
        A dict with 'success', 'exit_code', 'output', and 'results' keys
    """
    output_dir = CURRENT_OUTPUT_DIR  # 使用全局变量，不由 agent 控制
    import time
    
    gpu_id = GLOBAL_GPU_ID
    timeout = 90  # 5 minutes for dataset loading
    memory_limit = "16g"
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "load_dataset.py")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"\n{'='*60}")
    print(f"🐳 Testing dataset loading in Docker")
    print(f"📁 Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    hf_token = os.getenv("HF_TOKEN", "")
    
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{output_dir}:/workspace",
        "-w", "/workspace",
        "-m", memory_limit,
        "--gpus", f"device={gpu_id}",
        "-e", f"HF_TOKEN={hf_token}",
        "-e", "PYTHONPATH=/workspace",
        "simple-coder:latest",
        "bash", "-c",
        "python load_dataset.py"
    ]
    
    output_lines = []
    
    try:
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        start_time = time.time()
        for line in iter(process.stdout.readline, ''):
            if line:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] {line.rstrip()}")
                output_lines.append(line)
            
            if time.time() - start_time > timeout:
                process.kill()
                return {"success": False, "output": "Timeout", "results": {}}
        
        process.stdout.close()
        exit_code = process.wait()
        
        output = "".join(output_lines)
        
        # Check for results.json
        results_path = os.path.join(output_dir, "results.json")
        results = {}
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
        
        return {
            "success": exit_code == 0 and results.get("load_success", False),
            "exit_code": exit_code,
            "output": output[-3000:],
            "results": results
        }
    except Exception as e:
        return {"success": False, "exit_code": -1, "output": str(e), "results": {}}


def create_dataset_agent(model_id: str = "gpt-4o", max_steps: int = 8):
    """Create agent for generating dataset loading scripts."""
    model = LiteLLMModel(
        model_id=model_id,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = CodeAgent(
        tools=[run_code_in_docker],
        model=model,
        add_base_tools=True,
        max_steps=max_steps,
        verbosity_level=2,
    )
    
    return agent


def generate_dataset_loader(agent, dataset_id: str, output_dir: str):
    """Generate and validate a dataset loading script."""
    global CURRENT_OUTPUT_DIR
    
    # 设置全局输出目录（agent 无法修改）
    CURRENT_OUTPUT_DIR = output_dir
    
    prompt = f"""
You are an expert ML engineer. Generate a Python script that correctly loads the dataset `{dataset_id}` from HuggingFace.

REQUIREMENTS:
1. The script MUST define a reusable function `load_dataset()` that:
   - Takes no arguments (dataset info is hardcoded in the function)
   - Returns a dictionary with keys:
     - "dataset": the loaded dataset (HuggingFace Dataset or list of examples)
     - "premise_column": name of the premise column
     - "hypothesis_column": name of the hypothesis column  
     - "label_column": name of the label column
     - "label_mapping": dict mapping label values to standard names
     - "num_examples": total number of examples
   - Handles any special cases (filtering invalid labels, downloading zip files, etc.)

2. Example function signature:
   ```python
   def load_dataset():
       '''Load {dataset_id} dataset for NLI evaluation.'''
       # ... loading logic ...
       return {{
           "dataset": dataset,
           "premise_column": "premise",
           "hypothesis_column": "hypothesis",
           "label_column": "label",
           "label_mapping": {{0: "entailment", 1: "neutral", 2: "contradiction"}},
           "num_examples": len(dataset)
       }}
   ```

3. After defining the function, call it in a `if __name__ == "__main__":` block to test:
   - Print dataset info: number of examples, column names, first example
   - Save a JSON file 'results.json' with:
     {{"load_success": true, "num_examples": <int>, "columns": [...], "first_example": {{...}}}}

4. For datasets with multiple configs (like bAbI-NLI with 20 configs):
   - Load ALL configs and combine them in the function
   - Report total examples across all configs

5. Use `run_code_in_docker(code)` to test the script. The output directory is automatically set.

6. If the script fails, fix it and retry until it works.

IMPORTANT RULES:
- DO NOT create wrapper scripts that write to another file and then run it.
- DO NOT use textwrap.dedent, runpy, or similar tricks.
- The script you provide should be a DIRECT, STANDALONE Python file with `def load_dataset():` defined at the top level.
- The script should NOT create any additional .py files.

Return the final working script content.
"""
    
    print(f"\n{'='*80}")
    print(f"📦 Generating loader for: {dataset_id}")
    print(f"📁 Output dir: {output_dir}")
    print(f"{'='*80}")
    
    result = agent.run(prompt)
    
    # Save the final script
    script_path = os.path.join(output_dir, "load_dataset.py")
    if os.path.exists(script_path):
        print(f"✅ Dataset loader saved: {script_path}")
    else:
        print(f"❌ Failed to generate loader for {dataset_id}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset loading scripts")
    parser.add_argument("--output-dir", default="dataset_loaders", help="Output directory")
    parser.add_argument("--input-json", default=None, help="Input JSON file with model/dataset list")
    parser.add_argument("--llm-model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--dataset", default=None, help="Specific dataset to process (default: all)")
    parser.add_argument("--gpu-id", type=int, default=8, help="GPU device ID")
    parser.add_argument("--max-steps", type=int, default=3, help="Max agent steps")
    args = parser.parse_args()
    
    global GLOBAL_GPU_ID
    GLOBAL_GPU_ID = args.gpu_id
    
    script_dir = Path(__file__).parent
    output_base = script_dir / args.output_dir
    output_base.mkdir(exist_ok=True)
    
    # Load datasets from JSON file
    input_json = args.input_json or (script_dir / DEFAULT_INPUT_JSON)
    if not os.path.exists(input_json):
        print(f"❌ Input JSON file not found: {input_json}")
        sys.exit(1)
    
    all_datasets = load_datasets_from_json(str(input_json))
    print(f"📋 Loaded {len(all_datasets)} unique datasets from {input_json}")
    
    agent = create_dataset_agent(model_id=args.llm_model, max_steps=args.max_steps)
    
    # Filter datasets if specified
    if args.dataset:
        datasets_to_process = [d for d in all_datasets if args.dataset in d]
    else:
        datasets_to_process = all_datasets
    
    results = {}
    
    for dataset_id in datasets_to_process:
        # Create safe directory name
        safe_name = dataset_id.replace("/", "_")
        dataset_output_dir = output_base / safe_name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Check if already completed (results.json exists)
        results_path = dataset_output_dir / "results.json"
        if results_path.exists():
            print(f"⏭️  Skipping {dataset_id} - results.json already exists")
            results[dataset_id] = {
                "success": True, 
                "output_dir": str(dataset_output_dir),
                "skipped": True
            }
            continue
        
        # Generate loader
        try:
            result = generate_dataset_loader(
                agent, dataset_id, str(dataset_output_dir)
            )
            results[dataset_id] = {"success": True, "output_dir": str(dataset_output_dir)}
        except Exception as e:
            results[dataset_id] = {"success": False, "error": str(e)}
            print(f"❌ Error processing {dataset_id}: {e}")
    
    # Save summary
    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"📊 Summary saved to: {summary_path}")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results.values() if r.get("success"))
    print(f"✅ Success: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
