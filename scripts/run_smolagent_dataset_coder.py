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


# ============== NLI Dataset Specifications ==============
# 定义每个数据集的正确加载方式
NLI_DATASET_SPECS = {
    "stanfordnlp/snli": {
        "description": "Stanford Natural Language Inference dataset",
        "split": "test",
        "config": None,
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "labels": ["entailment", "neutral", "contradiction"],
        "label_mapping": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "notes": "Filter out label=-1 (invalid samples). Use ds = ds.filter(lambda x: x['label'] != -1)",
    },
    "nyu-mll/multi_nli": {
        "description": "Multi-Genre Natural Language Inference dataset",
        "split": "validation_matched",
        "config": None,
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "labels": ["entailment", "neutral", "contradiction"],
        "label_mapping": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "notes": "No test split available. Use validation_matched for evaluation.",
    },
    "facebook/anli": {
        "description": "Adversarial NLI dataset with 3 rounds",
        "split": ["test_r1", "test_r2", "test_r3"],
        "config": "plain_text",
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "labels": ["entailment", "neutral", "contradiction"],
        "label_mapping": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "notes": "Must use config='plain_text'. Combine all 3 rounds (r1, r2, r3) for evaluation.",
    },
    "allenai/scitail": {
        "description": "Science entailment dataset",
        "split": "test",
        "config": "snli_format",
        "columns": {"premise": "sentence1", "hypothesis": "sentence2", "label": "gold_label"},
        "labels": ["entails", "neutral"],
        "label_mapping": {"entails": 0, "neutral": 1},
        "notes": "Must use config='snli_format'. Only 2 classes (entails/neutral). Column names differ from standard NLI.",
    },
    "pietrolesci/nli_fever": {
        "description": "FEVER fact verification as NLI",
        "split": "dev",
        "config": None,
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "labels": ["entailment", "neutral", "contradiction"],
        "label_mapping": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "notes": "Use 'dev' split for evaluation.",
    },
    "kiddothe2b/contract-nli": {
        "description": "Contract NLI dataset",
        "split": None,
        "config": "zip",
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "labels": ["entailment", "contradiction", "neutral"],
        "label_mapping": None,
        "notes": "Requires downloading zip file. Use hf_hub_download to get contract_nli.zip, then unzip and load as JSON.",
    },
    "tasksource/babi_nli": {
        "description": "bAbI tasks reformulated as NLI",
        "split": "test",
        "config": "ALL_20_CONFIGS",
        "available_configs": [
            'agents-motivations', 'basic-coreference', 'basic-deduction', 'basic-induction',
            'compound-coreference', 'conjunction', 'counting', 'indefinite-knowledge',
            'lists-sets', 'path-finding', 'positional-reasoning', 'simple-negation',
            'single-supporting-fact', 'size-reasoning', 'three-arg-relations',
            'three-supporting-facts', 'time-reasoning', 'two-arg-relations',
            'two-supporting-facts', 'yes-no-questions'
        ],
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        "labels": ["not_entailment", "entailment"],
        "label_mapping": {0: "not_entailment", 1: "entailment"},
        "notes": "Binary classification. Must evaluate ALL 20 configs and report average accuracy.",
    },
    "tasksource/defeasible-nli": {
        "description": "Defeasible inference dataset",
        "split": "test",
        "config": "ALL_3_CONFIGS",
        "available_configs": ["atomic", "snli", "social"],
        "columns": {"premise": "Premise", "hypothesis": "Hypothesis", "update": "Update", "label": "UpdateType"},
        "labels": ["strengthener", "weakener"],
        "label_mapping": {"strengthener": 1, "weakener": 0},
        "notes": "Must evaluate ALL 3 configs (atomic, snli, social) and report average accuracy. Labels are strengthener/weakener.",
    },
    "marzieh-saeidi/SICK": {
        "description": "SICK dataset for semantic relatedness and entailment",
        "split": "test",
        "config": None,
        "columns": {"premise": "sentence_A", "hypothesis": "sentence_B", "label": "label"},
        "labels": ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
        "label_mapping": None,
        "notes": "Column names are sentence_A, sentence_B. Labels are uppercase strings.",
    },
    "allenai/scitail": {
        "description": "Science entailment dataset",
        "split": "test",
        "config": "snli_format",
        "columns": {"premise": "sentence1", "hypothesis": "sentence2", "label": "gold_label"},
        "labels": ["entails", "neutral"],
        "label_mapping": {"entails": 0, "neutral": 1},
        "notes": "Must use config='snli_format'. Only 2 classes (entails/neutral). Column names differ from standard NLI.",
    },
}


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
    timeout = 300  # 5 minutes for dataset loading
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
        "artifact-linker-verification:latest",
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


def generate_dataset_loader(agent, dataset_id: str, spec: dict, output_dir: str):
    """Generate and validate a dataset loading script."""
    global CURRENT_OUTPUT_DIR
    
    # 设置全局输出目录（agent 无法修改）
    CURRENT_OUTPUT_DIR = output_dir
    
    # Build the prompt
    spec_json = json.dumps(spec, indent=2, ensure_ascii=False)
    
    prompt = f"""
You are an expert ML engineer. Generate a Python script that correctly loads the dataset `{dataset_id}`.

DATASET SPECIFICATION:
{spec_json}

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

Return the final working script content.


DATASET LOADING REFERENCE (ONLY if no pre-verified loader exists):
- SNLI (`stanfordnlp/snli`): split="test". Filter out label=-1 (invalid samples).
- MNLI (`nyu-mll/multi_nli`): split="validation_matched" (No test split available).
- ANLI (`facebook/anli`): Use config="plain_text". Use split="test_r1", "test_r2", and "test_r3". Combine all 3 rounds.
- SciTail (`allenai/scitail`): Use config="snli_format", split="test". Columns: 'sentence1', 'sentence2', 'gold_label'. Only 2 classes: entails/neutral.
- FEVER-NLI (`pietrolesci/nli_fever`): split="dev" only.
- ContractNLI (`kiddothe2b/contract-nli`): Download zip, unzip, load jsonl.
- bAbI-NLI (`tasksource/babi_nli`): split="test". Must evaluate ALL 20 configs and report average accuracy. Binary classification.
- Defeasible-NLI (`tasksource/defeasible-nli`): split="test". Must evaluate ALL 3 configs and report average accuracy.
- SICK (`marzieh-saeidi/SICK`): split="test". Columns: 'sentence_A', 'sentence_B'. Labels are uppercase.
- QNLI/RTE/WNLI (`SetFit/qnli` etc.): Use split="validation".

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
    parser.add_argument("--llm-model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--dataset", default=None, help="Specific dataset to process (default: all)")
    parser.add_argument("--gpu-id", type=int, default=8, help="GPU device ID")
    parser.add_argument("--max-steps", type=int, default=8, help="Max agent steps")
    args = parser.parse_args()
    
    global GLOBAL_GPU_ID
    GLOBAL_GPU_ID = args.gpu_id
    
    script_dir = Path(__file__).parent
    output_base = script_dir / args.output_dir
    output_base.mkdir(exist_ok=True)
    
    agent = create_dataset_agent(model_id=args.llm_model, max_steps=args.max_steps)
    
    # Filter datasets if specified
    if args.dataset:
        datasets_to_process = {k: v for k, v in NLI_DATASET_SPECS.items() if args.dataset in k}
    else:
        datasets_to_process = NLI_DATASET_SPECS
    
    results = {}
    
    for dataset_id, spec in datasets_to_process.items():
        # Create safe directory name
        safe_name = dataset_id.replace("/", "_")
        dataset_output_dir = output_base / safe_name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Check if already successfully generated
        results_path = dataset_output_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    existing_results = json.load(f)
                if existing_results.get("load_success", False):
                    print(f"⏭️  Skipping {dataset_id} - already successfully generated")
                    results[dataset_id] = {
                        "success": True, 
                        "output_dir": str(dataset_output_dir),
                        "skipped": True
                    }
                    continue
            except Exception as e:
                print(f"⚠️ Failed to read existing results for {dataset_id}: {e}")
        
        # Save spec
        with open(dataset_output_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
        
        # Generate loader
        try:
            result = generate_dataset_loader(
                agent, dataset_id, spec, str(dataset_output_dir)
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
