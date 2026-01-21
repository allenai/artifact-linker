#!/usr/bin/env python3
"""
预先为每个目标模型生成正确的加载脚本。
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
MODEL_LOADERS_DIR = "model_loaders"  # 保存模型加载脚本的目录
CURRENT_OUTPUT_DIR = "/tmp/model_test"  # 当前输出目录（由脚本控制，不由 agent 控制）


# ============== NLI Model List ==============
NLI_MODELS = [
    # Zero-shot NLI models (classification head)
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "MoritzLaurer/roberta-large-zeroshot-v2.0-c",
    "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    
    # Standard NLI models (classification head)
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-base-mnli",
    "valhalla/distilbart-mnli-12-6",
    "joeddav/bart-large-mnli-yahoo-answers",
    "typeform/distilbert-base-uncased-mnli",
    
    # Modern models
    "tasksource/ModernBERT-large-nli",
    "tasksource/ModernBERT-base-nli",
    
    # Sentence embedding models
    "Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    
    # Generative models (need prompting)
    "google/gemma-2-2b",
    "google/gemma-2b",
    "google/gemma-3-1b-it",
    "EleutherAI/gpt-neo-1.3B",
]


@tool
def run_code_in_docker(code: str) -> dict:
    """
    Execute Python code inside a Docker container to test model loading.
    The output directory is automatically set by the system.
    
    Args:
        code: The Python source code to execute
    
    Returns:
        A dict with 'success', 'exit_code', 'output', and 'results' keys
    """
    import time
    
    gpu_id = GLOBAL_GPU_ID
    output_dir = CURRENT_OUTPUT_DIR  # 使用全局变量，不由 agent 控制
    timeout = 600  # 10 minutes for model downloading
    memory_limit = "32g"
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "load_model.py")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"\n{'='*60}")
    print(f"🐳 Testing model loading in Docker")
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
        "python load_model.py"
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


@tool
def get_model_metadata(model_id: str) -> dict:
    """
    Get metadata about a HuggingFace model.
    
    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/deberta-base-mnli")
    
    Returns:
        Dict with model metadata including pipeline_tag, config, id2label, etc.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
        
        api = HfApi()
        model_info = api.model_info(model_id)
        
        metadata = {
            "model_id": model_id,
            "pipeline_tag": model_info.pipeline_tag,
            "library_name": model_info.library_name,
            "tags": model_info.tags,
        }
        
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            metadata["config"] = {
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "num_labels": config.get("num_labels"),
                "id2label": config.get("id2label"),
                "label2id": config.get("label2id"),
            }
        except:
            metadata["config"] = None
        
        return metadata
    except Exception as e:
        return {"error": str(e)}


def create_model_agent(model_id: str = "gpt-4o", max_steps: int = 8):
    """Create agent for generating model loading scripts."""
    model = LiteLLMModel(
        model_id=model_id,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = CodeAgent(
        tools=[run_code_in_docker, get_model_metadata],
        model=model,
        add_base_tools=True,
        max_steps=max_steps,
        verbosity_level=2,
    )
    
    return agent


def generate_model_loader(agent, model_id: str, output_dir: str):
    """Generate and validate a model loading script."""
    global CURRENT_OUTPUT_DIR
    
    # 设置全局输出目录（agent 无法修改）
    CURRENT_OUTPUT_DIR = output_dir
    
    prompt = f"""
You are an expert ML engineer. Generate a Python script that correctly loads the model `{model_id}` for NLI evaluation.

REQUIREMENTS:
1. First, use `get_model_metadata("{model_id}")` to understand the model type and configuration.

2. The script MUST define a reusable function `load_model()` that:
   - Takes no arguments (model info is hardcoded in the function)
   - Returns a dictionary with keys:
     - "model": the loaded model (on CUDA, in eval mode)
     - "tokenizer": the loaded tokenizer
     - "model_type": string describing model architecture
     - "inference_type": "classification" or "generation"
     - "num_labels": number of output labels (for classification)
     - "id2label": dict mapping label indices to names
     - "label2id": dict mapping label names to indices
   - Handles all model-specific setup (special tokens, padding, etc.)

3. Example function signature:
   ```python
   def load_model():
       '''Load {model_id} model for NLI evaluation.'''
       # ... loading logic ...
       model = model.cuda().eval()
       return {{
           "model": model,
           "tokenizer": tokenizer,
           "model_type": "deberta-v3",
           "inference_type": "classification",
           "num_labels": 3,
           "id2label": {{0: "entailment", 1: "neutral", 2: "contradiction"}},
           "label2id": {{"entailment": 0, "neutral": 1, "contradiction": 2}}
       }}
   ```

4. After defining the function, call it in a `if __name__ == "__main__":` block to test:
   - Print model info: model type, num_labels, id2label mapping
   - Do a simple forward pass with a test example to verify it works
   - Save a JSON file 'results.json' with:
     {{
       "load_success": true, 
       "model_type": "...",
       "num_labels": ...,
       "id2label": {{...}},
       "inference_type": "classification" or "generation"
     }}

5. Determine the correct inference type:
   - If the model has a classification head (e.g., AutoModelForSequenceClassification), use "classification"
   - If the model is a generative model (e.g., AutoModelForCausalLM), use "generation"

6. Use `run_code_in_docker(code)` to test the script. The output directory is automatically set.

7. If the script fails, fix it and retry until it works.

Return the final working script content.
"""
    
    print(f"\n{'='*80}")
    print(f"🤖 Generating loader for: {model_id}")
    print(f"📁 Output dir: {output_dir}")
    print(f"{'='*80}")
    
    result = agent.run(prompt)
    
    # Save the final script
    script_path = os.path.join(output_dir, "load_model.py")
    if os.path.exists(script_path):
        print(f"✅ Model loader saved: {script_path}")
    else:
        print(f"❌ Failed to generate loader for {model_id}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate model loading scripts")
    parser.add_argument("--output-dir", default="model_loaders", help="Output directory")
    parser.add_argument("--llm-model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--model", default=None, help="Specific model to process (default: all)")
    parser.add_argument("--gpu-id", type=int, default=7, help="GPU device ID")
    parser.add_argument("--max-steps", type=int, default=8, help="Max agent steps")
    args = parser.parse_args()
    
    global GLOBAL_GPU_ID
    GLOBAL_GPU_ID = args.gpu_id
    
    script_dir = Path(__file__).parent
    output_base = script_dir / args.output_dir
    output_base.mkdir(exist_ok=True)
    
    agent = create_model_agent(model_id=args.llm_model, max_steps=args.max_steps)
    
    # Filter models if specified
    if args.model:
        models_to_process = [m for m in NLI_MODELS if args.model in m]
    else:
        models_to_process = NLI_MODELS
    
    results = {}
    
    for model_id in models_to_process:
        # Create safe directory name
        safe_name = model_id.replace("/", "_")
        model_output_dir = output_base / safe_name
        model_output_dir.mkdir(exist_ok=True)
        
        # Check if already successfully generated
        results_path = model_output_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    existing_results = json.load(f)
                if existing_results.get("load_success", False):
                    print(f"⏭️  Skipping {model_id} - already successfully generated")
                    results[model_id] = {
                        "success": True,
                        "output_dir": str(model_output_dir),
                        "inference_type": existing_results.get("inference_type"),
                        "num_labels": existing_results.get("num_labels"),
                        "skipped": True
                    }
                    continue
            except Exception as e:
                print(f"⚠️ Failed to read existing results for {model_id}: {e}")
        
        # Generate loader
        try:
            result = generate_model_loader(agent, model_id, str(model_output_dir))
            
            # Check if results.json exists and has load_success
            if results_path.exists():
                with open(results_path, "r") as f:
                    loader_results = json.load(f)
                results[model_id] = {
                    "success": loader_results.get("load_success", False),
                    "output_dir": str(model_output_dir),
                    "inference_type": loader_results.get("inference_type"),
                    "num_labels": loader_results.get("num_labels"),
                }
            else:
                results[model_id] = {"success": False, "error": "No results.json"}
                
        except Exception as e:
            results[model_id] = {"success": False, "error": str(e)}
            print(f"❌ Error processing {model_id}: {e}")
    
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
