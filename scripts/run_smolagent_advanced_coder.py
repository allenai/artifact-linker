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
                    "results": {},
                    "has_predictions": False
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
        
        # Check for predictions.json
        predictions_path = os.path.join(output_dir, "predictions.json")
        has_predictions = os.path.exists(predictions_path)
        if has_predictions:
            # Get prediction count without loading full file (could be large)
            try:
                with open(predictions_path, "r") as f:
                    predictions_data = json.load(f)
                pred_count = len(predictions_data) if isinstance(predictions_data, list) else "N/A"
                print(f"✅ Predictions saved: {predictions_path} ({pred_count} samples)")
            except:
                print(f"✅ Predictions saved: {predictions_path}")
        
        return {
            "success": exit_code == 0 and bool(results),
            "exit_code": exit_code,
            "output": output[-2000:],  # Truncate long outputs
            "results": results,
            "has_predictions": has_predictions
        }
    except Exception as e:
        print(f"\n❌ Docker execution failed: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "output": str(e),
            "results": {},
            "has_predictions": False
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


@tool
def check_code_legitimacy(code: str) -> dict:
    """
    Check if evaluation code is legitimate (actually runs model inference) or cheating 
    (hardcoding results without running the model).
    
    Use this tool BEFORE running code to verify it's doing real evaluation.
    
    A LEGITIMATE script should:
    - Load a model using transformers, torch, or similar ML libraries
    - Load a dataset
    - Actually run inference/predictions on the data
    - Compute accuracy from the predictions
    
    A CHEATING script typically:
    - Hardcodes accuracy values directly (e.g., `acc = 0.9361`)
    - Does not load any model
    - Does not run any inference
    - Just writes pre-determined results to a file
    
    Args:
        code: Python source code to analyze
    
    Returns:
        Dict with 'is_cheating', 'confidence', and 'reason' fields
    """
    from openai import OpenAI
    
    client = OpenAI()
    
    system_prompt = """You are an expert code reviewer. Your task is to analyze Python evaluation scripts and determine if they are legitimately running a machine learning model for evaluation, or if they are "cheating" by hardcoding/fabricating results.

A LEGITIMATE script should:
- Load a model using transformers, torch, or similar ML libraries
- Load a dataset (e.g., using datasets library)
- Actually run inference/predictions on the data
- Compute accuracy from the predictions

A CHEATING script typically:
- Hardcodes accuracy values directly (e.g., `acc = 0.9361`)
- Does not load any model
- Does not run any inference
- Just writes pre-determined results to a file

Respond with a JSON object with the following fields:
- "is_cheating": boolean (true if cheating, false if legitimate)
- "confidence": float between 0 and 1
- "reason": brief explanation (1-2 sentences)
"""

    user_prompt = f"""Analyze this evaluation code and determine if it's legitimately running model evaluation or cheating:

```python
{code}
```

Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Log the check result
        status = "🚨 CHEATING DETECTED" if result.get("is_cheating") else "✅ Code looks legitimate"
        print(f"\n{'='*60}")
        print(f"🔍 Code Legitimacy Check: {status}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Reason: {result.get('reason', 'N/A')}")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        return {
            "is_cheating": None,
            "confidence": 0,
            "reason": f"Error during analysis: {str(e)}",
            "error": str(e)
        }


@tool
def get_model_readme(model_id: str) -> str:
    """
    Fetch the README/model card content from a HuggingFace model.
    
    This helps understand how to use the model, what tasks it supports,
    and example code snippets.
    
    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased", "facebook/bart-large-cnn")
    
    Returns:
        The model's README content (model card) or error message
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        
        api = HfApi()
        
        # Try to get model card (README.md)
        try:
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Truncate if too long
            if len(content) > 15000:
                content = content[:15000] + "\n\n... [TRUNCATED - README too long] ..."
            
            return content
        except Exception as e:
            # Try to get model info instead
            model_info = api.model_info(model_id)
            return f"No README found. Model info:\n- Tags: {model_info.tags}\n- Pipeline tag: {model_info.pipeline_tag}\n- Library: {model_info.library_name}"
            
    except Exception as e:
        return f"Error fetching model README: {str(e)}"


@tool
def get_dataset_readme(dataset_id: str) -> str:
    """
    Fetch the README/dataset card content from a HuggingFace dataset.
    
    This helps understand the dataset structure, columns, splits,
    and how to load/use it.
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad", "glue", "SetFit/sst2")
    
    Returns:
        The dataset's README content (dataset card) or error message
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        
        api = HfApi()
        
        # Try to get dataset card (README.md)
        try:
            readme_path = hf_hub_download(repo_id=dataset_id, filename="README.md", repo_type="dataset")
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Truncate if too long
            if len(content) > 15000:
                content = content[:15000] + "\n\n... [TRUNCATED - README too long] ..."
            
            return content
        except Exception as e:
            # Try to get dataset info instead
            dataset_info = api.dataset_info(dataset_id)
            return f"No README found. Dataset info:\n- Tags: {dataset_info.tags}\n- Description: {dataset_info.description or 'N/A'}"
            
    except Exception as e:
        return f"Error fetching dataset README: {str(e)}"


@tool
def get_model_metadata(model_id: str) -> dict:
    """
    Fetch metadata about a HuggingFace model.
    
    Returns structured information including:
    - Pipeline tag (task type)
    - Library name (transformers, diffusers, etc.)
    - Tags
    - Config (model architecture details)
    
    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")
    
    Returns:
        Dict with model metadata
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
            "downloads": model_info.downloads,
            "likes": model_info.likes,
        }
        
        # Try to get config.json for architecture details
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            # Extract key config info
            metadata["config"] = {
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "num_labels": config.get("num_labels"),
                "vocab_size": config.get("vocab_size"),
                "hidden_size": config.get("hidden_size"),
                "task_specific_params": config.get("task_specific_params"),
            }
        except:
            metadata["config"] = None
        
        return metadata
        
    except Exception as e:
        return {"error": f"Error fetching model metadata: {str(e)}"}


@tool
def get_dataset_metadata(dataset_id: str) -> dict:
    """
    Fetch metadata about a HuggingFace dataset.
    
    Returns structured information including:
    - Available splits (train, test, validation)
    - Features/columns
    - Number of examples
    - Tags
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad", "glue")
    
    Returns:
        Dict with dataset metadata
    """
    try:
        from huggingface_hub import HfApi
        from datasets import load_dataset_builder
        
        api = HfApi()
        
        # Get basic info from Hub
        try:
            dataset_info = api.dataset_info(dataset_id)
            hub_info = {
                "dataset_id": dataset_id,
                "tags": dataset_info.tags,
                "downloads": dataset_info.downloads,
                "likes": dataset_info.likes,
            }
        except:
            hub_info = {"dataset_id": dataset_id}
        
        # Get detailed info from datasets library
        try:
            builder = load_dataset_builder(dataset_id)
            info = builder.info
            
            hub_info["description"] = info.description[:500] if info.description else None
            hub_info["features"] = str(info.features) if info.features else None
            hub_info["splits"] = {name: {"num_examples": split.num_examples} 
                                  for name, split in info.splits.items()} if info.splits else None
            hub_info["config_name"] = builder.config.name if hasattr(builder, 'config') else None
            
            # List available configs
            try:
                configs = builder.builder_configs
                if configs:
                    hub_info["available_configs"] = [c.name for c in configs.values()][:10]
            except:
                pass
                
        except Exception as e:
            hub_info["dataset_builder_error"] = str(e)
        
        return hub_info
        
    except Exception as e:
        return {"error": f"Error fetching dataset metadata: {str(e)}"}


# ============== Pre-generated Loader Tools ==============

SCRIPT_DIR = Path(__file__).parent
DATASET_LOADERS_DIR = SCRIPT_DIR / "dataset_loaders"
MODEL_LOADERS_DIR = SCRIPT_DIR / "model_loaders"


@tool
def get_dataset_loader(dataset_id: str) -> dict:
    """
    Get the pre-verified dataset loading script for a specific dataset.
    
    These scripts have been tested in Docker and are guaranteed to work correctly.
    Use this FIRST before writing any dataset loading code.
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "stanfordnlp/snli", "facebook/anli")
    
    Returns:
        Dict with:
        - "found": bool - whether a pre-verified loader exists
        - "script": str - the Python code for loading the dataset (if found)
        - "spec": dict - the dataset specification (split, config, columns, etc.)
    """
    safe_name = dataset_id.replace("/", "_")
    loader_dir = DATASET_LOADERS_DIR / safe_name
    
    result = {"found": False, "dataset_id": dataset_id}
    
    # Check for spec
    spec_path = loader_dir / "spec.json"
    if spec_path.exists():
        with open(spec_path, "r") as f:
            result["spec"] = json.load(f)
    
    # Check for loader script
    script_path = loader_dir / "load_dataset.py"
    if script_path.exists():
        with open(script_path, "r") as f:
            result["script"] = f.read()
        result["found"] = True
        print(f"✅ Found pre-verified loader for dataset: {dataset_id}")
    else:
        print(f"⚠️ No pre-verified loader found for dataset: {dataset_id}")
        # Return spec if available so agent knows the correct format
        if "spec" in result:
            print(f"   Using spec: split={result['spec'].get('split')}, config={result['spec'].get('config')}")
    
    return result


@tool
def get_model_loader(model_id: str) -> dict:
    """
    Get the pre-verified model loading script for a specific model.
    
    These scripts have been tested in Docker and are guaranteed to work correctly.
    Use this FIRST before writing any model loading code.
    
    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/deberta-base-mnli")
    
    Returns:
        Dict with:
        - "found": bool - whether a pre-verified loader exists
        - "script": str - the Python code for loading the model (if found)
        - "inference_type": "classification" or "generation"
        - "num_labels": int - number of output labels
        - "id2label": dict - mapping from label ids to label names
    """
    safe_name = model_id.replace("/", "_")
    loader_dir = MODEL_LOADERS_DIR / safe_name
    
    result = {"found": False, "model_id": model_id}
    
    # Check for results.json (contains model info)
    results_path = loader_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            model_results = json.load(f)
        result["inference_type"] = model_results.get("inference_type")
        result["num_labels"] = model_results.get("num_labels")
        result["id2label"] = model_results.get("id2label")
    
    # Check for loader script
    script_path = loader_dir / "load_model.py"
    if script_path.exists():
        with open(script_path, "r") as f:
            result["script"] = f.read()
        result["found"] = True
        print(f"✅ Found pre-verified loader for model: {model_id}")
        print(f"   Inference type: {result.get('inference_type')}, num_labels: {result.get('num_labels')}")
    else:
        print(f"⚠️ No pre-verified loader found for model: {model_id}")
    
    return result


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
    
    # Core tools + Pre-verified loaders + HuggingFace metadata tools + code legitimacy checker
    tools = [
        run_code_in_docker,
        read_file,
        save_file,
        get_dataset_loader,  # Pre-verified dataset loading scripts
        get_model_loader,    # Pre-verified model loading scripts
        check_code_legitimacy,
        get_model_readme,
        get_dataset_readme,
        get_model_metadata,
        get_dataset_metadata,
    ]
    print(f"🔧 Tools: run_code_in_docker, read_file, save_file, get_dataset_loader, get_model_loader, check_code_legitimacy, get_model_readme, get_dataset_readme, get_model_metadata, get_dataset_metadata + base tools")
    
    agent = CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=True,
        max_steps=max_steps,
        verbosity_level=2,  # 0=silent, 1=basic, 2=detailed (shows each step)
    )
    
    return agent


def evaluate_model(
    agent,
    model_name: str,
    dataset_name: str,
    metric: str = "accuracy",
    output_dir: str = "results",
    max_samples: int = 200
):
    """Run evaluation using the smolagents agent.
    
    Args:
        agent: The evaluation agent
        model_name: HuggingFace model ID
        dataset_name: HuggingFace dataset ID
        metric: Metric name to evaluate
        output_dir: Output directory for results
        max_samples: Maximum number of samples to evaluate (for large datasets), -1 means no limit
    """
    
    # Build sample limit instruction based on max_samples
    if max_samples > 0:
        sample_limit_instruction = f"- For large datasets with >{max_samples} examples, randomly sample up to {max_samples} examples"
    else:
        sample_limit_instruction = "- Use ALL samples in the dataset (no sampling limit)"
    
    prompt = f"""
You are an expert ML engineer. Your task is to evaluate the model `{model_name}` 
on the dataset `{dataset_name}` using the metric `{metric}`.

AVAILABLE TOOLS (in order of priority):
- `get_dataset_loader(dataset_id)` - ⭐ IMPORTANT: Get pre-verified dataset loading script (use this FIRST!)
- `get_model_loader(model_id)` - ⭐ IMPORTANT: Get pre-verified model loading script (use this FIRST!)
- `run_code_in_docker(code, output_dir)` - Execute Python code in Docker with GPU
- `check_code_legitimacy(code)` - Check if code actually runs model inference (NOT cheating)
- `get_model_metadata(model_id)` - Get model config, pipeline tag, architecture info
- `get_dataset_metadata(dataset_id)` - Get dataset splits, features, num examples
- `read_file(path)` / `save_file(path, content)` - File I/O

RECOMMENDED WORKFLOW:
1. ⭐ FIRST, use `get_dataset_loader("{dataset_name}")` to get the pre-verified dataset loading code.
2. ⭐ THEN, use `get_model_loader("{model_name}")` to get the pre-verified model loading code.
3. If pre-verified loaders are found, combine them into an evaluation script.
4. If NOT found, fall back to `get_model_metadata` and `get_dataset_metadata` to understand the format, then write your own loading code.
5. Write the full evaluation script that:
   - Loads the dataset using the verified method
   - Loads the model using the verified method
   - Runs batched inference on GPU
   - Computes accuracy
   - Saves results to 'results.json' in format: {{"{metric}": <value>}}
   - **IMPORTANT**: Also saves predictions to 'predictions.json' (see format below)
6. Use `run_code_in_docker(code, output_dir="{output_dir}")` to execute the script.
7. If it fails, analyze the error, fix the code and retry.


REQUIREMENTS:
- Use GPU: model.to("cuda"), inputs to GPU
- Use batched inference for speed
{sample_limit_instruction}
- Save results to results.json
- **Save predictions to predictions.json**: A list of dicts, each with at least:
  - "idx": sample index
  - "prediction": model's predicted label (string or int)
  - "ground_truth": actual label from dataset (string or int)
  - Optionally include "input_text" or other relevant fields
  Example format:
  ```json
  [
    {{"idx": 0, "prediction": "entailment", "ground_truth": "entailment"}},
    {{"idx": 1, "prediction": "neutral", "ground_truth": "contradiction"}},
    ...
  ]
  ```
- Make sure you load the model and do the evaluation instead of simply printing the results you collect.

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


def check_loader_prerequisites(model_id: str, dataset_id: str) -> Tuple[bool, str]:
    """
    Check if pre-verified loaders exist and were successful for both model and dataset.
    
    Args:
        model_id: HuggingFace model ID
        dataset_id: HuggingFace dataset ID
    
    Returns:
        Tuple of (success: bool, message: str)
        - success=True if both loaders exist and load_success=True
        - success=False with reason if either loader is missing or failed
    """
    issues = []
    
    # Check dataset loader
    dataset_safe_name = dataset_id.replace("/", "_")
    dataset_loader_dir = DATASET_LOADERS_DIR / dataset_safe_name
    dataset_results_path = dataset_loader_dir / "results.json"
    dataset_script_path = dataset_loader_dir / "load_dataset.py"
    
    if not dataset_script_path.exists():
        issues.append(f"Dataset loader script not found: {dataset_script_path}")
    elif dataset_results_path.exists():
        try:
            with open(dataset_results_path, "r") as f:
                dataset_results = json.load(f)
            if not dataset_results.get("load_success", False):
                issues.append(f"Dataset loader failed: load_success=False")
        except Exception as e:
            issues.append(f"Dataset results.json unreadable: {e}")
    else:
        issues.append(f"Dataset results.json not found: {dataset_results_path}")
    
    # Check model loader
    model_safe_name = model_id.replace("/", "_")
    model_loader_dir = MODEL_LOADERS_DIR / model_safe_name
    model_results_path = model_loader_dir / "results.json"
    model_script_path = model_loader_dir / "load_model.py"
    
    if not model_script_path.exists():
        issues.append(f"Model loader script not found: {model_script_path}")
    elif model_results_path.exists():
        try:
            with open(model_results_path, "r") as f:
                model_results = json.load(f)
            if not model_results.get("load_success", False):
                issues.append(f"Model loader failed: load_success=False")
        except Exception as e:
            issues.append(f"Model results.json unreadable: {e}")
    else:
        issues.append(f"Model results.json not found: {model_results_path}")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Both loaders verified successfully"


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
    output_dir: str,
    max_samples: int = 200
) -> Tuple[bool, str]:
    """
    Run a single model evaluation.
    
    Args:
        agent: The evaluation agent
        model: Model ID
        dataset: Dataset ID
        metric: Metric name
        output_dir: Output directory
        max_samples: Maximum number of samples to evaluate
    
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
                output_dir=output_dir,
                max_samples=max_samples
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
    max_steps: int = 10,
    max_samples: int = 200
):
    """Batch evaluate multiple model/dataset/metric combinations.
    
    Args:
        json_file: Path to JSON file with evaluation configurations
        llm_model: LLM model identifier
        output_dir: Output directory for results
        limit: Maximum number of triples to evaluate (0 = no limit)
        dataset_filter: Filter to match dataset names
        max_steps: Maximum CodeAct iteration steps
        max_samples: Maximum number of samples to evaluate per dataset
    """
    
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
        
        # Check if pre-verified loaders exist and were successful
        loaders_ok, loader_message = check_loader_prerequisites(model, dataset)
        if not loaders_ok:
            print(f"⏭️  Skipping - Loader prerequisites not met: {loader_message}")
            summary.append({
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "success": False,
                "message": f"Skipped. {loader_message}",
                "output_dir": None
            })
            continue
        
        print(f"✅ Loader prerequisites verified")
        
        # Run evaluation
        success, message = run_single_evaluation(agent, model, dataset, metric, out_dir, max_samples)
        
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
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Maximum number of samples to evaluate per dataset (default: 200, -1 for no limit)")
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
    
    # Show max samples info
    if args.max_samples > 0:
        print(f"📊 Max samples per dataset: {args.max_samples}")
    else:
        print(f"📊 Max samples per dataset: No limit (use all samples)")
    
    batch_evaluate(
        json_file=args.json_file,
        llm_model=args.llm_model,
        output_dir=args.output_dir,
        limit=args.limit,
        dataset_filter=args.dataset_name,
        max_steps=args.max_steps,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()