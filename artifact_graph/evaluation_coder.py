#!/usr/bin/env python3
"""
Core Evaluation Coder class supporting multiple tool configurations.

Four modes:
- oneturn_onetool: Single turn with only run_code_in_docker (max_steps=1)
- multiturn_onetool: Multi-turn with only run_code_in_docker  
- multiturn_metadatatool: Multi-turn with run/read/save + metadata tools + base_tools
- multiturn_cachefiletool: Multi-turn with all tools including cached loaders
"""

import os
import json
import subprocess
import time
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

import litellm
litellm.drop_params = True
os.environ["LITELLM_DROP_PARAMS"] = "true"

# Monkey-patch litellm.completion to drop 'stop' parameter
_original_completion = litellm.completion
def _patched_completion(*args, **kwargs):
    if 'stop' in kwargs:
        del kwargs['stop']
    return _original_completion(*args, **kwargs)
litellm.completion = _patched_completion

from smolagents import CodeAgent, LiteLLMModel, tool


class CoderMode(Enum):
    """Evaluation coder modes with different tool configurations."""
    ONETURN_ONETOOL = "oneturn_onetool"           # Single turn, only docker run
    MULTITURN_ONETOOL = "multiturn_onetool"       # Multi-turn, only docker run
    MULTITURN_METADATATOOL = "multiturn_metadatatool"   # Multi-turn, + metadata tools + base_tools
    MULTITURN_CACHEFILETOOL = "multiturn_cachefiletool" # Multi-turn, + cached loaders


# ============== Tool Definitions ==============

# Global GPU ID (will be set by EvaluationCoder)
_GLOBAL_GPU_ID = 0


def _set_gpu_id(gpu_id: int):
    """Set the global GPU ID for docker execution."""
    global _GLOBAL_GPU_ID
    _GLOBAL_GPU_ID = gpu_id


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
    gpu_id = _GLOBAL_GPU_ID
    timeout = 900  # 15 minutes
    memory_limit = "32g"
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "run_eval.py")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"\n{'='*60}")
    print(f"🐳 Docker execution started")
    print(f"📁 Output dir: {output_dir}")
    print(f"⏱️  Timeout: {timeout}s")
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
        "python run_eval.py"
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
                print(f"\n❌ Docker execution timed out after {timeout}s")
                return {
                    "success": False,
                    "exit_code": -1,
                    "output": f"Execution timed out after {timeout}s",
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
        
        results_path = os.path.join(output_dir, "results.json")
        results = {}
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            print(f"✅ Results: {results}")
        
        predictions_path = os.path.join(output_dir, "predictions.json")
        has_predictions = os.path.exists(predictions_path)
        if has_predictions:
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
            "output": output[-2000:],
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
def get_model_readme(model_id: str) -> str:
    """
    Fetch the README/model card content from a HuggingFace model.
    
    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")
    
    Returns:
        The model's README content or error message
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        
        api = HfApi()
        try:
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 15000:
                content = content[:15000] + "\n\n... [TRUNCATED - README too long] ..."
            return content
        except Exception:
            model_info = api.model_info(model_id)
            return f"No README found. Model info:\n- Tags: {model_info.tags}\n- Pipeline tag: {model_info.pipeline_tag}\n- Library: {model_info.library_name}"
    except Exception as e:
        return f"Error fetching model README: {str(e)}"


@tool
def get_dataset_readme(dataset_id: str) -> str:
    """
    Fetch the README/dataset card content from a HuggingFace dataset.
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad", "SetFit/sst2")
    
    Returns:
        The dataset's README content or error message
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        
        api = HfApi()
        try:
            readme_path = hf_hub_download(repo_id=dataset_id, filename="README.md", repo_type="dataset")
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 15000:
                content = content[:15000] + "\n\n... [TRUNCATED - README too long] ..."
            return content
        except Exception:
            dataset_info = api.dataset_info(dataset_id)
            return f"No README found. Dataset info:\n- Tags: {dataset_info.tags}\n- Description: {dataset_info.description or 'N/A'}"
    except Exception as e:
        return f"Error fetching dataset README: {str(e)}"


@tool
def get_model_metadata(model_id: str) -> dict:
    """
    Fetch metadata about a HuggingFace model.
    
    Args:
        model_id: HuggingFace model ID
    
    Returns:
        Dict with model metadata (pipeline_tag, library_name, config, etc.)
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
        
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
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
    
    Args:
        dataset_id: HuggingFace dataset ID
    
    Returns:
        Dict with dataset metadata (splits, features, num_examples, etc.)
    """
    try:
        from huggingface_hub import HfApi
        from datasets import load_dataset_builder
        
        api = HfApi()
        
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
        
        try:
            builder = load_dataset_builder(dataset_id)
            info = builder.info
            
            hub_info["description"] = info.description[:500] if info.description else None
            hub_info["features"] = str(info.features) if info.features else None
            hub_info["splits"] = {name: {"num_examples": split.num_examples} 
                                  for name, split in info.splits.items()} if info.splits else None
            hub_info["config_name"] = builder.config.name if hasattr(builder, 'config') else None
            
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


@tool
def check_code_legitimacy(code: str) -> dict:
    """
    Check if evaluation code is legitimate (actually runs model inference) or cheating.
    
    Args:
        code: Python source code to analyze
    
    Returns:
        Dict with 'is_cheating', 'confidence', and 'reason' fields
    """
    from openai import OpenAI
    
    client = OpenAI()
    
    system_prompt = """You are an expert code reviewer. Analyze Python evaluation scripts and determine if they legitimately run ML model evaluation or cheat by hardcoding results.

LEGITIMATE: Loads model, loads dataset, runs inference, computes metrics from predictions.
CHEATING: Hardcodes values (e.g., acc = 0.9361), no model/inference, just writes predetermined results.

Respond with JSON: {"is_cheating": bool, "confidence": float, "reason": "brief explanation"}"""

    user_prompt = f"""Analyze this evaluation code:

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
        status = "🚨 CHEATING DETECTED" if result.get("is_cheating") else "✅ Code looks legitimate"
        print(f"\n{'='*60}")
        print(f"🔍 Code Legitimacy Check: {status}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Reason: {result.get('reason', 'N/A')}")
        print(f"{'='*60}\n")
        
        return result
    except Exception as e:
        return {"is_cheating": None, "confidence": 0, "reason": f"Error: {str(e)}", "error": str(e)}


# ============== Cached Loader Tools ==============

def _create_get_dataset_loader(loaders_dir: Path):
    """Factory function to create get_dataset_loader tool with custom loaders directory."""
    
    @tool
    def get_dataset_loader(dataset_id: str) -> dict:
        """
        Get the pre-verified dataset loading script for a specific dataset.
        
        Args:
            dataset_id: HuggingFace dataset ID
        
        Returns:
            Dict with 'found', 'script', and 'spec' fields
        """
        safe_name = dataset_id.replace("/", "_")
        loader_dir = loaders_dir / safe_name
        
        result = {"found": False, "dataset_id": dataset_id}
        
        spec_path = loader_dir / "spec.json"
        if spec_path.exists():
            with open(spec_path, "r") as f:
                result["spec"] = json.load(f)
        
        script_path = loader_dir / "load_dataset.py"
        if script_path.exists():
            with open(script_path, "r") as f:
                result["script"] = f.read()
            result["found"] = True
            print(f"✅ Found pre-verified loader for dataset: {dataset_id}")
        else:
            print(f"⚠️ No pre-verified loader found for dataset: {dataset_id}")
            if "spec" in result:
                print(f"   Using spec: split={result['spec'].get('split')}, config={result['spec'].get('config')}")
        
        return result
    
    return get_dataset_loader


def _create_get_model_loader(loaders_dir: Path):
    """Factory function to create get_model_loader tool with custom loaders directory."""
    
    @tool
    def get_model_loader(model_id: str) -> dict:
        """
        Get the pre-verified model loading script for a specific model.
        
        Args:
            model_id: HuggingFace model ID
        
        Returns:
            Dict with 'found', 'script', 'inference_type', 'num_labels', 'id2label'
        """
        safe_name = model_id.replace("/", "_")
        loader_dir = loaders_dir / safe_name
        
        result = {"found": False, "model_id": model_id}
        
        results_path = loader_dir / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                model_results = json.load(f)
            result["inference_type"] = model_results.get("inference_type")
            result["num_labels"] = model_results.get("num_labels")
            result["id2label"] = model_results.get("id2label")
        
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
    
    return get_model_loader


# ============== Main EvaluationCoder Class ==============

class EvaluationCoder:
    """
    Evaluation Coder supporting multiple tool configurations.
    
    Four modes:
    - oneturn_onetool: Single turn with only run_code_in_docker (max_steps=1)
    - multiturn_onetool: Multi-turn with only run_code_in_docker
    - multiturn_metadatatool: Multi-turn with metadata tools + base_tools
    - multiturn_cachefiletool: Multi-turn with all tools including cached loaders
    """
    
    MODE_CONFIG = {
        CoderMode.ONETURN_ONETOOL: {
            "max_steps": 1,
            "tools": ["run_code_in_docker"],
            "add_base_tools": False,
        },
        CoderMode.MULTITURN_ONETOOL: {
            "max_steps": 10,
            "tools": ["run_code_in_docker"],
            "add_base_tools": False,
        },
        CoderMode.MULTITURN_METADATATOOL: {
            "max_steps": 10,
            "tools": ["run_code_in_docker", "read_file", "save_file", 
                     "get_model_readme", "get_dataset_readme",
                     "get_model_metadata", "get_dataset_metadata"],
            "add_base_tools": True,
        },
        CoderMode.MULTITURN_CACHEFILETOOL: {
            "max_steps": 10,
            "tools": ["run_code_in_docker", "read_file", "save_file",
                     "get_model_readme", "get_dataset_readme",
                     "get_model_metadata", "get_dataset_metadata",
                     "check_code_legitimacy", "get_dataset_loader", "get_model_loader"],
            "add_base_tools": True,
        },
    }
    
    # Tool name to function mapping
    TOOL_MAP = {
        "run_code_in_docker": run_code_in_docker,
        "read_file": read_file,
        "save_file": save_file,
        "get_model_readme": get_model_readme,
        "get_dataset_readme": get_dataset_readme,
        "get_model_metadata": get_model_metadata,
        "get_dataset_metadata": get_dataset_metadata,
        "check_code_legitimacy": check_code_legitimacy,
    }
    
    def __init__(
        self,
        mode: CoderMode,
        llm_model: str = "gpt-4o",
        gpu_id: int = 0,
        max_steps: Optional[int] = None,
        dataset_loaders_dir: Optional[str] = None,
        model_loaders_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
    ):
        """
        Initialize EvaluationCoder.
        
        Args:
            mode: One of CoderMode enum values
            llm_model: LLM model identifier (default: gpt-4o)
            gpu_id: GPU device ID for docker execution
            max_steps: Override max_steps for the mode (optional)
            dataset_loaders_dir: Directory with pre-verified dataset loaders
            model_loaders_dir: Directory with pre-verified model loaders
            api_key: API key for LLM
            temperature: LLM temperature
        """
        self.mode = mode
        self.llm_model = llm_model
        self.gpu_id = gpu_id
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Set global GPU ID
        _set_gpu_id(gpu_id)
        
        # Get mode configuration
        config = self.MODE_CONFIG[mode]
        self.max_steps = max_steps or config["max_steps"]
        self.add_base_tools = config["add_base_tools"]
        
        # Build tool list
        self.tools = []
        tool_names = config["tools"]
        
        for tool_name in tool_names:
            if tool_name == "get_dataset_loader":
                loaders_dir = Path(dataset_loaders_dir) if dataset_loaders_dir else Path(__file__).parent.parent / "scripts" / "dataset_loaders"
                self.tools.append(_create_get_dataset_loader(loaders_dir))
            elif tool_name == "get_model_loader":
                loaders_dir = Path(model_loaders_dir) if model_loaders_dir else Path(__file__).parent.parent / "scripts" / "model_loaders"
                self.tools.append(_create_get_model_loader(loaders_dir))
            else:
                self.tools.append(self.TOOL_MAP[tool_name])
        
        # Create agent
        self.agent = self._create_agent()
        
        print(f"🔧 EvaluationCoder initialized:")
        print(f"   Mode: {mode.value}")
        print(f"   LLM: {llm_model}")
        print(f"   GPU: {gpu_id}")
        print(f"   Max steps: {self.max_steps}")
        print(f"   Tools: {tool_names}")
        print(f"   Base tools: {self.add_base_tools}")
    
    def _create_agent(self) -> CodeAgent:
        """Create the smolagents CodeAgent."""
        model = LiteLLMModel(
            model_id=self.llm_model,
            temperature=self.temperature,
            api_key=self.api_key,
        )
        
        return CodeAgent(
            tools=self.tools,
            model=model,
            add_base_tools=self.add_base_tools,
            max_steps=self.max_steps,
            verbosity_level=2,
        )
    
    def evaluate(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int = 200,
    ) -> Dict[str, Any]:
        """
        Run evaluation using the configured agent.
        
        Args:
            model_name: HuggingFace model ID
            dataset_name: HuggingFace dataset ID
            metric: Metric name to evaluate
            output_dir: Output directory for results
            max_samples: Max samples to evaluate (-1 for no limit)
        
        Returns:
            Dict with evaluation results
        """
        # Build prompt based on mode
        prompt = self._build_prompt(model_name, dataset_name, metric, output_dir, max_samples)
        
        print(f"\n{'='*80}")
        print(f"🤖 Running evaluation: {model_name} | {dataset_name} | {metric}")
        print(f"📁 Output: {output_dir}")
        print(f"{'='*80}\n")
        
        result = self.agent.run(prompt)
        
        # Collect metadata
        os.makedirs(output_dir, exist_ok=True)
        
        # Get token usage
        try:
            token_usage = self.agent.monitor.get_total_token_counts()
            token_dict = token_usage.dict() if hasattr(token_usage, 'dict') else {}
        except Exception as e:
            print(f"⚠️ Failed to get token usage: {e}")
            token_dict = {}
        
        # Get step information
        try:
            steps = self.agent.memory.get_full_steps()
            num_steps = len(steps)
            
            # Count tool calls from each step
            tool_calls = {}
            for step in steps:
                if isinstance(step, dict):
                    # Check for tool_calls list in the step
                    step_tool_calls = step.get("tool_calls", [])
                    if step_tool_calls:
                        for tc in step_tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get("name") or tc.get("tool_name")
                            else:
                                # Could be a ToolCall object
                                tool_name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
                            if tool_name:
                                tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
                    
                    # Also check code_action for CodeAgent
                    if step.get("code_action"):
                        tool_calls["code_execution"] = tool_calls.get("code_execution", 0) + 1
        except Exception as e:
            print(f"⚠️ Failed to get step info: {e}")
            num_steps = self.agent.step_number
            steps = []
            tool_calls = {}
        
        # Calculate per-step token usage
        step_details = []
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                step_info = {
                    "step": i + 1,
                    "has_code_action": bool(step.get("code_action")),
                    "has_error": bool(step.get("error")),
                    "is_final_answer": step.get("is_final_answer", False),
                }
                # Get token usage for this step
                step_token = step.get("token_usage")
                if step_token:
                    if isinstance(step_token, dict):
                        step_info["token_usage"] = step_token
                    elif hasattr(step_token, "dict"):
                        step_info["token_usage"] = step_token.dict()
                step_details.append(step_info)
        
        # Build metadata
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metric": metric,
            "mode": self.mode.value,
            "llm_model": self.llm_model,
            "max_steps_config": self.max_steps,
            "actual_steps": num_steps,
            "step_number": self.agent.step_number,
            "token_usage": token_dict,
            "tool_calls": tool_calls,
            "tool_calls_total": sum(tool_calls.values()) if tool_calls else 0,
            "step_details": step_details,
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"📊 Metadata saved: {metadata_path}")
        except Exception as e:
            print(f"⚠️ Failed to save metadata: {e}")
        
        # Save agent response
        response_path = os.path.join(output_dir, "agent_response.json")
        try:
            response_data = result if isinstance(result, dict) else {"response": str(result)}
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Agent response saved: {response_path}")
        except Exception as e:
            print(f"⚠️ Failed to save agent response: {e}")
        
        return result
    
    def _build_prompt(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int,
    ) -> str:
        """Build the evaluation prompt based on mode."""
        
        # Sample limit instruction
        if max_samples > 0:
            sample_limit = f"- For large datasets with >{max_samples} examples, randomly sample up to {max_samples} examples"
        else:
            sample_limit = "- Use ALL samples in the dataset (no sampling limit)"
        
        # Base prompt
        base_prompt = f"""
You are an expert ML engineer. Your task is to evaluate the model `{model_name}` 
on the dataset `{dataset_name}` using the metric `{metric}`.

TASK:
1. Write a Python evaluation script that loads the model, evaluates it on the dataset, 
   and computes the {metric} metric.
2. Call `run_code_in_docker(code, output_dir="{output_dir}")` to execute the script.
   IMPORTANT: `run_code_in_docker` is a built-in tool - call it directly, do NOT import it!
3. The script MUST save results to 'results.json' in the format: {{"{metric}": <value>}}
4. Also save predictions to 'predictions.json'. The file needs to include the input data, the output prediction, and the ground-truth results.

REQUIREMENTS:
- Use GPU: model.to("cuda"), inputs to GPU
- Use batched inference for speed
{sample_limit}
- Save results to results.json
- Save predictions to predictions.json
- Do NOT use `from run_code_in_docker import ...` - all tools are already available!
"""
        
        # Add mode-specific instructions
        if self.mode == CoderMode.ONETURN_ONETOOL:
            base_prompt += """
NOTE: You have only ONE attempt. Make sure your code is correct on the first try.
"""
        elif self.mode == CoderMode.MULTITURN_ONETOOL:
            base_prompt += """
If the script fails, analyze the error, fix the code and retry.
"""
        elif self.mode == CoderMode.MULTITURN_METADATATOOL:
            base_prompt += f"""
AVAILABLE TOOLS:
- `run_code_in_docker(code, output_dir)` - Execute Python code in Docker with GPU
- `read_file(path)` / `save_file(path, content)` - File I/O
- `get_model_readme(model_id)` - Get model's README/model card
- `get_dataset_readme(dataset_id)` - Get dataset's README/dataset card
- `get_model_metadata(model_id)` - Get model config, pipeline tag, architecture info
- `get_dataset_metadata(dataset_id)` - Get dataset splits, features, num examples

RECOMMENDED: Use metadata tools first to understand the model and dataset format.
If the script fails, analyze the error, fix the code and retry.
"""
        elif self.mode == CoderMode.MULTITURN_CACHEFILETOOL:
            base_prompt += f"""
AVAILABLE TOOLS (in order of priority):
- `get_dataset_loader(dataset_id)` - ⭐ IMPORTANT: Get pre-verified dataset loading script (use this FIRST!)
- `get_model_loader(model_id)` - ⭐ IMPORTANT: Get pre-verified model loading script (use this FIRST!)
- `run_code_in_docker(code, output_dir)` - Execute Python code in Docker with GPU
- `check_code_legitimacy(code)` - Check if code actually runs model inference (NOT cheating)
- `get_model_readme(model_id)` / `get_dataset_readme(dataset_id)` - Get README content
- `get_model_metadata(model_id)` / `get_dataset_metadata(dataset_id)` - Get metadata
- `read_file(path)` / `save_file(path, content)` - File I/O

RECOMMENDED WORKFLOW:
1. ⭐ FIRST, use `get_dataset_loader("{dataset_name}")` to get pre-verified dataset loading code.
2. ⭐ THEN, use `get_model_loader("{model_name}")` to get pre-verified model loading code.
3. If pre-verified loaders are found, combine them into an evaluation script.
4. If NOT found, use metadata tools to understand the format, then write your own loading code.
5. Execute and iterate until successful.
"""
        
        base_prompt += """
Return the final evaluation results.
"""
        return base_prompt
    
    @classmethod
    def from_mode_string(cls, mode_str: str, **kwargs) -> "EvaluationCoder":
        """Create EvaluationCoder from mode string."""
        mode_map = {m.value: m for m in CoderMode}
        if mode_str not in mode_map:
            raise ValueError(f"Invalid mode: {mode_str}. Valid modes: {list(mode_map.keys())}")
        return cls(mode=mode_map[mode_str], **kwargs)
