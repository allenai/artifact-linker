#!/usr/bin/env python3
"""
smolagents-based EvaluationCoder implementation.

Uses smolagents CodeAgent with LiteLLMModel.
Tool logic is shared via evaluation_coder_tools.py;
this file only adds the smolagents @tool decorator and agent wiring.

Four modes:
- oneturn_onetool: Single turn with only run_code_in_docker (max_steps=1)
- multiturn_onetool: Multi-turn with only run_code_in_docker
- multiturn_metadatatool: Multi-turn with metadata tools + base_tools
- multiturn_cachefiletool: Multi-turn with all tools including cached loaders
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import litellm
litellm.drop_params = True
os.environ["LITELLM_DROP_PARAMS"] = "true"

# Monkey-patch litellm.completion to fix bugs with newer OpenAI models:
# 1. Drop 'stop' parameter (some models reject it)
# 2. Fix metadata=None bug in Responses API handler
#    (`"model_group" in kwargs.get("metadata", None)` raises TypeError when metadata is None)
# 3. Drop 'temperature' (not supported by gpt-5.x-codex, o1, o3, etc.)
_original_completion = litellm.completion
def _patched_completion(*args, **kwargs):
    if 'stop' in kwargs:
        del kwargs['stop']
    if kwargs.get('metadata') is None:
        kwargs['metadata'] = {}
    if 'temperature' in kwargs:
        del kwargs['temperature']
    return _original_completion(*args, **kwargs)
litellm.completion = _patched_completion

from smolagents import CodeAgent, LiteLLMModel, tool

# Import shared tool logic and types
from artifact_graph.evaluation_coder_tools import (
    CoderMode,
    _set_gpu_id,
    tool_run_code_in_docker,
    tool_read_file,
    tool_save_file,
    tool_get_model_readme,
    tool_get_dataset_readme,
    tool_get_model_metadata,
    tool_get_dataset_metadata,
    tool_check_code_legitimacy,
    make_dataset_loader_fn,
    make_model_loader_fn,
)


# ============== smolagents @tool Wrappers ==============

@tool
def run_code_in_docker(code: str, output_dir: str = "/tmp/eval_workspace") -> dict:
    """
    Execute Python code inside a Docker container with GPU support.
    The code will be saved to 'run_eval.py' and executed with 'python run_eval.py'.

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
    return tool_run_code_in_docker(code=code, output_dir=output_dir)


@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        The file contents as a string
    """
    return tool_read_file(file_path=file_path)


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
    return tool_save_file(file_path=file_path, content=content)


@tool
def get_model_readme(model_id: str) -> str:
    """
    Fetch the README/model card content from a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")

    Returns:
        The model's README content or error message
    """
    return tool_get_model_readme(model_id=model_id)


@tool
def get_dataset_readme(dataset_id: str) -> str:
    """
    Fetch the README/dataset card content from a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad", "SetFit/sst2")

    Returns:
        The dataset's README content or error message
    """
    return tool_get_dataset_readme(dataset_id=dataset_id)


@tool
def get_model_metadata(model_id: str) -> dict:
    """
    Fetch metadata about a HuggingFace model.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dict with model metadata (pipeline_tag, library_name, config, etc.)
    """
    return tool_get_model_metadata(model_id=model_id)


@tool
def get_dataset_metadata(dataset_id: str) -> dict:
    """
    Fetch metadata about a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID

    Returns:
        Dict with dataset metadata (splits, features, num_examples, etc.)
    """
    return tool_get_dataset_metadata(dataset_id=dataset_id)


@tool
def check_code_legitimacy(code: str) -> dict:
    """
    Check if evaluation code is legitimate (actually runs model inference) or cheating.

    Args:
        code: Python source code to analyze

    Returns:
        Dict with 'is_cheating', 'confidence', and 'reason' fields
    """
    return tool_check_code_legitimacy(code=code)


def _create_get_dataset_loader(loaders_dir: Path):
    """Factory: create smolagents @tool for dataset loader with specific loaders directory."""
    raw_fn = make_dataset_loader_fn(loaders_dir)

    @tool
    def get_dataset_loader(dataset_id: str) -> dict:
        """
        Get the pre-verified dataset loading script for a specific dataset.

        Args:
            dataset_id: HuggingFace dataset ID

        Returns:
            Dict with 'found', 'script', and 'spec' fields
        """
        return raw_fn(dataset_id)

    return get_dataset_loader


def _create_get_model_loader(loaders_dir: Path):
    """Factory: create smolagents @tool for model loader with specific loaders directory."""
    raw_fn = make_model_loader_fn(loaders_dir)

    @tool
    def get_model_loader(model_id: str) -> dict:
        """
        Get the pre-verified model loading script for a specific model.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Dict with 'found', 'script', 'inference_type', 'num_labels', 'id2label'
        """
        return raw_fn(model_id)

    return get_model_loader


# ============== Main EvaluationCoder Class ==============

class EvaluationCoder:
    """
    Evaluation Coder using smolagents CodeAgent + LiteLLMModel.

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

    # Tool name to smolagents tool object mapping
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
        self.mode = mode
        self.llm_model = llm_model
        self.gpu_id = gpu_id
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Set global GPU ID for docker tool
        _set_gpu_id(gpu_id)

        config = self.MODE_CONFIG[mode]
        self.max_steps = max_steps or config["max_steps"]
        self.add_base_tools = config["add_base_tools"]

        # Build tool list
        self.tools = []
        tool_names = config["tools"]

        for tool_name in tool_names:
            if tool_name == "get_dataset_loader":
                loaders_dir = (Path(dataset_loaders_dir) if dataset_loaders_dir
                               else Path(__file__).parent.parent / "scripts" / "dataset_loaders")
                self.tools.append(_create_get_dataset_loader(loaders_dir))
            elif tool_name == "get_model_loader":
                loaders_dir = (Path(model_loaders_dir) if model_loaders_dir
                               else Path(__file__).parent.parent / "scripts" / "model_loaders")
                self.tools.append(_create_get_model_loader(loaders_dir))
            else:
                self.tools.append(self.TOOL_MAP[tool_name])

        # Create smolagents agent
        self.agent = self._create_agent()

        print(f"🔧 EvaluationCoder (smolagents) initialized:")
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
        Run evaluation using the smolagents CodeAgent.

        Returns:
            Dict with evaluation results
        """
        prompt = self._build_prompt(model_name, dataset_name, metric, output_dir, max_samples)

        print(f"\n{'='*80}")
        print(f"🤖 Running evaluation: {model_name} | {dataset_name} | {metric}")
        print(f"📁 Output: {output_dir}")
        print(f"{'='*80}\n")

        result = self.agent.run(prompt)

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

            tool_calls: Dict[str, int] = {}
            for step in steps:
                if isinstance(step, dict):
                    for tc in step.get("tool_calls", []):
                        tool_name = (tc.get("name") or tc.get("tool_name")) if isinstance(tc, dict) \
                            else (getattr(tc, "name", None) or getattr(tc, "tool_name", None))
                        if tool_name:
                            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
                    if step.get("code_action"):
                        tool_calls["code_execution"] = tool_calls.get("code_execution", 0) + 1
        except Exception as e:
            print(f"⚠️ Failed to get step info: {e}")
            num_steps = self.agent.step_number
            steps = []
            tool_calls = {}

        # Per-step details
        step_details = []
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                step_info = {
                    "step": i + 1,
                    "has_code_action": bool(step.get("code_action")),
                    "has_error": bool(step.get("error")),
                    "is_final_answer": step.get("is_final_answer", False),
                }
                step_token = step.get("token_usage")
                if step_token:
                    step_info["token_usage"] = (
                        step_token if isinstance(step_token, dict) else step_token.dict()
                    )
                step_details.append(step_info)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metric": metric,
            "mode": self.mode.value,
            "backend": "smolagents",
            "llm_model": self.llm_model,
            "max_steps_config": self.max_steps,
            "actual_steps": num_steps,
            "step_number": self.agent.step_number,
            "token_usage": token_dict,
            "tool_calls": tool_calls,
            "tool_calls_total": sum(tool_calls.values()) if tool_calls else 0,
            "step_details": step_details,
        }
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
        if max_samples > 0:
            sample_limit = (f"- For large datasets with >{max_samples} examples, "
                            f"randomly sample up to {max_samples} examples")
        else:
            sample_limit = "- Use ALL samples in the dataset (no sampling limit)"

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

        if self.mode == CoderMode.ONETURN_ONETOOL:
            base_prompt += "\nNOTE: You have only ONE attempt. Make sure your code is correct on the first try.\n"
        elif self.mode == CoderMode.MULTITURN_ONETOOL:
            base_prompt += "\nIf the script fails, analyze the error, fix the code and retry.\n"
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

        base_prompt += "\nReturn the final evaluation results.\n"
        return base_prompt

    @classmethod
    def from_mode_string(cls, mode_str: str, **kwargs) -> "EvaluationCoder":
        """Create EvaluationCoder from mode string."""
        mode_map = {m.value: m for m in CoderMode}
        if mode_str not in mode_map:
            raise ValueError(f"Invalid mode: {mode_str}. Valid modes: {list(mode_map.keys())}")
        return cls(mode=mode_map[mode_str], **kwargs)
