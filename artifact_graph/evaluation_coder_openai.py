#!/usr/bin/env python3
"""
OpenAI Agents SDK-based EvaluationCoder implementation.

Uses openai-agents Runner with native function calling.
Tool logic is shared via evaluation_coder_tools.py;
this file only adds the @function_tool decorator and agent wiring.

Four modes:
- oneturn_onetool: Single turn with only run_code_in_docker (max_turns=1)
- multiturn_onetool: Multi-turn with only run_code_in_docker
- multiturn_metadatatool: Multi-turn with metadata tools
- multiturn_cachefiletool: Multi-turn with all tools including cached loaders
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from agents import Agent, Runner, function_tool, WebSearchTool

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


# ============== OpenAI Agents SDK @function_tool Wrappers ==============

@function_tool
def run_code_in_docker(code: str, output_dir: str = "/tmp/eval_workspace") -> str:
    """
    Execute Python code inside a Docker container with GPU support.
    The code will be saved to 'run_eval.py' and executed with 'python run_eval.py'.

    IMPORTANT: The 'code' parameter must be actual Python source code, NOT a shell command.

    Args:
        code: The actual Python source code to execute (NOT a shell command)
        output_dir: Directory to store scripts and results

    Returns:
        JSON string with 'success', 'exit_code', 'output', and 'results' keys
    """
    return json.dumps(tool_run_code_in_docker(code=code, output_dir=output_dir))


@function_tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        The file contents as a string
    """
    return tool_read_file(file_path=file_path)


@function_tool
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


@function_tool
def get_model_readme(model_id: str) -> str:
    """
    Fetch the README/model card content from a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")

    Returns:
        The model's README content or error message
    """
    return tool_get_model_readme(model_id=model_id)


@function_tool
def get_dataset_readme(dataset_id: str) -> str:
    """
    Fetch the README/dataset card content from a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID

    Returns:
        The dataset's README content or error message
    """
    return tool_get_dataset_readme(dataset_id=dataset_id)


@function_tool
def get_model_metadata(model_id: str) -> str:
    """
    Fetch metadata about a HuggingFace model.

    Args:
        model_id: HuggingFace model ID

    Returns:
        JSON string with model metadata (pipeline_tag, library_name, config, etc.)
    """
    return json.dumps(tool_get_model_metadata(model_id=model_id))


@function_tool
def get_dataset_metadata(dataset_id: str) -> str:
    """
    Fetch metadata about a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID

    Returns:
        JSON string with dataset metadata (splits, features, num_examples, etc.)
    """
    return json.dumps(tool_get_dataset_metadata(dataset_id=dataset_id))


@function_tool
def check_code_legitimacy(code: str) -> str:
    """
    Check if evaluation code is legitimate (actually runs model inference) or cheating.

    Args:
        code: Python source code to analyze

    Returns:
        JSON string with 'is_cheating', 'confidence', and 'reason' fields
    """
    return json.dumps(tool_check_code_legitimacy(code=code))


def _create_dataset_loader_tool(loaders_dir: Path):
    """Factory: create @function_tool for dataset loader with specific loaders directory."""
    raw_fn = make_dataset_loader_fn(loaders_dir)

    @function_tool
    def get_dataset_loader(dataset_id: str) -> str:
        """
        Get the pre-verified dataset loading script for a specific dataset.

        Args:
            dataset_id: HuggingFace dataset ID

        Returns:
            JSON string with 'found', 'script', and 'spec' fields
        """
        return json.dumps(raw_fn(dataset_id))

    return get_dataset_loader


def _create_model_loader_tool(loaders_dir: Path):
    """Factory: create @function_tool for model loader with specific loaders directory."""
    raw_fn = make_model_loader_fn(loaders_dir)

    @function_tool
    def get_model_loader(model_id: str) -> str:
        """
        Get the pre-verified model loading script for a specific model.

        Args:
            model_id: HuggingFace model ID

        Returns:
            JSON string with 'found', 'script', 'inference_type', 'num_labels', 'id2label'
        """
        return json.dumps(raw_fn(model_id))

    return get_model_loader


# ============== OpenAI Evaluation Coder ==============

class OpenAIEvaluationCoder:
    """
    Evaluation Coder using OpenAI Agents SDK (native function calling).

    Shares the same four modes and interface as EvaluationCoder (smolagents).
    Use this for native OpenAI model support without LiteLLM compatibility layers.
    """

    MODE_CONFIG = {
        CoderMode.ONETURN_ONETOOL: {
            "max_turns": 1,
            "tools": ["run_code_in_docker"],
        },
        CoderMode.MULTITURN_ONETOOL: {
            "max_turns": 10,
            "tools": ["run_code_in_docker"],
        },
        CoderMode.MULTITURN_METADATATOOL: {
            "max_turns": 10,
            "tools": ["run_code_in_docker", "read_file", "save_file",
                      "get_model_readme", "get_dataset_readme",
                      "get_model_metadata", "get_dataset_metadata",
                      "web_search"],
        },
        CoderMode.MULTITURN_CACHEFILETOOL: {
            "max_turns": 10,
            "tools": ["run_code_in_docker", "read_file", "save_file",
                      "get_model_readme", "get_dataset_readme",
                      "get_model_metadata", "get_dataset_metadata",
                      "check_code_legitimacy", "get_dataset_loader", "get_model_loader",
                      "web_search"],
        },
    }

    TOOL_MAP = {
        "run_code_in_docker": run_code_in_docker,
        "read_file": read_file,
        "save_file": save_file,
        "get_model_readme": get_model_readme,
        "get_dataset_readme": get_dataset_readme,
        "get_model_metadata": get_model_metadata,
        "get_dataset_metadata": get_dataset_metadata,
        "check_code_legitimacy": check_code_legitimacy,
        "web_search": WebSearchTool(),
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Set global GPU ID for docker tool
        _set_gpu_id(gpu_id)

        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        config = self.MODE_CONFIG[mode]
        self.max_turns = max_steps or config["max_turns"]

        # Build tool list
        tools = []
        tool_names = config["tools"]

        for tool_name in tool_names:
            if tool_name == "get_dataset_loader":
                loaders_dir = (Path(dataset_loaders_dir) if dataset_loaders_dir
                               else Path(__file__).parent.parent / "scripts" / "dataset_loaders")
                tools.append(_create_dataset_loader_tool(loaders_dir))
            elif tool_name == "get_model_loader":
                loaders_dir = (Path(model_loaders_dir) if model_loaders_dir
                               else Path(__file__).parent.parent / "scripts" / "model_loaders")
                tools.append(_create_model_loader_tool(loaders_dir))
            else:
                tools.append(self.TOOL_MAP[tool_name])

        self.tools = tools

        print(f"🔧 OpenAIEvaluationCoder initialized:")
        print(f"   Mode: {mode.value}")
        print(f"   LLM: {llm_model}")
        print(f"   GPU: {gpu_id}")
        print(f"   Max turns: {self.max_turns}")
        print(f"   Tools: {tool_names}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on mode."""
        base = "You are an expert ML engineer that evaluates HuggingFace models on datasets."

        if self.mode == CoderMode.ONETURN_ONETOOL:
            base += " You have ONE attempt only. Write correct evaluation code on the first try."
        elif self.mode == CoderMode.MULTITURN_ONETOOL:
            base += " If code execution fails, analyze the error, fix the code and retry."
        elif self.mode == CoderMode.MULTITURN_METADATATOOL:
            base += (" Use metadata tools (get_model_readme, get_dataset_readme, "
                     "get_model_metadata, get_dataset_metadata) first to understand "
                     "the model and dataset format before writing code. "
                     "If metadata is insufficient, use web_search to find usage examples. "
                     "If code fails, analyze the error and retry.")
        elif self.mode == CoderMode.MULTITURN_CACHEFILETOOL:
            base += (" ALWAYS start by calling get_dataset_loader and get_model_loader "
                     "to get pre-verified loading scripts. Combine them into evaluation code. "
                     "If not found, use metadata tools to understand the format. "
                     "If code fails, analyze the error and retry.")
        return base

    def _build_user_prompt(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int,
    ) -> str:
        """Build the per-evaluation user prompt."""
        if max_samples > 0:
            sample_limit = f"- For large datasets, randomly sample up to {max_samples} examples"
        else:
            sample_limit = "- Use ALL samples in the dataset (no limit)"

        return f"""Evaluate model `{model_name}` on dataset `{dataset_name}` using metric `{metric}`.

TASK:
1. Write a Python evaluation script that loads the model, runs inference on the dataset,
   and computes the {metric} metric.
2. Call run_code_in_docker(code, output_dir="{output_dir}") to execute the script.
3. The script MUST save results to 'results.json' in the format: {{"{metric}": <value>}}
4. Also save predictions to 'predictions.json' with input, prediction, and ground truth.

REQUIREMENTS:
- Use GPU: model.to("cuda"), move inputs to GPU
- Use batched inference for speed
{sample_limit}
- Save results to results.json
- Save predictions to predictions.json
"""

    def evaluate(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int = 200,
    ) -> Dict[str, Any]:
        """
        Run evaluation using OpenAI Agents SDK.

        Returns:
            Dict with evaluation results (same interface as EvaluationCoder)
        """
        return asyncio.run(self._evaluate_async(
            model_name=model_name,
            dataset_name=dataset_name,
            metric=metric,
            output_dir=output_dir,
            max_samples=max_samples,
        ))

    async def _evaluate_async(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int,
    ) -> Dict[str, Any]:
        """Async evaluation implementation."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🤖 [OpenAI SDK] Running evaluation: {model_name} | {dataset_name} | {metric}")
        print(f"📁 Output: {output_dir}")
        print(f"{'='*80}\n")

        agent = Agent(
            name="EvaluationCoder",
            model=self.llm_model,
            instructions=self._build_system_prompt(),
            tools=self.tools,
        )

        user_prompt = self._build_user_prompt(
            model_name=model_name,
            dataset_name=dataset_name,
            metric=metric,
            output_dir=output_dir,
            max_samples=max_samples,
        )

        result = await Runner.run(agent, user_prompt, max_turns=self.max_turns)

        final_output = result.final_output
        if isinstance(final_output, str):
            try:
                parsed = json.loads(final_output)
            except Exception:
                parsed = {"response": final_output}
        else:
            parsed = final_output if isinstance(final_output, dict) else {"response": str(final_output)}

        # Save metadata
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metric": metric,
            "mode": self.mode.value,
            "backend": "openai_agents_sdk",
            "llm_model": self.llm_model,
            "max_turns_config": self.max_turns,
            "new_items_count": len(result.new_items),
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
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"💾 Agent response saved: {response_path}")
        except Exception as e:
            print(f"⚠️ Failed to save agent response: {e}")

        # Return success based on results.json written by docker tool
        results_path = os.path.join(output_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                eval_results = json.load(f)
            return {"success": True, **eval_results}

        return parsed

    @classmethod
    def from_mode_string(cls, mode_str: str, **kwargs) -> "OpenAIEvaluationCoder":
        """Create OpenAIEvaluationCoder from mode string."""
        mode_map = {m.value: m for m in CoderMode}
        if mode_str not in mode_map:
            raise ValueError(f"Invalid mode: {mode_str}. Valid modes: {list(mode_map.keys())}")
        return cls(mode=mode_map[mode_str], **kwargs)
