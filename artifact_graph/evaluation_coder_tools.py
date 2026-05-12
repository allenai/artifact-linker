#!/usr/bin/env python3
"""
Shared tool implementations for EvaluationCoder backends.

This module contains:
- CoderMode enum (shared by smolagent and openai backends)
- Global GPU ID management
- Raw Python tool functions (NO framework-specific decorators)
- Factory functions for cached loader tools

Both evaluation_coder_smolagent.py and evaluation_coder_openai.py import
from here and apply their own framework decorators (@tool / @function_tool).
"""

import json
import os
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Dict


# ============== Shared Enum ==============

class CoderMode(Enum):
    """Evaluation coder modes with different tool configurations."""
    ONETURN_ONETOOL = "oneturn_onetool"                 # Single turn, only docker run
    MULTITURN_ONETOOL = "multiturn_onetool"             # Multi-turn, only docker run
    MULTITURN_METADATATOOL = "multiturn_metadatatool"   # Multi-turn, + metadata tools
    MULTITURN_CACHEFILETOOL = "multiturn_cachefiletool" # Multi-turn, + cached loaders


# ============== Global GPU ID ==============

_GLOBAL_GPU_ID = 0


def _set_gpu_id(gpu_id: int):
    """Set the global GPU ID for docker execution."""
    global _GLOBAL_GPU_ID
    _GLOBAL_GPU_ID = gpu_id


# ============== Raw Tool Functions ==============

def tool_run_code_in_docker(code: str, output_dir: str = "/tmp/eval_workspace") -> dict:
    """
    Execute Python code inside a Docker container with GPU support.
    The code will be saved to 'run_eval.py' and executed with 'python run_eval.py'.

    IMPORTANT: The 'code' parameter must be actual Python source code, NOT a shell command.

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
            except Exception:
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


def tool_read_file(file_path: str) -> str:
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


def tool_save_file(file_path: str, content: str) -> str:
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


def tool_get_model_readme(model_id: str) -> str:
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
            return (f"No README found. Model info:\n"
                    f"- Tags: {model_info.tags}\n"
                    f"- Pipeline tag: {model_info.pipeline_tag}\n"
                    f"- Library: {model_info.library_name}")
    except Exception as e:
        return f"Error fetching model README: {str(e)}"


def tool_get_dataset_readme(dataset_id: str) -> str:
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
            readme_path = hf_hub_download(
                repo_id=dataset_id, filename="README.md", repo_type="dataset"
            )
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 15000:
                content = content[:15000] + "\n\n... [TRUNCATED - README too long] ..."
            return content
        except Exception:
            dataset_info = api.dataset_info(dataset_id)
            return (f"No README found. Dataset info:\n"
                    f"- Tags: {dataset_info.tags}\n"
                    f"- Description: {dataset_info.description or 'N/A'}")
    except Exception as e:
        return f"Error fetching dataset README: {str(e)}"


def tool_get_model_metadata(model_id: str) -> dict:
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
        except Exception:
            metadata["config"] = None

        return metadata
    except Exception as e:
        return {"error": f"Error fetching model metadata: {str(e)}"}


def tool_get_dataset_metadata(dataset_id: str) -> dict:
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
            hub_info: Dict = {
                "dataset_id": dataset_id,
                "tags": dataset_info.tags,
                "downloads": dataset_info.downloads,
                "likes": dataset_info.likes,
            }
        except Exception:
            hub_info = {"dataset_id": dataset_id}

        try:
            builder = load_dataset_builder(dataset_id)
            info = builder.info

            hub_info["description"] = info.description[:500] if info.description else None
            hub_info["features"] = str(info.features) if info.features else None
            hub_info["splits"] = (
                {name: {"num_examples": split.num_examples}
                 for name, split in info.splits.items()}
                if info.splits else None
            )
            hub_info["config_name"] = builder.config.name if hasattr(builder, "config") else None

            try:
                configs = builder.builder_configs
                if configs:
                    hub_info["available_configs"] = [c.name for c in configs.values()][:10]
            except Exception:
                pass
        except Exception as e:
            hub_info["dataset_builder_error"] = str(e)

        return hub_info
    except Exception as e:
        return {"error": f"Error fetching dataset metadata: {str(e)}"}


def tool_check_code_legitimacy(code: str) -> dict:
    """
    Check if evaluation code is legitimate (actually runs model inference) or cheating.

    Args:
        code: Python source code to analyze

    Returns:
        Dict with 'is_cheating', 'confidence', and 'reason' fields
    """
    from openai import OpenAI

    client = OpenAI()

    system_prompt = (
        "You are an expert code reviewer. Analyze Python evaluation scripts and determine "
        "if they legitimately run ML model evaluation or cheat by hardcoding results.\n\n"
        "LEGITIMATE: Loads model, loads dataset, runs inference, computes metrics from predictions.\n"
        "CHEATING: Hardcodes values (e.g., acc = 0.9361), no model/inference, just writes predetermined results.\n\n"
        'Respond with JSON: {"is_cheating": bool, "confidence": float, "reason": "brief explanation"}'
    )

    user_prompt = f"Analyze this evaluation code:\n\n```python\n{code}\n```\n\nRespond with JSON only."

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


# ============== Augmentation Tools ==============


def tool_get_model_labels(model_id: str) -> dict:
    """
    Fetch id2label and label2id from a HuggingFace model's config.json.

    This is critical for label mapping — model label ordering often differs
    from dataset label ordering. Always check this before evaluation.

    Args:
        model_id: HuggingFace model ID (e.g., 'borisn70/bert-43-multilabel-emotion-detection')

    Returns:
        Dict with id2label, label2id, num_labels, and model_type
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        id2label = config.get("id2label", {})
        label2id = config.get("label2id", {})
        num_labels = config.get("num_labels")
        model_type = config.get("model_type")
        architectures = config.get("architectures", [])
        task_specific_params = config.get("task_specific_params")

        return {
            "model_id": model_id,
            "id2label": id2label,
            "label2id": label2id,
            "num_labels": num_labels,
            "model_type": model_type,
            "architectures": architectures,
            "task_specific_params": task_specific_params,
            "label_names": list(id2label.values()) if id2label else [],
        }
    except Exception as e:
        return {"model_id": model_id, "error": f"Error fetching model labels: {str(e)}"}


def tool_inspect_dataset_rows(dataset_id: str, config: str = "default", split: str = "test") -> dict:
    """
    Fetch a few sample rows from a HuggingFace dataset using the Dataset Viewer API.

    This is a fast REST API call (no dataset download). Use it to inspect actual
    column names, data types, label values, and detect issues like label=-1.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., 'rajpurkar/squad_v2')
        config: Dataset config name (default: 'default')
        split: Split to inspect (default: 'test')

    Returns:
        Dict with column_names, sample_rows, features, and label analysis
    """
    import requests

    base_url = "https://datasets-server.huggingface.co"

    # First try to get available configs/splits
    result = {"dataset_id": dataset_id, "config": config, "split": split}

    try:
        # Get first rows
        params = {"dataset": dataset_id, "config": config, "split": split}
        resp = requests.get(f"{base_url}/first-rows", params=params, timeout=15)

        if resp.status_code == 404 or resp.status_code == 400:
            # Try without config or with different config
            params_no_config = {"dataset": dataset_id, "split": split}
            resp = requests.get(f"{base_url}/first-rows", params=params_no_config, timeout=15)

        if resp.status_code != 200:
            # Try to get available splits info
            info_resp = requests.get(f"{base_url}/info", params={"dataset": dataset_id}, timeout=15)
            if info_resp.status_code == 200:
                result["available_info"] = info_resp.json()
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
            return result

        data = resp.json()
        features = data.get("features", [])
        rows = data.get("rows", [])

        result["features"] = features
        result["column_names"] = [f.get("column", f.get("name", "")) for f in features]
        result["num_rows_returned"] = len(rows)

        # Extract sample rows (first 3)
        sample_rows = []
        for row_data in rows[:3]:
            row = row_data.get("row", row_data)
            sample_rows.append(row)
        result["sample_rows"] = sample_rows

        # Analyze label fields
        label_analysis = {}
        for feat in features:
            col_name = feat.get("column", feat.get("name", ""))
            dtype = feat.get("dtype", feat.get("type", ""))

            # Check if this looks like a label field
            if col_name in ("label", "labels", "ner_tags", "fine_label", "coarse_label"):
                values = [r.get(col_name) for r in sample_rows if col_name in r]
                label_analysis[col_name] = {
                    "dtype": str(dtype),
                    "sample_values": values[:5],
                }
                # Check for -1 labels
                flat_vals = []
                for v in values:
                    if isinstance(v, list):
                        flat_vals.extend(v)
                    elif isinstance(v, (int, float)):
                        flat_vals.append(v)
                if flat_vals and all(v == -1 for v in flat_vals):
                    label_analysis[col_name]["warning"] = "ALL labels are -1 — use a different split"

        if label_analysis:
            result["label_analysis"] = label_analysis

        return result

    except requests.Timeout:
        return {**result, "error": "Timeout fetching dataset rows (15s)"}
    except Exception as e:
        return {**result, "error": f"Error: {str(e)}"}


def make_search_similar_evaluations_fn(results_dir: str):
    """Factory that returns a tool function for searching past successful evaluations."""

    def search_similar_evaluations(
        task_type: str = "",
        dataset_id: str = "",
        model_id: str = "",
        metric: str = "",
    ) -> dict:
        """
        Search past successful evaluations for similar tasks.

        Finds evaluations from prior runs that match the given criteria.
        Returns the plan and code patterns that worked, enabling self-evolving learning.

        Args:
            task_type: Task type to search for (e.g., 'ner', 'nli', 'qa', 'summarization',
                       'vision', 'multilabel', 'sentiment'). Partial match supported.
            dataset_id: Dataset ID to search for (partial match, e.g., 'squad' or 'conll2003')
            model_id: Model ID to search for (partial match, e.g., 'bert' or 'roberta')
            metric: Metric to search for (e.g., 'accuracy', 'f1', 'rouge2')

        Returns:
            Dict with matched evaluations including their plans and code patterns
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return {"matches": [], "error": f"Results directory not found: {results_dir}"}

        matches = []
        search_terms = {
            "task_type": task_type.lower() if task_type else "",
            "dataset_id": dataset_id.lower() if dataset_id else "",
            "model_id": model_id.lower() if model_id else "",
            "metric": metric.lower() if metric else "",
        }

        for case_dir in sorted(results_path.iterdir()):
            if not case_dir.is_dir():
                continue

            dir_name = case_dir.name.lower()

            # Check if results.json exists (only match successful runs)
            results_file = case_dir / "results.json"
            if not results_file.exists():
                continue

            try:
                with open(results_file, "r") as f:
                    results_data = json.load(f)
                # Skip if metric is 0.0 (failed evaluation)
                metric_vals = [v for v in results_data.values() if isinstance(v, (int, float))]
                if metric_vals and all(v == 0.0 for v in metric_vals):
                    continue
            except Exception:
                continue

            # Match against search terms
            score = 0
            if search_terms["dataset_id"] and search_terms["dataset_id"] in dir_name:
                score += 2
            if search_terms["model_id"] and search_terms["model_id"] in dir_name:
                score += 2
            if search_terms["metric"] and search_terms["metric"] in dir_name:
                score += 1

            # Check metadata for task_type match
            metadata_file = case_dir / "metadata.json"
            plan_data = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    plan_str = metadata.get("plan", "{}")
                    if isinstance(plan_str, str):
                        try:
                            plan_data = json.loads(plan_str)
                        except Exception:
                            pass
                    elif isinstance(plan_str, dict):
                        plan_data = plan_str

                    if search_terms["task_type"]:
                        plan_task = str(plan_data.get("task_type", "")).lower()
                        plan_framework = str(plan_data.get("framework", "")).lower()
                        if search_terms["task_type"] in plan_task or search_terms["task_type"] in plan_framework:
                            score += 3
                except Exception:
                    pass

            if score == 0 and any(search_terms.values()):
                continue

            # Extract code pattern (first 100 lines of run_eval.py)
            code_snippet = ""
            script_path = case_dir / "run_eval.py"
            if script_path.exists():
                try:
                    with open(script_path, "r") as f:
                        lines = f.readlines()
                    code_snippet = "".join(lines[:100])
                    if len(lines) > 100:
                        code_snippet += f"\n# ... ({len(lines) - 100} more lines)"
                except Exception:
                    pass

            matches.append({
                "case": case_dir.name,
                "score": score,
                "results": results_data,
                "plan": {
                    "framework": plan_data.get("framework", ""),
                    "task_type": plan_data.get("task_type", ""),
                    "pip_deps": plan_data.get("pip_deps", []),
                    "dataset_split": plan_data.get("dataset_split", ""),
                    "model_loading_hint": plan_data.get("model_loading_hint", ""),
                    "dataset_loading_hint": plan_data.get("dataset_loading_hint", ""),
                    "label_mapping_hint": plan_data.get("label_mapping_hint", ""),
                    "evaluation_strategy": plan_data.get("evaluation_strategy", ""),
                    "known_pitfalls": plan_data.get("known_pitfalls", []),
                    "generation_params": plan_data.get("generation_params"),
                },
                "code_snippet": code_snippet,
            })

        # Sort by score, return top 3
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:3]

        return {
            "query": search_terms,
            "total_matches": len(matches),
            "top_matches": top_matches,
        }

    return search_similar_evaluations


# ============== Cached Loader Factories ==============

def make_dataset_loader_fn(loaders_dir: Path):
    """Return a raw Python function that loads pre-verified dataset scripts."""
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
                print(f"   Using spec: split={result['spec'].get('split')}, "
                      f"config={result['spec'].get('config')}")

        return result

    return get_dataset_loader


def make_model_loader_fn(loaders_dir: Path):
    """Return a raw Python function that loads pre-verified model scripts."""
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
            print(f"   Inference type: {result.get('inference_type')}, "
                  f"num_labels: {result.get('num_labels')}")
        else:
            print(f"⚠️ No pre-verified loader found for model: {model_id}")

        return result

    return get_model_loader
