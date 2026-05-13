#!/usr/bin/env python3
"""
Shared utilities for the skills_multiagent EvaluationCoder.

Provides:
- CoderMode enum and MODE_CONFIG mapping
- Raw tool functions (tool_*) and their @function_tool wrappers
- Factory functions for cached/contextual tools
- Result inspection and plan-normalisation helpers
"""

import json
import os
import re
import subprocess
import time
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import function_tool, WebSearchTool


# ─────────────────────────── Mode enum ────────────────────────────────────────

class CoderMode(Enum):
    ONETURN_ONETOOL = "oneturn_onetool"
    MULTITURN_ONETOOL = "multiturn_onetool"
    MULTITURN_METADATATOOL = "multiturn_metadatatool"
    MULTITURN_CACHEFILETOOL = "multiturn_cachefiletool"


# ─────────────────────────── Global GPU ID ────────────────────────────────────

_GLOBAL_GPU_ID = 0


def _set_gpu_id(gpu_id: int):
    """Set the global GPU ID for docker execution."""
    global _GLOBAL_GPU_ID
    _GLOBAL_GPU_ID = gpu_id


# ─────────────────────────── Raw tool functions ───────────────────────────────


def tool_run_code_in_docker(code: str, output_dir: str = "/tmp/eval_workspace") -> dict:
    """Save `code` to run_eval.py and run it in a GPU-enabled Docker container."""
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
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def tool_save_file(file_path: str, content: str) -> str:
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


def tool_get_model_labels(model_id: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        id2label = config.get("id2label", {})
        label2id = config.get("label2id", {})

        return {
            "model_id": model_id,
            "id2label": id2label,
            "label2id": label2id,
            "num_labels": config.get("num_labels"),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
            "task_specific_params": config.get("task_specific_params"),
            "label_names": list(id2label.values()) if id2label else [],
        }
    except Exception as e:
        return {"model_id": model_id, "error": f"Error fetching model labels: {str(e)}"}


def tool_inspect_dataset_rows(dataset_id: str, config: str = "default", split: str = "test") -> dict:
    import requests

    base_url = "https://datasets-server.huggingface.co"
    result = {"dataset_id": dataset_id, "config": config, "split": split}

    try:
        params = {"dataset": dataset_id, "config": config, "split": split}
        resp = requests.get(f"{base_url}/first-rows", params=params, timeout=15)

        if resp.status_code in (400, 404):
            params_no_config = {"dataset": dataset_id, "split": split}
            resp = requests.get(f"{base_url}/first-rows", params=params_no_config, timeout=15)

        if resp.status_code != 200:
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

        sample_rows = [row_data.get("row", row_data) for row_data in rows[:3]]
        result["sample_rows"] = sample_rows

        label_analysis = {}
        for feat in features:
            col_name = feat.get("column", feat.get("name", ""))
            dtype = feat.get("dtype", feat.get("type", ""))
            if col_name in ("label", "labels", "ner_tags", "fine_label", "coarse_label"):
                values = [r.get(col_name) for r in sample_rows if col_name in r]
                label_analysis[col_name] = {
                    "dtype": str(dtype),
                    "sample_values": values[:5],
                }
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


# ─────────────────────────── Raw factory functions ────────────────────────────


def make_search_similar_evaluations_fn(results_dir: str):
    """Factory that returns a tool function for searching past successful evaluations."""

    def search_similar_evaluations(
        task_type: str = "",
        dataset_id: str = "",
        model_id: str = "",
        metric: str = "",
    ) -> dict:
        """Search past successful evaluations for similar tasks."""
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

            results_file = case_dir / "results.json"
            if not results_file.exists():
                continue

            try:
                with open(results_file, "r") as f:
                    results_data = json.load(f)
                metric_vals = [v for v in results_data.values() if isinstance(v, (int, float))]
                if metric_vals and all(v == 0.0 for v in metric_vals):
                    continue
            except Exception:
                continue

            score = 0
            if search_terms["dataset_id"] and search_terms["dataset_id"] in dir_name:
                score += 2
            if search_terms["model_id"] and search_terms["model_id"] in dir_name:
                score += 2
            if search_terms["metric"] and search_terms["metric"] in dir_name:
                score += 1

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

        matches.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query": search_terms,
            "total_matches": len(matches),
            "top_matches": matches[:3],
        }

    return search_similar_evaluations


def make_dataset_loader_fn(loaders_dir: Path):
    """Return a raw Python function that loads pre-verified dataset scripts."""
    def get_dataset_loader(dataset_id: str) -> dict:
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


# ─────────────────────────── @function_tool wrappers ──────────────────────────


@function_tool
def run_code_in_docker(code: str, output_dir: str = "/tmp/eval_workspace") -> str:
    """
    Execute Python code inside a Docker container with GPU support.
    The code will be saved to 'run_eval.py' and executed with 'python run_eval.py'.

    CRITICAL: Always add 'import subprocess; subprocess.run(["pip","install",<pkg>],check=True)'
    at the top of the script when the model requires non-standard libraries
    (e.g. flair, timm, sentence-transformers, setfit, clip).
    """
    return json.dumps(tool_run_code_in_docker(code=code, output_dir=output_dir))


@function_tool
def read_file(file_path: str) -> str:
    """Read a file from disk."""
    return tool_read_file(file_path=file_path)


@function_tool
def save_file(file_path: str, content: str) -> str:
    """Save content to a file."""
    return tool_save_file(file_path=file_path, content=content)


@function_tool
def get_model_readme(model_id: str) -> str:
    """Fetch model card / README from HuggingFace."""
    return tool_get_model_readme(model_id=model_id)


@function_tool
def get_dataset_readme(dataset_id: str) -> str:
    """Fetch dataset card / README from HuggingFace."""
    return tool_get_dataset_readme(dataset_id=dataset_id)


@function_tool
def get_model_metadata(model_id: str) -> str:
    """Fetch model metadata (pipeline_tag, library_name, config …)."""
    return json.dumps(tool_get_model_metadata(model_id=model_id))


@function_tool
def get_dataset_metadata(dataset_id: str) -> str:
    """Fetch dataset metadata (splits, features, num_examples …)."""
    return json.dumps(tool_get_dataset_metadata(dataset_id=dataset_id))


@function_tool
def check_code_legitimacy(code: str) -> str:
    """Check whether evaluation code genuinely runs model inference."""
    return json.dumps(tool_check_code_legitimacy(code=code))


@function_tool
def get_model_labels(model_id: str) -> str:
    """
    Fetch id2label and label2id from a model's config.json.
    Critical for label mapping — model label ordering often differs from dataset.
    Returns JSON with id2label, label2id, num_labels, label_names, and model_type.
    """
    return json.dumps(tool_get_model_labels(model_id=model_id))


@function_tool
def inspect_dataset_rows(dataset_id: str, config: str = "default", split: str = "test") -> str:
    """
    Fetch sample rows from a HuggingFace dataset via the Dataset Viewer REST API.
    Fast (~1-2s, no dataset download). Use to inspect column names, data types,
    label values, and detect issues like label=-1.
    """
    return json.dumps(tool_inspect_dataset_rows(
        dataset_id=dataset_id, config=config, split=split
    ), default=str)


def _create_search_similar_evaluations_tool(results_dir: str):
    """Factory: create @function_tool for searching past successful evaluations."""
    raw_fn = make_search_similar_evaluations_fn(results_dir)

    @function_tool
    def search_similar_evaluations(
        task_type: str = "",
        dataset_id: str = "",
        model_id: str = "",
        metric: str = "",
    ) -> str:
        """
        Search past successful evaluations for similar tasks.
        Returns plans and code patterns that worked for similar model/dataset/metric combos.
        Use this to learn from prior successes before writing new evaluation code.
        """
        return json.dumps(raw_fn(
            task_type=task_type, dataset_id=dataset_id,
            model_id=model_id, metric=metric,
        ), default=str)

    return search_similar_evaluations


def _create_dataset_loader_tool(loaders_dir: Path):
    """Factory: create @function_tool for cached dataset loader lookup."""
    raw_fn = make_dataset_loader_fn(loaders_dir)

    @function_tool
    def get_dataset_loader(dataset_id: str) -> str:
        """Get pre-verified dataset loading snippet for a dataset."""
        return json.dumps(raw_fn(dataset_id))

    return get_dataset_loader


def _create_model_loader_tool(loaders_dir: Path):
    """Factory: create @function_tool for cached model loader lookup."""
    raw_fn = make_model_loader_fn(loaders_dir)

    @function_tool
    def get_model_loader(model_id: str) -> str:
        """Get pre-verified model loading snippet for a model."""
        return json.dumps(raw_fn(model_id))

    return get_model_loader


# ─────────────────────────── Result inspection ────────────────────────────────


def _inspect_evaluation_results(output_dir: str, metric: str) -> Dict[str, Any]:
    """Inspect results.json and predictions.json and return structured findings."""
    issues: List[str] = []
    suggestions: List[str] = []
    metric_value = None
    n_predictions = 0
    degenerate = False
    cross_check_delta = None
    metric_name = (metric or "").strip().lower()
    is_accuracy_like = any(
        k in metric_name for k in ("accuracy", "acc", "top1", "top-1", "exact_match", "exact match")
    ) and not any(k in metric_name for k in ("f1", "rouge", "bleu", "auc", "mse", "mae", "rmse"))

    results_path = os.path.join(output_dir, "results.json")
    if not os.path.exists(results_path):
        issues.append("results.json is missing")
        suggestions.append("Ensure the evaluation script saves results to results.json "
                           f"as {{'{metric}': <float>}}")
    else:
        try:
            results = json.load(open(results_path))
            metric_value = next(
                (v for v in results.values() if isinstance(v, (int, float))), None
            )
            if metric_value is None:
                issues.append(f"results.json contains no numeric value: {results}")
                suggestions.append("Save metric as a plain float in results.json")
        except Exception as e:
            issues.append(f"results.json is not valid JSON: {e}")
            suggestions.append("Fix the results.json writing code")

    preds_path = os.path.join(output_dir, "predictions.json")
    if not os.path.exists(preds_path):
        issues.append("predictions.json is missing")
        suggestions.append("Save per-sample predictions to predictions.json as a list of "
                           "{'input':..., 'prediction':..., 'ground_truth':...} dicts")
    else:
        try:
            preds = json.load(open(preds_path))
            if not isinstance(preds, list) or len(preds) == 0:
                issues.append("predictions.json is empty or not a list")
                suggestions.append("Ensure predictions are written as a non-empty list")
            else:
                n_predictions = len(preds)

                pred_values = []
                for p in preds:
                    v = p.get("prediction") if isinstance(p, dict) else p
                    pred_values.append(str(v))

                counter = Counter(pred_values)
                most_common_pct = counter.most_common(1)[0][1] / len(pred_values)
                if most_common_pct >= 0.98:
                    degenerate = True
                    top_pred = counter.most_common(1)[0][0]
                    issues.append(
                        f"Degenerate predictions: {most_common_pct*100:.1f}% of predictions "
                        f"are identical ('{top_pred[:60]}')"
                    )
                    suggestions.append(
                        "The model's output is not varying across inputs. Possible causes:\n"
                        "  - Wrong model type used (e.g. base model used as classifier)\n"
                        "  - Image/text preprocessing is incorrect (all inputs look the same)\n"
                        "  - Label mapping is wrong (always maps to label 0)\n"
                        "  - For generative models: try log-probability scoring instead of generation\n"
                        "Try using the correct model class (check library_name and pipeline_tag "
                        "from get_model_metadata) and verify preprocessing."
                    )

                null_count = sum(
                    1 for p in preds
                    if isinstance(p, dict) and p.get("prediction") is None
                )
                if null_count > 0:
                    issues.append(f"{null_count}/{n_predictions} predictions are None/null")
                    suggestions.append(
                        "Handle edge cases where model output is None; "
                        "add a fallback default prediction"
                    )

                if metric_value is not None and metric_value == 0.0 and n_predictions > 10:
                    issues.append(
                        f"Metric is exactly 0.0 with {n_predictions} predictions – "
                        "likely a label/format mismatch"
                    )
                    suggestions.append(
                        "Metric is 0.0 even though code ran. Possible causes:\n"
                        "  - Predicted label IDs vs string labels mismatch\n"
                        "  - NER: tag scheme mismatch (BIO vs BIOES, different label names)\n"
                        "  - The model outputs probabilities but labels are compared as strings\n"
                        "Print a few (prediction, ground_truth) pairs to debug"
                    )

                valid_preds_for_check = [
                    p for p in preds
                    if isinstance(p, dict)
                    and p.get("prediction") is not None
                    and p.get("ground_truth") is not None
                ]
                if len(valid_preds_for_check) >= 5:
                    pred_types = {type(p["prediction"]).__name__ for p in valid_preds_for_check[:20]}
                    gt_types = {type(p["ground_truth"]).__name__ for p in valid_preds_for_check[:20]}
                    sample_preds = [str(p["prediction"]) for p in valid_preds_for_check[:5]]
                    sample_gts = [str(p["ground_truth"]) for p in valid_preds_for_check[:5]]
                    # QA tasks: str pred + list gt is normal — exclude from type-mismatch flag
                    is_qa_style = pred_types <= {"str"} and gt_types <= {"list"}
                    if pred_types != gt_types and not (pred_types <= {"int", "float"} and gt_types <= {"int", "float"}) and not is_qa_style:
                        issues.append(
                            f"Label type mismatch: predictions are {pred_types} "
                            f"but ground_truth are {gt_types}. "
                            f"Sample preds={sample_preds}, gts={sample_gts}"
                        )
                        suggestions.append(
                            "Prediction type does not match ground_truth type — likely a label mapping bug.\n"
                            "  - If model outputs integer IDs, convert via model.config.id2label[pred_id]\n"
                            "  - If dataset labels are integers but model outputs strings, cast accordingly\n"
                            "  - Check TODO 4 (label mapping check) was correctly implemented"
                        )
                    pred_set = set(sample_preds)
                    gt_set = set(sample_gts)
                    all_pred_numeric = all(s.lstrip("-").isdigit() for s in pred_set)
                    all_gt_alpha = any(not s.lstrip("-").isdigit() for s in gt_set)
                    if all_pred_numeric and all_gt_alpha:
                        issues.append(
                            f"Predictions appear to be raw label IDs ({sorted(pred_set)[:5]}) "
                            f"while ground_truth are label strings ({sorted(gt_set)[:5]}). "
                            "Forgot to apply id2label mapping."
                        )
                        suggestions.append(
                            "Apply model.config.id2label[predicted_id] to convert integer class indices "
                            "to label strings before comparing with ground_truth."
                        )

                if is_accuracy_like and metric_value is not None and n_predictions > 0:
                    valid_preds = [
                        p for p in preds
                        if isinstance(p, dict)
                        and p.get("prediction") is not None
                        and p.get("ground_truth") is not None
                    ]
                    if len(valid_preds) >= 10:
                        def _pred_matches_gt(pred, gt):
                            if isinstance(gt, list):
                                return any(str(pred).strip().lower() == str(g).strip().lower() for g in gt)
                            return str(pred) == str(gt)
                        recomputed = sum(
                            1 for p in valid_preds
                            if _pred_matches_gt(p["prediction"], p["ground_truth"])
                        ) / len(valid_preds)
                        cross_check_delta = abs(metric_value - recomputed)
                        if cross_check_delta > 0.15:
                            issues.append(
                                f"Cross-check mismatch: reported={metric_value:.3f} "
                                f"recomputed_accuracy={recomputed:.3f} "
                                f"(delta={cross_check_delta:.3f}). "
                                "The script may be computing the metric incorrectly."
                            )
                            suggestions.append(
                                "Check how the metric is calculated. For non-accuracy metrics "
                                "(F1, ROUGE etc.) this delta is expected; for accuracy it should be ~0."
                            )

        except Exception as e:
            issues.append(f"predictions.json is not valid JSON: {e}")
            suggestions.append("Fix the predictions.json writing code")

    valid = len(issues) == 0
    return {
        "valid": valid,
        "issues": issues,
        "suggestions": suggestions,
        "metric_value": metric_value,
        "n_predictions": n_predictions,
        "degenerate": degenerate,
        "cross_check_delta": cross_check_delta,
    }


# ─────────────────────────── Mode → tool mapping ──────────────────────────────


MODE_CONFIG = {
    CoderMode.ONETURN_ONETOOL: {
        "max_exec_turns": 1,
        "planning_tools": ["get_model_metadata", "get_dataset_metadata"],
        "execution_tools": ["run_code_in_docker"],
    },
    CoderMode.MULTITURN_ONETOOL: {
        "max_exec_turns": 10,
        "planning_tools": ["get_model_metadata", "get_dataset_metadata"],
        "execution_tools": ["run_code_in_docker"],
    },
    CoderMode.MULTITURN_METADATATOOL: {
        "max_exec_turns": 10,
        "planning_tools": [
            "get_model_readme", "get_dataset_readme",
            "get_model_metadata", "get_dataset_metadata",
            "get_model_labels", "inspect_dataset_rows",
            "search_similar_evaluations",
            "web_search",
        ],
        "execution_tools": [
            "run_code_in_docker", "read_file", "save_file",
            "get_model_readme", "get_dataset_readme",
            "get_model_metadata", "get_dataset_metadata",
            "get_model_labels", "inspect_dataset_rows",
            "search_similar_evaluations",
            "web_search"
        ],
    },
    CoderMode.MULTITURN_CACHEFILETOOL: {
        "max_exec_turns": 10,
        "planning_tools": [
            "get_dataset_loader", "get_model_loader",
            "get_model_readme", "get_dataset_readme",
            "get_model_metadata", "get_dataset_metadata",
            "get_model_labels", "inspect_dataset_rows",
            "search_similar_evaluations",
            "web_search",
        ],
        "execution_tools": [
            "run_code_in_docker", "read_file", "save_file",
            "get_dataset_loader", "get_model_loader",
            "get_model_readme", "get_dataset_readme",
            "get_model_metadata", "get_dataset_metadata",
            "get_model_labels", "inspect_dataset_rows",
            "search_similar_evaluations",
            "check_code_legitimacy", "web_search"
        ],
    },
}


# ─────────────────────────── Module-level helpers ─────────────────────────────


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON robustly from raw model output text."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _normalize_plan(plan_text: str) -> Dict[str, Any]:
    """Normalize planner output into a stable schema with defaults."""
    parsed_obj = _safe_json_parse(plan_text)
    parsed = parsed_obj or {}
    defaults: Dict[str, Any] = {
        "framework": "other",
        "pip_deps": [],
        "dataset_split": "test",
        "input_fields": [],
        "label_field": "label",
        "trust_remote_code": False,
        "model_loading_hint": "",
        "dataset_loading_hint": "",
        "label_mapping_hint": "",
        "known_pitfalls": [],
        "model_metadata_summary": {},
        "dataset_metadata_summary": {},
    }
    for k, v in defaults.items():
        parsed.setdefault(k, v)
    if not isinstance(parsed.get("pip_deps"), list):
        parsed["pip_deps"] = []
    if not isinstance(parsed.get("known_pitfalls"), list):
        parsed["known_pitfalls"] = []
    if not isinstance(parsed.get("input_fields"), list):
        parsed["input_fields"] = []
    if not isinstance(parsed.get("model_metadata_summary"), dict):
        parsed["model_metadata_summary"] = {}
    if not isinstance(parsed.get("dataset_metadata_summary"), dict):
        parsed["dataset_metadata_summary"] = {}
    parsed["_plan_parse_ok"] = bool(parsed_obj)
    return parsed


def _extract_usage(result: Any) -> Dict[str, int]:
    """Extract token usage from a RunResult, summing across all raw responses."""
    inp = out = 0
    try:
        for item in result.raw_responses:
            usage = getattr(item, "usage", None)
            if usage:
                inp += getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
                out += getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
    except Exception:
        pass
    return {"input": inp, "output": out, "total": inp + out}
