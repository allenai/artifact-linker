#!/usr/bin/env python3
"""
Skills-augmented EvaluationCoder using OpenAI Agents SDK with ShellTool.

Two agents (Planner + Executor) + programmatic validation via
_inspect_evaluation_results. Simple retry loop: plan → execute → validate → fix.
"""

import asyncio
import base64
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agents import (
    Agent,
    Runner,
    ShellTool,
    ShellToolInlineSkill,
    ShellToolSkillReference,
    WebSearchTool,
)

from artifact_graph.coder_tools import (
    CoderMode,
    MODE_CONFIG,
    _create_dataset_loader_tool,
    _create_model_loader_tool,
    _create_search_similar_evaluations_tool,
    _extract_usage,
    _inspect_evaluation_results,
    _normalize_plan,
    _set_gpu_id,
    get_dataset_metadata,
    get_dataset_readme,
    get_model_labels,
    get_model_metadata,
    get_model_readme,
    inspect_dataset_rows,
    check_code_legitimacy,
    read_file,
    run_code_in_docker,
    save_file,
    tool_check_code_legitimacy,
)


# ─────────────────────────── Config ───────────────────────────────────────────

DEFAULT_HF_SKILLS_DIR = str(Path(__file__).parent.parent / "skills")

RELEVANT_SKILLS = [
    "hugging-face-dataset-viewer",
    "hugging-face-evaluation",
    "hugging-face-datasets",
    "eval-templates",
]


# ─────────────────────────── Agent instructions ──────────────────────────────

PLANNER_INSTRUCTIONS = """You are an ML evaluation planning assistant.

WORKFLOW — do these steps in order:
1. Call search_similar_evaluations(task_type=..., dataset_id=..., model_id=..., metric=...)
   to find similar past evaluations. If matches found, use their plan and code as a starting point.

2. Call get_model_labels(model_id) to get id2label, label2id, num_labels.
   Call inspect_dataset_rows(dataset_id, config, split) to see actual sample rows,
   column names, and detect issues like label=-1.

3. Call get_dataset_metadata(dataset_id) and get_model_metadata(model_id) for full metadata.
   Call get_model_readme(model_id) and get_dataset_readme(dataset_id) for usage hints.

4. For non-standard libraries (flair, timm, sentence-transformers, setfit, clip, etc.),
   call web_search() for usage examples.

5. Optionally call get_dataset_loader/get_model_loader for cached snippets.

=== TASK-TYPE REFERENCE (brief — detailed templates in eval-task-skills) ===
- extractive_qa: pipeline('question-answering'), SQuAD v2 ~50% unanswerable (threshold score<0.2→""), normalize_answer
- nli/classification: AutoModelForSequenceClassification, check id2label ordering, filter label=-1, transformers>=4.48 for ModernBERT
- zero_shot: pipeline('zero-shot-classification'), define candidate_labels + hypothesis_template
- multilabel: For "accuracy" metric → single-label approach (filter to 1-hot examples, argmax). GoEmotions: 0/1 columns per emotion
- summarization: Seq2SeqLM + generate(), evaluate.load("rouge"), CNN/DM config='3.0.0', num_beams=4
- ner: AutoModelForTokenClassification, is_split_into_words=True, seqeval, conll2003→revision="refs/convert/parquet" if script error
- vision: AutoImageProcessor + AutoModelForImageClassification, grayscale→RGB, CIFAR-100: "img"/"fine_label"
- sentiment: Standard classification, check actual column names (e.g., "verse_text" for poem_sentiment)
- multiple_choice: HellaSwag→vLLM + chat_template + conditional log-likelihood, gpu_memory_utilization=0.5 for FP8 models

=== CROSS-CUTTING RULES ===
- Always set trust_remote_code=True in the plan.
- Always check actual label values — if all -1 or invalid, note different split needed.
- Label mapping: ALWAYS check model.config.id2label/label2id at runtime. Never assume identity mapping.
- Sampling: shuffle(seed=42).select(range(min(1000, len(ds)))).

Output ONLY JSON:
{
  "framework": "transformers|flair|timm|sentence-transformers|setfit|clip|other",
  "pip_deps": ["transformers>=4.48.0", ...],
  "dataset_split": "test (or 'dev'/'validation' if test labels are invalid)",
  "dataset_config": "default (or actual config name from API inspection)",
  "input_fields": ["premise", "hypothesis"], "label_field": "label",
  "label_names": {"0": "entailment", ...},
  "task_type": "classification|nli|extractive_qa|multiple_choice|generation|ner|vision|multilabel|zero_shot",
  "trust_remote_code": true,
  "model_loading_hint": "...", "dataset_loading_hint": "...",
  "label_mapping_hint": "...", "evaluation_strategy": "...",
  "generation_params": {"num_beams": 4, "max_length": 142, ...},
  "known_pitfalls": ["..."],
  "sample_data": "1-2 actual rows from shell inspection",
  "model_metadata_summary": {...}, "dataset_metadata_summary": {...}
}"""

EXECUTOR_INSTRUCTIONS = """You are an expert ML engineer. Write and run evaluation scripts.

Read the plan carefully. Write a COMPLETE Python script and run it via run_code_in_docker.

=== SETUP ===
1. Install deps at script TOP:
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
       "transformers", "datasets", "evaluate", "huggingface_hub<1.0", "pyarrow", "fsspec", ...])
   CRITICAL: Pin huggingface_hub<1.0. Always upgrade transformers+datasets+evaluate together.
2. ALWAYS use trust_remote_code=True in load_dataset() calls.
3. Follow model_loading_hint and dataset_loading_hint from the plan EXACTLY.

=== CRITICAL TASK-SPECIFIC PATTERNS ===

** multiple_choice (e.g. HellaSwag) — MUST use vLLM for FP8/instruct models **
  pip install: vllm, compressed-tensors, torch, accelerate, safetensors
  from vllm import LLM, SamplingParams
  llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.5, trust_remote_code=True, max_model_len=4096)
  sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
  For instruct models: use tokenizer.apply_chat_template() with system="You are a helpful assistant"
  Score: LENGTH-NORMALIZED avg logprob of continuation tokens (total_logprob / n_tokens).
  Use out.prompt_token_ids for alignment (don't re-tokenize). Batch all options (batch_size=64).
  DO NOT use AutoModelForCausalLM for FP8 models — it will fail with compressed_tensors errors.

** ner (e.g. CoNLL2003) — parquet fallback for legacy scripts **
  try: load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT, trust_remote_code=True)
  except: load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT, revision="refs/convert/parquet")
  Use seqeval for entity-level F1. Use is_split_into_words=True for tokenization.
  Map model label IDs to dataset label names via normalized BIO tag matching.

** multilabel (e.g. GoEmotions) — single-label approach for "accuracy" metric **
  For "accuracy" on multilabel datasets: filter to examples with exactly ONE positive label
  (exclude neutral/no-emotion). Use logits.argmax(dim=-1) instead of sigmoid+threshold.

** extractive_qa (e.g. SQuAD v2) **
  Use pipeline('question-answering'). For SQuAD v2 (has unanswerable questions):
  set handle_impossible_answer=True, threshold score<0.2 → empty string "".
  Ground truth: answers["text"] is a list — match ANY answer.

=== GENERAL REQUIREMENTS ===
- GPU: model.to("cuda"). Batched inference (batch>=8). torch.no_grad(). Fallback to CPU on OOM.
- Sampling: ds.shuffle(seed=42).select(range(min(1000, len(ds)))).
- Save results.json: {"metric_name": float_value} (e.g. {"accuracy": 0.812})
- Save predictions.json: [{"input":.., "prediction":.., "ground_truth":..}]
- Print first 5 (prediction, ground_truth) pairs for debugging.
- Label mapping: ALWAYS check model.config.id2label/label2id. Never assume identity mapping.

=== ON FAILURE ===
- Read the error, fix the specific issue, retry.
- "Dataset scripts no longer supported": use trust_remote_code=True, then revision="refs/convert/parquet".
- "HfFolder"/"is_offline_mode" ImportError: upgrade huggingface_hub with -U.
- "BuilderConfig 'default' not found": use actual config name (often same as dataset name).
- GPU OOM: reduce batch size or fall back to CPU.
- Float8/FP8/compressed_tensors error: use vLLM (pip install vllm compressed-tensors) with gpu_memory_utilization=0.5.
- vLLM memory error: lower gpu_memory_utilization (0.4, 0.3).
- Don't repeat the same mistake."""


# ─────────────────────────── Skill packaging ─────────────────────────────────


def _parse_skill_frontmatter(skill_dir: str) -> Dict[str, str]:
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    name = os.path.basename(skill_dir)
    description = f"HuggingFace skill: {name}"
    if not os.path.exists(skill_md_path):
        return {"name": name, "description": description}
    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {"name": name, "description": description}
    fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            line = line.strip()
            if line.lower().startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"').strip("'")
    return {"name": name, "description": description}


def _zip_directory_to_bytes(directory: str) -> bytes:
    buf = io.BytesIO()
    base = Path(directory)
    top_dir = base.name
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(base.rglob("*")):
            if file_path.is_file():
                arcname = f"{top_dir}/{file_path.relative_to(base)}"
                zf.write(file_path, arcname)
    return buf.getvalue()


def _package_skill_inline(skill_dir: str) -> ShellToolInlineSkill:
    meta = _parse_skill_frontmatter(skill_dir)
    zip_bytes = _zip_directory_to_bytes(skill_dir)
    b64_data = base64.b64encode(zip_bytes).decode("ascii")
    return {
        "type": "inline",
        "name": meta["name"],
        "description": meta["description"],
        "source": {"type": "base64", "media_type": "application/zip", "data": b64_data},
    }


def _upload_skill_hosted(skill_dir: str, api_key: str) -> ShellToolSkillReference:
    meta = _parse_skill_frontmatter(skill_dir)
    zip_bytes = _zip_directory_to_bytes(skill_dir)
    response = requests.post(
        "https://api.openai.com/v1/skills",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"files": (f"{meta['name']}.zip", zip_bytes, "application/zip")},
        timeout=120,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to upload skill '{meta['name']}': HTTP {response.status_code}")
    result = response.json()
    skill_id = result.get("id") or result.get("skill_id")
    if not skill_id:
        raise RuntimeError(f"No skill_id returned for '{meta['name']}': {result}")
    return {"type": "skill_reference", "skill_id": skill_id}


# ─────────────────────────── Main class ───────────────────────────────────────


class SkillsMultiAgentEvaluationCoder:
    """Two-agent EvaluationCoder with HF skills and programmatic validation."""

    MODE_CONFIG = MODE_CONFIG

    def __init__(
        self,
        mode: CoderMode = CoderMode.MULTITURN_CACHEFILETOOL,
        llm_model: str = "gpt-4o",
        gpu_id: int = 0,
        max_steps: Optional[int] = None,
        max_exec_turns: Optional[int] = None,
        max_retry_rounds: int = 3,
        dataset_loaders_dir: Optional[str] = None,
        model_loaders_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        skill_mode: str = "inline",
        hf_skills_dir: str = DEFAULT_HF_SKILLS_DIR,
    ):
        self.mode = mode
        self.llm_model = llm_model
        self.gpu_id = gpu_id
        self.max_retry_rounds = max_retry_rounds
        self.skill_mode = skill_mode
        self.hf_skills_dir = hf_skills_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        _set_gpu_id(gpu_id)
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        config = self.MODE_CONFIG[mode]
        self.max_exec_turns = max_steps or max_exec_turns or config["max_exec_turns"]

        # ── Build tools ───────────────────────────────────────────────────
        dataset_loader_tool = _create_dataset_loader_tool(
            Path(dataset_loaders_dir) if dataset_loaders_dir
            else Path(__file__).parent.parent / "scripts" / "dataset_loaders"
        )
        model_loader_tool = _create_model_loader_tool(
            Path(model_loaders_dir) if model_loaders_dir
            else Path(__file__).parent.parent / "scripts" / "model_loaders"
        )
        default_results_dir = str(
            Path(__file__).parent.parent / "scripts" / "skills_v14_hard_10"
        )
        similar_eval_tool = _create_search_similar_evaluations_tool(default_results_dir)

        tool_map = {
            "run_code_in_docker": run_code_in_docker,
            "read_file": read_file,
            "save_file": save_file,
            "get_model_readme": get_model_readme,
            "get_dataset_readme": get_dataset_readme,
            "get_model_metadata": get_model_metadata,
            "get_dataset_metadata": get_dataset_metadata,
            "get_model_labels": get_model_labels,
            "inspect_dataset_rows": inspect_dataset_rows,
            "search_similar_evaluations": similar_eval_tool,
            "check_code_legitimacy": check_code_legitimacy,
            "get_dataset_loader": dataset_loader_tool,
            "get_model_loader": model_loader_tool,
            "web_search": WebSearchTool(),
        }

        # ── Package HF skills & create ShellTool ─────────────────────────
        skills, skill_names = self._prepare_skills()
        self.skill_names = skill_names
        shell_tool = None
        if skills:
            shell_tool = ShellTool(environment={
                "type": "container_auto",
                "skills": skills,
            })

        # ── Planning agent: research tools only (no ShellTool — avoids API bug) ──
        planning_tools = [tool_map[n] for n in config["planning_tools"]]
        self.planning_tool_names = list(config["planning_tools"])

        self.planning_agent = Agent(
            name="PlanningAgent",
            model=llm_model,
            instructions=PLANNER_INSTRUCTIONS,
            tools=planning_tools,
        )

        # ── Execution agent: coding tools + ShellTool ─────────────────────
        execution_tools = [tool_map[n] for n in config["execution_tools"]]
        if shell_tool:
            execution_tools.append(shell_tool)
        self.execution_tool_names = list(config["execution_tools"]) + (["shell_tool"] if shell_tool else [])

        self.execution_agent = Agent(
            name="ExecutionAgent",
            model=llm_model,
            instructions=EXECUTOR_INSTRUCTIONS,
            tools=execution_tools,
        )

        print(f"[Skills] SkillsMultiAgentEvaluationCoder initialized:")
        print(f"   Mode: {mode.value} | LLM: {llm_model} | GPU: {gpu_id}")
        print(f"   Skills: {skill_names} | Retry rounds: {max_retry_rounds}")
        print(f"   Planning: {self.planning_tool_names}")
        print(f"   Execution: {self.execution_tool_names}")

    def _prepare_skills(self) -> tuple:
        skills = []
        skill_names = []
        for skill_name in RELEVANT_SKILLS:
            skill_dir = os.path.join(self.hf_skills_dir, skill_name)
            if not os.path.isdir(skill_dir):
                continue
            try:
                if self.skill_mode == "hosted":
                    skill = _upload_skill_hosted(skill_dir, self.api_key)
                else:
                    skill = _package_skill_inline(skill_dir)
                skills.append(skill)
                skill_names.append(skill_name)
                print(f"   [Skills] Packaged ({self.skill_mode}): {skill_name}")
            except Exception as e:
                print(f"   [Skills] Error: {skill_name}: {e}")
        return skills, skill_names

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int = 200,
    ) -> Dict[str, Any]:
        return asyncio.run(self._evaluate_async(
            model_name, dataset_name, metric, output_dir, max_samples
        ))

    # ── Core loop ─────────────────────────────────────────────────────────

    @staticmethod
    def _analyze_predictions(output_dir: str, metric: str) -> str:
        """Analyze predictions.json to produce diagnostic feedback for the agent."""
        preds_path = os.path.join(output_dir, "predictions.json")
        if not os.path.exists(preds_path):
            return ""
        try:
            import json as _json
            with open(preds_path, "r") as f:
                preds = _json.load(f)
            if not preds or not isinstance(preds, list):
                return ""

            diagnostics = []
            n = len(preds)

            # Check prediction vs ground_truth type/format
            sample_pred = preds[0].get("prediction")
            sample_gt = preds[0].get("ground_truth")

            # Multilabel: check label count distribution
            if isinstance(sample_pred, list) and isinstance(sample_gt, list):
                pred_counts = [len(p.get("prediction", [])) for p in preds]
                gt_counts = [len(p.get("ground_truth", [])) for p in preds]
                avg_pred = sum(pred_counts) / n if n else 0
                avg_gt = sum(gt_counts) / n if n else 0
                if avg_pred > avg_gt * 3 and avg_gt > 0:
                    diagnostics.append(
                        f"OVER-PREDICTION: Model predicts avg {avg_pred:.1f} labels/sample "
                        f"but ground truth has avg {avg_gt:.1f}. "
                        f"The sigmoid threshold (0.5) is too low for this model. "
                        f"Try threshold=0.8 or 0.9, or use per-label calibration."
                    )
                elif avg_pred == 0:
                    diagnostics.append(
                        f"UNDER-PREDICTION: Model predicts 0 labels/sample on average. "
                        f"Check label mapping — model outputs may not be aligned to dataset labels."
                    )

            # Classification: check if predictions are all the same
            if isinstance(sample_pred, (str, int)):
                unique_preds = set(str(p.get("prediction", "")) for p in preds[:200])
                if len(unique_preds) == 1:
                    diagnostics.append(
                        f"DEGENERATE: All predictions are the same value '{unique_preds.pop()}'. "
                        f"Model may not be loaded correctly or input preprocessing is wrong."
                    )

            # QA: check empty predictions ratio
            if isinstance(sample_pred, str) and isinstance(sample_gt, str):
                empty_preds = sum(1 for p in preds if not p.get("prediction", "").strip())
                empty_gts = sum(1 for p in preds if not p.get("ground_truth", "").strip())
                if empty_preds > n * 0.8 and empty_gts < n * 0.6:
                    diagnostics.append(
                        f"TOO MANY EMPTY PREDICTIONS: {empty_preds}/{n} predictions are empty "
                        f"but only {empty_gts}/{n} ground truths are empty. "
                        f"The no-answer threshold may be too aggressive."
                    )
                elif empty_preds < n * 0.1 and empty_gts > n * 0.3:
                    diagnostics.append(
                        f"TOO FEW EMPTY PREDICTIONS: {empty_preds}/{n} predictions are empty "
                        f"but {empty_gts}/{n} ground truths are empty (unanswerable). "
                        f"Add no-answer detection with confidence threshold < 0.2."
                    )

            # Show first 3 mismatches
            mismatches = []
            for p in preds[:20]:
                pred_str = str(p.get("prediction", ""))
                gt_str = str(p.get("ground_truth", ""))
                if pred_str != gt_str:
                    mismatches.append(f"  pred={pred_str[:80]} | gt={gt_str[:80]}")
                if len(mismatches) >= 3:
                    break
            if mismatches:
                diagnostics.append("Sample mismatches:\n" + "\n".join(mismatches))

            return "\n".join(diagnostics)
        except Exception:
            return ""

    @staticmethod
    def _extract_docker_errors(output_dir: str) -> str:
        """Extract relevant error lines from run.log."""
        run_log_path = os.path.join(output_dir, "run.log")
        if not os.path.exists(run_log_path):
            return ""
        try:
            with open(run_log_path, "r") as f:
                log_lines = f.readlines()
            error_lines = []
            for line in log_lines:
                stripped = line.strip()
                if any(kw in stripped for kw in ("Traceback", "Error", "error", "Exception",
                                                   "ImportError", "ModuleNotFoundError",
                                                   "RuntimeError", "ValueError")):
                    error_lines.append(stripped)
            if error_lines:
                return "Docker errors: " + "; ".join(error_lines[-5:])
        except Exception:
            pass
        return ""

    async def _evaluate_async(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🤖 [SkillsEval] {model_name} | {dataset_name} | {metric}")
        print(f"📁 Output: {output_dir}")
        print(f"{'='*80}\n")

        sample_note = (
            f"Randomly sample up to {max_samples} examples for large datasets."
            if max_samples > 0 else "Use ALL samples (no limit)."
        )

        total_tokens = {"input": 0, "output": 0, "total": 0}
        t_start = time.time()
        validation_feedback = ""
        plan_str = None
        replan_needed = True
        # Lessons-learned buffer: prevents repeating the same mistakes across rounds
        lessons_learned = []

        for round_idx in range(self.max_retry_rounds):
            print(f"\n── Round {round_idx + 1}/{self.max_retry_rounds} ──────────────────────")

            # ── STEP 1: Plan ──────────────────────────────────────────────
            if replan_needed or not plan_str:
                plan_prompt = (
                    f"Plan evaluation of model `{model_name}` on dataset `{dataset_name}` (metric: `{metric}`).\n"
                    f"Sampling: {sample_note}\n\n"
                    f"FIRST: Call search_similar_evaluations() to find past successful evaluations.\n"
                    f"THEN: Call get_model_labels() and inspect_dataset_rows() to check label mapping and data.\n"
                    f"THEN: Call get_model_metadata/get_dataset_metadata for full context.\n"
                    f"Output ONLY JSON plan."
                )
                if validation_feedback:
                    plan_prompt += f"\n\nPREVIOUS FAILURE:\n{validation_feedback}\nFix the issues."
                if lessons_learned:
                    plan_prompt += (
                        f"\n\nLESSONS FROM PRIOR ROUNDS (do NOT repeat these mistakes):\n"
                        + "\n".join(f"- {l}" for l in lessons_learned)
                    )

                print("📋 [PlanningAgent] Researching …")
                t0 = time.time()
                try:
                    plan_result = await Runner.run(self.planning_agent, plan_prompt, max_turns=8)
                except Exception as e:
                    print(f"   ⚠️ PlanningAgent error: {e}")
                    plan_data = _normalize_plan("{}")
                    plan_str = json.dumps(plan_data, ensure_ascii=False, indent=2)
                    lessons_learned.append(f"PlanningAgent crashed: {str(e)[:100]}")
                    continue
                plan_elapsed = time.time() - t0
                plan_usage = _extract_usage(plan_result)
                for k in ("input", "output", "total"):
                    total_tokens[k] += plan_usage.get(k, 0)

                raw_plan = str(plan_result.final_output or "{}")
                plan_data = _normalize_plan(raw_plan)
                plan_str = json.dumps(plan_data, ensure_ascii=False, indent=2)
                print(f"   Plan: {plan_str[:300]}")
                print(f"   [tokens] in={plan_usage['input']} out={plan_usage['output']} | {plan_elapsed:.1f}s")
            else:
                print("📋 [PlanningAgent] Skipped (reusing plan)")

            # ── STEP 2: Execute ───────────────────────────────────────────
            exec_prompt = f"""Evaluate model `{model_name}` on dataset `{dataset_name}` (metric: `{metric}`).

PLAN:
{plan_str}

REQUIREMENTS:
- {sample_note}
- Save results.json: {{"{metric}": <float>}} and predictions.json: [{{"input":..,"prediction":..,"ground_truth":..}}]
- Call run_code_in_docker(code, output_dir="{output_dir}"). Don't hardcode paths in script."""

            if validation_feedback:
                exec_prompt += f"\n\nPREVIOUS FAILURE:\n{validation_feedback}\nFix the specific issue."
            if lessons_learned:
                exec_prompt += (
                    f"\n\nLESSONS FROM PRIOR ROUNDS (do NOT repeat these mistakes):\n"
                    + "\n".join(f"- {l}" for l in lessons_learned)
                )

            print("⚙️  [ExecutionAgent] Writing and running code …")
            t0 = time.time()
            try:
                exec_result = await Runner.run(self.execution_agent, exec_prompt, max_turns=self.max_exec_turns)
            except Exception as e:
                print(f"   ⚠️ ExecutionAgent error: {e}")
                validation_feedback = f"Agent crashed: {str(e)[:200]}"
                lessons_learned.append(f"ExecutionAgent crashed: {str(e)[:100]}")
                replan_needed = False
                continue
            exec_elapsed = time.time() - t0
            exec_usage = _extract_usage(exec_result)
            for k in ("input", "output", "total"):
                total_tokens[k] += exec_usage.get(k, 0)

            print(f"   Output: {str(exec_result.final_output)[:200]}")
            print(f"   [tokens] in={exec_usage['input']} out={exec_usage['output']} | {exec_elapsed:.1f}s")

            # ── STEP 3: Legitimacy check ──────────────────────────────────
            script_path = os.path.join(output_dir, "run_eval.py")
            if os.path.exists(script_path):
                try:
                    with open(script_path, "r") as f:
                        code_text = f.read()
                    legit = tool_check_code_legitimacy(code_text)
                    if legit.get("is_cheating") and float(legit.get("confidence", 0)) >= 0.5:
                        reason = legit.get("reason", "suspicious code")
                        print(f"🚫 Legitimacy failed: {reason}")
                        validation_feedback = f"LEGITIMACY FAILED: {reason}. Use real model inference."
                        lessons_learned.append(f"Code flagged as cheating: {reason}")
                        replan_needed = False
                        continue
                    print(f"✅ Legitimacy OK (conf={legit.get('confidence', '?')})")
                except Exception as e:
                    print(f"⚠️  Legitimacy check error: {e}")

            # ── STEP 4: Programmatic validation + prediction diagnostics ──
            raw_val = _inspect_evaluation_results(output_dir=output_dir, metric=metric)

            if raw_val.get("valid"):
                mv = raw_val.get("metric_value")
                np_ = raw_val.get("n_predictions", 0)
                print(f"✅ VALID: {metric}={mv} (n={np_} predictions)")
                break
            else:
                issues = raw_val.get("issues", [])
                suggestions = raw_val.get("suggestions", [])
                print(f"❌ INVALID: {'; '.join(issues[:2])}")

                # Analyze predictions for diagnostic feedback
                pred_diagnostics = self._analyze_predictions(output_dir, metric)
                if pred_diagnostics:
                    print(f"🔍 Prediction diagnostics: {pred_diagnostics[:200]}")

                # Extract Docker errors
                docker_error = self._extract_docker_errors(output_dir)

                # Determine if replanning needed
                issue_text = " ".join(issues + suggestions).lower()
                replan_keywords = ("wrong model", "label mapping", "id2label", "degenerate",
                                   "label type mismatch", "raw label ids",
                                   "dataset scripts are no longer supported",
                                   "trust_remote_code")
                replan_needed = any(kw in issue_text for kw in replan_keywords)

                # Force replan on metric=0.0 (structural failure)
                metric_value = raw_val.get("metric_value")
                if metric_value is not None and float(metric_value) == 0.0:
                    replan_needed = True
                    print("   ⚠️ Metric=0.0 → forcing replan")

                # Build rich validation feedback
                validation_feedback = f"Issues: {json.dumps(issues)}\nSuggestions: {json.dumps(suggestions[:2])}"
                if pred_diagnostics:
                    validation_feedback += f"\nPrediction diagnostics: {pred_diagnostics}"
                if docker_error:
                    validation_feedback += f"\n{docker_error}"

                # Record lesson learned for this round
                lesson = "; ".join(issues[:2])
                if pred_diagnostics:
                    lesson += f" | {pred_diagnostics.split(chr(10))[0]}"
                lessons_learned.append(f"Round {round_idx+1}: {lesson}")

        total_elapsed = time.time() - t_start
        print(f"\n📊 {round_idx + 1} round(s) | tokens in={total_tokens['input']} "
              f"out={total_tokens['output']} | {total_elapsed:.1f}s")

        return self._collect_results(
            output_dir, model_name, dataset_name, metric, plan_str,
            usage_summary={
                "rounds_used": round_idx + 1,
                "total_tokens": total_tokens,
                "total_elapsed_s": round(total_elapsed, 1),
            },
        )

    def _collect_results(
        self,
        output_dir: str,
        model_name: str,
        dataset_name: str,
        metric: str,
        plan_str: Optional[str] = None,
        usage_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results_path = os.path.join(output_dir, "results.json")
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metric": metric,
            "backend": "skills_multiagent",
            "mode": self.mode.value,
            "llm_model": self.llm_model,
            "max_retry_rounds": self.max_retry_rounds,
            "planning_tools": self.planning_tool_names,
            "execution_tools": self.execution_tool_names,
            "skills": self.skill_names,
            "plan": plan_str,
            "usage": usage_summary,
        }
        try:
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass

        if os.path.exists(results_path):
            try:
                eval_results = json.load(open(results_path))
                return {"success": True, **eval_results}
            except Exception:
                pass
        return {"success": False, "error": "No results.json produced"}

    @classmethod
    def from_mode_string(cls, mode_str: str, **kwargs) -> "SkillsMultiAgentEvaluationCoder":
        mode_map = {m.value: m for m in CoderMode}
        if mode_str not in mode_map:
            raise ValueError(f"Invalid mode: {mode_str}. Valid: {list(mode_map.keys())}")
        kwargs.pop("temperature", None)
        kwargs["mode"] = mode_map[mode_str]
        return cls(**kwargs)
