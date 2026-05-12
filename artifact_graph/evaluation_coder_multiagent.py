#!/usr/bin/env python3
"""
Multi-Agent EvaluationCoder using OpenAI Agents SDK.

Architecture:
  Orchestrator
    ├── PlanningAgent   – reads model/dataset metadata, determines strategy & pip deps
    ├── ExecutionAgent  – writes & runs evaluation code in Docker
    └── ValidationAgent – checks results.json AND predictions.json for quality issues

Validation checks:
  - predictions.json exists and is non-empty
  - predictions are NOT all the same (degenerate)
  - no null / None predictions
  - metric value is not suspiciously 0 when sample count > 0
  - cross-check: recompute metric from predictions and compare to results.json
  - trigger re-plan with specific feedback if any check fails
"""

import asyncio
import json
import os
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from agents import Agent, Runner, function_tool, WebSearchTool

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
    tool_get_model_labels,
    tool_inspect_dataset_rows,
    make_dataset_loader_fn,
    make_model_loader_fn,
    make_search_similar_evaluations_fn,
)


# ─────────────────────────── shared @function_tool wrappers ──────────────────


@function_tool
def run_code_in_docker(code: str, output_dir: str = "/tmp/eval_workspace") -> str:
    """
    Execute Python code inside a Docker container with GPU support.
    The code will be saved to 'run_eval.py' and executed with 'python run_eval.py'.

    CRITICAL: Always add 'import subprocess; subprocess.run(["pip","install",<pkg>],check=True)'
    at the top of the script when the model requires non-standard libraries
    (e.g. flair, timm, sentence-transformers, setfit, clip).

    Args:
        code: Python source code to execute
        output_dir: Directory to store scripts and results

    Returns:
        JSON string with 'success', 'exit_code', 'output', and 'results' keys
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

    Args:
        dataset_id: HuggingFace dataset ID
        config: Dataset config name (default: 'default')
        split: Split to inspect (default: 'test')
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

        Args:
            task_type: e.g. 'ner', 'nli', 'qa', 'summarization', 'vision', 'multilabel'
            dataset_id: partial match, e.g. 'squad' or 'conll2003'
            model_id: partial match, e.g. 'bert' or 'roberta'
            metric: e.g. 'accuracy', 'f1', 'rouge2'
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
        """
        Get pre-verified dataset loading snippet for a dataset.

        Returns:
            JSON string with fields like 'found', 'script', and 'spec'.
        """
        return json.dumps(raw_fn(dataset_id))

    return get_dataset_loader


def _create_model_loader_tool(loaders_dir: Path):
    """Factory: create @function_tool for cached model loader lookup."""
    raw_fn = make_model_loader_fn(loaders_dir)

    @function_tool
    def get_model_loader(model_id: str) -> str:
        """
        Get pre-verified model loading snippet for a model.

        Returns:
            JSON string with fields like 'found', 'script', 'inference_type', 'id2label'.
        """
        return json.dumps(raw_fn(model_id))

    return get_model_loader


# ─────────────────────────── Validation tool ─────────────────────────────────


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

    # ── 1. results.json ──────────────────────────────────────────────────────
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

    # ── 2. predictions.json ───────────────────────────────────────────────────
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

                # ── 3. Degenerate check ───────────────────────────────────────
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

                # ── 4. Null check ─────────────────────────────────────────────
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

                # ── 5. Zero metric when predictions exist ─────────────────────
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

                # ── 6. Label type/mapping mismatch check ─────────────────────
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
                    # int vs string mismatch (skip str vs list — QA tasks use str pred + list gt)
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
                    # Check if predictions look like raw integer IDs while GT are label strings
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

                # ── 7. Cross-check accuracy-like metrics only ─────────────────
                if is_accuracy_like and metric_value is not None and n_predictions > 0:
                    valid_preds = [
                        p for p in preds
                        if isinstance(p, dict)
                        and p.get("prediction") is not None
                        and p.get("ground_truth") is not None
                    ]
                    if len(valid_preds) >= 10:
                        def _pred_matches_gt(pred, gt):
                            """Handle QA-style (str pred vs list gt) and standard equality."""
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


@function_tool
def check_evaluation_results(output_dir: str, metric: str) -> str:
    """
    Inspect results.json and predictions.json in output_dir to detect quality issues.

    Checks performed:
      1. results.json exists and contains a numeric metric value
      2. predictions.json exists and is non-empty
      3. Predictions are NOT all identical (degenerate output)
      4. No null / None values in predictions
      5. Metric value is not 0.0 when there are valid predictions  (suspicious)
      6. Cross-check: if predictions carry ground_truth + prediction fields,
         recompute accuracy/f1 and compare to the reported value

    Returns JSON with:
      {
        "valid": bool,
        "issues": [str],          # list of problem descriptions
        "suggestions": [str],     # actionable fix hints for re-planning
        "metric_value": float | null,
        "n_predictions": int,
        "degenerate": bool,
        "cross_check_delta": float | null   # |reported - recomputed|
      }
    """
    return json.dumps(_inspect_evaluation_results(output_dir=output_dir, metric=metric), indent=2)


# ─────────────────────────── Agent factories ─────────────────────────────────


def _make_planning_agent(llm_model: str, tools: List[Any]) -> Agent:
    return Agent(
        name="PlanningAgent",
        model=llm_model,
        instructions="""You are an ML evaluation research assistant.

Your job is to quickly gather the key information needed to correctly load and
evaluate a HuggingFace model on a dataset — especially for non-standard libraries
where the ExecutionAgent might get the API wrong.

Steps (do them in order, stop early if you already have enough info):
1. Call search_similar_evaluations(task_type=..., dataset_id=..., model_id=..., metric=...)
   to find past evaluations for similar tasks. If matches found, use their plan and code as reference.
2. Call get_model_labels(model_id) to get id2label, label2id, num_labels.
   Call inspect_dataset_rows(dataset_id, config, split) to see actual sample rows and detect issues.
3. If available, call get_dataset_loader(dataset_id) and get_model_loader(model_id).
   If found=true, prioritize these cached snippets as primary loading hints.
4. get_model_metadata(model_id)  — check library_name, pipeline_tag.
5. get_model_readme(model_id)    — find loading snippet, pip deps, label mapping.
6. get_dataset_readme(dataset_id) — find split name, input/label column names.
7. If library_name is NOT "transformers" (e.g. flair, timm, sentence-transformers,
   setfit, etc.) OR the README has no clear usage example, call web_search() with
   queries like:
     - "<model_name> python inference example"
     - "<library> load model evaluate huggingface"
   to find the correct API and any known loading gotchas.

=== TASK-SPECIFIC SKILLS (learned from successful evaluations) ===

## Extractive QA (SQuAD-like)
- Use pipeline('question-answering') — handles sliding window internally.
- Gold answers = ex['answers']['text'] (a LIST). Correct if normalized pred matches ANY.
- SQuAD v2: ~50% unanswerable (empty list). Apply no-answer threshold (score < 0.20 → "").
- Normalize: lowercase, strip punctuation, remove articles, collapse whitespace.
- Dataset config: specify explicitly (e.g., 'squad_v2').

## NLI / Classification
- Use AutoTokenizer + AutoModelForSequenceClassification, NOT zero-shot-classification pipeline.
- Tokenize pairwise for NLI: tokenizer(premises, hypotheses, ...).
- Filter out label == -1 before evaluation. If ALL test labels are -1, use validation split.
- Check model.config.id2label — label ordering can differ between model and dataset.
- For DeBERTa-v3: install sentencepiece. For ModernBERT: transformers>=4.48.

## Zero-Shot Classification
- Use pipeline("zero-shot-classification") with meaningful candidate_labels and hypothesis_template.

## Multilabel (GoEmotions etc.)
- sigmoid(logits) > 0.5 (NOT softmax+argmax). Exact-match accuracy.
- Map labels by NAME not index. model.config.id2label → dataset columns.
- GoEmotions: check if labels are individual 0/1 columns or list-of-IDs.

## Summarization
- AutoModelForSeq2SeqLM + model.generate(). Use model's task_specific_params.
- BART-CNN: num_beams=4, max_length=142, min_length=56. CNN/DailyMail config='3.0.0'.
- evaluate.load("rouge"), use_stemmer=True. Install rouge-score.

## NER
- AutoModelForTokenClassification. NEVER use pipeline('ner').
- is_split_into_words=True. Align with word_ids(): first subword only.
- evaluate.load("seqeval"). Dataset conll2003 needs trust_remote_code=True.
- For Flair: SequenceTagger.load(), Sentence(tokens, use_tokenizer=False).

## Vision
- AutoImageProcessor + AutoModelForImageClassification. Grayscale → .convert("RGB").
- CIFAR-100: field="img", use fine_label. Food-101: validation split, field="image".

## Sentiment / Text Classification
- Check actual column names (e.g., "verse_text" not "text").

=== CROSS-CUTTING RULES ===
- ALWAYS check model.config.id2label/label2id. Never assume identity mapping.
- Always set trust_remote_code=True when the plan says so.

Output ONLY valid JSON (no markdown, no extra text):
{
  "framework": "transformers|flair|timm|sentence-transformers|setfit|clip|other",
  "pip_deps": ["flair", "datasets", ...],
  "dataset_split": "test|validation|train",
  "input_fields": ["text"],
  "label_field": "label",
  "trust_remote_code": false,
  "model_loading_hint": "exact Python snippet or description of how to load THIS model",
  "dataset_loading_hint": "exact Python snippet or description of how to load THIS dataset",
  "label_mapping_hint": "how to convert model output to dataset label (id2label, etc.)",
  "known_pitfalls": ["list of known issues / common mistakes for this model or library"],
  "generation_params": {"num_beams": 4, "max_length": 142, ...},
  "model_metadata_summary": {
    "pipeline_tag": "...",
    "library_name": "...",
    "architectures": ["..."],
    "num_labels": 2
  },
  "dataset_metadata_summary": {
    "splits": ["train", "validation", "test"],
    "features": ["text", "label"],
    "label_candidates": ["..."]
  }
}

Keep it concise. The most valuable fields are model_loading_hint, dataset_loading_hint,
label_mapping_hint, and known_pitfalls — these are what the ExecutionAgent needs most.""",
        tools=tools,
    )


def _make_execution_agent(llm_model: str, tools: List[Any]) -> Agent:
    return Agent(
        name="ExecutionAgent",
        model=llm_model,
        instructions="""You are an expert ML engineer writing HuggingFace evaluation scripts.

You receive a JSON hint from the PlanningAgent with loading snippets and known pitfalls.
Write a complete Python evaluation script and run it in Docker.

=== SETUP ===
1. If available, call get_dataset_loader(dataset_id) and get_model_loader(model_id)
   before coding. Prefer cached loaders when found=true.
2. Validate model/dataset assumptions against metadata from the plan.
3. Install deps at script TOP:
   ```python
   import subprocess, sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
       "transformers", "datasets", "evaluate", "huggingface_hub<1.0", "pyarrow", "fsspec",
       ...other_deps_from_plan...])
   ```
   CRITICAL: Pin huggingface_hub<1.0 — the container has 1.6.0 which breaks datasets/transformers.
4. Use model_loading_hint / dataset_loading_hint EXACTLY as described.
5. Apply label_mapping_hint — never leave model outputs as raw integer IDs.
6. Check known_pitfalls before writing — avoid those mistakes.
7. trust_remote_code=True in plan → pass to load_dataset() AND from_pretrained().

=== TASK-SPECIFIC IMPLEMENTATION PATTERNS ===

## Extractive QA (SQuAD v2)
- Use pipeline('question-answering', model=..., tokenizer=..., device=0).
- No-answer threshold: score < 0.2 → predict "". Gold = ex['answers']['text'] (list).
- Normalize: lowercase, strip punctuation, remove articles, collapse whitespace.
- ground_truth = first answer text or "" if unanswerable.

## NLI / Classification
- AutoTokenizer + AutoModelForSequenceClassification. Tokenize pairwise for NLI.
- ALWAYS check model.config.id2label — build explicit remapping to dataset label IDs.
- Filter out label == -1 before evaluation. For DeBERTa-v3: install sentencepiece.

## Multilabel (GoEmotions etc.)
- sigmoid(logits) > 0.5 (NOT softmax+argmax). Map labels by NAME not index.
- Exact-match accuracy: (pred_vector == gold_vector).all(dim=1).float().mean().
- GoEmotions: check if labels are individual 0/1 columns or list-of-IDs.

## Summarization
- AutoModelForSeq2SeqLM + model.generate(). Use generation_params from plan.
- Truncate inputs: max_length=1024. Install rouge-score. evaluate.load("rouge"), use_stemmer=True.
- batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).

## NER
- AutoModelForTokenClassification. NEVER use pipeline('ner').
- is_split_into_words=True. Align with word_ids(): first subword → label, rest → -100.
- evaluate.load("seqeval") → result["overall_f1"]. conll2003: trust_remote_code=True.

## Vision
- AutoImageProcessor + AutoModelForImageClassification. Grayscale → .convert("RGB").
- CIFAR-100: field "img", use fine_label. Food-101: validation split.

## Sentiment / Text Classification
- Check actual column names (e.g., "verse_text" not "text" for poem_sentiment).

=== GENERAL REQUIREMENTS ===
8. Use GPU: model.to("cuda"), inputs.to("cuda"). Batched inference (batch_size >= 8).
9. Sampling: ds.shuffle(seed=42).select(range(min(1000, len(ds)))).
10. Save results.json: {"<metric>": <float>}
11. Save predictions.json: [{"input":..., "prediction":..., "ground_truth":...}]
12. Print first 5 (prediction, ground_truth) pairs for debugging.
13. ALWAYS check model.config.id2label/label2id. Never assume identity mapping.

=== ON FAILURE ===
- Exit code != 0 → read the error carefully, fix the specific issue, retry.
- "Dataset scripts are no longer supported" → trust_remote_code=True.
- "HfFolder" or "is_offline_mode" ImportError → pip install -U huggingface_hub<1.0.
- "BuilderConfig 'default' not found" → use the available config name.
- GPU OOM → reduce batch size or fall back to CPU.
- Don't repeat the same mistake.""",
        tools=tools,
    )


def _make_validation_agent(llm_model: str) -> Agent:
    return Agent(
        name="ValidationAgent",
        model=llm_model,
        instructions="""You are a result quality inspector.

If the prompt already contains RAW_VALIDATION_JSON, reason from it directly.
Otherwise call check_evaluation_results(output_dir, metric) and analyse the JSON it returns.

Checks performed automatically:
  - results.json and predictions.json exist and are valid
  - Predictions are not degenerate (all identical)
  - No null predictions
  - Metric is not suspiciously 0.0
  - Label type mismatch (int IDs vs label strings)
  - Cross-check: recomputed accuracy vs reported value

If "valid" is True → reply with exactly:
  VALID: <metric>=<value> (n=<n_predictions> predictions)

If "valid" is False → reply with exactly this structure:
  INVALID: <one-line summary of main issue>
  SUGGESTIONS:
    1. <specific fix — include what loading hint or label mapping to correct>
    2. <specific fix>
  REPLAN_NEEDED: true|false
    (true  = wrong model/dataset loading strategy; PlanningAgent should search again)
    (false = code bug that ExecutionAgent can fix directly)

Be concise and actionable.""",
        tools=[check_evaluation_results, read_file],
    )


# ─────────────────────────── Orchestrator ────────────────────────────────────


class MultiAgentEvaluationCoder:
    """
    Multi-agent EvaluationCoder: Planning → Execution → Validation loop.

    Same external interface as OpenAIEvaluationCoder / EvaluationCoder.
    """

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
    ):
        self.mode = mode
        self.llm_model = llm_model
        self.gpu_id = gpu_id
        self.max_retry_rounds = max_retry_rounds
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        _set_gpu_id(gpu_id)
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        config = self.MODE_CONFIG[mode]
        self.max_exec_turns = max_steps or max_exec_turns or config["max_exec_turns"]

        dataset_loader_tool = _create_dataset_loader_tool(
            Path(dataset_loaders_dir) if dataset_loaders_dir
            else Path(__file__).parent.parent / "scripts" / "dataset_loaders"
        )
        model_loader_tool = _create_model_loader_tool(
            Path(model_loaders_dir) if model_loaders_dir
            else Path(__file__).parent.parent / "scripts" / "model_loaders"
        )
        # Use GPT-5.2 multiagent results as the knowledge base for similar evaluations
        default_results_dir = str(
            Path(__file__).parent.parent / "scripts" / "multiagent_results_v3_hard_gpt-5.2_multiturn_metadatatool"
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
        planning_tools = [tool_map[name] for name in config["planning_tools"]]
        execution_tools = [tool_map[name] for name in config["execution_tools"]]
        self.planning_tool_names = list(config["planning_tools"])
        self.execution_tool_names = list(config["execution_tools"])

        self.planning_agent = _make_planning_agent(llm_model, planning_tools)
        self.execution_agent = _make_execution_agent(llm_model, execution_tools)
        self.validation_agent = _make_validation_agent(llm_model)

        print(f"🔧 MultiAgentEvaluationCoder initialized:")
        print(f"   Mode: {mode.value}")
        print(f"   LLM: {llm_model} | GPU: {gpu_id}")
        print(f"   Max exec turns: {self.max_exec_turns} | Max retry rounds: {max_retry_rounds}")
        print(f"   Planning tools: {self.planning_tool_names}")
        print(f"   Execution tools: {self.execution_tool_names}")

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        max_samples: int = 200,
    ) -> Dict[str, Any]:
        return asyncio.run(self._evaluate_async(
            model_name=model_name,
            dataset_name=dataset_name,
            metric=metric,
            output_dir=output_dir,
            max_samples=max_samples,
        ))

    # ── internal async pipeline ───────────────────────────────────────────────

    @staticmethod
    def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON robustly from raw model output text."""
        if not text:
            return None
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        # Fallback: extract the largest {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_plan(plan_text: str) -> Dict[str, Any]:
        """Normalize planner output into a stable schema with defaults."""
        parsed_obj = MultiAgentEvaluationCoder._safe_json_parse(plan_text)
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

    @staticmethod
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

    @staticmethod
    async def _run_agent(agent: Agent, prompt: str, max_turns: int) -> Tuple[Any, Dict[str, int], float]:
        """Run one agent call and return (result, usage, elapsed_seconds)."""
        import time
        t0 = time.time()
        result = await Runner.run(agent, prompt, max_turns=max_turns)
        elapsed = time.time() - t0
        usage = MultiAgentEvaluationCoder._extract_usage(result)
        return result, usage, elapsed

    @staticmethod
    def _parse_validation_output(text: str) -> Dict[str, Any]:
        """Extract validity, replan flag, and suggestions from validator text output."""
        output = (text or "").strip()
        valid = output.startswith("VALID")
        replan_needed: Optional[bool] = None
        if not valid:
            m = re.search(r"REPLAN_NEEDED:\s*([A-Za-z]+)", output, flags=re.IGNORECASE)
            if m:
                flag = m.group(1).lower()
                if flag.startswith("t"):
                    replan_needed = True
                elif flag.startswith("f"):
                    replan_needed = False
        summary = ""
        m_sum = re.search(r"INVALID:\s*(.+)", output)
        if m_sum:
            summary = m_sum.group(1).strip()
        suggestions = re.findall(r"^\s*\d+\.\s*(.+)$", output, flags=re.MULTILINE)
        return {
            "valid": valid,
            "replan_needed": replan_needed,
            "summary": summary,
            "suggestions": suggestions,
        }

    @staticmethod
    def _classify_error_type(summary: str) -> str:
        s = (summary or "").lower()
        if "degenerate" in s:
            return "degenerate_predictions"
        if "label" in s and "mapping" in s:
            return "label_mapping"
        if "metric" in s and "mismatch" in s:
            return "metric_mismatch"
        if "missing" in s:
            return "missing_outputs"
        return "unknown"

    @staticmethod
    def _file_fingerprint(path: str) -> Optional[Tuple[int, int]]:
        """Return a stable file fingerprint based on mtime and size."""
        try:
            stat = os.stat(path)
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    @staticmethod
    def _infer_replan_needed(
        raw_validation: Dict[str, Any],
        parsed_validation: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Prefer explicit validator flag; otherwise infer from issue keywords."""
        if parsed_validation and parsed_validation.get("replan_needed") is not None:
            return bool(parsed_validation["replan_needed"])

        parts: List[str] = []
        if parsed_validation:
            parts.append(parsed_validation.get("summary", ""))
            parts.extend(parsed_validation.get("suggestions", []) or [])
        parts.extend(raw_validation.get("issues", []) or [])
        parts.extend(raw_validation.get("suggestions", []) or [])
        text = " ".join(parts).lower()
        replan_keywords = (
            "wrong model type",
            "correct model class",
            "label mapping",
            "id2label",
            "trust_remote_code",
            "raw label ids",
            "label type mismatch",
            "degenerate predictions",
            "output is not varying",
            "preprocessing is incorrect",
            "all inputs look the same",
        )
        return any(keyword in text for keyword in replan_keywords)

    @staticmethod
    def _format_validation_output(
        *,
        valid: bool,
        metric: str,
        metric_value: Optional[float] = None,
        n_predictions: int = 0,
        summary: str = "",
        suggestions: Optional[List[str]] = None,
        replan_needed: bool = False,
    ) -> str:
        """Normalize validation logs into one stable text shape."""
        if valid:
            return f"VALID: {metric}={metric_value} (n={n_predictions} predictions)"

        lines = [f"INVALID: {summary or 'validation failed'}", "SUGGESTIONS:"]
        final_suggestions = [s for s in (suggestions or []) if s][:2]
        if not final_suggestions:
            final_suggestions = ["Inspect run.log and fix the evaluation script before retrying."]
        for idx, suggestion in enumerate(final_suggestions, start=1):
            lines.append(f"  {idx}. {suggestion}")
        lines.append(f"REPLAN_NEEDED: {'true' if replan_needed else 'false'}")
        return "\n".join(lines)

    def _bootstrap_dependencies(self, output_dir: str, pip_deps: List[str]) -> Tuple[bool, str]:
        """
        No-op bootstrap: each docker invocation is an ephemeral container, so a separate
        dependency-only run does not persist installed packages to later evaluation runs.
        Running it also risks overwriting run_eval.py with a non-evaluation script.

        Returns (ok, message).
        """
        deps = [str(d).strip() for d in (pip_deps or []) if str(d).strip()]
        if not deps:
            return True, "No pip_deps in plan"
        return True, (
            "Skipped bootstrap for pip_deps="
            f"{deps}: evaluation runs use fresh `docker run --rm` containers, so "
            "dependencies must be installed inside the final evaluation script."
        )

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
        print(f"🤖 [MultiAgent] {model_name} | {dataset_name} | {metric}")
        print(f"📁 Output: {output_dir}")
        print(f"{'='*80}\n")

        sample_note = (
            f"Randomly sample up to {max_samples} examples for large datasets."
            if max_samples > 0
            else "Use ALL samples (no limit)."
        )

        validation_feedback = ""
        plan_str = None
        replan_needed = True
        failure_memory: Dict[str, Any] = {}
        bootstrapped_deps: set[str] = set()

        # Stats tracking
        rounds_stats: List[Dict[str, Any]] = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        import time
        t_start = time.time()

        final_round = 0
        for round_idx in range(self.max_retry_rounds):
            final_round = round_idx + 1
            print(f"\n── Round {round_idx + 1}/{self.max_retry_rounds} ──────────────────────")
            round_stat: Dict[str, Any] = {"round": round_idx + 1, "agents": {}}

            plan_usage = {"input": 0, "output": 0, "total": 0}
            plan_elapsed = 0.0
            if replan_needed or not plan_str:
                # ── STEP 1: Plan ──────────────────────────────────────────────
                plan_prompt = self._build_plan_prompt(
                    model_name, dataset_name, metric, sample_note, validation_feedback
                )
                print("📋 [PlanningAgent] Analysing model/dataset …")
                plan_result, plan_usage, plan_elapsed = await self._run_agent(
                    self.planning_agent, plan_prompt, max_turns=8
                )
                raw_plan_str = str(plan_result.final_output or "{}")
                plan_data = self._normalize_plan(raw_plan_str)
                plan_str = json.dumps(plan_data, ensure_ascii=False, indent=2)
                print(f"   Plan(parse_ok={plan_data.get('_plan_parse_ok', False)}): {plan_str[:400]}")
                print(f"   [tokens] in={plan_usage['input']} out={plan_usage['output']} | {plan_elapsed:.1f}s")
                planned_deps = [str(d).strip() for d in plan_data.get("pip_deps", []) if str(d).strip()]
                new_deps = [d for d in planned_deps if d not in bootstrapped_deps]
                if new_deps:
                    ok, msg = self._bootstrap_dependencies(output_dir, new_deps)
                    if ok:
                        bootstrapped_deps.update(new_deps)
                        print(f"   ℹ️  {msg}")
                    else:
                        print(f"   ⚠️  Bootstrap deps failed for {new_deps}: {msg[:200]}")
            else:
                print("📋 [PlanningAgent] Skipped (REPLAN_NEEDED=false); reusing previous plan.")
            round_stat["agents"]["planning"] = {**plan_usage, "elapsed_s": round(plan_elapsed, 1)}

            # ── STEP 2: Execute ───────────────────────────────────────────────
            script_path = os.path.join(output_dir, "run_eval.py")
            results_path = os.path.join(output_dir, "results.json")
            predictions_path = os.path.join(output_dir, "predictions.json")
            fingerprints_before = {
                "script": self._file_fingerprint(script_path),
                "results": self._file_fingerprint(results_path),
                "predictions": self._file_fingerprint(predictions_path),
            }
            exec_prompt = self._build_exec_prompt(
                model_name, dataset_name, metric, output_dir, sample_note, plan_str, failure_memory
            )
            print("⚙️  [ExecutionAgent] Writing and running code …")
            exec_result, exec_usage, exec_elapsed = await self._run_agent(
                self.execution_agent, exec_prompt, max_turns=self.max_exec_turns
            )
            fingerprints_after = {
                "script": self._file_fingerprint(script_path),
                "results": self._file_fingerprint(results_path),
                "predictions": self._file_fingerprint(predictions_path),
            }
            script_changed = (
                fingerprints_after["script"] is not None
                and fingerprints_after["script"] != fingerprints_before["script"]
            )
            outputs_changed = any(
                fingerprints_after[name] != fingerprints_before[name]
                for name in ("results", "predictions")
            )
            round_stat["agents"]["execution"] = {**exec_usage, "elapsed_s": round(exec_elapsed, 1)}
            print(f"   Execution output: {str(exec_result.final_output)[:200]}")
            print(f"   [tokens] in={exec_usage['input']} out={exec_usage['output']} | {exec_elapsed:.1f}s")

            # ── STEP 3: Legitimacy gate + Validate ───────────────────────────
            val_output = ""
            val_usage = {"input": 0, "output": 0, "total": 0}
            val_elapsed = 0.0
            if not script_changed and not outputs_changed:
                val_output = self._format_validation_output(
                    valid=False,
                    metric=metric,
                    summary="Execution did not update run_eval.py, results.json, or predictions.json",
                    suggestions=[
                        "Ensure the ExecutionAgent calls run_code_in_docker with actual Python evaluation code.",
                        "If the current plan is correct, patch the script-writing logic instead of reusing the same failed response.",
                    ],
                    replan_needed=False,
                )
                print("⚠️  [ExecutionCheck] No evaluation artifacts changed in this round.")
            elif script_changed and os.path.exists(script_path):
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        code_text = f.read()
                    legit = tool_check_code_legitimacy(code_text)
                    if legit.get("is_cheating") and float(legit.get("confidence", 0)) >= 0.5:
                        reason = legit.get("reason", "Code legitimacy check failed")
                        val_output = self._format_validation_output(
                            valid=False,
                            metric=metric,
                            summary=f"legitimacy check failed ({reason})",
                            suggestions=[
                                "Ensure real model forward pass is used for every prediction.",
                                "Do not use shortcut labels or dataset leakage.",
                            ],
                            replan_needed=False,
                        )
                        print(f"🚫 [LegitimacyGate] {reason}")
                except Exception as e:
                    print(f"⚠️  [LegitimacyGate] Failed to run: {e}")
            elif os.path.exists(script_path):
                print("ℹ️  [LegitimacyGate] Skipped because run_eval.py was not updated this round.")

            if not val_output:
                raw_validation = _inspect_evaluation_results(output_dir=output_dir, metric=metric)
                if raw_validation.get("valid"):
                    val_output = self._format_validation_output(
                        valid=True,
                        metric=metric,
                        metric_value=raw_validation.get("metric_value"),
                        n_predictions=int(raw_validation.get("n_predictions", 0) or 0),
                    )
                else:
                    parsed_validation = None
                    val_prompt = (
                        f"Validate the evaluation results in: {output_dir}\n"
                        f"Expected metric: {metric}\n"
                        f"RAW_VALIDATION_JSON:\n{json.dumps(raw_validation, ensure_ascii=False, indent=2)}"
                    )
                    print("🔍 [ValidationAgent] Checking results and predictions …")
                    try:
                        val_result, val_usage, val_elapsed = await self._run_agent(
                            self.validation_agent, val_prompt, max_turns=4
                        )
                        parsed_validation = self._parse_validation_output(str(val_result.final_output))
                    except Exception as e:
                        print(f"⚠️  [ValidationAgent] Failed to run: {e}")

                    summary = (
                        (parsed_validation or {}).get("summary")
                        or (raw_validation.get("issues") or ["Validation failed"])[0]
                    )
                    suggestions = (
                        (parsed_validation or {}).get("suggestions")
                        or raw_validation.get("suggestions")
                        or []
                    )
                    replan_needed = self._infer_replan_needed(raw_validation, parsed_validation)
                    val_output = self._format_validation_output(
                        valid=False,
                        metric=metric,
                        summary=summary,
                        suggestions=suggestions,
                        replan_needed=replan_needed,
                    )
            round_stat["agents"]["validation"] = {**val_usage, "elapsed_s": round(val_elapsed, 1)}

            # Round totals (planning + execution + validation)
            round_in  = plan_usage["input"]  + exec_usage["input"]  + val_usage["input"]
            round_out = plan_usage["output"] + exec_usage["output"] + val_usage["output"]
            round_stat["round_tokens"] = {"input": round_in, "output": round_out, "total": round_in + round_out}
            rounds_stats.append(round_stat)

            # Accumulate global totals
            for k in ("input", "output", "total"):
                total_tokens[k] += round_stat["round_tokens"].get(k, 0)

            print(f"   Validation: {val_output[:300]}")
            print(f"   [tokens] in={val_usage['input']} out={val_usage['output']} | {val_elapsed:.1f}s")
            print(f"   [round {round_idx+1} total tokens] in={round_in} out={round_out} total={round_in+round_out}")

            val_info = self._parse_validation_output(val_output)
            resolved_replan_needed = (
                False if val_info.get("valid") else (
                    bool(val_info["replan_needed"]) if val_info.get("replan_needed") is not None else True
                )
            )
            round_stat["validation_decision"] = {
                "valid": bool(val_info.get("valid", False)),
                "replan_needed": resolved_replan_needed,
                "summary": val_info.get("summary", ""),
                "suggestions": val_info.get("suggestions", []),
            }
            if val_info["valid"]:
                print("✅ Validation passed!")
                break
            # Extract feedback for next round
            validation_feedback = val_output
            replan_needed = resolved_replan_needed
            last_code_hash = None
            try:
                if os.path.exists(script_path):
                    with open(script_path, "rb") as f:
                        last_code_hash = hashlib.md5(f.read()).hexdigest()
            except Exception:
                pass
            failure_memory = {
                "last_error_type": self._classify_error_type(val_info.get("summary", "")),
                "last_validation_summary": val_info.get("summary", ""),
                "last_suggestions": val_info.get("suggestions", []),
                "last_replan_needed": replan_needed,
                "last_execution_output": str(exec_result.final_output)[:400],
                "last_code_hash": last_code_hash,
            }
            if replan_needed:
                print(f"⚠️  Validation failed (round {round_idx+1}), re-planning …")
            else:
                print(f"⚠️  Validation failed (round {round_idx+1}), patching with existing plan …")

        total_elapsed = time.time() - t_start
        usage_summary = {
            "rounds_used": final_round,
            "max_rounds": self.max_retry_rounds,
            "total_elapsed_s": round(total_elapsed, 1),
            "total_tokens": total_tokens,
            "rounds": rounds_stats,
            "final_validation_decision": (
                rounds_stats[-1].get("validation_decision", {}) if rounds_stats else {}
            ),
        }
        print(f"\n📊 Usage summary: {final_round} round(s) | "
              f"total tokens in={total_tokens['input']} out={total_tokens['output']} "
              f"| {total_elapsed:.1f}s elapsed")

        # ── Collect final results ─────────────────────────────────────────────
        return self._collect_results(
            output_dir, model_name, dataset_name, metric, plan_str, usage_summary
        )

    # ── prompt builders ───────────────────────────────────────────────────────

    def _build_plan_prompt(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        sample_note: str,
        validation_feedback: str,
    ) -> str:
        prompt = (
            f"Plan how to evaluate HuggingFace model `{model_name}` "
            f"on dataset `{dataset_name}` using metric `{metric}`.\n"
            f"Sampling: {sample_note}\n\n"
            f"Output ONLY a JSON plan with loading hints and known pitfalls."
        )
        if validation_feedback:
            prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED.\n"
                f"Validation feedback:\n{validation_feedback}\n\n"
                f"Revise the plan to fix the reported issues. "
                f"Pay attention to the SUGGESTIONS from the feedback and update the "
                f"label_mapping_hint or loading hints accordingly."
            )
        return prompt

    def _build_exec_prompt(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        output_dir: str,
        sample_note: str,
        plan_str: str,
        failure_memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        failure_block = ""
        if failure_memory:
            failure_block = (
                "\nPREVIOUS FAILURE MEMORY (use this to avoid repeating mistakes):\n"
                f"{json.dumps(failure_memory, ensure_ascii=False, indent=2)}\n"
            )
        return f"""Evaluate model `{model_name}` on dataset `{dataset_name}` (metric: `{metric}`).

PLANNING HINTS:
{plan_str}
{failure_block}

FIXED REQUIREMENTS:
- {sample_note}
- Before coding, verify model/dataset assumptions using plan metadata summaries.
  If those summaries are missing/ambiguous, call get_model_metadata("{model_name}")
  and get_dataset_metadata("{dataset_name}") before writing code.
- The Python script MUST save results directly to `results.json` (i.e. in the current working directory). Format: {{"{metric}": <float>}}
- The Python script MUST save predictions directly to `predictions.json` (i.e. in the current working directory). Format: list of {{"input":..., "prediction":..., "ground_truth":...}}
- CRITICAL: When calling the `run_code_in_docker` tool, you MUST pass exactly `output_dir="{output_dir}"`. Do NOT use or hardcode this path inside your Python script! The script runs inside a Docker container where the current directory is already mapped to the output dir.
"""

    # ── result collection ─────────────────────────────────────────────────────

    def _collect_results(
        self,
        output_dir: str,
        model_name: str,
        dataset_name: str,
        metric: str,
        plan_str: Optional[str],
        usage_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results_path = os.path.join(output_dir, "results.json")
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metric": metric,
            "backend": "multiagent",
            "mode": self.mode.value,
            "llm_model": self.llm_model,
            "max_retry_rounds": self.max_retry_rounds,
            "max_exec_turns": self.max_exec_turns,
            "planning_tools": self.planning_tool_names,
            "execution_tools": self.execution_tool_names,
            "plan": plan_str,
            "usage": usage_summary,   # rounds, tokens, elapsed
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

    # ── class method for compatibility ────────────────────────────────────────

    @classmethod
    def from_mode_string(cls, mode_str: str, **kwargs) -> "MultiAgentEvaluationCoder":
        """Create MultiAgentEvaluationCoder from mode string."""
        mode_map = {m.value: m for m in CoderMode}
        if mode_str not in mode_map:
            raise ValueError(f"Invalid mode: {mode_str}. Valid modes: {list(mode_map.keys())}")
        kwargs.pop("temperature", None)
        kwargs["mode"] = mode_map[mode_str]
        return cls(**kwargs)
