#!/usr/bin/env python3
"""
Analyze gaps between predicted results and ground truth.
Read run.log and results.json to understand why there are discrepancies.
"""

import os
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

# Set up LiteLLM
import litellm
litellm.drop_params = True
os.environ["LITELLM_DROP_PARAMS"] = "true"

SCRIPTS_DIR = Path(__file__).parent

MODE_DIRS = {
    "oneturn_onetool": "smolagent_results_v3_hard_oneturn_onetool",
    "multiturn_onetool": "smolagent_results_v3_hard_multiturn_onetool",
    "multiturn_metadatatool": "smolagent_results_v3_hard_multiturn_metadatatool",
    "multiturn_cachefiletool": "smolagent_results_v3_hard_multiturn_cachefiletool",
}

GROUND_TRUTH_FILE = SCRIPTS_DIR / "perfect_model_dataset_metrics_v3_0120_coding_agent_filtered_hard_both_successful.json"

# Gap categories
GAP_CATEGORIES = {
    "sampling_difference": "Different sampling strategy (random seed, sample size)",
    "preprocessing_difference": "Different data preprocessing or tokenization",
    "evaluation_metric": "Different metric calculation method",
    "label_mapping": "Incorrect label mapping or class order",
    "model_loading": "Model loaded incorrectly (wrong weights, config)",
    "dataset_split": "Using different dataset split",
    "batch_processing": "Batching or padding issues",
    "numerical_precision": "Floating point precision differences",
    "incomplete_evaluation": "Evaluation didn't complete on all samples",
    "wrong_task": "Evaluated on wrong task type",
    "cheating": "Result is hardcoded or not from actual inference",
    "unknown": "Cannot determine the cause",
}

ANALYSIS_PROMPT = """You are an expert ML engineer analyzing why an evaluation result differs from ground truth.

Task: Evaluate model `{model_id}` on dataset `{dataset_id}` for metric `{metric}`.

Ground truth value: {gt_value}
Predicted value: {pred_value}
Absolute error: {abs_error:.4f}
Relative error: {rel_error:.1%}

Here is the evaluation code that was run:
```python
{eval_code}
```

Here are the last parts of the run log:
```
{run_log}
```

Analyze why there might be a gap between the predicted and ground truth values.
Classify the likely cause into ONE of these categories:
{categories}

Respond in JSON format:
{{
    "category": "<category_key>",
    "reason": "<detailed explanation of why the gap exists>",
    "confidence": "<high/medium/low>",
    "fix_suggestion": "<how to potentially fix this to match ground truth>"
}}

Only output the JSON, nothing else.
"""


def load_ground_truth() -> dict:
    """Load ground truth metrics."""
    if not GROUND_TRUTH_FILE.exists():
        print(f"⚠️ Ground truth file not found: {GROUND_TRUTH_FILE}")
        return {}
    
    with open(GROUND_TRUTH_FILE) as f:
        data = json.load(f)
    
    gt = {}
    for item in data.get("results", []):
        model_id = item.get("model_id", "")
        dataset_id = item.get("dataset_id", "")
        metrics = item.get("metrics", {})
        gt[(model_id, dataset_id)] = metrics
    
    return gt


def get_evaluations_with_gaps(mode_dir: Path, ground_truth: dict, 
                               min_error: float = 0.05, max_samples: int = 0) -> list:
    """Get evaluations with significant gaps from ground truth."""
    gaps = []
    
    for item in sorted(mode_dir.iterdir()):
        if not item.is_dir() or item.name.startswith('.'):
            continue
        
        metadata_file = item / "metadata.json"
        results_file = item / "results.json"
        run_log_file = item / "run.log"
        eval_script = item / "run_eval.py"
        
        if not metadata_file.exists() or not results_file.exists():
            continue
        
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            with open(results_file) as f:
                results = json.load(f)
            
            model_id = metadata.get("model_name", "")
            dataset_id = metadata.get("dataset_name", "")
            metric = metadata.get("metric", "")
            
            if not model_id or not dataset_id or not metric:
                continue
            
            key = (model_id, dataset_id)
            if key not in ground_truth:
                continue
            
            gt_metrics = ground_truth[key]
            if metric not in gt_metrics:
                continue
            
            gt_value = float(gt_metrics[metric])
            pred_value = results.get(metric)
            
            if pred_value is None:
                continue
            
            pred_value = float(pred_value)
            abs_error = abs(pred_value - gt_value)
            rel_error = abs_error / abs(gt_value) if gt_value != 0 else abs_error
            
            if abs_error < min_error:
                continue
            
            # Read run log
            run_log = ""
            if run_log_file.exists():
                try:
                    run_log = run_log_file.read_text(encoding="utf-8", errors="ignore")
                except:
                    pass
            
            # Read eval script
            eval_code = ""
            if eval_script.exists():
                try:
                    eval_code = eval_script.read_text(encoding="utf-8")
                except:
                    pass
            
            gaps.append({
                "name": item.name,
                "path": str(item),
                "model_id": model_id,
                "dataset_id": dataset_id,
                "metric": metric,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "run_log": run_log,
                "eval_code": eval_code,
            })
            
            if max_samples > 0 and len(gaps) >= max_samples:
                break
                
        except Exception as e:
            continue
    
    # Sort by absolute error (descending)
    gaps.sort(key=lambda x: -x["abs_error"])
    return gaps


def analyze_with_llm(gap_info: dict, llm_model: str = "gpt-4o-mini") -> dict:
    """Use LLM to analyze a gap."""
    categories_str = "\n".join([f"- {k}: {v}" for k, v in GAP_CATEGORIES.items()])
    
    # Truncate logs for LLM
    run_log = gap_info["run_log"]
    if len(run_log) > 8000:
        run_log = run_log[:3000] + "\n\n... [truncated] ...\n\n" + run_log[-3000:]
    
    eval_code = gap_info["eval_code"]
    if len(eval_code) > 4000:
        eval_code = eval_code[:4000] + "\n# ... [truncated] ..."
    
    prompt = ANALYSIS_PROMPT.format(
        model_id=gap_info["model_id"],
        dataset_id=gap_info["dataset_id"],
        metric=gap_info["metric"],
        gt_value=gap_info["gt_value"],
        pred_value=gap_info["pred_value"],
        abs_error=gap_info["abs_error"],
        rel_error=gap_info["rel_error"],
        eval_code=eval_code if eval_code else "(not available)",
        run_log=run_log if run_log else "(not available)",
        categories=categories_str,
    )
    
    try:
        response = litellm.completion(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        result = json.loads(content)
        return {
            "category": result.get("category", "unknown"),
            "reason": result.get("reason", ""),
            "confidence": result.get("confidence", "low"),
            "fix_suggestion": result.get("fix_suggestion", ""),
        }
    except Exception as e:
        return {
            "category": "unknown",
            "reason": f"LLM analysis failed: {str(e)}",
            "confidence": "low",
            "fix_suggestion": "",
        }


def analyze_without_llm(gap_info: dict) -> dict:
    """Quick rule-based analysis without LLM."""
    run_log = (gap_info.get("run_log", "") or "").lower()
    eval_code = (gap_info.get("eval_code", "") or "").lower()
    abs_error = gap_info.get("abs_error", 0)
    rel_error = gap_info.get("rel_error", 0)
    
    # Check for common patterns
    
    # Sampling issues
    if "sample" in run_log or "random" in run_log:
        if "1000" in run_log or "max_samples" in eval_code:
            return {
                "category": "sampling_difference",
                "reason": "Uses sampling (likely 1000 examples), may differ from full evaluation",
                "confidence": "medium",
                "fix_suggestion": "Run on full dataset or use same sampling seed",
            }
    
    # Label mapping issues
    if "label" in run_log and ("map" in run_log or "id2label" in run_log):
        return {
            "category": "label_mapping",
            "reason": "Possible label mapping mismatch",
            "confidence": "medium",
            "fix_suggestion": "Check id2label mapping matches dataset",
        }
    
    # Split issues
    if "split" in run_log:
        for split in ["train", "validation", "val", "test"]:
            if split in run_log:
                return {
                    "category": "dataset_split",
                    "reason": f"Using {split} split, may not match ground truth split",
                    "confidence": "medium",
                    "fix_suggestion": "Verify using same dataset split as ground truth",
                }
    
    # Very small error - likely numerical precision
    if abs_error < 0.01:
        return {
            "category": "numerical_precision",
            "reason": "Small difference likely due to floating point precision",
            "confidence": "high",
            "fix_suggestion": "Acceptable difference, no action needed",
        }
    
    # Very large error - might be wrong task or cheating
    if rel_error > 0.5:
        # Check for hardcoded values
        pred_str = str(gap_info.get("pred_value", ""))
        if pred_str in eval_code:
            return {
                "category": "cheating",
                "reason": "Predicted value appears hardcoded in code",
                "confidence": "high",
                "fix_suggestion": "Regenerate evaluation code",
            }
        
        return {
            "category": "wrong_task",
            "reason": "Very large error suggests fundamental evaluation issue",
            "confidence": "medium",
            "fix_suggestion": "Review if model/dataset/metric are correctly matched",
        }
    
    # Check for incomplete evaluation
    if "error" in run_log or "exception" in run_log or "failed" in run_log:
        return {
            "category": "incomplete_evaluation",
            "reason": "Errors during evaluation may have affected results",
            "confidence": "medium",
            "fix_suggestion": "Check logs for errors and rerun",
        }
    
    return {
        "category": "unknown",
        "reason": "Cannot determine cause from available information",
        "confidence": "low",
        "fix_suggestion": "Manual inspection of code and logs required",
    }


def print_gap_details(gap: dict, analysis: dict):
    """Print detailed gap information."""
    print(f"\n{'='*80}")
    print(f"📊 {gap['model_id']} | {gap['dataset_id']} | {gap['metric']}")
    print(f"{'='*80}")
    print(f"  Ground truth: {gap['gt_value']:.4f}")
    print(f"  Predicted:    {gap['pred_value']:.4f}")
    print(f"  Abs error:    {gap['abs_error']:.4f}")
    print(f"  Rel error:    {gap['rel_error']:.1%}")
    print(f"\n  Category:     {analysis['category']}")
    print(f"  Confidence:   {analysis.get('confidence', 'N/A')}")
    print(f"  Reason:       {analysis['reason']}")
    print(f"  Fix:          {analysis.get('fix_suggestion', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze gaps between results and ground truth")
    parser.add_argument("--mode", choices=list(MODE_DIRS.keys()) + ["all"], default="all",
                        help="Mode to analyze")
    parser.add_argument("--min-error", type=float, default=0.05,
                        help="Minimum absolute error to include (default: 0.05)")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM for detailed analysis")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of gaps to analyze per mode (0 = all)")
    parser.add_argument("--output", default="result_gap_analysis.json",
                        help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed gap information")
    args = parser.parse_args()
    
    print("Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    all_results = {}
    
    modes_to_analyze = [args.mode] if args.mode != "all" else list(MODE_DIRS.keys())
    
    for mode in modes_to_analyze:
        mode_dir = SCRIPTS_DIR / MODE_DIRS[mode]
        if not mode_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {mode}...")
        print(f"{'='*60}")
        
        gaps = get_evaluations_with_gaps(
            mode_dir, ground_truth, 
            min_error=args.min_error,
            max_samples=args.limit
        )
        
        print(f"Found {len(gaps)} evaluations with abs_error >= {args.min_error}")
        
        results = {
            "total_gaps": len(gaps),
            "by_category": defaultdict(list),
            "details": [],
        }
        
        for i, gap in enumerate(gaps):
            if args.verbose or args.llm:
                print(f"  [{i+1}/{len(gaps)}] Analyzing {gap['name'][:50]}...")
            
            if args.llm:
                analysis = analyze_with_llm(gap, args.llm_model)
            else:
                analysis = analyze_without_llm(gap)
            
            result = {
                "name": gap["name"],
                "model_id": gap["model_id"],
                "dataset_id": gap["dataset_id"],
                "metric": gap["metric"],
                "gt_value": gap["gt_value"],
                "pred_value": gap["pred_value"],
                "abs_error": gap["abs_error"],
                "rel_error": gap["rel_error"],
                **analysis,
            }
            
            results["details"].append(result)
            results["by_category"][analysis["category"]].append({
                "name": gap["name"],
                "gt_value": gap["gt_value"],
                "pred_value": gap["pred_value"],
                "abs_error": gap["abs_error"],
                "reason": analysis["reason"][:100],
            })
            
            if args.verbose:
                print_gap_details(gap, analysis)
        
        results["by_category"] = dict(results["by_category"])
        all_results[mode] = results
        
        # Print summary
        print(f"\n📊 {mode.upper()} GAP SUMMARY")
        print("-" * 40)
        if results["total_gaps"] > 0:
            for category, items in sorted(results["by_category"].items(), key=lambda x: -len(x[1])):
                pct = len(items) / results["total_gaps"] * 100
                print(f"  {category}: {len(items)} ({pct:.1f}%)")
                # Show first 3 examples with gt/pred values
                for item in items[:3]:
                    gt = item.get("gt_value", 0)
                    pred = item.get("pred_value", 0)
                    err = item.get("abs_error", 0)
                    name_short = item["name"][:40] + "..." if len(item["name"]) > 40 else item["name"]
                    print(f"    - {name_short}")
                    print(f"      GT: {gt:.4f} | Pred: {pred:.4f} | Err: {err:.4f}")
                if len(items) > 3:
                    print(f"    ... and {len(items) - 3} more")
        else:
            print("  No significant gaps found")
    
    # Save results
    output_path = SCRIPTS_DIR / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Results saved to: {output_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL GAP SUMMARY")
    print(f"{'='*60}")
    
    total_by_category = defaultdict(int)
    for mode, results in all_results.items():
        for category, items in results.get("by_category", {}).items():
            total_by_category[category] += len(items)
    
    total = sum(total_by_category.values())
    if total > 0:
        print(f"\nTotal gaps analyzed: {total}")
        print(f"\nBy category:")
        for category, count in sorted(total_by_category.items(), key=lambda x: -x[1]):
            cat_desc = GAP_CATEGORIES.get(category, category)
            print(f"  {category}: {count} ({count/total*100:.1f}%)")
            print(f"    → {cat_desc}")


if __name__ == "__main__":
    main()
