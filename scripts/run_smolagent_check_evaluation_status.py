#!/usr/bin/env python3
"""
Check evaluation status for all four modes.
Shows how many evaluations ran and how many succeeded for each mode.
Also computes MAE (Mean Absolute Error) against ground truth metrics.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# Base directory for results
SCRIPTS_DIR = Path(__file__).parent

# Four mode directories
MODE_DIRS = {
    "oneturn_onetool": "smolagent_results_v3_hard_oneturn_onetool",
    "multiturn_onetool": "smolagent_results_v3_hard_multiturn_onetool",
    "multiturn_metadatatool": "smolagent_results_v3_hard_multiturn_metadatatool",
    "multiturn_cachefiletool": "smolagent_results_v3_hard_multiturn_cachefiletool",
}

# Ground truth file
GROUND_TRUTH_FILE = SCRIPTS_DIR / "perfect_model_dataset_metrics_v3_0120_coding_agent_filtered_hard_both_successful.json"


def load_ground_truth() -> dict:
    """Load ground truth metrics from JSON file.
    
    Returns a dict mapping (model_id, dataset_id) -> {metric_name: value}
    """
    ground_truth = {}
    if not GROUND_TRUTH_FILE.exists():
        print(f"⚠️ Ground truth file not found: {GROUND_TRUTH_FILE}")
        return ground_truth
    
    with open(GROUND_TRUTH_FILE) as f:
        data = json.load(f)
    
    for item in data.get("results", []):
        model_id = item.get("model_id", "")
        dataset_id = item.get("dataset_id", "")
        metrics = item.get("metrics", {})
        key = (model_id, dataset_id)
        ground_truth[key] = metrics
    
    return ground_truth


def load_metadata(eval_dir: Path) -> dict:
    """Load metadata.json for an evaluation directory."""
    metadata_file = eval_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file) as f:
            return json.load(f)
    except Exception:
        return None


def check_evaluation_dir(eval_dir: Path, ground_truth: dict = None) -> dict:
    """Check a single evaluation directory and return status."""
    status = {
        "exists": eval_dir.exists(),
        "has_results": False,
        "has_predictions": False,
        "has_metadata": False,
        "has_run_log": False,
        "results": None,
        "metadata": None,
        "error": None,
        "predicted_value": None,
        "ground_truth_value": None,
        "absolute_error": None,
        "signed_error": None,  # pred - gt (positive = pred higher, negative = pred lower)
        "model_id": None,
        "dataset_id": None,
        "metric_name": None,
        "meta_mode": None,
        "meta_llm_model": None,
        "meta_max_steps_config": None,
        "meta_actual_steps": None,
        "meta_step_number": None,
        "meta_token_input": None,
        "meta_token_output": None,
        "meta_token_total": None,
        "meta_tool_calls": None,
        "meta_tool_calls_total": None,
        "meta_error_steps": None,
        "meta_step_details_count": None,
    }
    
    if not eval_dir.exists():
        return status
    
    # Check for results.json
    results_file = eval_dir / "results.json"
    if results_file.exists():
        status["has_results"] = True
        try:
            with open(results_file) as f:
                status["results"] = json.load(f)
        except Exception as e:
            status["error"] = f"Failed to parse results.json: {e}"
    
    # Check for predictions.json
    predictions_file = eval_dir / "predictions.json"
    status["has_predictions"] = predictions_file.exists()
    
    # Check for metadata.json
    metadata_file = eval_dir / "metadata.json"
    if metadata_file.exists():
        status["has_metadata"] = True
        metadata = load_metadata(eval_dir)
        if metadata:
            status["metadata"] = metadata
            status["model_id"] = metadata.get("model_name", "")
            status["dataset_id"] = metadata.get("dataset_name", "")
            status["metric_name"] = metadata.get("metric", "")
            status["meta_mode"] = metadata.get("mode")
            status["meta_llm_model"] = metadata.get("llm_model")
            status["meta_max_steps_config"] = metadata.get("max_steps_config")
            status["meta_actual_steps"] = metadata.get("actual_steps", metadata.get("step_number"))
            status["meta_step_number"] = metadata.get("step_number")
            
            token_usage = metadata.get("token_usage") or {}
            if isinstance(token_usage, dict):
                status["meta_token_input"] = token_usage.get("input_tokens")
                status["meta_token_output"] = token_usage.get("output_tokens")
                status["meta_token_total"] = token_usage.get("total_tokens")
            
            tool_calls = metadata.get("tool_calls") or {}
            if isinstance(tool_calls, dict):
                status["meta_tool_calls"] = tool_calls
            status["meta_tool_calls_total"] = metadata.get("tool_calls_total")
            
            step_details = metadata.get("step_details") or []
            if isinstance(step_details, list):
                status["meta_step_details_count"] = len(step_details)
                status["meta_error_steps"] = sum(
                    1 for step in step_details
                    if isinstance(step, dict) and step.get("has_error")
                )
        else:
            if status["error"]:
                status["error"] += "; Failed to parse metadata.json"
            else:
                status["error"] = "Failed to parse metadata.json"
    
    # Check for run.log
    run_log = eval_dir / "run.log"
    status["has_run_log"] = run_log.exists()
    
    # Compute absolute error if we have results and ground truth
    if status["has_results"] and status["results"] and ground_truth:
        model_id = status["model_id"]
        dataset_id = status["dataset_id"]
        metric_name = status["metric_name"]
        
        if model_id and dataset_id:
            key = (model_id, dataset_id)
            if key in ground_truth:
                gt_metrics = ground_truth[key]
                # Find matching metric
                if metric_name and metric_name in gt_metrics:
                    gt_value = gt_metrics[metric_name]
                    # Get predicted value from results
                    pred_value = status["results"].get(metric_name)
                    if pred_value is not None and gt_value is not None:
                        try:
                            pred_value = float(pred_value)
                            gt_value = float(gt_value)
                            status["predicted_value"] = pred_value
                            status["ground_truth_value"] = gt_value
                            status["absolute_error"] = abs(pred_value - gt_value)
                            status["signed_error"] = pred_value - gt_value  # positive = higher, negative = lower
                            # Compute relative error (percentage)
                            if gt_value != 0:
                                status["relative_error"] = abs(pred_value - gt_value) / abs(gt_value)
                            else:
                                # If ground truth is 0, use absolute error as relative
                                status["relative_error"] = abs(pred_value - gt_value)
                        except (ValueError, TypeError):
                            pass
    
    return status


def analyze_mode(mode_name: str, mode_dir_name: str, ground_truth: dict = None) -> dict:
    """Analyze all evaluations for a mode."""
    mode_dir = SCRIPTS_DIR / mode_dir_name
    
    stats = {
        "mode": mode_name,
        "dir": mode_dir_name,
        "exists": mode_dir.exists(),
        "total": 0,
        "total_with_id": 0,
        "total_no_id": 0,
        "success": 0,
        "success_with_id": 0,
        "has_predictions": 0,
        "has_metadata": 0,
        "failed": 0,
        "failed_with_id": 0,
        "not_started": 0,
        "pending_with_id": 0,
        "evaluations": [],
        # MAE statistics
        "mae_count": 0,  # number of evaluations with valid MAE
        "mae_sum": 0.0,  # sum of absolute errors
        "mae_values": [],  # list of all absolute errors
        # Signed error statistics (pred - gt)
        "signed_error_values": [],  # list of all signed errors
        # Relative error statistics
        "relative_error_values": [],  # list of all relative errors
        # Success threshold statistics (pred >= threshold * gt)
        "pred_gt_pairs": [],  # list of (pred, gt) tuples for threshold analysis
        # Ground truth coverage
        "gt_total": len(ground_truth) if ground_truth else 0,
        "gt_matched": 0,  # unique keys matched in ground truth
        "gt_missing": 0,
        "gt_matched_dirs": 0,  # dirs mapped to ground truth (includes duplicates)
        "gt_unexpected": 0,  # dirs with ids not in ground truth
        "gt_unknown": 0,  # dirs without ids
        "gt_duplicates": 0,  # dirs mapping to already seen key
        # Metadata aggregation (only for dirs with valid ids)
        "meta_count": 0,
        "meta_missing": 0,
        "meta_llm_models": defaultdict(int),
        "meta_max_steps_configs": defaultdict(int),
        "meta_actual_steps_values": [],
        "meta_token_total_values": [],
        "meta_tool_calls_total_values": [],
        "meta_error_steps_values": [],
        "meta_tool_calls": defaultdict(int),
    }
    
    if not mode_dir.exists():
        return stats
    
    matched_keys = set() if ground_truth else None
    
    def _add_numeric(values: list, value):
        if value is None:
            return
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return
    
    # List all subdirectories (each is an evaluation)
    for item in sorted(mode_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            stats["total"] += 1
            eval_status = check_evaluation_dir(item, ground_truth)
            has_ids = bool(eval_status["model_id"] and eval_status["dataset_id"])
            
            if eval_status["has_results"]:
                stats["success"] += 1
            elif eval_status["has_run_log"]:
                stats["failed"] += 1
            else:
                stats["not_started"] += 1
            
            if has_ids:
                stats["total_with_id"] += 1
                if eval_status["has_results"]:
                    stats["success_with_id"] += 1
                elif eval_status["has_run_log"]:
                    stats["failed_with_id"] += 1
                else:
                    stats["pending_with_id"] += 1
            else:
                stats["total_no_id"] += 1
            
            if eval_status["has_predictions"]:
                stats["has_predictions"] += 1
            
            if eval_status["has_metadata"]:
                stats["has_metadata"] += 1
                if has_ids:
                    stats["meta_count"] += 1
                    if eval_status["meta_llm_model"]:
                        stats["meta_llm_models"][eval_status["meta_llm_model"]] += 1
                    if eval_status["meta_max_steps_config"] is not None:
                        stats["meta_max_steps_configs"][str(eval_status["meta_max_steps_config"])] += 1
                    _add_numeric(stats["meta_actual_steps_values"], eval_status.get("meta_actual_steps"))
                    _add_numeric(stats["meta_token_total_values"], eval_status.get("meta_token_total"))
                    _add_numeric(stats["meta_tool_calls_total_values"], eval_status.get("meta_tool_calls_total"))
                    _add_numeric(stats["meta_error_steps_values"], eval_status.get("meta_error_steps"))
                    tool_calls = eval_status.get("meta_tool_calls") or {}
                    if isinstance(tool_calls, dict):
                        for tool_name, count in tool_calls.items():
                            try:
                                stats["meta_tool_calls"][tool_name] += int(count)
                            except (TypeError, ValueError):
                                continue
            
            # Ground truth coverage accounting
            if ground_truth:
                if eval_status["model_id"] and eval_status["dataset_id"]:
                    key = (eval_status["model_id"], eval_status["dataset_id"])
                    if key in ground_truth:
                        stats["gt_matched_dirs"] += 1
                        if key in matched_keys:
                            stats["gt_duplicates"] += 1
                        else:
                            matched_keys.add(key)
                    else:
                        stats["gt_unexpected"] += 1
                else:
                    stats["gt_unknown"] += 1
            
            # Collect MAE statistics
            if eval_status["absolute_error"] is not None:
                stats["mae_count"] += 1
                stats["mae_sum"] += eval_status["absolute_error"]
                stats["mae_values"].append(eval_status["absolute_error"])
            
            # Collect signed error statistics
            if eval_status.get("signed_error") is not None:
                stats["signed_error_values"].append(eval_status["signed_error"])
            
            # Collect relative error statistics
            if eval_status.get("relative_error") is not None:
                stats["relative_error_values"].append(eval_status["relative_error"])
            
            # Collect pred/gt pairs for threshold analysis
            if eval_status.get("predicted_value") is not None and eval_status.get("ground_truth_value") is not None:
                stats["pred_gt_pairs"].append((eval_status["predicted_value"], eval_status["ground_truth_value"]))
            
            stats["evaluations"].append({
                "name": item.name,
                **eval_status
            })
    
    if ground_truth:
        stats["gt_matched"] = len(matched_keys)
        stats["gt_missing"] = max(0, stats["gt_total"] - stats["gt_matched"])
    stats["meta_missing"] = max(0, stats["total_with_id"] - stats["meta_count"])
    
    return stats


def compute_mae_stats(stats: dict) -> dict:
    """Compute MAE statistics from collected values."""
    mae_values = stats.get("mae_values", [])
    if not mae_values:
        return {
            "mae": None,
            "mae_count": 0,
            "mae_std": None,
            "mae_min": None,
            "mae_max": None,
            "mae_median": None,
        }
    
    import statistics
    
    mae = statistics.mean(mae_values)
    mae_std = statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0
    mae_min = min(mae_values)
    mae_max = max(mae_values)
    mae_median = statistics.median(mae_values)
    
    return {
        "mae": mae,
        "mae_count": len(mae_values),
        "mae_std": mae_std,
        "mae_min": mae_min,
        "mae_max": mae_max,
        "mae_median": mae_median,
    }


def compute_signed_error_stats(stats: dict) -> dict:
    """Compute signed error statistics (pred - gt).
    
    Positive values mean prediction is higher than ground truth.
    Negative values mean prediction is lower than ground truth.
    """
    signed_errors = stats.get("signed_error_values", [])
    if not signed_errors:
        return {
            "me": None,  # Mean Error (signed)
            "me_count": 0,
            "me_std": None,
            "me_min": None,
            "me_max": None,
            "me_median": None,
            "pred_higher_count": 0,
            "pred_lower_count": 0,
            "pred_equal_count": 0,
            "pred_higher_avg": None,
            "pred_lower_avg": None,
        }
    
    import statistics
    
    me = statistics.mean(signed_errors)
    me_std = statistics.stdev(signed_errors) if len(signed_errors) > 1 else 0.0
    me_min = min(signed_errors)
    me_max = max(signed_errors)
    me_median = statistics.median(signed_errors)
    
    # Breakdown by direction
    higher = [e for e in signed_errors if e > 0.001]
    lower = [e for e in signed_errors if e < -0.001]
    equal = [e for e in signed_errors if abs(e) <= 0.001]
    
    return {
        "me": me,
        "me_count": len(signed_errors),
        "me_std": me_std,
        "me_min": me_min,
        "me_max": me_max,
        "me_median": me_median,
        "pred_higher_count": len(higher),
        "pred_lower_count": len(lower),
        "pred_equal_count": len(equal),
        "pred_higher_pct": len(higher) / len(signed_errors) * 100 if signed_errors else 0,
        "pred_lower_pct": len(lower) / len(signed_errors) * 100 if signed_errors else 0,
        "pred_higher_avg": statistics.mean(higher) if higher else None,
        "pred_lower_avg": statistics.mean(lower) if lower else None,
    }


def compute_relative_error_stats(stats: dict, threshold: float = 0.2) -> dict:
    """Compute relative error statistics.
    
    Args:
        stats: Mode statistics dict
        threshold: Relative error threshold (default 0.2 = 20%)
    
    Returns:
        Dict with within_threshold count, rate, and other stats
    """
    rel_errors = stats.get("relative_error_values", [])
    if not rel_errors:
        return {
            "rel_count": 0,
            "within_threshold": 0,
            "within_threshold_rate": None,
            "rel_mean": None,
            "rel_median": None,
        }
    
    import statistics
    
    within = sum(1 for e in rel_errors if e <= threshold)
    rate = within / len(rel_errors) if rel_errors else 0
    
    return {
        "rel_count": len(rel_errors),
        "within_threshold": within,
        "within_threshold_rate": rate,
        "rel_mean": statistics.mean(rel_errors),
        "rel_median": statistics.median(rel_errors),
    }


def compute_meta_stats(stats: dict) -> dict:
    """Compute metadata summary statistics."""
    import statistics
    
    def _avg(values):
        return statistics.mean(values) if values else None
    
    return {
        "meta_count": stats.get("meta_count", 0),
        "meta_missing": stats.get("meta_missing", 0),
        "avg_steps": _avg(stats.get("meta_actual_steps_values", [])),
        "avg_error_steps": _avg(stats.get("meta_error_steps_values", [])),
        "avg_token_total": _avg(stats.get("meta_token_total_values", [])),
        "avg_tool_calls_total": _avg(stats.get("meta_tool_calls_total_values", [])),
        "llm_models": dict(stats.get("meta_llm_models", {})),
        "max_steps_configs": dict(stats.get("meta_max_steps_configs", {})),
        "tool_calls": dict(stats.get("meta_tool_calls", {})),
    }


def print_metadata_summary(all_stats: list):
    """Print metadata summary (turns, tokens, tools, etc.)."""
    if not all_stats:
        return
    
    print("\n" + "=" * 120)
    print("🧾 METADATA SUMMARY")
    print("=" * 120)
    print(f"\n{'Mode':<30} {'Meta':>6} {'Missing':>7} {'AvgSteps':>9} {'MaxCfg':>7} {'AvgErr':>7} {'AvgTok':>9} {'AvgTool':>8} {'LLM':>12}")
    print("-" * 120)
    
    def _format_avg(value, width, decimals=1, integer=False):
        if value is None:
            return "N/A".rjust(width)
        if integer:
            return f"{int(round(value))}".rjust(width)
        return f"{value:.{decimals}f}".rjust(width)
    
    def _format_top(counter: dict, width: int):
        if not counter:
            return "N/A".rjust(width)
        top_value = max(counter.items(), key=lambda x: x[1])[0]
        others = len(counter) - 1
        value_str = str(top_value)
        if others > 0:
            value_str = f"{value_str}+{others}"
        if len(value_str) > width:
            value_str = value_str[:max(0, width - 3)] + "..."
        return value_str.rjust(width)
    
    for stats in all_stats:
        meta_stats = compute_meta_stats(stats)
        meta_count = meta_stats["meta_count"]
        meta_missing = meta_stats["meta_missing"]
        avg_steps = _format_avg(meta_stats["avg_steps"], 9, decimals=1)
        max_cfg = _format_top(meta_stats["max_steps_configs"], 7)
        avg_err = _format_avg(meta_stats["avg_error_steps"], 7, decimals=1)
        avg_tok = _format_avg(meta_stats["avg_token_total"], 9, integer=True)
        avg_tool = _format_avg(meta_stats["avg_tool_calls_total"], 8, decimals=1)
        llm_str = _format_top(meta_stats["llm_models"], 12)
        
        print(f"{stats['mode']:<30} {meta_count:>6} {meta_missing:>7} {avg_steps} {max_cfg} {avg_err} {avg_tok} {avg_tool} {llm_str}")
    
    print("Note: Metadata stats are computed for dirs with valid model_id/dataset_id.")


def print_tool_calls_summary(all_stats: list):
    """Print tool calls breakdown by mode."""
    if not all_stats:
        return
    
    # Collect all unique tool names
    all_tools = set()
    for stats in all_stats:
        tool_calls = stats.get("meta_tool_calls", {})
        if isinstance(tool_calls, dict):
            all_tools.update(tool_calls.keys())
    
    if not all_tools:
        return
    
    # Sort tools by total usage
    tool_totals = {}
    for tool in all_tools:
        total = sum(
            stats.get("meta_tool_calls", {}).get(tool, 0)
            for stats in all_stats
        )
        tool_totals[tool] = total
    sorted_tools = sorted(all_tools, key=lambda t: tool_totals[t], reverse=True)
    
    print("\n" + "=" * 120)
    print("🔧 TOOL CALLS BREAKDOWN")
    print("=" * 120)
    
    # Dynamic header based on tools
    header = f"{'Mode':<30}"
    for tool in sorted_tools:
        # Shorten tool name if needed
        short_name = tool[:12] if len(tool) > 12 else tool
        header += f" {short_name:>12}"
    header += f" {'Total':>10}"
    print(f"\n{header}")
    print("-" * 120)
    
    grand_total = 0
    tool_grand_totals = {tool: 0 for tool in sorted_tools}
    
    for stats in all_stats:
        tool_calls = stats.get("meta_tool_calls", {})
        if not isinstance(tool_calls, dict):
            tool_calls = {}
        
        row = f"{stats['mode']:<30}"
        mode_total = 0
        for tool in sorted_tools:
            count = tool_calls.get(tool, 0)
            row += f" {count:>12}"
            mode_total += count
            tool_grand_totals[tool] += count
        row += f" {mode_total:>10}"
        grand_total += mode_total
        print(row)
    
    # Total row
    print("-" * 120)
    total_row = f"{'TOTAL':<30}"
    for tool in sorted_tools:
        total_row += f" {tool_grand_totals[tool]:>12}"
    total_row += f" {grand_total:>10}"
    print(total_row)
    
    # Average per successful evaluation
    print("\n" + "-" * 120)
    print("Average per successful evaluation:")
    avg_row = f"{'AVG/SUCCESS':<30}"
    total_success = sum(stats.get("success_with_id", 0) for stats in all_stats)
    for tool in sorted_tools:
        avg_val = tool_grand_totals[tool] / total_success if total_success > 0 else 0
        avg_row += f" {avg_val:>12.2f}"
    avg_all = grand_total / total_success if total_success > 0 else 0
    avg_row += f" {avg_all:>10.2f}"
    print(avg_row)
    
    print("=" * 120)


def print_ground_truth_coverage(all_stats: list):
    """Print coverage summary against ground truth entries."""
    if not all_stats:
        return
    gt_total = all_stats[0].get("gt_total", 0)
    if not gt_total:
        return
    
    print("\n" + "=" * 120)
    print("🧮 GROUND TRUTH COVERAGE")
    print("=" * 120)
    print(f"\n{'Mode':<30} {'GT':>6} {'Matched':>8} {'Missing':>8} {'DirMatch':>8} {'Dup':>6} {'Unexp':>7} {'NoID':>6}")
    print("-" * 120)
    
    for stats in all_stats:
        gt_total = stats.get("gt_total", 0)
        gt_matched = stats.get("gt_matched", 0)
        gt_missing = stats.get("gt_missing", 0)
        gt_dir_match = stats.get("gt_matched_dirs", 0)
        gt_dup = stats.get("gt_duplicates", 0)
        gt_unexp = stats.get("gt_unexpected", 0)
        gt_noid = stats.get("gt_unknown", 0)
        
        print(f"{stats['mode']:<30} {gt_total:>6} {gt_matched:>8} {gt_missing:>8} "
              f"{gt_dir_match:>8} {gt_dup:>6} {gt_unexp:>7} {gt_noid:>6}")
    
    print("Note: Matched counts unique GT keys; DirMatch counts dirs mapped to GT (includes duplicates).")


def print_summary(all_stats: list):
    """Print a summary table."""
    print("\n" + "=" * 120)
    print("📊 EVALUATION STATUS SUMMARY")
    print("=" * 120)
    
    # Header
    print(f"\n{'Mode':<30} {'Total':>8} {'Success':>10} {'Failed':>8} {'Pending':>8} {'Success%':>10} {'MAE':>10} {'MAE_n':>8}")
    print("-" * 120)
    
    total_all = 0
    success_all = 0
    failed_all = 0
    pending_all = 0
    all_mae_values = []
    
    for stats in all_stats:
        total = stats["total_with_id"]
        success = stats["success_with_id"]
        failed = stats["failed_with_id"]
        pending = stats["pending_with_id"]
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        # Compute MAE for this mode
        mae_stats = compute_mae_stats(stats)
        mae_str = f"{mae_stats['mae']:.4f}" if mae_stats['mae'] is not None else "N/A"
        mae_count = mae_stats['mae_count']
        
        print(f"{stats['mode']:<30} {total:>8} {success:>10} {failed:>8} {pending:>8} {success_rate:>9.1f}% {mae_str:>10} {mae_count:>8}")
        
        total_all += total
        success_all += success
        failed_all += failed
        pending_all += pending
        all_mae_values.extend(stats.get("mae_values", []))
    
    # Total row
    print("-" * 120)
    success_rate_all = (success_all / total_all * 100) if total_all > 0 else 0
    
    # Compute overall MAE
    if all_mae_values:
        import statistics
        overall_mae = statistics.mean(all_mae_values)
        overall_mae_str = f"{overall_mae:.4f}"
    else:
        overall_mae_str = "N/A"
    
    print(f"{'TOTAL':<30} {total_all:>8} {success_all:>10} {failed_all:>8} {pending_all:>8} {success_rate_all:>9.1f}% {overall_mae_str:>10} {len(all_mae_values):>8}")
    print("=" * 120)
    print("Note: Totals/Success/Failed/Pending exclude dirs without model_id/dataset_id.")
    
    # Print ground truth coverage
    print_ground_truth_coverage(all_stats)
    
    # Print metadata summary (turns, tokens, tools, etc.)
    print_metadata_summary(all_stats)
    
    # Print tool calls breakdown
    print_tool_calls_summary(all_stats)
    
    # Print detailed MAE statistics
    print("\n" + "=" * 120)
    print("📈 MAE STATISTICS (Mean Absolute Error vs Ground Truth)")
    print("=" * 120)
    print(f"\n{'Mode':<30} {'MAE':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10} {'Count':>8}")
    print("-" * 120)
    
    for stats in all_stats:
        mae_stats = compute_mae_stats(stats)
        if mae_stats['mae'] is not None:
            print(f"{stats['mode']:<30} "
                  f"{mae_stats['mae']:>10.4f} "
                  f"{mae_stats['mae_std']:>10.4f} "
                  f"{mae_stats['mae_min']:>10.4f} "
                  f"{mae_stats['mae_max']:>10.4f} "
                  f"{mae_stats['mae_median']:>10.4f} "
                  f"{mae_stats['mae_count']:>8}")
        else:
            print(f"{stats['mode']:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>8}")
    
    print("=" * 120)
    
    # Print signed error statistics (ME = Mean Error)
    print("\n" + "=" * 120)
    print("📉 SIGNED ERROR STATISTICS (Mean Error = pred - gt)")
    print("    Negative ME means predictions are systematically LOWER than ground truth")
    print("=" * 120)
    print(f"\n{'Mode':<30} {'ME':>10} {'Std':>10} {'Median':>10} {'Higher':>10} {'Lower':>10} {'AvgHigh':>10} {'AvgLow':>10}")
    print(f"{'':<30} {'(signed)':>10} {'':>10} {'':>10} {'(n/%)':>10} {'(n/%)':>10} {'':>10} {'':>10}")
    print("-" * 120)
    
    for stats in all_stats:
        se_stats = compute_signed_error_stats(stats)
        if se_stats['me'] is not None:
            higher_str = f"{se_stats['pred_higher_count']}/{se_stats['pred_higher_pct']:.0f}%"
            lower_str = f"{se_stats['pred_lower_count']}/{se_stats['pred_lower_pct']:.0f}%"
            avg_high = f"{se_stats['pred_higher_avg']:+.4f}" if se_stats['pred_higher_avg'] is not None else "N/A"
            avg_low = f"{se_stats['pred_lower_avg']:+.4f}" if se_stats['pred_lower_avg'] is not None else "N/A"
            print(f"{stats['mode']:<30} "
                  f"{se_stats['me']:>+10.4f} "
                  f"{se_stats['me_std']:>10.4f} "
                  f"{se_stats['me_median']:>+10.4f} "
                  f"{higher_str:>10} "
                  f"{lower_str:>10} "
                  f"{avg_high:>10} "
                  f"{avg_low:>10}")
        else:
            print(f"{stats['mode']:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    print("=" * 120)
    print("Note: ME = mean(pred - gt). Negative = predictions lower than ground truth.")
    
    # Print success threshold statistics (pred >= threshold * gt)
    print("\n" + "=" * 120)
    print("✅ SUCCESS THRESHOLD STATISTICS (pred >= threshold × gt)")
    print("    Success = prediction is at least X% of ground truth value")
    print("=" * 120)
    print(f"\n{'Mode':<30} {'Count':>8} {'≥90%':>8} {'≥95%':>8} {'≥100%':>8} {'≥90%':>10} {'≥95%':>10} {'≥100%':>10}")
    print(f"{'':<30} {'':>8} {'(n)':>8} {'(n)':>8} {'(n)':>8} {'(rate)':>10} {'(rate)':>10} {'(rate)':>10}")
    print("-" * 120)
    
    all_pairs = []
    for stats in all_stats:
        pairs = stats.get("pred_gt_pairs", [])
        all_pairs.extend(pairs)
        
        if pairs:
            count = len(pairs)
            # pred >= threshold * gt means success
            success_90 = sum(1 for pred, gt in pairs if pred >= 0.90 * gt)
            success_95 = sum(1 for pred, gt in pairs if pred >= 0.95 * gt)
            success_100 = sum(1 for pred, gt in pairs if pred >= gt)
            rate_90 = success_90 / count * 100
            rate_95 = success_95 / count * 100
            rate_100 = success_100 / count * 100
            
            print(f"{stats['mode']:<30} {count:>8} {success_90:>8} {success_95:>8} {success_100:>8} "
                  f"{rate_90:>9.1f}% {rate_95:>9.1f}% {rate_100:>9.1f}%")
        else:
            print(f"{stats['mode']:<30} {0:>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} "
                  f"{'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    # Total row
    print("-" * 120)
    if all_pairs:
        total_count = len(all_pairs)
        total_90 = sum(1 for pred, gt in all_pairs if pred >= 0.90 * gt)
        total_95 = sum(1 for pred, gt in all_pairs if pred >= 0.95 * gt)
        total_100 = sum(1 for pred, gt in all_pairs if pred >= gt)
        total_rate_90 = total_90 / total_count * 100
        total_rate_95 = total_95 / total_count * 100
        total_rate_100 = total_100 / total_count * 100
        print(f"{'TOTAL':<30} {total_count:>8} {total_90:>8} {total_95:>8} {total_100:>8} "
              f"{total_rate_90:>9.1f}% {total_rate_95:>9.1f}% {total_rate_100:>9.1f}%")
    
    print("=" * 120)
    print("Note: Success means pred >= threshold × gt (e.g., ≥90% means pred is at least 90% of ground truth)")


def analyze_loader_errors(loader_type: str) -> dict:
    """Analyze loader errors for models or datasets."""
    from collections import Counter
    
    loader_dir = SCRIPTS_DIR / f"{loader_type}_loaders"
    script_name = "load_model.py" if loader_type == "model" else "load_dataset.py"
    
    stats = {
        "type": loader_type,
        "total": 0,
        "has_script": 0,
        "has_results": 0,
        "success": 0,
        "failed": 0,
        "no_validation": 0,
        "error_categories": Counter(),
        "error_samples": defaultdict(list),
    }
    
    if not loader_dir.exists():
        return stats
    
    for item in sorted(loader_dir.iterdir()):
        if not item.is_dir() or item.name.startswith('.'):
            continue
        
        stats["total"] += 1
        loader_id = item.name.replace("_", "/", 1)
        
        script_path = item / script_name
        results_path = item / "results.json"
        validation_log = item / "validation.log"
        
        if not script_path.exists():
            continue
        stats["has_script"] += 1
        
        if not results_path.exists():
            stats["no_validation"] += 1
            continue
        
        stats["has_results"] += 1
        
        try:
            with open(results_path) as f:
                results = json.load(f)
            
            if results.get("load_success", False):
                stats["success"] += 1
            else:
                stats["failed"] += 1
                
                # Try to categorize error
                error_msg = str(results.get("error", "")).lower()
                log_content = ""
                if validation_log.exists():
                    try:
                        log_content = validation_log.read_text(errors='ignore').lower()
                    except:
                        pass
                
                combined = error_msg + " " + log_content
                
                # Categorize
                if "gated" in combined or "401" in combined or "access" in combined:
                    category = "gated_access"
                elif "404" in combined or "not found" in combined or "doesn't exist" in combined:
                    category = "not_found"
                elif "timeout" in combined or "connection" in combined or "429" in combined:
                    category = "network_error"
                elif "modulenotfounderror" in combined or "no module named" in combined:
                    category = "missing_dependency"
                elif "out of memory" in combined or "oom" in combined or "cuda out of memory" in combined:
                    category = "memory_error"
                elif "cuda" in combined and "error" in combined:
                    category = "cuda_error"
                elif "syntaxerror" in combined:
                    category = "syntax_error"
                elif "typeerror" in combined or "attributeerror" in combined or "keyerror" in combined:
                    category = "runtime_error"
                elif "valueerror" in combined:
                    category = "value_error"
                elif "importerror" in combined:
                    category = "import_error"
                else:
                    category = "other"
                
                stats["error_categories"][category] += 1
                if len(stats["error_samples"][category]) < 3:
                    stats["error_samples"][category].append(loader_id)
        except:
            stats["failed"] += 1
            stats["error_categories"]["parse_error"] += 1
    
    return stats


def print_loader_summary():
    """Print loader validation summary table."""
    model_stats = analyze_loader_errors("model")
    dataset_stats = analyze_loader_errors("dataset")
    
    print("\n" + "=" * 100)
    print("📦 LOADER VALIDATION SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Metric':<30} {'Models':>15} {'Datasets':>15} {'Total':>15}")
    print("-" * 100)
    print(f"{'Total directories':<30} {model_stats['total']:>15} {dataset_stats['total']:>15} {model_stats['total']+dataset_stats['total']:>15}")
    print(f"{'Has load script':<30} {model_stats['has_script']:>15} {dataset_stats['has_script']:>15} {model_stats['has_script']+dataset_stats['has_script']:>15}")
    print(f"{'Has results.json':<30} {model_stats['has_results']:>15} {dataset_stats['has_results']:>15} {model_stats['has_results']+dataset_stats['has_results']:>15}")
    print(f"{'Load success ✅':<30} {model_stats['success']:>15} {dataset_stats['success']:>15} {model_stats['success']+dataset_stats['success']:>15}")
    print(f"{'Load failed ❌':<30} {model_stats['failed']:>15} {dataset_stats['failed']:>15} {model_stats['failed']+dataset_stats['failed']:>15}")
    print(f"{'No validation yet':<30} {model_stats['no_validation']:>15} {dataset_stats['no_validation']:>15} {model_stats['no_validation']+dataset_stats['no_validation']:>15}")
    
    # Success rate
    m_rate = model_stats['success'] / model_stats['has_results'] * 100 if model_stats['has_results'] else 0
    d_rate = dataset_stats['success'] / dataset_stats['has_results'] * 100 if dataset_stats['has_results'] else 0
    t_rate = (model_stats['success'] + dataset_stats['success']) / (model_stats['has_results'] + dataset_stats['has_results']) * 100 if (model_stats['has_results'] + dataset_stats['has_results']) else 0
    print("-" * 100)
    print(f"{'Success rate':<30} {m_rate:>14.1f}% {d_rate:>14.1f}% {t_rate:>14.1f}%")
    
    # Error categories
    print("\n" + "=" * 100)
    print("❌ LOADER ERROR CATEGORIES")
    print("=" * 100)
    
    all_categories = set(model_stats['error_categories'].keys()) | set(dataset_stats['error_categories'].keys())
    all_categories = sorted(all_categories, key=lambda x: -(model_stats['error_categories'][x] + dataset_stats['error_categories'][x]))
    
    if all_categories:
        print(f"\n{'Error Category':<25} {'Models':>10} {'Datasets':>10} {'Total':>10} {'%':>8}")
        print("-" * 70)
        
        total_errors = model_stats['failed'] + dataset_stats['failed']
        for cat in all_categories:
            m = model_stats['error_categories'][cat]
            d = dataset_stats['error_categories'][cat]
            t = m + d
            pct = t / total_errors * 100 if total_errors else 0
            print(f"{cat:<25} {m:>10} {d:>10} {t:>10} {pct:>7.1f}%")
        
        print("-" * 70)
        print(f"{'TOTAL':<25} {model_stats['failed']:>10} {dataset_stats['failed']:>10} {total_errors:>10} {100.0:>7.1f}%")
        
        # Sample errors
        print("\n" + "-" * 70)
        print("Sample errors by category:")
        for cat in all_categories[:5]:
            samples = model_stats['error_samples'][cat] + dataset_stats['error_samples'][cat]
            if samples:
                print(f"  {cat}: {', '.join(samples[:2])}")
    else:
        print("\n  No loader errors found! 🎉")
    
    print("=" * 100)
    
    return model_stats, dataset_stats


def print_failed_details(all_stats: list, show_all: bool = False):
    """Print details of failed evaluations."""
    print("\n" + "=" * 100)
    print("❌ FAILED EVALUATIONS (started but no results.json)")
    print("=" * 100)
    
    for stats in all_stats:
        failed_evals = [e for e in stats["evaluations"] if e["has_run_log"] and not e["has_results"]]
        if failed_evals:
            print(f"\n📁 {stats['mode']} ({len(failed_evals)} failed):")
            for eval_info in failed_evals[:10 if not show_all else None]:
                print(f"   - {eval_info['name']}")
            if len(failed_evals) > 10 and not show_all:
                print(f"   ... and {len(failed_evals) - 10} more")


def print_high_error_details(all_stats: list, threshold: float = 0.1, show_all: bool = False):
    """Print details of evaluations with high absolute error."""
    print("\n" + "=" * 120)
    print(f"⚠️ HIGH ERROR EVALUATIONS (Absolute Error > {threshold})")
    print("=" * 120)
    
    for stats in all_stats:
        high_error_evals = [
            e for e in stats["evaluations"] 
            if e.get("absolute_error") is not None and e["absolute_error"] > threshold
        ]
        high_error_evals.sort(key=lambda x: x["absolute_error"], reverse=True)
        
        if high_error_evals:
            print(f"\n📁 {stats['mode']} ({len(high_error_evals)} high error):")
            for eval_info in high_error_evals[:10 if not show_all else None]:
                pred = eval_info.get("predicted_value", "?")
                gt = eval_info.get("ground_truth_value", "?")
                ae = eval_info.get("absolute_error", 0)
                print(f"   - {eval_info['name'][:60]:<60} | pred={pred:.4f} gt={gt:.4f} AE={ae:.4f}")
            if len(high_error_evals) > 10 and not show_all:
                print(f"   ... and {len(high_error_evals) - 10} more")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check evaluation status for all modes")
    parser.add_argument("--details", "-d", action="store_true", help="Show detailed failed evaluations")
    parser.add_argument("--all", "-a", action="store_true", help="Show all failed evaluations (not just first 10)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--high-error", "-e", action="store_true", help="Show high error evaluations")
    parser.add_argument("--error-threshold", type=float, default=0.1, help="Threshold for high error (default: 0.1)")
    parser.add_argument("--loaders", "-l", action="store_true", help="Show loader validation summary")
    args = parser.parse_args()
    
    # Load ground truth for MAE computation
    print("Loading ground truth data...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth entries.")
    
    # Analyze all modes
    all_stats = []
    for mode_name, mode_dir in MODE_DIRS.items():
        stats = analyze_mode(mode_name, mode_dir, ground_truth)
        all_stats.append(stats)
    
    if args.json:
        # Output as JSON
        summary = {}
        for stats in all_stats:
            meta_stats = compute_meta_stats(stats)
            mae_stats = compute_mae_stats(stats)
            summary[stats["mode"]] = {
                "total": stats["total"],
                "total_with_id": stats.get("total_with_id", 0),
                "total_no_id": stats.get("total_no_id", 0),
                "success": stats["success"],
                "success_with_id": stats.get("success_with_id", 0),
                "failed": stats["failed"],
                "failed_with_id": stats.get("failed_with_id", 0),
                "pending": stats["not_started"],
                "pending_with_id": stats.get("pending_with_id", 0),
                "has_metadata": stats["has_metadata"],
                "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
                "success_rate_with_id": (
                    stats.get("success_with_id", 0) / stats.get("total_with_id", 0)
                    if stats.get("total_with_id", 0) > 0 else 0
                ),
                "gt_total": stats.get("gt_total", 0),
                "gt_matched": stats.get("gt_matched", 0),
                "gt_missing": stats.get("gt_missing", 0),
                "gt_matched_dirs": stats.get("gt_matched_dirs", 0),
                "gt_duplicates": stats.get("gt_duplicates", 0),
                "gt_unexpected": stats.get("gt_unexpected", 0),
                "gt_unknown": stats.get("gt_unknown", 0),
                "meta_count": meta_stats.get("meta_count", 0),
                "meta_missing": meta_stats.get("meta_missing", 0),
                "meta_avg_steps": meta_stats.get("avg_steps"),
                "meta_avg_error_steps": meta_stats.get("avg_error_steps"),
                "meta_avg_token_total": meta_stats.get("avg_token_total"),
                "meta_avg_tool_calls_total": meta_stats.get("avg_tool_calls_total"),
                "meta_llm_models": meta_stats.get("llm_models"),
                "meta_max_steps_configs": meta_stats.get("max_steps_configs"),
                "meta_tool_calls": meta_stats.get("tool_calls"),
                **mae_stats,
                # Success threshold stats (pred >= threshold * gt)
                "success_threshold_count": len(stats.get("pred_gt_pairs", [])),
                "success_ge_90pct": sum(1 for pred, gt in stats.get("pred_gt_pairs", []) if pred >= 0.90 * gt),
                "success_ge_95pct": sum(1 for pred, gt in stats.get("pred_gt_pairs", []) if pred >= 0.95 * gt),
                "success_ge_100pct": sum(1 for pred, gt in stats.get("pred_gt_pairs", []) if pred >= gt),
                "success_rate_90pct": (
                    sum(1 for pred, gt in stats.get("pred_gt_pairs", []) if pred >= 0.90 * gt) / len(stats.get("pred_gt_pairs", []))
                    if stats.get("pred_gt_pairs") else None
                ),
                "success_rate_95pct": (
                    sum(1 for pred, gt in stats.get("pred_gt_pairs", []) if pred >= 0.95 * gt) / len(stats.get("pred_gt_pairs", []))
                    if stats.get("pred_gt_pairs") else None
                ),
                "success_rate_100pct": (
                    sum(1 for pred, gt in stats.get("pred_gt_pairs", []) if pred >= gt) / len(stats.get("pred_gt_pairs", []))
                    if stats.get("pred_gt_pairs") else None
                ),
                # Signed error stats (pred - gt)
                **compute_signed_error_stats(stats),
            }
        output = {"summary": summary}
        print(json.dumps(output, indent=2))
    else:
        # Print summary table
        print_summary(all_stats)
        
        if args.details:
            print_failed_details(all_stats, show_all=args.all)
        
        if args.high_error:
            print_high_error_details(all_stats, threshold=args.error_threshold, show_all=args.all)
        
        if args.loaders:
            print_loader_summary()
        
        # Quick tips
        print("\n💡 Tips:")
        print("   - Run with --details (-d) to see failed evaluation names")
        print("   - Run with --high-error (-e) to see high error evaluations")
        print("   - Run with --error-threshold 0.05 to change error threshold")
        print("   - Run with --loaders (-l) to see loader validation summary")
        print("   - Run with --json (-j) to get JSON output")
        print("   - MAE = Mean Absolute Error between predicted and ground truth metrics")


if __name__ == "__main__":
    main()
