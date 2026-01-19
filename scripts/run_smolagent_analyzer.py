#!/usr/bin/env python3
"""
Analyze smolagent coding agent results.
Compare agent-generated results with ground truth metrics.

Logic: Start from ground truth JSON, check if each task has a corresponding result.
Saves a comparable JSON file and calculates statistics from it.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def normalize_dir_name(s: str) -> str:
    """Normalize string for directory name matching."""
    return s.replace('/', '_').replace(' ', '_')


def load_agent_result(result_file: Path) -> Tuple[Optional[Dict], Optional[float]]:
    """Load agent result from results.json and return (result_dict, value)."""
    if not result_file.exists():
        return None, None
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        if data:
            metric_name = list(data.keys())[0]
            value = data[metric_name]
            return data, value
    except:
        pass
    return None, None


def normalize_scale(agent_value: float, gt_value: float) -> Tuple[float, float, str]:
    """
    Normalize agent and GT values to the same scale.
    
    Handles cases where GT is in percentage (0-100) and agent is in decimal (0-1) or vice versa.
    
    Returns:
        Tuple of (normalized_agent, normalized_gt, scale_note)
    """
    if agent_value is None or gt_value is None:
        return agent_value, gt_value, "none"
    
    # Case 1: GT > 1 and agent <= 1 (GT is percentage, agent is decimal)
    if gt_value > 1.0 and agent_value <= 1.0:
        # Convert agent to percentage scale
        return agent_value * 100, gt_value, "agent_scaled_up"
    
    # Case 2: GT <= 1 and agent > 1 (GT is decimal, agent is percentage)
    if gt_value <= 1.0 and agent_value > 1.0:
        # Convert agent to decimal scale
        return agent_value / 100, gt_value, "agent_scaled_down"
    
    # Case 3: Same scale
    return agent_value, gt_value, "same_scale"


def calculate_relative_error(agent_value: float, gt_value: float, normalize: bool = True) -> Optional[float]:
    """
    Calculate relative error between agent result and ground truth.
    
    Args:
        agent_value: Value from agent evaluation
        gt_value: Ground truth value
        normalize: If True, normalize scales before comparison
    """
    if agent_value is None:
        return None
    
    if normalize:
        agent_value, gt_value, _ = normalize_scale(agent_value, gt_value)
    
    if gt_value == 0:
        return float('inf') if agent_value != 0 else 0.0
    return abs(agent_value - gt_value) / abs(gt_value)


def calculate_stats_from_comparisons(comparisons: List[Dict]) -> Dict:
    """Calculate statistics from a list of comparison records."""
    stats = {
        "total": len(comparisons),
        "completed": 0,
        "missing": 0,
        "valid_results": 0,
        "within_20_percent": 0,
        "within_10_percent": 0,
        "within_5_percent": 0,
        "exact_match": 0,
    }
    
    for c in comparisons:
        if c["status"] == "missing":
            stats["missing"] += 1
        else:
            stats["completed"] += 1
            
            rel_error = c.get("relative_error")
            if rel_error is not None and rel_error != "inf" and not (isinstance(rel_error, float) and rel_error != rel_error):
                stats["valid_results"] += 1
                if rel_error <= 0.20:
                    stats["within_20_percent"] += 1
                if rel_error <= 0.10:
                    stats["within_10_percent"] += 1
                if rel_error <= 0.05:
                    stats["within_5_percent"] += 1
                if rel_error == 0:
                    stats["exact_match"] += 1
    
    # Calculate percentages
    if stats["total"] > 0:
        stats["completion_rate"] = stats["completed"] / stats["total"]
    if stats["valid_results"] > 0:
        stats["accuracy_20pct"] = stats["within_20_percent"] / stats["valid_results"]
        stats["accuracy_10pct"] = stats["within_10_percent"] / stats["valid_results"]
        stats["accuracy_5pct"] = stats["within_5_percent"] / stats["valid_results"]
    
    return stats


def analyze_part(json_file: str, result_dir: str) -> Tuple[Dict, List[Dict]]:
    """
    Analyze results for a single part, starting from ground truth.
    
    Returns:
        Tuple of (analysis_dict, comparisons_list)
        - analysis_dict: summary statistics
        - comparisons_list: list of comparable records with GT and agent values
    """
    result_path = Path(result_dir)
    comparisons = []  # List of comparable records
    
    if not os.path.exists(json_file):
        return {"error": f"JSON file not found: {json_file}"}, []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    expected_dirs = set()
    extra_results = []
    
    # Iterate through ground truth tasks
    for item in data['results']:
        model_id = item['model_id']
        dataset_id = item['dataset_id']
        
        for metric_name, gt_value in item['metrics'].items():
            # Build expected directory name
            model_dir = normalize_dir_name(model_id)
            dataset_dir = normalize_dir_name(dataset_id)
            metric_dir = normalize_dir_name(metric_name)
            expected_dir = f"{model_dir}_{dataset_dir}_{metric_dir}"
            expected_dirs.add(expected_dir)
            
            # Also try with original metric name (space not converted)
            metric_dir_orig = metric_name.replace('/', '_')
            expected_dir_orig = f"{model_dir}_{dataset_dir}_{metric_dir_orig}"
            expected_dirs.add(expected_dir_orig)
            
            # Check if results exist
            results_file = result_path / expected_dir / "results.json"
            results_file_orig = result_path / expected_dir_orig / "results.json"
            
            actual_dir = None
            if results_file.exists():
                actual_dir = expected_dir
            elif results_file_orig.exists():
                actual_dir = expected_dir_orig
                results_file = results_file_orig
            
            # Create comparison record
            comparison = {
                "model": model_id,
                "dataset": dataset_id,
                "metric": metric_name,
                "ground_truth": gt_value,
                "agent_result": None,
                "relative_error": None,
                "status": "missing",
                "output_dir": expected_dir
            }
            
            if actual_dir is not None:
                # Load agent result
                agent_result, agent_value = load_agent_result(results_file)
                comparison["status"] = "completed"
                
                if agent_value is not None and not (isinstance(agent_value, float) and agent_value != agent_value):
                    comparison["agent_result"] = agent_value
                    
                    # Normalize scales and calculate error
                    norm_agent, norm_gt, scale_note = normalize_scale(agent_value, gt_value)
                    comparison["agent_normalized"] = norm_agent
                    comparison["gt_normalized"] = norm_gt
                    comparison["scale_adjustment"] = scale_note
                    
                    rel_error = calculate_relative_error(agent_value, gt_value, normalize=True)
                    if rel_error == float('inf'):
                        comparison["relative_error"] = "inf"
                    else:
                        comparison["relative_error"] = rel_error
            
            comparisons.append(comparison)
    
    # Find extra results (results.json that don't match any GT task)
    if result_path.exists():
        for subdir in result_path.iterdir():
            if not subdir.is_dir() or subdir.name == "__pycache__":
                continue
            if (subdir / "results.json").exists():
                if subdir.name not in expected_dirs:
                    extra_results.append(subdir.name)
    
    # Calculate stats from comparisons
    stats = calculate_stats_from_comparisons(comparisons)
    stats["extra_results"] = extra_results
    
    # For backward compatibility, also include detailed lists
    stats["completed_details"] = [c for c in comparisons if c["status"] == "completed"]
    stats["missing_details"] = [c for c in comparisons if c["status"] == "missing"]
    stats["total_tasks"] = stats["total"]
    
    return stats, comparisons


def print_summary(part_name: str, analysis: Dict):
    """Print summary for a part."""
    print(f"\n{'='*70}")
    print(f"{part_name}")
    print('='*70)
    
    if "error" in analysis:
        print(f"  Error: {analysis['error']}")
        return
    
    total = analysis["total"]
    completed = analysis["completed"]
    missing = analysis["missing"]
    valid = analysis.get("valid_results", completed)
    within_20 = analysis["within_20_percent"]
    within_10 = analysis["within_10_percent"]
    within_5 = analysis["within_5_percent"]
    exact = analysis["exact_match"]
    extra = len(analysis.get("extra_results", []))
    
    print(f"  Total tasks in ground truth: {total}")
    print(f"  Completed (with results.json): {completed} ({completed/total*100:.1f}%)")
    print(f"  Valid numeric results: {valid}")
    print(f"  Missing: {missing} ({missing/total*100:.1f}%)")
    if extra > 0:
        print(f"  Extra results (not in GT): {extra}")
    print()
    
    if valid > 0:
        print(f"  Results within error thresholds (of {valid} valid results):")
        print(f"    ≤ 20% error: {within_20} ({within_20/valid*100:.1f}%)")
        print(f"    ≤ 10% error: {within_10} ({within_10/valid*100:.1f}%)")
        print(f"    ≤  5% error: {within_5} ({within_5/valid*100:.1f}%)")
        print(f"    Exact match: {exact} ({exact/valid*100:.1f}%)")


def print_detailed_results(analysis: Dict, show_missing: bool = True, show_all: bool = False):
    """Print detailed results."""
    details = analysis.get("completed_details", [])
    
    # Sort by relative error
    sorted_details = sorted(
        [d for d in details if d.get("agent_value") is not None and d.get("relative_error") is not None],
        key=lambda x: x.get("relative_error", float('inf'))
    )
    
    if sorted_details:
        print(f"\n  Top results (sorted by error):")
        print(f"  {'Model':<35} {'Dataset':<25} {'Metric':<15} {'Agent':>10} {'GT':>10} {'Error':>8}")
        print(f"  {'-'*110}")
        
        for detail in sorted_details[:10] if not show_all else sorted_details:
            model = detail.get("model", "")[:35]
            dataset = detail.get("dataset", "")[:25]
            metric = detail.get("metric", "")[:15]
            agent = detail.get("agent_value", 0)
            gt = detail.get("gt_value", 0)
            err = detail.get("rel_error_pct", "N/A")
            print(f"  {model:<35} {dataset:<25} {metric:<15} {agent:>10.4f} {gt:>10.4f} {err:>8}")
        
        # Show worst results
        if len(sorted_details) > 10 and not show_all:
            print(f"\n  Worst results (highest error):")
            print(f"  {'Model':<35} {'Dataset':<25} {'Metric':<15} {'Agent':>10} {'GT':>10} {'Error':>8}")
            print(f"  {'-'*110}")
            for detail in sorted_details[-5:]:
                model = detail.get("model", "")[:35]
                dataset = detail.get("dataset", "")[:25]
                metric = detail.get("metric", "")[:15]
                agent = detail.get("agent_value", 0)
                gt = detail.get("gt_value", 0)
                err = detail.get("rel_error_pct", "N/A")
                print(f"  {model:<35} {dataset:<25} {metric:<15} {agent:>10.4f} {gt:>10.4f} {err:>8}")
    
    # Show missing tasks
    if show_missing:
        missing = analysis.get("missing_details", [])
        if missing:
            print(f"\n  Missing tasks ({len(missing)}):")
            for m in missing[:10]:
                print(f"    - {m['model']} | {m['dataset']} | {m['metric']}")
            if len(missing) > 10:
                print(f"    ... and {len(missing)-10} more")
    
    # Show extra results
    extra = analysis.get("extra_results", [])
    if extra:
        print(f"\n  Extra results not in ground truth ({len(extra)}):")
        for e in extra[:5]:
            print(f"    - {e}")
        if len(extra) > 5:
            print(f"    ... and {len(extra)-5} more")


def save_comparable_json(output_file: str, all_comparisons: List[Dict], all_stats: Dict, mode: str):
    """Save a structured JSON file with comparable GT and agent results."""
    
    # Recalculate overall stats from all comparisons
    overall_stats = calculate_stats_from_comparisons(all_comparisons)
    
    output_data = {
        "metadata": {
            "mode": mode,
            "description": "Comparison of ground truth vs agent evaluation results"
        },
        "overall_summary": {
            "total_tasks": overall_stats["total"],
            "completed": overall_stats["completed"],
            "missing": overall_stats["missing"],
            "valid_numeric_results": overall_stats["valid_results"],
            "within_20_percent": overall_stats["within_20_percent"],
            "within_10_percent": overall_stats["within_10_percent"],
            "within_5_percent": overall_stats["within_5_percent"],
            "exact_match": overall_stats["exact_match"],
            "completion_rate": overall_stats.get("completion_rate", 0),
            "accuracy_20pct": overall_stats.get("accuracy_20pct", 0),
            "accuracy_10pct": overall_stats.get("accuracy_10pct", 0),
            "accuracy_5pct": overall_stats.get("accuracy_5pct", 0),
        },
        "per_part_summary": all_stats,
        "comparisons": all_comparisons
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    return output_data


def load_and_calculate_from_json(json_file: str) -> Dict:
    """Load a comparable JSON file and recalculate statistics."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    comparisons = data.get("comparisons", [])
    stats = calculate_stats_from_comparisons(comparisons)
    
    return {
        "loaded_from": json_file,
        "recalculated_stats": stats,
        "original_stats": data.get("overall_summary", {})
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze smolagent results")
    parser.add_argument("--mode", "-m", choices=["basic", "advanced", "only_three_tools", "oneshot"], default="basic",
                        help="Analysis mode: 'basic', 'advanced', 'only_three_tools', or 'oneshot' (default: basic)")
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="Show detailed results")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all detailed results (not just top/bottom)")
    parser.add_argument("--no-missing", action="store_true",
                        help="Don't show missing tasks")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file for comparable results (default: auto-generated)")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Calculate stats from existing comparable JSON file instead of scanning results")
    args = parser.parse_args()
    
    base_dir = "/data/haofeiy2/artifact-graph/scripts"
    
    # If loading from existing JSON file
    if args.from_json:
        print(f"\nLoading and recalculating from: {args.from_json}")
        result = load_and_calculate_from_json(args.from_json)
        stats = result["recalculated_stats"]
        
        print("\n" + "="*70)
        print("RECALCULATED STATISTICS FROM JSON")
        print("="*70)
        print(f"  Total tasks: {stats['total']}")
        print(f"  Completed: {stats['completed']} ({stats.get('completion_rate', 0)*100:.1f}%)")
        print(f"  Valid numeric results: {stats['valid_results']}")
        print(f"  Missing: {stats['missing']}")
        print()
        if stats['valid_results'] > 0:
            print(f"  Within 20% error: {stats['within_20_percent']} ({stats.get('accuracy_20pct', 0)*100:.1f}%)")
            print(f"  Within 10% error: {stats['within_10_percent']} ({stats.get('accuracy_10pct', 0)*100:.1f}%)")
            print(f"  Within  5% error: {stats['within_5_percent']} ({stats.get('accuracy_5pct', 0)*100:.1f}%)")
            print(f"  Exact match: {stats['exact_match']}")
        return
    
    # Select result directories based on mode
    if args.mode == "advanced":
        suffix = "_advanced"
        mode_label = "Advanced"
    elif args.mode == "only_three_tools":
        suffix = "_only_three_tools"
        mode_label = "Only Three Tools"
    elif args.mode == "oneshot":
        suffix = "_oneshot"
        mode_label = "One-Shot"
    else:
        suffix = "_basic"
        mode_label = "Basic"
    
    parts = [
        (f"Part 1 ({mode_label})", 
         f"{base_dir}/smolagent_results_coding_agent_part1{suffix}",
         f"{base_dir}/perfect_model_dataset_metrics_v2_1125_coding_agent_part1.json"),
        (f"Part 2 ({mode_label})", 
         f"{base_dir}/smolagent_results_coding_agent_part2{suffix}",
         f"{base_dir}/perfect_model_dataset_metrics_v2_1125_coding_agent_part2.json"),
        (f"Part 3 ({mode_label})", 
         f"{base_dir}/smolagent_results_coding_agent_part3{suffix}",
         f"{base_dir}/perfect_model_dataset_metrics_v2_1125_coding_agent_part3.json"),
    ]
    
    all_analysis = {}
    all_comparisons = []
    per_part_stats = {}
    
    print("\n" + "="*70)
    print(f"SMOLAGENT CODING AGENT RESULTS ANALYSIS ({mode_label.upper()} MODE)")
    print("="*70)
    
    for part_name, result_dir, json_file in parts:
        analysis, comparisons = analyze_part(json_file, result_dir)
        all_analysis[part_name] = analysis
        
        # Add part info to each comparison
        for c in comparisons:
            c["part"] = part_name
        all_comparisons.extend(comparisons)
        
        # Store per-part stats
        per_part_stats[part_name] = {
            "total": analysis.get("total", 0),
            "completed": analysis.get("completed", 0),
            "missing": analysis.get("missing", 0),
            "valid_results": analysis.get("valid_results", 0),
            "within_20_percent": analysis.get("within_20_percent", 0),
            "within_10_percent": analysis.get("within_10_percent", 0),
            "within_5_percent": analysis.get("within_5_percent", 0),
        }
        
        print_summary(part_name, analysis)
        
        if args.detailed:
            print_detailed_results(analysis, show_missing=not args.no_missing, show_all=args.all)
    
    # Calculate overall stats from all comparisons
    overall_stats = calculate_stats_from_comparisons(all_comparisons)
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"  Total tasks in ground truth: {overall_stats['total']}")
    print(f"  Completed (with results.json): {overall_stats['completed']} ({overall_stats.get('completion_rate', 0)*100:.1f}%)")
    print(f"  Valid numeric results: {overall_stats['valid_results']}")
    print(f"  Missing: {overall_stats['missing']}")
    print()
    if overall_stats['valid_results'] > 0:
        print(f"  Within 20% error: {overall_stats['within_20_percent']} ({overall_stats.get('accuracy_20pct', 0)*100:.1f}%)")
        print(f"  Within 10% error: {overall_stats['within_10_percent']} ({overall_stats.get('accuracy_10pct', 0)*100:.1f}%)")
        print(f"  Within  5% error: {overall_stats['within_5_percent']} ({overall_stats.get('accuracy_5pct', 0)*100:.1f}%)")
        print(f"  Exact match: {overall_stats['exact_match']}")
    
    # Save comparable JSON (default or specified)
    output_file = args.output
    if output_file is None:
        output_file = f"{base_dir}/smolagent_comparable_results_{args.mode}.json"
    
    save_comparable_json(output_file, all_comparisons, per_part_stats, args.mode)
    print(f"\n✓ Comparable results saved to: {output_file}")
    print(f"  Use --from-json {output_file} to recalculate stats from this file")


if __name__ == "__main__":
    main()
