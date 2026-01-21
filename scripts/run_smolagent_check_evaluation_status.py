#!/usr/bin/env python3
"""
Check evaluation status for all four modes.
Shows how many evaluations ran and how many succeeded for each mode.
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


def check_evaluation_dir(eval_dir: Path) -> dict:
    """Check a single evaluation directory and return status."""
    status = {
        "exists": eval_dir.exists(),
        "has_results": False,
        "has_predictions": False,
        "has_metadata": False,
        "has_run_log": False,
        "results": None,
        "error": None,
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
    status["has_metadata"] = metadata_file.exists()
    
    # Check for run.log
    run_log = eval_dir / "run.log"
    status["has_run_log"] = run_log.exists()
    
    return status


def analyze_mode(mode_name: str, mode_dir_name: str) -> dict:
    """Analyze all evaluations for a mode."""
    mode_dir = SCRIPTS_DIR / mode_dir_name
    
    stats = {
        "mode": mode_name,
        "dir": mode_dir_name,
        "exists": mode_dir.exists(),
        "total": 0,
        "success": 0,
        "has_predictions": 0,
        "has_metadata": 0,
        "failed": 0,
        "not_started": 0,
        "evaluations": [],
    }
    
    if not mode_dir.exists():
        return stats
    
    # List all subdirectories (each is an evaluation)
    for item in sorted(mode_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            stats["total"] += 1
            eval_status = check_evaluation_dir(item)
            
            if eval_status["has_results"]:
                stats["success"] += 1
            elif eval_status["has_run_log"]:
                stats["failed"] += 1
            else:
                stats["not_started"] += 1
            
            if eval_status["has_predictions"]:
                stats["has_predictions"] += 1
            
            if eval_status["has_metadata"]:
                stats["has_metadata"] += 1
            
            stats["evaluations"].append({
                "name": item.name,
                **eval_status
            })
    
    return stats


def print_summary(all_stats: list):
    """Print a summary table."""
    print("\n" + "=" * 100)
    print("📊 EVALUATION STATUS SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"\n{'Mode':<30} {'Total':>8} {'Success':>10} {'Failed':>8} {'Pending':>8} {'Success%':>10} {'Meta':>8}")
    print("-" * 100)
    
    total_all = 0
    success_all = 0
    failed_all = 0
    pending_all = 0
    meta_all = 0
    
    for stats in all_stats:
        total = stats["total"]
        success = stats["success"]
        failed = stats["failed"]
        pending = stats["not_started"]
        meta = stats["has_metadata"]
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        print(f"{stats['mode']:<30} {total:>8} {success:>10} {failed:>8} {pending:>8} {success_rate:>9.1f}% {meta:>8}")
        
        total_all += total
        success_all += success
        failed_all += failed
        pending_all += pending
        meta_all += meta
    
    # Total row
    print("-" * 100)
    success_rate_all = (success_all / total_all * 100) if total_all > 0 else 0
    print(f"{'TOTAL':<30} {total_all:>8} {success_all:>10} {failed_all:>8} {pending_all:>8} {success_rate_all:>9.1f}% {meta_all:>8}")
    print("=" * 100)


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check evaluation status for all modes")
    parser.add_argument("--details", "-d", action="store_true", help="Show detailed failed evaluations")
    parser.add_argument("--all", "-a", action="store_true", help="Show all failed evaluations (not just first 10)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    # Analyze all modes
    all_stats = []
    for mode_name, mode_dir in MODE_DIRS.items():
        stats = analyze_mode(mode_name, mode_dir)
        all_stats.append(stats)
    
    if args.json:
        # Output as JSON
        output = {
            "summary": {
                stats["mode"]: {
                    "total": stats["total"],
                    "success": stats["success"],
                    "failed": stats["failed"],
                    "pending": stats["not_started"],
                    "has_metadata": stats["has_metadata"],
                    "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
                }
                for stats in all_stats
            }
        }
        print(json.dumps(output, indent=2))
    else:
        # Print summary table
        print_summary(all_stats)
        
        if args.details:
            print_failed_details(all_stats, show_all=args.all)
        
        # Quick tips
        print("\n💡 Tips:")
        print("   - Run with --details (-d) to see failed evaluation names")
        print("   - Run with --json (-j) to get JSON output")
        print("   - 'Meta' column shows how many have metadata.json (new feature)")


if __name__ == "__main__":
    main()
