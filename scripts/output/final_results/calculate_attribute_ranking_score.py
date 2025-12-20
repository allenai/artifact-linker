#!/usr/bin/env python3
"""
Script to calculate attribute ranking metrics from LLM ranking results.
Follows the same evaluation logic as rank_attribute_llm.py
"""

import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import evaluation utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from artifact_graph.utils.evaluation_utils import (
    calculate_ndcg,
    calculate_map,
    calculate_ranking_correlation,
)


def calculate_attribute_ranking_metrics(file_path, model_name):
    """Calculate attribute ranking metrics for a single ranking file."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        print(f"\n=== {model_name} ===")
        print(f"Total ranking results: {len(data)}")
        
        # Filter valid results (same logic as original script)
        valid_results = [r for r in data if r and r.get("ranked_pairs")]
        
        if not valid_results:
            print(f"❌ No valid rankings found in {file_path}")
            return None
        
        print(f"Valid rankings: {len(valid_results)}/{len(data)} ({len(valid_results)/len(data)*100:.1f}%)")
        
        # Calculate metrics (same logic as original script)
        all_ndcg = []
        all_map = []
        all_kendall_tau = []
        all_spearman = []
        
        for result in valid_results:
            if "ranked_pairs" in result:
                ranked_pairs = result["ranked_pairs"]
                ground_truth_pairs = result.get("ground_truth", [])
                
                # Extract predicted and ground truth items
                predicted_items = [pair[0] for pair in ranked_pairs]  # model_ids
                
                # Create ground truth dictionary from pairs
                ground_truth = {}
                for model_id, true_value in ground_truth_pairs:
                    ground_truth[model_id] = true_value
                
                # Create predicted items with scores for correlation
                predicted_items_with_scores = {}
                for model_id, predicted_score in ranked_pairs:
                    predicted_items_with_scores[model_id] = predicted_score
                
                relevant_items = set(ground_truth.keys())

                # Calculate NDCG and MAP (using top 10 as default)
                all_ndcg.append(calculate_ndcg(predicted_items, relevant_items, k=min(10, len(predicted_items))))
                all_map.append(calculate_map(predicted_items, relevant_items, k=min(10, len(predicted_items))))
                
                # Calculate ranking correlation
                try:
                    correlation_metrics = calculate_ranking_correlation(
                        predicted_items_with_scores, ground_truth
                    )
                    if "kendall_tau" in correlation_metrics:
                        all_kendall_tau.append(correlation_metrics["kendall_tau"])
                    if "spearman" in correlation_metrics:
                        all_spearman.append(correlation_metrics["spearman"])
                except Exception as e:
                    print(f"Warning: Could not calculate correlation metrics: {e}")
        
        # Calculate average metrics (same format as original script)
        metrics = {}
        
        if all_ndcg: 
            metrics["ndcg@10"] = sum(all_ndcg) / len(all_ndcg)
            print(f"  - NDCG@10: {metrics['ndcg@10']:.4f}")
        
        if all_map: 
            metrics["map@10"] = sum(all_map) / len(all_map)
            print(f"  - MAP@10: {metrics['map@10']:.4f}")
        
        if all_kendall_tau: 
            metrics["kendall_tau"] = sum(all_kendall_tau) / len(all_kendall_tau)
            print(f"  - Kendall's Tau: {metrics['kendall_tau']:.4f}")
        
        if all_spearman: 
            metrics["spearman"] = sum(all_spearman) / len(all_spearman)
            print(f"  - Spearman's Rho: {metrics['spearman']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def main():
    """Calculate metrics for attribute ranking files."""
    
    # Auto-discover attribute ranking files
    current_dir = Path(".")
    files_to_process = []
    
    # Look for files matching pattern: llm_attribute_rankings_*
    for file_path in current_dir.glob("llm_attribute_rankings_*.json"):
        file_str = str(file_path)
        
        # Extract model name and other info from filename
        if "openai_gpt-4o" in file_str:
            model_base = "GPT-4o"
        elif "Qwen2.5-72B-Instruct-Turbo" in file_str:
            model_base = "Qwen2.5-72B"
        else:
            model_base = "Unknown"
        
        # Extract hop and metric information
        hop_info = "unknown"
        if "_0hop_" in file_str:
            hop_info = "0hop"
        elif "_1hop_" in file_str:
            hop_info = "1hop"
        
        # Extract metric name if present
        metric_info = ""
        parts = file_str.split("_")
        for i, part in enumerate(parts):
            if part.endswith(".json"):
                metric_part = part.replace(".json", "")
                if metric_part and metric_part not in ["auto", model_base.lower()]:
                    metric_info = f"_{metric_part}"
                break
        
        model_name = f"{model_base}-{hop_info}{metric_info}"
        files_to_process.append((str(file_path), model_name))
    
    if not files_to_process:
        print("❌ No attribute ranking files found. Looking for files matching:")
        print("  - llm_attribute_rankings_*.json")
        return
    
    results = {}
    
    for file_path, model_name in files_to_process:
        # Create unique key for each file (in case multiple metrics/hops)
        file_key = f"{model_name}_{Path(file_path).stem}"
        metrics = calculate_attribute_ranking_metrics(file_path, file_key)
        if metrics:
            results[file_key] = metrics
    
    # Compare results if multiple files have valid results
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        result_keys = list(results.keys())
        
        # Get all metrics that exist
        all_metric_names = set()
        for metrics in results.values():
            all_metric_names.update(metrics.keys())
        
        # Sort metrics logically
        metric_order = ["ndcg@10", "map@10", "kendall_tau", "spearman"]
        sorted_metrics = [m for m in metric_order if m in all_metric_names]
        
        # Print comparison table
        header = f"{'File':<40}"
        for metric in sorted_metrics:
            header += f"{metric.upper():<12}"
        print(header)
        print("-" * (40 + 12 * len(sorted_metrics)))
        
        for file_key in result_keys:
            metrics = results[file_key]
            row = f"{file_key:<40}"
            for metric in sorted_metrics:
                val = metrics.get(metric, 0.0)
                row += f"{val:<12.4f}"
            print(row)
        
        # If exactly two models, show difference
        if len(results) == 2:
            print("\n" + "-" * (40 + 12 * len(sorted_metrics)))
            file1, file2 = result_keys[0], result_keys[1]
            row = f"{'Difference (File2 - File1)':<40}"
            for metric in sorted_metrics:
                val1 = results[file1].get(metric, 0.0)
                val2 = results[file2].get(metric, 0.0)
                diff = val2 - val1
                row += f"{diff:<12.4f}"
            print(row)


if __name__ == "__main__":
    main()
