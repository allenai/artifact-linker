#!/usr/bin/env python3
"""
Script to calculate link ranking metrics from LLM ranking results.
Follows the same evaluation logic as rank_link_llm.py
"""

import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import evaluation utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from artifact_graph.utils.evaluation_utils import (
    calculate_recall_at_k,
    calculate_precision_at_k,
)


def calculate_ranking_metrics(file_path, model_name):
    """Calculate ranking metrics for a single ranking file."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        print(f"\n=== {model_name} ===")
        print(f"Total ranking results: {len(data)}")
        
        # Filter valid results (same logic as original script)
        valid_results = [r for r in data if r and r.get("ranked_model_ids")]
        
        if not valid_results:
            print(f"❌ No valid rankings found in {file_path}")
            return None
        
        print(f"Valid rankings: {len(valid_results)}/{len(data)} ({len(valid_results)/len(data)*100:.1f}%)")
        
        # Calculate metrics for different K values (same as original script)
        k_values = [1, 3, 5, 10]
        
        all_metrics = {f"recall@{k}": [] for k in k_values}
        all_metrics.update({f"precision@{k}": [] for k in k_values})
        
        for result in valid_results:
            if "ranked_model_ids" in result:
                predicted_model_ids = result["ranked_model_ids"]
                positive_models = set(result["positive_models"])
                
                # Calculate metrics for different K values
                for k in k_values:
                    recall_k = calculate_recall_at_k(predicted_model_ids, positive_models, k)
                    precision_k = calculate_precision_at_k(predicted_model_ids, positive_models, k)
                    
                    all_metrics[f"recall@{k}"].append(recall_k)
                    all_metrics[f"precision@{k}"].append(precision_k)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                avg_metrics[metric_name] = avg_value
                print(f"  - {metric_name.upper()}: {avg_value:.4f}")
        
        return avg_metrics
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def main():
    """Calculate metrics for link ranking files."""
    
    # Auto-discover link ranking files
    current_dir = Path(".")
    files_to_process = []
    
    # Look for files matching pattern: llm_link_rankings_*
    for file_path in current_dir.glob("llm_link_rankings_*.json"):
        file_str = str(file_path)
        
        # Extract model name and hop info from filename
        if "openai_gpt-4o" in file_str:
            model_base = "GPT-4o"
        elif "Qwen2.5-72B-Instruct-Turbo" in file_str:
            model_base = "Qwen2.5-72B"
        else:
            model_base = "Unknown"
        
        # Extract hop information
        if "_0hop_" in file_str:
            hop_info = "0hop"
        elif "_1hop_" in file_str:
            hop_info = "1hop"
        else:
            hop_info = "unknown"
        
        model_name = f"{model_base}-{hop_info}"
        files_to_process.append((str(file_path), model_name))
    
    if not files_to_process:
        print("❌ No link ranking files found. Looking for files matching:")
        print("  - llm_link_rankings_*.json")
        return
    
    results = {}
    
    for file_path, model_name in files_to_process:
        metrics = calculate_ranking_metrics(file_path, model_name)
        if metrics:
            results[model_name] = metrics
    
    # Compare results if multiple models have valid results
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        model_names = list(results.keys())
        
        # Get all unique metrics
        all_metric_names = set()
        for model_metrics in results.values():
            all_metric_names.update(model_metrics.keys())
        
        # Sort metrics logically
        sorted_metrics = []
        for k in [1, 3, 5, 10]:
            if f"recall@{k}" in all_metric_names:
                sorted_metrics.append(f"recall@{k}")
        for k in [1, 3, 5, 10]:
            if f"precision@{k}" in all_metric_names:
                sorted_metrics.append(f"precision@{k}")
        
        # Print comparison table
        header = f"{'Metric':<15}"
        for model_name in model_names:
            header += f"{model_name:<15}"
        if len(model_names) == 2:
            header += "Difference"
        print(header)
        print("-" * (15 + 15 * len(model_names) + (12 if len(model_names) == 2 else 0)))
        
        for metric in sorted_metrics:
            row = f"{metric.upper():<15}"
            values = []
            for model_name in model_names:
                val = results[model_name].get(metric, 0.0)
                values.append(val)
                row += f"{val:<15.4f}"
            
            # Add difference for two models
            if len(values) == 2:
                diff = values[1] - values[0]
                row += f"{diff:+.4f}"
            
            print(row)


if __name__ == "__main__":
    main()
