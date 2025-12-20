#!/usr/bin/env python3
"""
Script to calculate attribute prediction metrics from LLM prediction results.
Follows the same evaluation logic as predict_attribute_llm.py
"""

import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import evaluation utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from artifact_graph.utils.evaluation_utils import (
    calculate_mse,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r_squared,
    calculate_mean_absolute_difference,
)


def calculate_regression_metrics(file_path, model_name):
    """Calculate regression metrics for a single prediction file."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        print(f"\n=== {model_name} ===")
        print(f"Total predictions: {len(data)}")
        
        # Extract valid predictions (same logic as original script)
        valid_predictions = []
        valid_true_metrics = []
        
        for row in data:
            if row.get("status") == "Success" and row.get("predicted_metric") is not None:
                valid_predictions.append(row["predicted_metric"])
                valid_true_metrics.append(row["true_metric"])
        
        if not valid_predictions:
            print(f"❌ No valid predictions found in {file_path}")
            return None
        
        print(f"Valid predictions: {len(valid_predictions)}/{len(data)} ({len(valid_predictions)/len(data)*100:.1f}%)")
        
        # Calculate comprehensive regression metrics (same as original script)
        metrics = {
            "mse": calculate_mse(valid_predictions, valid_true_metrics),
            "mae": calculate_mae(valid_predictions, valid_true_metrics),
            "rmse": calculate_rmse(valid_predictions, valid_true_metrics),
            "mape": calculate_mape(valid_predictions, valid_true_metrics),
            "r_squared": calculate_r_squared(valid_predictions, valid_true_metrics),
            "mean_abs_diff": calculate_mean_absolute_difference(valid_predictions, valid_true_metrics),
        }
        
        # Print results (same format as original script)
        print("--- Regression Metrics ---")
        print(f"  - MSE (Mean Squared Error): {metrics['mse']:.4f}")
        print(f"  - MAE (Mean Absolute Error): {metrics['mae']:.4f}")
        print(f"  - RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        if metrics['mape'] != float('inf'):
            print(f"  - MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
        else:
            print(f"  - MAPE: Undefined (zero true values)")
        print(f"  - R² (R-squared): {metrics['r_squared']:.4f}")
        print(f"  - Mean Absolute Difference: {metrics['mean_abs_diff']:.4f}")
        print("--------------------------")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def main():
    """Calculate metrics for attribute prediction files."""
    
    # Auto-discover attribute prediction files
    current_dir = Path(".")
    files_to_process = []
    
    # Look for files matching pattern: llm_attribute_predictions_*
    for file_path in current_dir.glob("llm_attribute_predictions_*.json"):
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
        print("❌ No attribute prediction files found. Looking for files matching:")
        print("  - llm_attribute_predictions_*.json")
        return
    
    results = {}
    
    for file_path, model_name in files_to_process:
        # Create unique key for each file (in case multiple metrics/hops)
        file_key = f"{model_name}_{Path(file_path).stem}"
        metrics = calculate_regression_metrics(file_path, file_key)
        if metrics:
            results[file_key] = metrics
    
    # Compare results if multiple files have valid results
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        result_keys = list(results.keys())
        
        # Get all metrics
        metric_names = ["mse", "mae", "rmse", "r_squared", "mean_abs_diff"]
        
        # Print comparison table
        header = f"{'File':<40}{'MSE':<10}{'MAE':<10}{'RMSE':<10}{'R²':<10}{'MAD':<10}"
        print(header)
        print("-" * 90)
        
        for file_key in result_keys:
            metrics = results[file_key]
            row = f"{file_key:<40}"
            row += f"{metrics['mse']:<10.4f}"
            row += f"{metrics['mae']:<10.4f}"
            row += f"{metrics['rmse']:<10.4f}"
            row += f"{metrics['r_squared']:<10.4f}"
            row += f"{metrics['mean_abs_diff']:<10.4f}"
            print(row)
        
        # If exactly two models, show difference
        if len(results) == 2:
            print("\n" + "-" * 90)
            file1, file2 = result_keys[0], result_keys[1]
            row = f"{'Difference (File2 - File1)':<40}"
            for metric in metric_names:
                diff = results[file2][metric] - results[file1][metric]
                row += f"{diff:<10.4f}"
            print(row)


if __name__ == "__main__":
    main()
