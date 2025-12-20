import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import evaluation utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from artifact_graph.utils.evaluation_utils import evaluate_binary_classification


def calculate_metrics(file_path, model_name):
    """Calculate metrics for a single prediction file."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        print(f"\n=== {model_name} ===")
        print(f"Total predictions: {len(data)}")
        
        # Extract labels, filtering out None predictions correctly
        valid_pairs = []
        for row in data:
            if row.get("predicted_label") is not None:
                valid_pairs.append((row["true_label"], row["predicted_label"]))
        
        if not valid_pairs:
            print(f"❌ No valid predictions found in {file_path}")
            return None
        
        true_labels, predicted_labels = zip(*valid_pairs)
        
        print(f"Valid predictions: {len(valid_pairs)}/{len(data)} ({len(valid_pairs)/len(data)*100:.1f}%)")
        
        # Calculate metrics using the same evaluation function as the original script
        metrics = evaluate_binary_classification(list(true_labels), list(predicted_labels))
        
        # Print results
        for metric_name, value in metrics.items():
            print(f"  - {metric_name.upper()}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def main():
    """Calculate metrics for link prediction files."""
    
    # Auto-discover link prediction files
    current_dir = Path(".")
    files_to_process = []
    
    # Look for files matching pattern: llm_link_predictions_*
    for file_path in current_dir.glob("llm_link_predictions_*.json"):
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
        print("❌ No link prediction files found. Looking for files matching:")
        print("  - llm_link_predictions_*.json")
        return
    
    results = {}
    
    for file_path, model_name in files_to_process:
        metrics = calculate_metrics(file_path, model_name)
        if metrics:
            results[model_name] = metrics
    
    # Compare results if both models have valid results
    if len(results) == 2:
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print(f"{'='*50}")
        
        model_names = list(results.keys())
        model1, model2 = model_names[0], model_names[1]
        
        print(f"{'Metric':<12} {model1:<15} {model2:<15} {'Difference':<12}")
        print("-" * 60)
        
        for metric in ["accuracy", "f1", "precision", "recall"]:
            val1 = results[model1][metric]
            val2 = results[model2][metric]
            diff = val2 - val1
            diff_str = f"{diff:+.4f}"
            
            print(f"{metric.upper():<12} {val1:<15.4f} {val2:<15.4f} {diff_str:<12}")


if __name__ == "__main__":
    main()