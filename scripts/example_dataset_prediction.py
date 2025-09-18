#!/usr/bin/env python3
"""
Example script showing how to use the enhanced predict_links_llm.py
to get predictions for all models given a specific dataset.
"""

import json
import subprocess
import sys

def find_available_datasets():
    """Find all available datasets in the perfect metrics file."""
    try:
        with open("output/perfect_model_dataset_metrics.json", "r") as f:
            data = json.load(f)
        
        datasets = set()
        for item in data.get("results", []):
            datasets.add(item["dataset_id"])
        
        return sorted(list(datasets))
    except FileNotFoundError:
        print("Error: perfect_model_dataset_metrics.json not found")
        return []

def run_prediction_for_dataset(dataset_name, mode="zero-shot", metric="accuracy"):
    """Run LLM prediction for a specific dataset."""
    print(f"\n🚀 Running LLM prediction for dataset: {dataset_name}")
    print(f"   Mode: {mode}")
    print(f"   Metric: {metric}")
    
    # Run the prediction script
    cmd = [
        "python", "scripts/predict_links_llm.py",
        "--dataset", dataset_name,
        "--mode", mode,
        "--metric", metric
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Prediction completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Prediction failed!")
        print("Error output:")
        print(e.stderr)
        return False

def main():
    print("🔍 Finding available datasets...")
    datasets = find_available_datasets()
    
    if not datasets:
        print("No datasets found!")
        return
    
    print(f"Found {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets[:10]):  # Show first 10
        print(f"  {i+1}. {dataset}")
    
    if len(datasets) > 10:
        print(f"  ... and {len(datasets) - 10} more")
    
    # Example: Run prediction for the first dataset
    if datasets:
        example_dataset = datasets[0]
        print(f"\n📊 Example: Running prediction for '{example_dataset}'")
        success = run_prediction_for_dataset(
            dataset_name=example_dataset,
            mode="zero-shot",
            metric="accuracy"
        )
        
        if success:
            output_file = f"output/llm_predictions_{example_dataset}_zero-shot.json"
            print(f"\n📁 Results saved to: {output_file}")
            
            # Show summary of results
            try:
                with open(output_file, "r") as f:
                    results = json.load(f)
                
                successful = sum(1 for r in results if r["status"] == "Success")
                with_ground_truth = sum(1 for r in results if r["has_ground_truth"])
                
                print(f"\n📈 Summary:")
                print(f"   Total models tested: {len(results)}")
                print(f"   Successful predictions: {successful}")
                print(f"   Models with ground truth: {with_ground_truth}")
                
            except Exception as e:
                print(f"Could not read results: {e}")

if __name__ == "__main__":
    main()
