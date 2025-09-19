#!/usr/bin/env python3
"""
Example usage of predict_models_for_dataset.py

This script demonstrates how to use the model prediction functionality
for specific datasets.
"""

import json
import subprocess
import sys


def run_prediction_example():
    """Run a simple prediction example."""

    print("🧪 EXAMPLE: Predicting models for a specific dataset")
    print("=" * 60)

    # Example 1: Predict accuracy for SQuAD dataset (limited to 5 models for testing)
    print("\n📋 Example 1: Predicting accuracy for 'squad' dataset")
    print(
        "Command: python scripts/predict_models_for_dataset.py --dataset squad --metric accuracy --limit 5"
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/predict_models_for_dataset.py",
                "--dataset",
                "squad",
                "--metric",
                "accuracy",
                "--limit",
                "5",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Exit code: {result.returncode}")

    except Exception as e:
        print(f"❌ Failed to run example: {e}")

    print("\n" + "=" * 60)
    print("📚 Other usage examples:")
    print()
    print("# Predict F1 score for GLUE SST-2:")
    print("python scripts/predict_models_for_dataset.py --dataset 'glue/sst2' --metric f1")
    print()
    print("# Predict BLEU for WMT translation:")
    print("python scripts/predict_models_for_dataset.py --dataset 'wmt14/de-en' --metric bleu")
    print()
    print("# Use neighborhood mode for better context:")
    print("python scripts/predict_models_for_dataset.py --dataset squad --mode neighborhood")
    print()
    print("# Use different LLM model:")
    print("python scripts/predict_models_for_dataset.py --dataset squad --llm gpt-3.5-turbo")
    print()
    print("# Save to specific output file:")
    print("python scripts/predict_models_for_dataset.py --dataset squad -o my_predictions.json")


def analyze_prediction_results(results_file: str):
    """Analyze results from a prediction run."""

    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in results file: {results_file}")
        return

    summary = data.get("summary", {})
    predictions = data.get("predictions", [])

    print(f"\n📊 ANALYSIS: {results_file}")
    print("=" * 60)

    # Summary stats
    print(f"Dataset: {summary.get('dataset', 'Unknown')}")
    print(f"Metric: {summary.get('metric', 'Unknown')}")
    print(f"Total models evaluated: {summary.get('total_models', 0)}")
    print(f"Successful predictions: {summary.get('successful_predictions', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1%}")

    # Prediction statistics
    stats = summary.get("prediction_stats")
    if stats:
        print("\n🎯 Prediction Statistics:")
        print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Range: {stats['min']:.3f} to {stats['max']:.3f}")
        print(f"  Median: {stats['median']:.3f}")

    # Top and bottom predictions
    successful_predictions = [
        p
        for p in predictions
        if p.get("status") == "success" and p.get("predicted_metric") is not None
    ]

    if successful_predictions:
        # Sort by predicted metric
        sorted_predictions = sorted(
            successful_predictions, key=lambda x: x["predicted_metric"], reverse=True
        )

        print("\n🏆 Top 5 predicted models:")
        for i, pred in enumerate(sorted_predictions[:5]):
            print(f"  {i+1}. {pred['model_id']}: {pred['predicted_metric']:.3f}")

        print("\n📉 Bottom 5 predicted models:")
        for i, pred in enumerate(sorted_predictions[-5:]):
            print(
                f"  {len(sorted_predictions)-4+i}. {pred['model_id']}: {pred['predicted_metric']:.3f}"
            )

    # Error analysis
    failed_predictions = [p for p in predictions if p.get("status") != "success"]
    if failed_predictions:
        print(f"\n⚠️  Failed predictions: {len(failed_predictions)}")
        error_types = {}
        for pred in failed_predictions:
            error_type = pred.get("status", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")


def main():
    """Main function to run examples or analyze results."""

    if len(sys.argv) > 1:
        # Analyze results file if provided
        results_file = sys.argv[1]
        analyze_prediction_results(results_file)
    else:
        # Run example
        run_prediction_example()


if __name__ == "__main__":
    main()
