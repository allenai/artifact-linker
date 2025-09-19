#!/usr/bin/env python3
"""
Compare results.json files with expected results from perfect_model_dataset_metrics.json

Usage: python compare_results.py [results_directory] [reference_json_path]
"""

import glob
import json
import os
import sys


def load_reference_data(json_path):
    """Load reference data file"""
    if not os.path.exists(json_path):
        print(f"❌ Reference file not found: {json_path}")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create lookup table: (model_id, dataset_id, metric_name) -> expected_value
    lookup = {}
    for result in data.get("results", []):
        model_id = result.get("model_id", "")
        dataset_id = result.get("dataset_id", "")
        metrics = result.get("metrics", {})

        for metric_name, expected_value in metrics.items():
            if expected_value is not None and not isinstance(expected_value, dict):
                lookup[(model_id, dataset_id, metric_name)] = expected_value

    return lookup


def parse_directory_name(dir_name):
    """Parse model, dataset, and metric information from directory name"""
    # Directory format: model_dataset_metric
    parts = dir_name.split("_")
    if len(parts) < 3:
        return None, None, None

    # Try to find the boundary between dataset and metric
    # Assume metric is the last part
    metric = parts[-1]

    # Middle parts are dataset, beginning parts are model
    # Need smarter parsing since both model and dataset names may contain underscores
    model_parts = []
    dataset_parts = []

    # Simple strategy: first half is model, middle is dataset, last is metric
    mid_point = len(parts) // 2
    model_parts = parts[:mid_point]
    dataset_parts = parts[mid_point:-1]

    model = "_".join(model_parts) if model_parts else ""
    dataset = "_".join(dataset_parts) if dataset_parts else ""

    return model, dataset, metric


def compare_results(results_dir, reference_json):
    """Compare actual results with reference data"""
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return

    print(f"📁 Results directory: {results_dir}")
    print(f"📋 Reference file: {reference_json}")
    print("=" * 80)

    # Load reference data
    reference_lookup = load_reference_data(reference_json)
    if not reference_lookup:
        print("❌ Failed to load reference data")
        return

    print(f"📊 Total metrics in reference data: {len(reference_lookup)}")
    print()

    # Find all results.json files
    results_pattern = os.path.join(results_dir, "**/results.json")
    results_files = glob.glob(results_pattern, recursive=True)

    if not results_files:
        print("❌ No results.json files found")
        return

    print(f"📊 Found {len(results_files)} results.json files")
    print()

    matches = []
    mismatches = []
    not_found = []

    for results_file in sorted(results_files):
        try:
            # Parse directory name
            dir_name = os.path.basename(os.path.dirname(results_file))
            model, dataset, metric = parse_directory_name(dir_name)

            # Read actual results
            with open(results_file, "r", encoding="utf-8") as f:
                actual_data = json.load(f)

            actual_model = actual_data.get("model_name", model)
            actual_dataset = actual_data.get("dataset_name", dataset)

            # Get primary metric value
            primary_metric = None
            actual_value = None

            # Try to find the primary metric
            if metric in actual_data:
                primary_metric = metric
                actual_value = actual_data[metric]
            else:
                # Find the first numeric metric
                for key, value in actual_data.items():
                    if isinstance(value, (int, float)) and key not in [
                        "total_samples",
                        "processing_time",
                    ]:
                        primary_metric = key
                        actual_value = value
                        break

            if primary_metric is None or actual_value is None:
                not_found.append(
                    {"file": results_file, "reason": "Unable to find valid metric value"}
                )
                continue

            # Search in reference data
            # Try multiple matching strategies
            expected_value = None
            lookup_key = None

            # Strategy 1: Exact match
            key1 = (actual_model, actual_dataset, primary_metric)
            if key1 in reference_lookup:
                expected_value = reference_lookup[key1]
                lookup_key = key1
            else:
                # Strategy 2: Fuzzy match - check containment relationships
                for (ref_model, ref_dataset, ref_metric), ref_value in reference_lookup.items():
                    if (
                        primary_metric == ref_metric
                        and (actual_model in ref_model or ref_model in actual_model)
                        and (actual_dataset in ref_dataset or ref_dataset in actual_dataset)
                    ):
                        expected_value = ref_value
                        lookup_key = (ref_model, ref_dataset, ref_metric)
                        break

            if expected_value is not None:
                # Handle percentage format differences
                # If actual value is 0-1 and expected value > 1, expected might be percentage
                adjusted_expected = expected_value
                percentage_conversion = False

                if 0 <= actual_value <= 1 and expected_value > 1:
                    # Expected value might be percentage, convert to decimal
                    adjusted_expected = expected_value / 100
                    percentage_conversion = True
                elif actual_value > 1 and 0 <= expected_value <= 1:
                    # Actual value might be percentage, convert expected value
                    adjusted_expected = expected_value * 100
                    percentage_conversion = True

                # Calculate difference
                diff = abs(actual_value - adjusted_expected)
                relative_diff = (
                    diff / abs(adjusted_expected) if adjusted_expected != 0 else float("inf")
                )

                result_info = {
                    "file": results_file,
                    "dir_name": dir_name,
                    "model": actual_model,
                    "dataset": actual_dataset,
                    "metric": primary_metric,
                    "actual_value": actual_value,
                    "expected_value": expected_value,
                    "adjusted_expected": adjusted_expected,
                    "percentage_conversion": percentage_conversion,
                    "diff": diff,
                    "relative_diff": relative_diff,
                    "lookup_key": lookup_key,
                }

                # Determine if it's a match (allow 20% numerical error)
                if relative_diff < 0.20:  # 20% error range
                    matches.append(result_info)
                else:
                    mismatches.append(result_info)
            else:
                not_found.append(
                    {
                        "file": results_file,
                        "dir_name": dir_name,
                        "model": actual_model,
                        "dataset": actual_dataset,
                        "metric": primary_metric,
                        "actual_value": actual_value,
                        "reason": "No matching entry found in reference data",
                    }
                )

        except Exception as e:
            not_found.append({"file": results_file, "reason": f"Error processing file: {e}"})

    # Display results
    print("🎯 Comparison results:")
    print(f"  ✅ Matches: {len(matches)}")
    print(f"  ❌ Mismatches: {len(mismatches)}")
    print(f"  ❓ Not found: {len(not_found)}")
    print()

    if matches:
        print("✅ Matching results:")
        for i, match in enumerate(matches, 1):
            if match["percentage_conversion"]:
                print(
                    f"  {i:2d}. {match['metric']}: {match['actual_value']:.4f} ≈ {match['adjusted_expected']:.4f} (original: {match['expected_value']:.1f}%)"
                )
            else:
                print(
                    f"  {i:2d}. {match['metric']}: {match['actual_value']:.4f} ≈ {match['expected_value']:.4f}"
                )
            print(f"      📁 {match['dir_name']}")
        print()

    if mismatches:
        print("❌ Mismatched results:")
        for i, mismatch in enumerate(mismatches[:20], 1):
            if mismatch["percentage_conversion"]:
                print(
                    f"  {i:2d}. {mismatch['metric']}: {mismatch['actual_value']:.4f} vs {mismatch['adjusted_expected']:.4f} (original: {mismatch['expected_value']:.1f}%, diff: {mismatch['relative_diff']:.2%})"
                )
            else:
                print(
                    f"  {i:2d}. {mismatch['metric']}: {mismatch['actual_value']:.4f} vs {mismatch['expected_value']:.4f} (diff: {mismatch['relative_diff']:.2%})"
                )
            print(f"      📁 {mismatch['dir_name']}")
        if len(mismatches) > 10:
            print(f"      ... ({len(mismatches)-10} more mismatches)")
        print()

    if not_found:
        print("❓ Not found results:")
        for i, nf in enumerate(not_found[:10], 1):
            print(f"  {i:2d}. {nf.get('reason', 'Unknown reason')}")
            print(f"      📁 {nf.get('dir_name', os.path.basename(nf['file']))}")
        if len(not_found) > 10:
            print(f"      ... ({len(not_found)-10} more not found)")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "simple_results_0801"
    reference_json = sys.argv[2] if len(sys.argv) > 2 else "perfect_model_dataset_metrics.json"

    # Convert to absolute paths
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.getcwd(), results_dir)
    if not os.path.isabs(reference_json):
        reference_json = os.path.join(os.getcwd(), reference_json)

    compare_results(results_dir, reference_json)


if __name__ == "__main__":
    main()
