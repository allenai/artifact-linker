#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from artifact_graph.collectors.metric_collector import MetricCollector


def extract_dataset_name_from_path(path: str) -> str:
    """
    Extract dataset name from accuracy path.
    
    Examples:
    - "GPQA-D.Accuracy" -> "GPQA-D"
    - "AIME 2025.Accuracy" -> "AIME 2025"
    - "MATH-500.Accuracy" -> "MATH-500"
    - "boolq.Accuracy" -> "boolq"
    """
    # Split by dot and take the first part (dataset name)
    if '.' in path:
        return path.split('.')[0]
    return path


def split_dataset_names(dataset_names: List[str]) -> List[str]:
    """
    Split dataset names by _, /, or - and flatten the results.
    
    Examples:
    - "GPQA-D" -> ["GPQA", "D"]
    - "AIME_2025" -> ["AIME", "2025"]
    - "user/dataset-name" -> ["user", "dataset", "name"]
    """
    all_parts = set()
    
    for name in dataset_names:
        # Split by _, /, or -
        parts = re.split(r'[_/\-]', name)
        for part in parts:
            # Clean and filter parts
            part = part.strip()
            if part and len(part) > 1:  # Only keep non-empty parts with length > 1
                all_parts.add(part)
    
    return sorted(list(all_parts))


def find_accuracy_in_metrics(metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search for accuracy values in metrics data.
    Returns list of accuracy entries found.
    """
    accuracy_entries = []
    
    def search_recursive(obj, path=""):
        """Recursively search for accuracy keys in nested dictionaries."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if key contains 'accuracy' or 'acc' (case insensitive)
                key_lower = key.lower()
                if 'accuracy' in key_lower or 'acc' in key_lower:
                    # Check if value is a number
                    if isinstance(value, (int, float)):
                        accuracy_entries.append({
                            'path': current_path,
                            'key': key,
                            'value': value
                        })
                    elif isinstance(value, dict):
                        # If accuracy key points to a dict, look for numeric values inside
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                accuracy_entries.append({
                                    'path': f"{current_path}.{sub_key}",
                                    'key': f"{key}.{sub_key}",
                                    'value': sub_value
                                })
                
                # Continue recursive search
                if isinstance(value, (dict, list)):
                    search_recursive(value, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                search_recursive(item, current_path)
    
    search_recursive(metrics_data)
    return accuracy_entries


def find_datasets_with_accuracy(metrics_dir: str = "output/metrics") -> Dict[str, List[Dict[str, Any]]]:
    """
    Find all datasets that have accuracy metrics.
    
    Args:
        metrics_dir: Directory containing metric files
        
    Returns:
        Dictionary mapping dataset names to their accuracy entries
    """
    metrics_path = Path(metrics_dir)
    datasets_with_accuracy = {}
    
    if not metrics_path.exists():
        print(f"Metrics directory '{metrics_dir}' does not exist.")
        return datasets_with_accuracy
    
    print(f"Searching for accuracy metrics in '{metrics_dir}'...")
    
    # Iterate through all JSON files in metrics directory
    json_files = list(metrics_path.glob("*.json"))
    print(f"Found {len(json_files)} metric files to check.")
    
    for json_file in json_files:
        try:
            # Extract dataset name from filename
            dataset_name = json_file.stem.replace("__", "/")
            
            # Load metrics data
            with open(json_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # Search for accuracy in metrics
            accuracy_entries = find_accuracy_in_metrics(metrics_data)
            
            if accuracy_entries:
                datasets_with_accuracy[dataset_name] = accuracy_entries
                print(f"✓ Found accuracy in {dataset_name}: {len(accuracy_entries)} entries")
        
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}")
            continue
    
    return datasets_with_accuracy


def extract_unique_datasets_from_paths(results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Extract unique dataset names from accuracy entry paths."""
    unique_datasets = set()
    
    for model_name, accuracy_entries in results.items():
        for entry in accuracy_entries:
            dataset_name = extract_dataset_name_from_path(entry['path'])
            unique_datasets.add(dataset_name)
    
    return sorted(list(unique_datasets))


def save_results(results: Dict[str, List[Dict[str, Any]]], output_file: str = "datasets_with_accuracy.json"):
    """Save the results to a JSON file."""
    output_path = Path(output_file)
    
    # Extract unique dataset names from paths
    unique_datasets_from_paths = extract_unique_datasets_from_paths(results)
    
    # Split dataset names
    split_dataset_names_list = split_dataset_names(unique_datasets_from_paths)
    
    # Save only the path dataset names as independent file
    path_dataset_list_file = output_path.stem + "_path_dataset_names.json"
    with open(path_dataset_list_file, 'w', encoding='utf-8') as f:
        json.dump(unique_datasets_from_paths, f, indent=2, ensure_ascii=False)
    
    print(f"Path dataset names list saved to {path_dataset_list_file}")
    
    # Save split dataset names as independent file
    split_dataset_list_file = output_path.stem + "_split_dataset_names.json"
    with open(split_dataset_list_file, 'w', encoding='utf-8') as f:
        json.dump(split_dataset_names_list, f, indent=2, ensure_ascii=False)
    
    print(f"Split dataset names list saved to {split_dataset_list_file}")


def print_summary(results: Dict[str, List[Dict[str, Any]]]):
    """Print a summary of findings."""
    unique_datasets_from_paths = extract_unique_datasets_from_paths(results)
    split_datasets = split_dataset_names(unique_datasets_from_paths)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found {len(results)} models with accuracy metrics")
    print(f"Unique datasets from metric paths: {len(unique_datasets_from_paths)}")
    print(f"Split dataset parts: {len(split_datasets)}")
    print(f"{'='*60}")
    
    if not results:
        print("No datasets with accuracy metrics found.")
        return
    
    # Print unique dataset names from paths
    print(f"\n📋 Unique Dataset Names from Metric Paths ({len(unique_datasets_from_paths)}):")
    for i, dataset in enumerate(unique_datasets_from_paths, 1):
        print(f"  {i:2d}. {dataset}")
    
    # Print split dataset names
    print(f"\n📋 Split Dataset Names ({len(split_datasets)}):")
    for i, dataset in enumerate(split_datasets, 1):
        print(f"  {i:2d}. {dataset}")
    
    # Sort by number of accuracy entries (descending)
    sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n📊 Top Models with Most Accuracy Entries:")
    for dataset_name, accuracy_entries in sorted_results[:5]:  # Show top 5
        path_datasets = [extract_dataset_name_from_path(entry['path']) for entry in accuracy_entries]
        unique_path_datasets = list(set(path_datasets))
        
        print(f"\n   {dataset_name}")
        print(f"   Path datasets: {', '.join(unique_path_datasets)}")
        print(f"   Accuracy entries: {len(accuracy_entries)}")
        
        # Show first few accuracy values
        for entry in accuracy_entries[:3]:  # Show first 3
            path_dataset = extract_dataset_name_from_path(entry['path'])
            print(f"   - {path_dataset}: {entry['key']} = {entry['value']}")
        
        if len(accuracy_entries) > 3:
            print(f"   ... and {len(accuracy_entries) - 3} more")


def main():
    """Main function to run the accuracy dataset finder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find datasets with accuracy metrics")
    parser.add_argument(
        "--metrics-dir", 
        default="output/metrics",
        help="Directory containing metric files (default: output/metrics)"
    )
    parser.add_argument(
        "--output", 
        default="datasets_with_accuracy.json",
        help="Output file for results (default: datasets_with_accuracy.json)"
    )
    parser.add_argument(
        "--download-first", 
        action="store_true",
        help="Download metrics first using MetricCollector"
    )
    parser.add_argument(
        "--dataset-names-only", 
        action="store_true",
        help="Only print the unique dataset names list from paths"
    )
    parser.add_argument(
        "--split-names", 
        action="store_true",
        help="Print split dataset names instead of original names"
    )
    
    args = parser.parse_args()
    
    # Optionally download metrics first
    if args.download_first:
        print("Downloading metrics first...")
        try:
            collector = MetricCollector()
            # This would need to be implemented in MetricCollector
            print("Note: Automatic metrics download not implemented yet.")
            print("Please run metric collection manually first.")
        except Exception as e:
            print(f"Error during metrics download: {e}")
            return
    
    # Find datasets with accuracy
    results = find_datasets_with_accuracy(args.metrics_dir)
    
    if args.dataset_names_only:
        # Just print the dataset names
        unique_datasets = extract_unique_datasets_from_paths(results)
        if args.split_names:
            split_datasets = split_dataset_names(unique_datasets)
            print(json.dumps(split_datasets, indent=2))
        else:
            print(json.dumps(unique_datasets, indent=2))
    else:
        # Print full summary
        print_summary(results)
        
        # Save results
        if results:
            save_results(results, args.output)
    
    print(f"\nDone! Found {len(results)} models with accuracy metrics.")


if __name__ == "__main__":
    main()
