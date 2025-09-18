#!/usr/bin/env python3
"""
Helper script to explore available datasets and metrics for prediction.
"""

import json
from collections import defaultdict, Counter
import argparse


def analyze_available_data(json_file: str = "output/perfect_model_dataset_metrics.json"):
    """Analyze what datasets and metrics are available in the data."""
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        results = data.get("results", [])
    except FileNotFoundError:
        print(f"❌ Data file not found: {json_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {json_file}")
        return
    
    if not results:
        print("❌ No results found in the data file")
        return
    
    print(f"📊 ANALYZING DATA FROM: {json_file}")
    print("=" * 60)
    
    # Collect statistics
    datasets = set()
    models = set()
    metrics = defaultdict(int)
    dataset_model_count = defaultdict(int)
    model_dataset_count = defaultdict(int)
    
    for item in results:
        model_id = item.get("model_id")
        dataset_id = item.get("dataset_id")
        item_metrics = item.get("metrics", {})
        
        if model_id:
            models.add(model_id)
        if dataset_id:
            datasets.add(dataset_id)
            
        if model_id and dataset_id:
            dataset_model_count[dataset_id] += 1
            model_dataset_count[model_id] += 1
        
        # Count available metrics
        for metric_name in item_metrics.keys():
            metrics[metric_name] += 1
    
    # Print summary
    print(f"📈 SUMMARY STATISTICS")
    print(f"  Total entries: {len(results)}")
    print(f"  Unique models: {len(models)}")
    print(f"  Unique datasets: {len(datasets)}")
    print(f"  Available metrics: {len(metrics)}")
    
    # Print available metrics
    print(f"\n🎯 AVAILABLE METRICS")
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    for metric_name, count in sorted_metrics:
        print(f"  {metric_name:15s}: {count:4d} entries ({count/len(results)*100:.1f}%)")
    
    # Print top datasets by number of models
    print(f"\n📚 TOP DATASETS (by number of model evaluations)")
    sorted_datasets = sorted(dataset_model_count.items(), key=lambda x: x[1], reverse=True)
    for i, (dataset_name, count) in enumerate(sorted_datasets[:15]):
        print(f"  {i+1:2d}. {dataset_name:30s}: {count:3d} models")
    
    if len(sorted_datasets) > 15:
        print(f"     ... and {len(sorted_datasets) - 15} more datasets")
    
    # Print top models by number of datasets
    print(f"\n🤖 TOP MODELS (by number of dataset evaluations)")
    sorted_models = sorted(model_dataset_count.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, count) in enumerate(sorted_models[:10]):
        print(f"  {i+1:2d}. {model_name:40s}: {count:3d} datasets")
    
    if len(sorted_models) > 10:
        print(f"     ... and {len(sorted_models) - 10} more models")
    
    return {
        'datasets': sorted(datasets),
        'models': sorted(models),
        'metrics': dict(metrics),
        'dataset_model_count': dict(dataset_model_count),
        'model_dataset_count': dict(model_dataset_count)
    }


def search_datasets(query: str, json_file: str = "output/perfect_model_dataset_metrics.json"):
    """Search for datasets matching a query."""
    
    data_info = analyze_available_data(json_file)
    if not data_info:
        return
    
    datasets = data_info['datasets']
    dataset_model_count = data_info['dataset_model_count']
    
    # Search for datasets matching the query
    query_lower = query.lower()
    matching_datasets = [
        d for d in datasets 
        if query_lower in d.lower()
    ]
    
    print(f"\n🔍 SEARCH RESULTS for '{query}'")
    print("=" * 60)
    
    if matching_datasets:
        print(f"Found {len(matching_datasets)} matching datasets:")
        for dataset in matching_datasets:
            model_count = dataset_model_count.get(dataset, 0)
            print(f"  📚 {dataset:40s}: {model_count:3d} models")
        
        print(f"\n💡 USAGE EXAMPLES:")
        for dataset in matching_datasets[:3]:  # Show examples for first 3
            print(f"  python scripts/predict_models_for_dataset.py --dataset '{dataset}' --metric accuracy")
    else:
        print(f"❌ No datasets found matching '{query}'")
        print(f"\n💡 Try searching for common dataset types:")
        print("  • 'squad' for reading comprehension")
        print("  • 'glue' for GLUE benchmark tasks")
        print("  • 'wmt' for translation tasks")
        print("  • 'cnn' for summarization")
        print("  • 'imdb' for sentiment analysis")


def main():
    parser = argparse.ArgumentParser(description="Explore available datasets and metrics")
    parser.add_argument("--search", "-s", help="Search for datasets matching a query")
    parser.add_argument("--file", "-f", default="output/perfect_model_dataset_metrics.json",
                       help="JSON file to analyze")
    
    args = parser.parse_args()
    
    if args.search:
        search_datasets(args.search, args.file)
    else:
        analyze_available_data(args.file)


if __name__ == "__main__":
    main()
