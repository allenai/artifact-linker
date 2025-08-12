#!/usr/bin/env python3

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def check_file_coverage(cache_file, metadata_dir, readme_dir, entity_type="dataset", download_threshold=100):
    """
    Check which entities with >download_threshold downloads have their metadata and readme files stored.
    
    Args:
        cache_file: Path to the cached JSON file
        metadata_dir: Directory containing metadata files
        readme_dir: Directory containing readme files
        entity_type: "dataset" or "model"
        download_threshold: Minimum downloads to consider
    
    Returns:
        Dictionary with coverage statistics
    """
    print(f"\n{'='*60}")
    print(f"Checking {entity_type} coverage from {cache_file}")
    print(f"{'='*60}")
    
    # Load cached data
    with open(cache_file, 'r') as f:
        entities = json.load(f)
    
    # Filter entities with downloads above threshold
    high_download_entities = [entity for entity in entities if entity.get('downloads', 0) > download_threshold]
    print(f"Found {len(high_download_entities)} {entity_type}s with >{download_threshold} downloads")
    
    # Check file availability
    entities_with_metadata = []
    entities_with_readme = []
    entities_with_both = []
    entities_missing_both = []
    
    for entity in high_download_entities:
        entity_id = entity['id'].replace('/', '__')
        
        # Check metadata file
        if entity_type == "dataset":
            metadata_file = os.path.join(metadata_dir, f"{entity_id}.json")
        else:  # model
            metadata_file = os.path.join(metadata_dir, f"{entity_id}.json")
        
        # Check readme file
        if entity_type == "dataset":
            readme_file = os.path.join(readme_dir, f"{entity_id}.md")
        else:  # model
            readme_file = os.path.join(readme_dir, f"{entity_id}.md")
        
        has_metadata = os.path.exists(metadata_file)
        has_readme = os.path.exists(readme_file)
        
        if has_metadata:
            entities_with_metadata.append(entity_id)
        if has_readme:
            entities_with_readme.append(entity_id)
        if has_metadata and has_readme:
            entities_with_both.append(entity_id)
        if not has_metadata and not has_readme:
            entities_missing_both.append(entity_id)
    
    # Print statistics
    total = len(high_download_entities)
    print(f"\n{entity_type.capitalize()}s with >{download_threshold} downloads:")
    print(f"  - Total: {total}")
    print(f"  - With metadata: {len(entities_with_metadata)} ({len(entities_with_metadata)/total*100:.1f}%)")
    print(f"  - With readme: {len(entities_with_readme)} ({len(entities_with_readme)/total*100:.1f}%)")
    print(f"  - With both: {len(entities_with_both)} ({len(entities_with_both)/total*100:.1f}%)")
    print(f"  - Missing both: {len(entities_missing_both)} ({len(entities_missing_both)/total*100:.1f}%)")
    
    # Show examples of missing files
    if entities_missing_both:
        print(f"\n{entity_type.capitalize()}s missing both metadata and readme files:")
        for entity_id in entities_missing_both[:10]:  # Show first 10
            entity = next(e for e in high_download_entities if e['id'].replace('/', '__') == entity_id)
            print(f"  - {entity['id']} ({entity['downloads']} downloads)")
        if len(entities_missing_both) > 10:
            print(f"  ... and {len(entities_missing_both) - 10} more")
    
    return {
        'total': total,
        'with_metadata': len(entities_with_metadata),
        'with_readme': len(entities_with_readme),
        'with_both': len(entities_with_both),
        'missing_both': len(entities_missing_both),
        'metadata_percentage': len(entities_with_metadata)/total*100 if total > 0 else 0,
        'readme_percentage': len(entities_with_readme)/total*100 if total > 0 else 0,
        'both_percentage': len(entities_with_both)/total*100 if total > 0 else 0,
        'missing_both_percentage': len(entities_missing_both)/total*100 if total > 0 else 0
    }

def plot_coverage_comparison(dataset_stats, model_stats):
    """Plot coverage comparison between datasets and models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset coverage
    labels = ['Metadata', 'Readme', 'Both', 'Missing Both']
    dataset_values = [
        dataset_stats['with_metadata'],
        dataset_stats['with_readme'], 
        dataset_stats['with_both'],
        dataset_stats['missing_both']
    ]
    dataset_percentages = [
        dataset_stats['metadata_percentage'],
        dataset_stats['readme_percentage'],
        dataset_stats['both_percentage'],
        dataset_stats['missing_both_percentage']
    ]
    
    # Model coverage
    model_values = [
        model_stats['with_metadata'],
        model_stats['with_readme'],
        model_stats['with_both'],
        model_stats['missing_both']
    ]
    model_percentages = [
        model_stats['metadata_percentage'],
        model_stats['readme_percentage'],
        model_stats['both_percentage'],
        model_stats['missing_both_percentage']
    ]
    
    # Plot 1: Dataset coverage (counts)
    ax1.bar(labels, dataset_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    ax1.set_title('Dataset Coverage (Counts)')
    ax1.set_ylabel('Number of Datasets')
    for i, v in enumerate(dataset_values):
        ax1.text(i, v + max(dataset_values)*0.01, str(v), ha='center', va='bottom')
    
    # Plot 2: Model coverage (counts)
    ax2.bar(labels, model_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    ax2.set_title('Model Coverage (Counts)')
    ax2.set_ylabel('Number of Models')
    for i, v in enumerate(model_values):
        ax2.text(i, v + max(model_values)*0.01, str(v), ha='center', va='bottom')
    
    # Plot 3: Dataset coverage (percentages)
    ax3.bar(labels, dataset_percentages, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    ax3.set_title('Dataset Coverage (Percentages)')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_ylim(0, 100)
    for i, v in enumerate(dataset_percentages):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Model coverage (percentages)
    ax4.bar(labels, model_percentages, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    ax4.set_title('Model Coverage (Percentages)')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_ylim(0, 100)
    for i, v in enumerate(model_percentages):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to check both datasets and models coverage."""
    
    # Check datasets
    dataset_stats = check_file_coverage(
        cache_file='cached_datasets.json',
        metadata_dir='../data/output/datasets/metadata',
        readme_dir='../data/output/datasets/readmes',
        entity_type="dataset",
        download_threshold=100
    )
    
    # Check models
    model_stats = check_file_coverage(
        cache_file='cached_models.json',
        metadata_dir='output/models/metadata',
        readme_dir='output/models/readmes',
        entity_type="model",
        download_threshold=100
    )
    
    # Plot comparison
    plot_coverage_comparison(dataset_stats, model_stats)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Datasets with >100 downloads: {dataset_stats['total']}")
    print(f"  - Complete coverage: {dataset_stats['both_percentage']:.1f}%")
    print(f"  - Missing files: {dataset_stats['missing_both_percentage']:.1f}%")
    print(f"\nModels with >100 downloads: {model_stats['total']}")
    print(f"  - Complete coverage: {model_stats['both_percentage']:.1f}%")
    print(f"  - Missing files: {model_stats['missing_both_percentage']:.1f}%")

if __name__ == "__main__":
    main() 