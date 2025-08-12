#!/usr/bin/env python3
"""
Keep only: the highest download dataset_id and its metrics for perfect fuzzy match (score=100) for each model
"""

import json
import sys
from typing import Dict, Any, Tuple, Optional, List


def load_metric_pairings(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Cannot parse JSON file {file_path}")
        sys.exit(1)


def has_perfect_fuzzy_match(pairing: Dict[str, Any]) -> bool:
    """Check if there exists a fuzzy match with score==100"""
    for m in pairing.get('top_fuzzy_matches', []):
        if isinstance(m, dict) and float(m.get('score', 0)) == 100.0:
            return True
    return False


def get_max_download_dataset(pairing: Dict[str, Any], min_downloads: int = 100) -> Tuple[Optional[str], int]:
    """
    Find the dataset id with highest downloads from pairing['best_match']['ids']
    Only consider datasets with downloads > min_downloads
    Try field order: id / dataset_id / repo_id
    Returns (dataset_id, downloads)
    """
    best_match = pairing.get('best_match') or {}
    ids = best_match.get('ids', [])
    best_id = None
    best_downloads = -1
    for id_info in ids:
        if not isinstance(id_info, dict):
            continue
        downloads = int(id_info.get('downloads', 0))
        # Only consider datasets with downloads > min_downloads
        if downloads <= min_downloads:
            continue
        # Compatible with different field names
        ds_id = id_info.get('id') or id_info.get('dataset_id') or id_info.get('repo_id')
        if ds_id and downloads > best_downloads:
            best_id = ds_id
            best_downloads = downloads
    return best_id, best_downloads


def load_cached_data(cache_file: str) -> Dict[str, int]:
    """
    Load cached data and return a mapping of id -> downloads
    """
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        return {entity['id']: entity.get('downloads', 0) for entity in entities}
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load {cache_file}")
        return {}


def get_model_downloads(model_id: str, model_downloads_map: Dict[str, int]) -> int:
    """
    Get model downloads from the cached models data
    """
    return model_downloads_map.get(model_id, 0)


def extract_model_dataset_metrics(data: Dict[str, Any], min_downloads: int = 100, 
                                  model_downloads_map: Dict[str, int] = None) -> List[Dict[str, Any]]:
    """
    Return list: each model -> {model_id, model_downloads, dataset_id, dataset_downloads, metrics}
    Rule: only consider pairings with perfect fuzzy match and both model/dataset downloads > min_downloads;
    if multiple for same model, take the one with highest downloads.
    """
    if model_downloads_map is None:
        model_downloads_map = {}
        
    model_best: Dict[str, Dict[str, Any]] = {}

    for pairing in data.get('pairings', []):
        if not has_perfect_fuzzy_match(pairing):
            continue

        model_id = pairing.get('model_id', 'Unknown')
        
        # Check if model has enough downloads
        model_downloads = get_model_downloads(model_id, model_downloads_map)
        if model_downloads <= min_downloads:
            continue
            
        dataset_id, dataset_downloads = get_max_download_dataset(pairing, min_downloads)
        if not dataset_id:  # No dataset with sufficient downloads
            continue

        metrics = pairing.get('raw_llm_output', {}).get('metrics', {})

        # Update if this model not recorded yet or current downloads higher
        prev = model_best.get(model_id)
        if (prev is None) or (dataset_downloads > prev['dataset_downloads']):
            model_best[model_id] = {
                "model_id": model_id,
                "model_downloads": model_downloads,
                "dataset_id": dataset_id,
                "dataset_downloads": dataset_downloads,
                "metrics": metrics
            }

    return list(model_best.values())


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    out = {
        "results": results,
        "total_count": len(results),
        "extraction_criteria": {
            "perfect_fuzzy_match": "Only keep pairings with fuzzy match score=100",
            "download_threshold": "Only consider models and datasets with >100 downloads",
            "dataset_selection": "When multiple for same model, select dataset_id with highest downloads"
        }
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract perfect fuzzy matched dataset with highest downloads for each model')
    parser.add_argument('-i', '--input', default='output/metric_pairings.json', help='Input file path')
    parser.add_argument('-o', '--output', default='output/perfect_model_dataset_metrics.json', help='Output file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--min-downloads', type=int, default=100, help='Minimum downloads threshold for models and datasets (default: 100)')
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    data = load_metric_pairings(args.input)
    
    print("Loading cached model downloads...")
    model_downloads_map = load_cached_data('cached_models.json')
    print(f"Loaded {len(model_downloads_map)} models from cache")

    print("Extracting...")
    results = extract_model_dataset_metrics(data, args.min_downloads, model_downloads_map)

    print(f"Total pairings: {len(data.get('pairings', []))}")
    print(f"Models meeting criteria: {len(results)}")
    print(f"Match rate: {len(results) / max(1, len(data.get('pairings', []))) * 100:.2f}%")

    if args.verbose:
        for r in results[:10]:
            print(f"\nModel: {r['model_id']} (downloads={r['model_downloads']})")
            print(f"Dataset: {r['dataset_id']} (downloads={r['dataset_downloads']})")
            if r['metrics']:
                print("Metrics:")
                for k, v in r['metrics'].items():
                    print(f"  {k}: {v}")

    print(f"Saving to {args.output}")
    save_results(results, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
