#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import aiohttp
from tqdm import tqdm

from artifact_graph.collectors.dataset_collector import DatasetCollector


def load_cached_datasets(cache_file: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load cached dataset list from JSON file.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        List of dataset objects or None if file doesn't exist
    """
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            datasets = json.load(f)
        print(f"Loaded {len(datasets)} datasets from cache: {cache_file}")
        return datasets
    except FileNotFoundError:
        print(f"Cache file not found: {cache_file}")
        return None
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def save_cached_datasets(datasets: List[Any], cache_file: str) -> None:
    """
    Save dataset list to JSON cache file.
    
    Args:
        datasets: List of dataset objects
        cache_file: Path to cache file
    """
    try:
        # Convert dataset objects to dictionaries
        dataset_dicts = []
        for dataset in datasets:
            dataset_dict = {
                'id': dataset.id,
                'downloads': getattr(dataset, 'downloads', 0),
                'likes': getattr(dataset, 'likes', 0),
                'author': getattr(dataset, 'author', ''),
                'tags': getattr(dataset, 'tags', []),
                'private': getattr(dataset, 'private', False),
                'last_modified': str(getattr(dataset, 'last_modified', '')) if hasattr(dataset, 'last_modified') else '',
            }
            dataset_dicts.append(dataset_dict)
        
        # Save to file
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_dicts, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(datasets)} datasets to cache: {cache_file}")
    
    except Exception as e:
        print(f"Error saving cache: {e}")


async def download_dataset_async(session: aiohttp.ClientSession, dataset_id: str, 
                                collector: DatasetCollector, metadata_dir: str, 
                                readme_dir: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Download a single dataset asynchronously.
    
    Args:
        session: aiohttp session
        dataset_id: Dataset ID to download
        collector: DatasetCollector instance
        metadata_dir: Directory to save metadata
        readme_dir: Directory to save readme
        semaphore: Semaphore to limit concurrent requests
        
    Returns:
        Dictionary with download result
    """
    async with semaphore:
        try:
            # Check if already exists
            metadata_path = Path(metadata_dir) / f"{dataset_id.replace('/', '__')}.json"
            if metadata_path.exists():
                return {"dataset_id": dataset_id, "status": "skipped", "reason": "already exists"}
            
            # Download metadata and readme
            data = collector.collect_one(dataset_id)
            if "error" not in data:
                collector.save_metadata(dataset_id, data["metadata"], metadata_dir=metadata_dir)
                if data["readme"]:
                    collector.save_readme(dataset_id, data["readme"], readme_dir=readme_dir)
                return {"dataset_id": dataset_id, "status": "success"}
            else:
                return {"dataset_id": dataset_id, "status": "error", "reason": data["error"]}
                
        except Exception as e:
            return {"dataset_id": dataset_id, "status": "error", "reason": str(e)}


async def download_datasets_async(datasets: List[Dict[str, Any]], 
                                 collector: DatasetCollector, metadata_dir: str, 
                                 readme_dir: str, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Download multiple datasets concurrently.
    
    Args:
        datasets: List of datasets to download
        collector: DatasetCollector instance
        metadata_dir: Directory to save metadata
        readme_dir: Directory to save readme
        max_concurrent: Maximum number of concurrent downloads
        
    Returns:
        List of download results
    """
    # Create directories
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)
    Path(readme_dir).mkdir(parents=True, exist_ok=True)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create tasks for all datasets
        tasks = []
        for dataset in datasets:
            dataset_id = dataset['id'] if isinstance(dataset, dict) else dataset.id
            task = download_dataset_async(session, dataset_id, collector, metadata_dir, readme_dir, semaphore)
            tasks.append(task)
        
        # Execute all tasks with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Downloading datasets") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                
                # Update progress bar description with current status
                success_count = sum(1 for r in results if r["status"] == "success")
                error_count = sum(1 for r in results if r["status"] == "error")
                skipped_count = sum(1 for r in results if r["status"] == "skipped")
                pbar.set_postfix({
                    "success": success_count,
                    "error": error_count,
                    "skipped": skipped_count
                })
    
    return results


def main(limit: int, hf_token: str, cache_file: str = "cached_datasets.json", 
         force_refresh: bool = False, max_concurrent: int = 10) -> None:
    """
    Download metadata and READMEs for top Hugging Face datasets.
    """
    metadata_dir = "output/datasets/metadata"
    readme_dir = "output/datasets/readmes"
    
    collector = DatasetCollector(hf_token=hf_token)
    
    # Load or fetch dataset list
    all_datasets = None
    if not force_refresh:
        all_datasets = load_cached_datasets(cache_file)
    
    if all_datasets is None:
        print("Fetching dataset list from HuggingFace Hub...")
        try:
            all_datasets = list(collector.api.list_datasets(sort="downloads", full=True))
            save_cached_datasets(all_datasets, cache_file)
        except Exception as e:
            print(f"Error fetching datasets: {e}")
            return
    
    # Limit to top N datasets
    if limit:
        all_datasets = all_datasets[:limit]
    
    print(f"Downloading top {len(all_datasets)} datasets by downloads...")
    
    # Download datasets asynchronously
    print(f"Starting async download with {max_concurrent} concurrent requests...")
    results = asyncio.run(download_datasets_async(
        all_datasets, collector, metadata_dir, readme_dir, max_concurrent
    ))
    
    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⚠️  Skipped: {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    
    if error_count > 0:
        print(f"\n❌ Failed downloads:")
        for result in results:
            if result["status"] == "error":
                print(f"   - {result['dataset_id']}: {result['reason']}")

    print(f"\n✅ Download complete.")
    print(f"   - Metadata saved to: {metadata_dir}")
    print(f"   - READMEs saved to: {readme_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download metadata and READMEs for top Hugging Face datasets."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Number of top datasets to download, sorted by downloads (default: 1000).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="cached_datasets.json",
        help="Path to cache file for dataset list (default: cached_datasets.json).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh the dataset list cache.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent downloads (default: 10).",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set HF_TOKEN or pass --hf_token.")

    main(args.limit, args.hf_token, args.cache_file, args.force_refresh, args.max_concurrent) 