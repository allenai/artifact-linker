import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import asyncio
import aiohttp


def ensure_directory(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


class DatasetCollector:
    """
    Collects metadata and README content for Hugging Face datasets.
    """

    def __init__(self, hf_token: Optional[str] = None) -> None:
        self.token = hf_token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN must be set or provided.")
        self.api = HfApi(token=self.token)

    def _format_date(self, val: Any) -> str:
        if isinstance(val, datetime):
            return val.isoformat()
        return str(val) if val is not None else ""

    def collect_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch metadata dict for a single dataset without saving."""
        try:
            ds_info = self.api.dataset_info(dataset_id)
            return {
                "datasetId": ds_info.id,
                "sha": getattr(ds_info, "sha", None),
                "createdAt": self._format_date(getattr(ds_info, "created_at", None)),
                "lastModified": self._format_date(getattr(ds_info, "last_modified", None)),
                "tags": getattr(ds_info, "tags", []),
                "downloads": getattr(ds_info, "downloads", 0),
                "likes": getattr(ds_info, "likes", 0),
                "private": getattr(ds_info, "private", False),
                "author": getattr(ds_info, "author", ""),
            }
        except Exception as e:
            raise ValueError(f"Failed to fetch dataset '{dataset_id}': {str(e)}")

    def collect_readme(self, dataset_id: str) -> Optional[bytes]:
        """Download README.md for a dataset and return its bytes, without saving."""
        try:
            file_path = hf_hub_download(
                repo_id=dataset_id,
                filename="README.md",
                repo_type="dataset",
                token=self.token,
            )
            return Path(file_path).read_bytes()
        except Exception:
            return None

    def save_metadata(
        self,
        dataset_id: str,
        metadata: Dict[str, Any],
        metadata_dir: str = "dataset_metadata",
    ) -> Path:
        """Save metadata dict to a JSON file."""
        out_dir = Path(metadata_dir)
        ensure_directory(out_dir)
        fname = dataset_id.replace("/", "__") + ".json"
        path = out_dir / fname
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_readme(
        self,
        dataset_id: str,
        readme_bytes: bytes,
        readme_dir: str = "dataset_readmes",
    ) -> Path:
        """Save README bytes to a markdown file."""
        out_dir = Path(readme_dir)
        ensure_directory(out_dir)
        fname = dataset_id.replace("/", "__") + ".md"
        path = out_dir / fname
        path.write_bytes(readme_bytes)
        return path

    def collect_one(
        self,
        dataset_id: str,
    ) -> Dict[str, Union[Dict[str, Any], Optional[bytes]]]:
        """Collect metadata and readme content without saving files."""
        metadata = self.collect_metadata(dataset_id)
        readme_bytes = self.collect_readme(dataset_id)
        return {"metadata": metadata, "readme": readme_bytes}

    def collect_batch(
        self,
        dataset_ids: List[str],
        pause: float = 0.2,
    ) -> Dict[str, Any]:
        """Collect metadata and readmes for multiple datasets without saving."""
        results: Dict[str, Any] = {}
        for did in dataset_ids:
            try:
                results[did] = self.collect_one(did)
            except Exception as e:
                results[did] = {"error": str(e)}
            time.sleep(pause)
        return results

    def collect_all(
        self,
        sort: str = "downloads",
        limit: Optional[int] = None,
        metadata_dir: str = "output/datasets/metadata",
        readme_dir: str = "output/datasets/readmes",
        max_workers: int = 5,
    ) -> None:
        """Collects all datasets, sorted and optionally limited, and saves them progressively in parallel."""
        print("Fetching dataset list...")
        start_time = time.time()
        datasets = list(self.api.list_datasets(sort=sort, full=True))
        if limit:
            datasets = datasets[:limit]
        print(f"Finished fetching dataset list in {time.time() - start_time:.2f} seconds.")

        Path(metadata_dir).mkdir(parents=True, exist_ok=True)
        Path(readme_dir).mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(
                self._process_and_save_item,
                item_type="dataset",
                metadata_dir=metadata_dir,
                readme_dir=readme_dir,
            )
            list(
                tqdm(
                    executor.map(process_func, datasets),
                    total=len(datasets),
                    desc="Collecting and saving datasets",
                )
            )

    def _process_and_save_item(
        self, item: Any, item_type: str, metadata_dir: str, readme_dir: str
    ) -> None:
        """Helper function to process and save a single item (model or dataset)."""
        item_id = item.id

        # Skip if metadata file already exists
        metadata_path = Path(metadata_dir) / f"{item_id.replace('/', '__')}.json"
        if metadata_path.exists():
            return

        try:
            data = self.collect_one(item_id)
            if "error" not in data:
                self.save_metadata(item_id, data["metadata"], metadata_dir=metadata_dir)
                if data["readme"]:
                    self.save_readme(item_id, data["readme"], readme_dir=readme_dir)
        except Exception as e:
            print(f"Error processing {item_id}: {e}")

    async def _download_dataset_async(
        self,
        dataset_id: str,
        metadata_dir: str,
        readme_dir: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """
        Download a single dataset asynchronously.
        """
        async with semaphore:
            try:
                metadata_path = Path(metadata_dir) / f"{dataset_id.replace('/', '__')}.json"
                if metadata_path.exists():
                    return {"dataset_id": dataset_id, "status": "skipped", "reason": "already exists"}

                data = self.collect_one(dataset_id)
                if "error" not in data:
                    self.save_metadata(dataset_id, data["metadata"], metadata_dir=metadata_dir)
                    if data["readme"]:
                        self.save_readme(dataset_id, data["readme"], readme_dir=readme_dir)
                    return {"dataset_id": dataset_id, "status": "success"}
                else:
                    return {"dataset_id": dataset_id, "status": "error", "reason": data["error"]}

            except Exception as e:
                return {"dataset_id": dataset_id, "status": "error", "reason": str(e)}

    async def _download_all_async(
        self,
        datasets: List[Dict[str, Any]],
        metadata_dir: str,
        readme_dir: str,
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Download multiple datasets concurrently.
        """
        Path(metadata_dir).mkdir(parents=True, exist_ok=True)
        Path(readme_dir).mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrent)

        tasks = []
        for dataset in datasets:
            dataset_id = dataset["id"] if isinstance(dataset, dict) else dataset.id
            task = self._download_dataset_async(
                dataset_id, metadata_dir, readme_dir, semaphore
            )
            tasks.append(task)

        results = []
        success_count = 0
        error_count = 0
        skipped_count = 0
        with tqdm(total=len(tasks), desc="Downloading datasets") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                elif result["status"] == "error":
                    error_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1

                pbar.update(1)
                pbar.set_postfix(
                    {"success": success_count, "error": error_count, "skipped": skipped_count}
                )
        return results

    def collect_top_datasets(
        self,
        limit: int,
        metadata_dir: str,
        readme_dir: str,
        max_concurrent: int,
        cache_file: str = "cached_datasets.json",
        force_refresh: bool = False,
    ) -> None:
        """
        Downloads metadata and READMEs for top Hugging Face datasets.
        """
        all_datasets = None
        if not force_refresh:
            all_datasets = self.load_cached_datasets(cache_file)

        if all_datasets is None:
            print("Fetching dataset list from HuggingFace Hub...")
            try:
                all_datasets = list(self.api.list_datasets(sort="downloads", full=True))
                self.save_cached_datasets(all_datasets, cache_file)
            except Exception as e:
                print(f"Error fetching datasets: {e}")
                return

        if limit:
            all_datasets = all_datasets[:limit]

        print(f"Downloading top {len(all_datasets)} datasets by downloads...")
        print(f"Starting download with {max_concurrent} concurrent requests...")

        results = asyncio.run(
            self._download_all_async(all_datasets, metadata_dir, readme_dir, max_concurrent)
        )

        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")

        print("\n📊 Download Summary:")
        print(f"   ✅ Success: {success_count}")
        print(f"   ⚠️  Skipped: {skipped_count}")
        print(f"   ❌ Errors: {error_count}")

        if error_count > 0:
            print("\n❌ Failed downloads:")
            for result in results:
                if result["status"] == "error":
                    print(f"   - {result['dataset_id']}: {result['reason']}")

        print("\n✅ Download complete.")
        print(f"   - Metadata saved to: {metadata_dir}")
        print(f"   - READMEs saved to: {readme_dir}")

    @staticmethod
    def load_cached_datasets(cache_file: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached dataset list from JSON file.
        """
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                datasets = json.load(f)
            print(f"Loaded {len(datasets)} datasets from cache: {cache_file}")
            return datasets
        except FileNotFoundError:
            print(f"Cache file not found: {cache_file}")
            return None
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    @staticmethod
    def save_cached_datasets(datasets: List[Any], cache_file: str) -> None:
        """
        Save dataset list to JSON cache file.
        """
        try:
            dataset_dicts = []
            for dataset in datasets:
                dataset_dict = {
                    "id": dataset.id,
                    "downloads": getattr(dataset, "downloads", 0),
                    "likes": getattr(dataset, "likes", 0),
                    "author": getattr(dataset, "author", ""),
                    "tags": getattr(dataset, "tags", []),
                    "private": getattr(dataset, "private", False),
                    "last_modified": str(getattr(dataset, "last_modified", ""))
                    if hasattr(dataset, "last_modified")
                    else "",
                }
                dataset_dicts.append(dataset_dict)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(dataset_dicts, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(datasets)} datasets to cache: {cache_file}")

        except Exception as e:
            print(f"Error saving cache: {e}")

    @staticmethod
    def load_metadata(
        dataset_id: str, metadata_dir: str = "dataset_metadata"
    ) -> Optional[Dict[str, Any]]:
        """Load saved metadata JSON for a given dataset ID."""
        fname = dataset_id.replace("/", "__") + ".json"
        path = Path(metadata_dir) / fname
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_readme(dataset_id: str, readme_dir: str = "dataset_readmes") -> Optional[bytes]:
        """Load saved README file for a given dataset ID."""
        fname = dataset_id.replace("/", "__") + "_README.md"
        path = Path(readme_dir) / fname
        if not path.exists():
            return None
        return path.read_bytes()

    @staticmethod
    def load_all_metadata(
        metadata_dir: str = "dataset_metadata", min_downloads: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """Load all metadata files from directory and filter by downloads."""
        results = {}
        metadata_path = Path(metadata_dir)
        if not metadata_path.exists():
            return results

        for file in metadata_path.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if data.get("downloads", 0) >= min_downloads:
                    dataset_id = file.stem.replace("__", "/")
                    results[dataset_id] = data
            except Exception:
                continue
        return results
