import contextlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


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
            # Suppress the verbose download progress bar
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
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

    def _process_and_save_item(
        self, item: Any, metadata_dir: str, readme_dir: str
    ) -> Dict[str, Any]:
        """Helper function to process and save a single item (dataset)."""
        item_id = item.id if not isinstance(item, dict) else item["id"]

        # Skip if metadata file already exists
        metadata_path = Path(metadata_dir) / f"{item_id.replace('/', '__')}.json"
        if metadata_path.exists():
            return {"dataset_id": item_id, "status": "skipped"}

        try:
            data = self.collect_one(item_id)
            if "error" not in data:
                self.save_metadata(item_id, data["metadata"], metadata_dir=metadata_dir)
                if data["readme"]:
                    self.save_readme(item_id, data["readme"], readme_dir=readme_dir)
                return {"dataset_id": item_id, "status": "success"}
            else:
                return {
                    "dataset_id": item_id,
                    "status": "error",
                    "reason": data.get("error", "Unknown error"),
                }
        except Exception as e:
            return {"dataset_id": item_id, "status": "error", "reason": str(e)}

    def collect_all(
        self,
        datasets: List[Any],
        metadata_dir: str,
        readme_dir: str,
        max_concurrent: int,
    ) -> List[Dict[str, Any]]:
        """Collects all datasets, saving them progressively in parallel using threads."""
        ensure_directory(Path(metadata_dir))
        ensure_directory(Path(readme_dir))

        results = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            process_func = partial(
                self._process_and_save_item,
                metadata_dir=metadata_dir,
                readme_dir=readme_dir,
            )
            with tqdm(total=len(datasets), desc="Downloading datasets") as pbar:
                for result in executor.map(process_func, datasets):
                    results.append(result)
                    pbar.update(1)

                    # Update postfix with live stats
                    success_count = sum(1 for r in results if r["status"] == "success")
                    error_count = sum(1 for r in results if r["status"] == "error")
                    skipped_count = sum(1 for r in results if r["status"] == "skipped")
                    pbar.set_postfix(
                        {"success": success_count, "errors": error_count, "skipped": skipped_count}
                    )
        return results

    def collect_top_datasets(
        self,
        min_downloads: int = 100,
        metadata_dir: str = "output/datasets/metadata",
        readme_dir: str = "output/datasets/readmes",
        max_concurrent: int = 10,
        cache_file: str = "cached_datasets.json",
        force_refresh: bool = False,
    ) -> None:
        """
        Downloads metadata and READMEs for Hugging Face datasets with downloads >= min_downloads.

        Args:
            min_downloads: Minimum number of downloads required (default: 100)
            metadata_dir: Directory to save metadata files
            readme_dir: Directory to save README files
            max_concurrent: Maximum concurrent downloads
            cache_file: Path to cache file for dataset list
            force_refresh: Force refresh the dataset list cache
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

        # Filter datasets by download threshold
        filtered_datasets = []
        total_datasets = len(all_datasets)

        print(f"Filtering {total_datasets} datasets by download threshold (>= {min_downloads})...")

        for dataset in all_datasets:
            downloads = (
                dataset["downloads"]
                if isinstance(dataset, dict)
                else getattr(dataset, "downloads", 0)
            )
            if downloads >= min_downloads:
                filtered_datasets.append(dataset)

        print(f"Found {len(filtered_datasets)} datasets with >= {min_downloads} downloads")

        # Check for already downloaded datasets
        print(f"Checking for existing datasets in {metadata_dir}...")
        metadata_path = Path(metadata_dir)
        ensure_directory(metadata_path)
        existing_dataset_files = {f.stem for f in metadata_path.glob("*.json")}

        to_download = []
        for dataset in filtered_datasets:
            dataset_id = dataset.id if not isinstance(dataset, dict) else dataset["id"]
            dataset_fname = dataset_id.replace("/", "__")
            if dataset_fname not in existing_dataset_files:
                to_download.append(dataset)

        print(
            f"Found {len(existing_dataset_files)} existing datasets. Need to download {len(to_download)} new datasets."
        )

        if not to_download:
            print("All datasets are already up to date.")
            return

        print(f"Downloading {len(to_download)} new datasets...")
        print(f"Starting download with {max_concurrent} concurrent requests...")

        results = self.collect_all(to_download, metadata_dir, readme_dir, max_concurrent)

        # The rest of the summary logic...
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")

        print("\n📊 Download Summary:")
        print(f"   ✅ Success: {success_count}")
        print(f"   ⚠️  Skipped: {skipped_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   📊 Total datasets processed: {len(to_download)}")
        print(f"   🎯 Download threshold: >= {min_downloads}")

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
        fname = dataset_id.replace("/", "__") + ".md"
        path = Path(readme_dir) / fname
        print(path)
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
