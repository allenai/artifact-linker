import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial


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

    def collect_all(self, sort: str = "downloads", limit: Optional[int] = None, metadata_dir: str = "output/datasets/metadata", readme_dir: str = "output/datasets/readmes", max_workers: int = 5) -> None:
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
            process_func = partial(self._process_and_save_item, item_type='dataset', metadata_dir=metadata_dir, readme_dir=readme_dir)
            list(tqdm(executor.map(process_func, datasets), total=len(datasets), desc="Collecting and saving datasets"))

    def _process_and_save_item(self, item: Any, item_type: str, metadata_dir: str, readme_dir: str) -> None:
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
