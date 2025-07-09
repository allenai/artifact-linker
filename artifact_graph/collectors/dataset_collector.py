import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download


def ensure_directory(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


class DatasetCollector:
    def __init__(
        self,
        overview_json: str,
        hf_token: Optional[str] = None,
    ) -> None:
        self.overview_path = Path(overview_json)
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
        all_ds = self.api.list_datasets(full=True)
        match = next((d for d in all_ds if d.id == dataset_id), None)
        if not match:
            raise ValueError(f"Dataset '{dataset_id}' not found.")
        ds = match
        return {
            "datasetId": ds.id,
            "sha": getattr(ds, "sha", None),
            "createdAt": self._format_date(getattr(ds, "created_at", None)),
            "lastModified": self._format_date(getattr(ds, "last_modified", None)),
            "tags": getattr(ds, "tags", []),
            "downloads": getattr(ds, "downloads", 0),
            "likes": getattr(ds, "likes", 0),
            "private": getattr(ds, "private", False),
            "author": getattr(ds, "author", ""),
        }

    def download_readme(self, dataset_id: str) -> Optional[bytes]:
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
        fname = dataset_id.replace("/", "__") + "_README.md"
        path = out_dir / fname
        path.write_bytes(readme_bytes)
        return path

    def collect_one(
        self,
        dataset_id: str,
    ) -> Dict[str, Union[Dict[str, Any], Optional[bytes]]]:
        """Collect metadata and readme content without saving files."""
        metadata = self.collect_metadata(dataset_id)
        readme_bytes = self.download_readme(dataset_id)
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
