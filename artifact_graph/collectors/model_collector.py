import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


class ModelCollector:
    """
    Collects metadata and README files for Hugging Face models.
    """

    def __init__(self, hf_token: Optional[str] = None) -> None:
        self.token = hf_token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN must be provided or set as env var.")
        self.api = HfApi(token=self.token)

    def _extract_model_info(self, model) -> Dict[str, Any]:
        """Extracts selected fields from a model listing entry."""
        return {
            "modelId": model.id,
            "sha": getattr(model, "sha", None),
            "lastModified": getattr(model, "lastModified", ""),
            "pipeline_tag": getattr(model, "pipeline_tag", None),
            "tags": getattr(model, "tags", []),
            "likes": getattr(model, "likes", None),
            "downloads": getattr(model, "downloads", None),
            "private": getattr(model, "private", False),
            "author": getattr(model, "author", None),
        }

    def _safe_card_field(self, info: Any, field: str) -> Any:
        """Safely retrieve a field from model card metadata."""
        try:
            return getattr(info.card_data, field)
        except Exception:
            return None

    def collect_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Fetches and returns metadata for a single model.
        """
        try:
            detailed = self.api.model_info(model_id)
            meta = {
                "modelId": detailed.id,
                "sha": getattr(detailed, "sha", None),
                "lastModified": getattr(detailed, "lastModified", ""),
                "pipeline_tag": getattr(detailed, "pipeline_tag", None),
                "tags": getattr(detailed, "tags", []),
                "likes": getattr(detailed, "likes", None),
                "downloads": getattr(detailed, "downloads", None),
                "private": getattr(detailed, "private", False),
                "author": getattr(detailed, "author", None),
            }
            trained_ds = self._safe_card_field(detailed, "datasets")
            base = self._safe_card_field(detailed, "base_model")
            meta["trainedDataset"] = (
                trained_ds if isinstance(trained_ds, list) else ([trained_ds] if trained_ds else [])
            )
            meta["baseModel"] = base
            return meta
        except Exception as e:
            raise ValueError(f"Failed to fetch model '{model_id}': {str(e)}")

    def collect_readme(self, model_id: str) -> Optional[bytes]:
        """Download README.md for a model and return its bytes, without saving."""
        try:
            src = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                repo_type="model",
                token=self.token,
            )
            return Path(src).read_bytes()
        except Exception:
            return None

    def save_metadata(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        metadata_dir: str = "model_metadata",
    ) -> Path:
        """Save metadata dict to a JSON file."""
        out_dir = Path(metadata_dir)
        ensure_directory(out_dir)
        fname = model_id.replace("/", "__") + ".json"
        path = out_dir / fname
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_readme(
        self,
        model_id: str,
        readme_bytes: bytes,
        readme_dir: str = "model_readmes",
    ) -> Path:
        """Save README bytes to a markdown file."""
        out_dir = Path(readme_dir)
        ensure_directory(out_dir)
        fname = model_id.replace("/", "__") + ".md"
        path = out_dir / fname
        path.write_bytes(readme_bytes)
        return path

    def collect_one(
        self,
        model_id: str,
    ) -> Dict[str, Union[Dict[str, Any], Optional[bytes]]]:
        """Collect metadata and readme content without saving files."""
        metadata = self.collect_metadata(model_id)
        readme_bytes = self.collect_readme(model_id)
        return {"metadata": metadata, "readme": readme_bytes}

    def collect_batch(
        self,
        model_ids: List[str],
        pause: float = 0.2,
    ) -> Dict[str, Any]:
        """Iterates over a list of model IDs and collects each one."""
        results = {}
        for mid in model_ids:
            try:
                results[mid] = self.collect_one(mid)
            except Exception as e:
                results[mid] = {"error": str(e)}
            time.sleep(pause)
        return results

    def collect_all(self, sort: str = "downloads", limit: Optional[int] = None, metadata_dir: str = "output/models/metadata", readme_dir: str = "output/models/readmes", max_workers: int = 5) -> None:
        """Collects all models, sorted and optionally limited, and saves them progressively in parallel."""
        print("Fetching model list...")
        start_time = time.time()
        models = list(self.api.list_models(sort=sort, full=True))
        if limit:
            models = models[:limit]
        print(f"Finished fetching model list in {time.time() - start_time:.2f} seconds.")

        Path(metadata_dir).mkdir(parents=True, exist_ok=True)
        Path(readme_dir).mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(self._process_and_save_item, item_type='model', metadata_dir=metadata_dir, readme_dir=readme_dir)
            list(tqdm(executor.map(process_func, models), total=len(models), desc="Collecting and saving models"))

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
        model_id: str, metadata_dir: str = "model_metadata"
    ) -> Optional[Dict[str, Any]]:
        """Load saved metadata JSON for a given model ID."""
        fname = model_id.replace("/", "__") + ".json"
        path = Path(metadata_dir) / fname
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_readme(model_id: str, readme_dir: str = "model_readmes") -> Optional[bytes]:
        """Load saved README file for a given model ID."""
        fname = model_id.replace("/", "__") + ".md"
        path = Path(readme_dir) / fname
        if not path.exists():
            return None
        return path.read_bytes()

    @staticmethod
    def load_all_metadata(
        metadata_dir: str = "model_metadata", min_downloads: int = 1
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
                    model_id = file.stem.replace("__", "/")
                    results[model_id] = data
            except Exception:
                continue
        return results
