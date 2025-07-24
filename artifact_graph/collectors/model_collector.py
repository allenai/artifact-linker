import contextlib
import json
import os
import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


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
                "lastModified": str(detailed.lastModified)
                if hasattr(detailed, "lastModified") and detailed.lastModified
                else None,
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
            return {"error": f"Failed to fetch model '{model_id}': {str(e)}"}

    def collect_readme(self, model_id: str) -> Optional[bytes]:
        """Download README.md for a model, returning its bytes without saving."""
        try:
            # Suppress the verbose download progress bar
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    file_path = hf_hub_download(
                        repo_id=model_id,
                        filename="README.md",
                        repo_type="model",
                        token=self.token,
                    )
            return Path(file_path).read_bytes()
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
    ) -> Dict[str, Union[Dict[str, Any], Optional[bytes], str]]:
        """Collect metadata and readme content for one model."""
        metadata = self.collect_metadata(model_id)
        if "error" in metadata:
            return {"model_id": model_id, "status": "error", "reason": metadata["error"]}

        readme_bytes = self.collect_readme(model_id)
        return {
            "model_id": model_id,
            "status": "success",
            "metadata": metadata,
            "readme": readme_bytes,
        }

    def _process_and_save_item(
        self, model: Any, metadata_dir: str, readme_dir: str
    ) -> Dict[str, Any]:
        """Helper function to process and save a single item for parallel execution."""
        model_id = model["id"] if isinstance(model, dict) else model.id
        metadata_path = Path(metadata_dir) / f"{model_id.replace('/', '__')}.json"
        if metadata_path.exists():
            return {"model_id": model_id, "status": "skipped", "reason": "already exists"}

        result = self.collect_one(model_id)

        if result["status"] == "success":
            self.save_metadata(model_id, result["metadata"], metadata_dir)
            if result["readme"]:
                self.save_readme(model_id, result["readme"], readme_dir)
        return result

    def collect_all(
        self,
        models: List[Any],
        metadata_dir: str,
        readme_dir: str,
        max_concurrent: int,
    ) -> List[Dict[str, Any]]:
        """Collects all models, saving them progressively in parallel using threads."""
        ensure_directory(Path(metadata_dir))
        ensure_directory(Path(readme_dir))

        results = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            process_func = partial(
                self._process_and_save_item,
                metadata_dir=metadata_dir,
                readme_dir=readme_dir,
            )
            with tqdm(total=len(models), desc="Downloading models") as pbar:
                for result in executor.map(process_func, models):
                    results.append(result)
                    pbar.update(1)
        return results

    def collect_top_models(
        self,
        min_downloads: int = 100,
        metadata_dir: str = "output/models/metadata",
        readme_dir: str = "output/models/readmes",
        max_concurrent: int = 10,
        cache_file: str = "cached_models.json",
        force_refresh: bool = False,
    ) -> None:
        """
        Downloads metadata and READMEs for Hugging Face models with downloads >= min_downloads.

        Args:
            min_downloads: Minimum number of downloads required (default: 100)
            metadata_dir: Directory to save metadata files
            readme_dir: Directory to save README files
            max_concurrent: Maximum concurrent downloads
            cache_file: Path to cache file for model list
            force_refresh: Force refresh the model list cache
        """
        all_models = None
        if not force_refresh:
            all_models = self.load_cached_models(cache_file)

        if all_models is None:
            print("Fetching model list from HuggingFace Hub...")
            try:
                all_models = list(self.api.list_models(sort="downloads", full=True))
                self.save_cached_models(all_models, cache_file)
            except Exception as e:
                print(f"Error fetching models: {e}")
                return

        # Filter models by download threshold
        filtered_models = []
        total_models = len(all_models)

        print(f"Filtering {total_models} models by download threshold (>= {min_downloads})...")

        for model in all_models:
            downloads = model["downloads"]
            if downloads >= min_downloads:
                filtered_models.append(model)

        print(f"Found {len(filtered_models)} models with >= {min_downloads} downloads")

        # Check for already downloaded models
        print(f"Checking for existing models in {metadata_dir}...")
        metadata_path = Path(metadata_dir)
        ensure_directory(metadata_path)
        existing_model_files = {f.stem for f in metadata_path.glob("*.json")}

        to_download = []
        for model in filtered_models:
            model_id = model.id if not isinstance(model, dict) else model["id"]
            model_fname = model_id.replace("/", "__")
            if model_fname not in existing_model_files:
                to_download.append(model)

        print(
            f"Found {len(existing_model_files)} existing models. Need to download {len(to_download)} new models."
        )

        if not to_download:
            print("All models are already up to date.")
            return

        print(f"Downloading {len(to_download)} new models...")
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
        print(f"   📊 Total models processed: {len(to_download)}")
        print(f"   🎯 Download threshold: >= {min_downloads}")

        if error_count > 0:
            print("\n❌ Failed downloads:")
            for result in results:
                if result["status"] == "error":
                    print(f"   - {result['model_id']}: {result['reason']}")

        print("\n✅ Download complete.")
        print(f"   - Metadata saved to: {metadata_dir}")
        print(f"   - READMEs saved to: {readme_dir}")

    @staticmethod
    def load_cached_models(cache_file: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached model list from JSON file.
        """
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                models = json.load(f)
            print(f"Loaded {len(models)} models from cache: {cache_file}")
            return models
        except FileNotFoundError:
            print(f"Cache file not found: {cache_file}")
            return None
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    @staticmethod
    def save_cached_models(models: List[Any], cache_file: str) -> None:
        """
        Save model list to JSON cache file.
        """
        try:
            model_dicts = []
            for model in models:
                model_dict = {
                    "id": model.id,
                    "downloads": getattr(model, "downloads", 0),
                    "likes": getattr(model, "likes", 0),
                    "author": getattr(model, "author", ""),
                    "tags": getattr(model, "tags", []),
                    "private": getattr(model, "private", False),
                    "last_modified": str(getattr(model, "last_modified", ""))
                    if hasattr(model, "last_modified")
                    else "",
                }
                model_dicts.append(model_dict)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(model_dicts, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(models)} models to cache: {cache_file}")

        except Exception as e:
            print(f"Error saving cache: {e}")

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
