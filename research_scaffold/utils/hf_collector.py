from huggingface_hub import HfApi, hf_hub_download
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any


def safe_dataset_info(dataset) -> Dict[str, Any]:
    return {
        "datasetId": dataset.id,
        "sha": getattr(dataset, "sha", None),
        "lastModified": (
            dataset.last_modified.isoformat()
            if isinstance(dataset.last_modified, datetime)
            else str(dataset.last_modified)
        ),
        "tags": getattr(dataset, "tags", []),
        "downloads": getattr(dataset, "downloads", None),
        "likes": getattr(dataset, "likes", None),
        "private": getattr(dataset, "private", False),
        "author": getattr(dataset, "author", None),
    }

def collect_datasets(
    hf_token: str,
    output_dir: str = ".",
    save_metadata: bool = True,
    save_readmes: bool = True,
    full: bool = True,
) -> None:
    """
    Fetch all HuggingFace datasets metadata (and optionally READMEs) and save them locally.
    
    Args:
      hf_token:    Your HF_TOKEN env var or plain token string.
      output_dir:  Directory under which subfolders will be created.
      save_metadata: If True, writes per-dataset JSON metadata files.
      save_readmes:  If True, downloads each dataset’s README.md (if present).
      full:        If True, calls list_datasets(full=True) to get sha, tags, downloads, etc.
    """
    api = HfApi(token=hf_token)
    
    # prepare folders
    meta_dir   = os.path.join(output_dir, "dataset_metadata")
    readme_dir = os.path.join(output_dir, "dataset_readmes")
    os.makedirs(meta_dir,   exist_ok=True)
    os.makedirs(readme_dir, exist_ok=True)
    
    # fetch
    print("Fetching dataset list…")
    datasets = api.list_datasets(full=full)
    
    # overview
    overview: List[Dict[str, Any]] = []
    for ds in datasets:
        overview.append({
            "id": ds.id,
            "downloads": getattr(ds, "downloads", None),
            "trending_score": getattr(ds, "trending_score", None),
            "likes": getattr(ds, "likes", None),
            "created_at": (
                ds.created_at.isoformat()
                if isinstance(ds.created_at, datetime)
                else str(ds.created_at)
            ),
            "last_modified": (
                ds.last_modified.isoformat()
                if isinstance(ds.last_modified, datetime)
                else str(ds.last_modified)
            ),
        })
    
    # save overview JSON
    overview_path = os.path.join(output_dir, "dataset_info.json")
    with open(overview_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved overview of {len(overview)} datasets to {overview_path}")
    
    # per-dataset metadata + readme
    for ds in datasets:
        ds_id = ds.id.replace("/", "__")  # avoid nested folders
        if save_metadata:
            info = safe_dataset_info(ds)
            with open(os.path.join(meta_dir, f"{ds_id}.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        if save_readmes:
            try:
                readme_path = hf_hub_download(
                    repo_id=ds.id, filename="README.md", repo_type="dataset", token=hf_token
                )
                os.replace(readme_path, os.path.join(readme_dir, f"{ds_id}_README.md"))
            except Exception:
                # no README or download failed
                pass
    
    print("Done collecting all dataset metadata and READMEs.")




# —– your existing helpers —–
def safe_model_info(model) -> Dict[str, Any]:
    return {
        "modelId": model.id,
        "sha": getattr(model, "sha", None),
        "lastModified": str(getattr(model, "lastModified", "")),
        "pipeline_tag": getattr(model, "pipeline_tag", None),
        "tags": getattr(model, "tags", []),
        "likes": getattr(model, "likes", None),
        "downloads": getattr(model, "downloads", None),
        "private": getattr(model, "private", False),
        "author": getattr(model, "author", None),
    }

def get_trained_dataset(info) -> List[str]:
    try:
        ds = info.card_data.datasets
        return ds if isinstance(ds, list) else [ds]
    except Exception:
        return []

def get_base_model(info) -> Optional[str]:
    try:
        return info.card_data.base_model
    except Exception:
        return None

# —– new wrapper function —–
def collect_models(
    hf_token: str,
    readme_dir: str = "model_readmes_download_ranks",
    metadata_dir: str = "model_metadata_download_ranks",
    sort: str = "downloads",
    pause: float = 0.2,
) -> None:
    """
    Download each model's metadata and README from HuggingFace.

    Args:
      hf_token:      your HF_TOKEN string
      readme_dir:    where to save README.md files
      metadata_dir:  where to save per-model JSON metadata
      sort:          ordering key for list_models (e.g. 'downloads', 'likes')
      pause:         seconds to sleep between requests
    """
    api = HfApi(token=hf_token)

    os.makedirs(readme_dir,   exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    def process_model(model):
        model_id = model.id
        safe_id = model_id.replace("/", "__")
        meta_path = os.path.join(metadata_dir, f"{safe_id}.json")
        readme_path = os.path.join(readme_dir, f"{safe_id}.md")

        if os.path.exists(meta_path):
            print(f"↪ Skipping {model_id} (metadata exists)")
        else:
            data = safe_model_info(model)
            try:
                info = api.model_info(model_id)
                data["trainedDataset"] = get_trained_dataset(info)
                data["baseModel"]      = get_base_model(info)
            except Exception as e:
                print(f"✗ Failed to fetch info for {model_id}: {e}")
                data["trainedDataset"] = []
                data["baseModel"]      = None

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved metadata for {model_id}")

        if os.path.exists(readme_path):
            print(f"↪ README exists for {model_id}, skipping.")
        else:
            try:
                tmp = hf_hub_download(
                    repo_id=model_id,
                    filename="README.md",
                    repo_type="model",
                    token=hf_token,
                    force_download=True,
                )
                with open(tmp,     "r", encoding="utf-8") as src, \
                     open(readme_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                print(f"✓ Saved README for {model_id}")
            except Exception as e:
                print(f"✗ No README for {model_id}: {e}")

        time.sleep(pause)

    # main loop
    for mdl in api.list_models(sort=sort):
        print(f"Processing {mdl.id} …")
        process_model(mdl)

# —– call it from your script’s entry point —–
if __name__ == "__main__":
    import os
    token = os.getenv("HF_TOKEN") or "<your-token-here>"
    collect_models(hf_token=token)
