from huggingface_hub import HfApi, hf_hub_download
import os
import json
import time
from datetime import datetime

# Authenticate API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Output directories
os.makedirs("model_readmes_download_ranks", exist_ok=True)
os.makedirs("model_metadata_download_ranks", exist_ok=True)

def safe_model_info(model):
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

def get_trained_dataset(info):
    try:
        datasets = info.card_data.datasets
        return datasets if isinstance(datasets, list) else [datasets]
    except Exception:
        return []

def get_base_model(info):
    try:
        return info.card_data.base_model
    except Exception:
        return None

def process_model(model):
    model_id = model.modelId
    filename_safe_id = model_id.replace("/", "__")

    metadata_path = f"model_metadata_download_ranks/{filename_safe_id}.json"
    readme_path = f"model_readmes_download_ranks/{filename_safe_id}.md"

    # Skip if metadata already exists
    if os.path.exists(metadata_path):
        print(f"↪ Skipping {model_id} (already processed)")
        return

    # Save metadata
    metadata = safe_model_info(model)

    try:
        model_info = api.model_info(model_id)
        metadata["trainedDataset"] = get_trained_dataset(model_info)
        metadata["baseModel"] = get_base_model(model_info)
    except Exception as e:
        print(f"✗ Error fetching info for {model_id}: {e}")
        metadata["trainedDataset"] = []
        metadata["baseModel"] = None

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save README
    if not os.path.exists(readme_path):
        try:
            hf_path = hf_hub_download(repo_id=model.modelId, filename="README.md", repo_type="model", force_download=True)
            print(readme_path)
            with open(hf_path, "r", encoding="utf-8") as f_in, open(readme_path, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            print(f"✓ Saved README for {hf_path}")
        except Exception as e:
            print(f"✗ No README for {model_id}: {e}")
    else:
        print(f"↪ README already exists for {model_id}, skipping.")

    time.sleep(0.2)

# Main loop
models = api.list_models(sort='downloads')
for model in models:
    print(f"Processing {model.modelId}")
    process_model(model)

