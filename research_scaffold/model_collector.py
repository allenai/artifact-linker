from huggingface_hub import HfApi, hf_hub_download
import os
import json
import time
from datetime import datetime

api = HfApi(token=os.getenv("HF_TOKEN"))

# Output folders
os.makedirs("model_readmes", exist_ok=True)
os.makedirs("model_metadata", exist_ok=True)

# Pagination
page = 0
limit = 100  # max per page

def safe_model_info(model):
    """Manually extract serializable fields from model."""
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

while True:
    models = api.list_models(filter="text-classification")
    if not models:
        break  # No more models
    
    for model in models:
        model_id = model.modelId
        print(f"Processing {model_id}")
        filename_safe_id = model_id.replace("/", "__")

        # Save metadata
        metadata_path = f"model_metadata/{filename_safe_id}.json"
        metadata = safe_model_info(model)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save README.md
        try:
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md", repo_type="model")
            output_path = f"model_readmes/{filename_safe_id}.md"
            with open(readme_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            print(f"✓ Saved README for {model_id}")
        except Exception as e:
            print(f"✗ No README for {model_id}: {e}")

        time.sleep(0.2)  # Respect rate limits

    page += 1
