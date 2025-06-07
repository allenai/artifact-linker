from huggingface_hub import HfApi, hf_hub_download
import os
import json
import time
from datetime import datetime

# Initialize API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create output directories
os.makedirs("dataset_metadata", exist_ok=True)
os.makedirs("dataset_readmes", exist_ok=True)

# Helper to safely extract dataset metadata
def safe_dataset_info(dataset):
    return {
        "datasetId": dataset.id,
        "sha": getattr(dataset, "sha", None),
        "lastModified": str(getattr(dataset, "lastModified", "")),
        "tags": getattr(dataset, "tags", []),
        "downloads": getattr(dataset, "downloads", None),
        "likes": getattr(dataset, "likes", None),
        "private": getattr(dataset, "private", False),
        "author": getattr(dataset, "author", None),
    }

# First: Collect overview info
print("Fetching dataset list...")
datasets = api.list_datasets(full=True)

dataset_overview = []
for dataset in datasets:
    dataset_overview.append({
        "id": dataset.id,
        "downloads": getattr(dataset, "downloads", None),
        "trending_score": getattr(dataset, "trending_score", None),
        "likes": getattr(dataset, "likes", None),
        "created_at": dataset.created_at.isoformat() if isinstance(dataset.created_at, datetime) else str(dataset.created_at),
        "last_modified": dataset.last_modified.isoformat() if isinstance(dataset.last_modified, datetime) else str(dataset.last_modified),
    })

# Save overview
with open("dataset_info.json", "w", encoding="utf-8") as f:
    json.dump(dataset_overview, f, indent=2, ensure_ascii=False)
print(f"✓ Saved overview of {len(dataset_overview)} datasets.")