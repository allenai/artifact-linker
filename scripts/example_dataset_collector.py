# example_dataset_collect_and_save.py

import os
from pathlib import Path
from artifact_graph.collectors.dataset_collector import DatasetCollector

# Ensure token and output dirs
Path("output/datasets/metadata").mkdir(parents=True, exist_ok=True)
Path("output/datasets/readmes").mkdir(parents=True, exist_ok=True)

# Initialize collector (overview.json can be empty or existing)
dc = DatasetCollector(
    overview_json="output/datasets/overview.json",
    hf_token=os.getenv("HF_TOKEN"),
)

# 1) Collect metadata & README in memory
print("Collecting 'squad' dataset...")
data = dc.collect_one("rajpurkar/squad")
meta = data["metadata"]
readme_bytes = data["readme"]

meta_path = dc.save_metadata("rajpurkar/squad", meta, metadata_dir="output/datasets/metadata")
readme_path = dc.save_readme("rajpurkar/squad", readme_bytes, readme_dir="output/datasets/readmes")
print(f"Saved metadata -> {meta_path}")
print(f"Saved README   -> {readme_path}")