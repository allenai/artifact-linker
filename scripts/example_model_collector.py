import os
from pathlib import Path
from artifact_graph.collectors.model_collector import ModelCollector

# Ensure token and output dirs
Path("output/models/metadata").mkdir(parents=True, exist_ok=True)
Path("output/models/readmes").mkdir(parents=True, exist_ok=True)

# Initialize collector
mc = ModelCollector(
    metadata_dir="output/models/metadata",
    hf_token=os.getenv("HF_TOKEN"),
)

# 1) Collect metadata & README in memory
print("Collecting 'microsoft/DialoGPT-medium' model...")
data = mc.collect_one("microsoft/DialoGPT-medium")
meta = data["metadata"]
readme_bytes = data["readme"]

meta_path = mc.save_metadata("microsoft/DialoGPT-medium", meta, metadata_dir="output/models/metadata")
readme_path = mc.save_readme("microsoft/DialoGPT-medium", readme_bytes, readme_dir="output/models/readmes")
print(f"Saved metadata -> {meta_path}")
print(f"Saved README   -> {readme_path}")