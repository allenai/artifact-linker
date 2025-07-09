from pathlib import Path

from artifact_graph.collectors.metric_collector import MetricCollector

metrics_dir = Path("output/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)

mc = MetricCollector()

print("Processing 'microsoft__DialoGPT-medium_README.md' file…")
readme_path = Path("output/models/readmes/microsoft__DialoGPT-medium_README.md")
data = mc.collect_one(readme_path)
metrics = data["metrics"]

metrics_path = mc.save_metrics_file(
    "microsoft/DialoGPT-medium", metrics, metrics_dir="output/metrics"
)
print(f"Saved metrics → {metrics_path}")
