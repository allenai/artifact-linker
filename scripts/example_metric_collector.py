import json
from pathlib import Path
from artifact_graph.collectors.metric_collector import MetricCollector

# 1) Ensure output directory exists
metrics_dir = Path("output/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)

# 2) Initialize the collector (point it at our metrics directory)
mc = MetricCollector(data_dir=metrics_dir)

# 3) Process a single README and get metrics back in memory
print("Processing 'microsoft__DialoGPT-medium_README.md' file…")
readme_path = Path("output/models/readmes/microsoft__DialoGPT-medium_README.md")
metrics = mc.process_readme(readme_path)
print("Extracted metrics:", metrics)

# 4) Save those metrics to a per‐model JSON file
out_path = metrics_dir / f"{readme_path.stem}.json"
out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved metrics → {out_path}")