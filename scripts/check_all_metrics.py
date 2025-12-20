#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    edge_metadata_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("output/artifact_graph_data/edge_metadata.json")
    )

    with open(edge_metadata_file) as f:
        edge_metadata = json.load(f)

    all_metrics = set()
    metric_counts = defaultdict(int)
    metric_value_ranges = defaultdict(
        lambda: {"min": float("inf"), "max": float("-inf"), "values": []}
    )

    for edge_key, edge_data in edge_metadata.items():
        if "metrics" not in edge_data:
            continue

        for metric_name, metric_value in edge_data["metrics"].items():
            all_metrics.add(metric_name)
            metric_counts[metric_name] += 1

            try:
                val = float(metric_value)
                metric_value_ranges[metric_name]["min"] = min(
                    metric_value_ranges[metric_name]["min"], val
                )
                metric_value_ranges[metric_name]["max"] = max(
                    metric_value_ranges[metric_name]["max"], val
                )
                metric_value_ranges[metric_name]["values"].append(val)
            except:
                pass

    print(f"Total edges: {len(edge_metadata)}")
    print(f"Total unique metrics: {len(all_metrics)}")
    print("\nAll metrics found:")

    for metric in sorted(all_metrics):
        count = metric_counts[metric]
        if metric in metric_value_ranges:
            min_val = metric_value_ranges[metric]["min"]
            max_val = metric_value_ranges[metric]["max"]
            values = metric_value_ranges[metric]["values"]
            avg_val = sum(values) / len(values) if values else 0
            print(
                f"  {metric:<25} count: {count:<6} range: [{min_val:.6f}, {max_val:.6f}] avg: {avg_val:.6f}"
            )
        else:
            print(f"  {metric:<25} count: {count:<6} (non-numeric)")


if __name__ == "__main__":
    main()
