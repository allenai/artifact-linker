#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main():
    input_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("output/artifact_graph_data/edge_metadata.json")
    )
    output_file = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("output/artifact_graph_data/edge_metadata_normalized.json")
    )

    with open(input_file) as f:
        edge_metadata = json.load(f)

    # Metrics that should be normalized from percentage (0-100) to decimal (0-1)
    percentage_metrics = ["wer", "rouge-2", "rouge-l", "chrf"]

    normalization_stats = {metric: {"total": 0, "normalized": 0} for metric in percentage_metrics}

    for edge_key, edge_data in edge_metadata.items():
        if "metrics" not in edge_data:
            continue

        for metric_name in percentage_metrics:
            if metric_name in edge_data["metrics"]:
                normalization_stats[metric_name]["total"] += 1
                metric_value = edge_data["metrics"][metric_name]

                # If metric > 1.0, assume it's in percentage form and normalize to 0-1
                if metric_value > 1.0:
                    normalized_value = metric_value / 100.0
                    edge_data["metrics"][metric_name] = normalized_value
                    normalization_stats[metric_name]["normalized"] += 1

                    print(
                        f"Normalized {metric_name}: {metric_value} -> {normalized_value:.6f} for edge {edge_key}"
                    )

    with open(output_file, "w") as f:
        json.dump(edge_metadata, f, indent=2)

    print("\nNormalization Summary:")
    for metric_name, stats in normalization_stats.items():
        print(f"  {metric_name}: {stats['normalized']}/{stats['total']} values normalized")
    print(f"Normalized metadata saved to: {output_file}")


if __name__ == "__main__":
    main()
