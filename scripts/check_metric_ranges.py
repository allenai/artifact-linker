#!/usr/bin/env python3

import json
import sys
from pathlib import Path

METRIC_RANGES = {
    "accuracy": [0.0, 1.0],
    "precision": [0.0, 1.0],
    "recall": [0.0, 1.0],
    "f1": [0.0, 1.0],
    "balanced_accuracy": [0.0, 1.0],
    "matthews_correlation": [-1.0, 1.0],
    "auc": [0.0, 1.0],
    "pearson": [-1.0, 1.0],
    "spearman": [-1.0, 1.0],
    "cosine_similarity": [-1.0, 1.0],
    "bleu": [0.0, 1.0],
    "rouge-1": [0.0, 100.0],
    "rouge-2": [0.0, 1.0],
    "rouge-l": [0.0, 1.0],
    "chrf": [0.0, 1.0],
    "meteor": [0.0, 1.0],
    "pass@k": [0.0, 1.0],
    "success_rate": [0.0, 1.0],
    "win_rate": [0.0, 1.0],
    "wer": [0.0, 1.0],
}


def main():
    edge_metadata_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("output/artifact_graph_data/edge_metadata.json")
    )

    with open(edge_metadata_file) as f:
        edge_metadata = json.load(f)

    total_edges = len(edge_metadata)
    total_metrics_checked = 0
    violations = []

    for edge_key, edge_data in edge_metadata.items():
        if "metrics" not in edge_data:
            continue

        for metric_name, metric_value in edge_data["metrics"].items():
            if metric_name not in METRIC_RANGES:
                continue

            total_metrics_checked += 1
            min_val, max_val = METRIC_RANGES[metric_name]

            if not (min_val <= metric_value <= max_val):
                violations.append(
                    {
                        "edge": edge_key,
                        "metric": metric_name,
                        "value": metric_value,
                        "expected_range": [min_val, max_val],
                    }
                )

    print(f"Total edges: {total_edges}")
    print(f"Total metrics checked: {total_metrics_checked}")
    print(f"Violations found: {len(violations)}")

    if violations:
        print("\nViolations:")
        for v in violations[:10]:
            print(f"  {v['edge']}: {v['metric']} = {v['value']} (expected {v['expected_range']})")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
    else:
        print("All metrics are within expected ranges!")


if __name__ == "__main__":
    main()
