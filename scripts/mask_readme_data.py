#!/usr/bin/env python3
"""
Masks dataset names, metric names, and metric values in README files.
"""

import argparse
import json
import os
import re
from typing import Set


def load_model_data_from_json(file_path: str) -> dict:
    """Load model data mapping from a .json file."""
    model_data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle if the JSON has a 'results' key containing the list
            if isinstance(data, dict) and "results" in data:
                entries = data["results"]
            elif isinstance(data, list):
                entries = data
            else:
                entries = [data]

            for entry in entries:
                model_id = entry.get("model_id")
                if model_id:
                    if model_id not in model_data:
                        model_data[model_id] = {
                            "datasets": set(),
                            "metric_names": set(),
                            "metric_values": set(),
                        }

                    # Add dataset name
                    if entry.get("dataset_id"):
                        model_data[model_id]["datasets"].add(entry["dataset_id"].lower())
                        dataset_id = entry["dataset_id"].split("/")[-1]
                        model_data[model_id]["datasets"].add(dataset_id.lower())

                    # Add metric names and values
                    if "metrics" in entry and isinstance(entry["metrics"], dict):
                        model_data[model_id]["metric_names"].update(
                            {k.lower() for k in entry["metrics"].keys()}
                        )
                        for value in entry["metrics"].values():
                            if isinstance(value, (int, float)):
                                model_data[model_id]["metric_values"].add(str(value))

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
    return model_data


def load_dataset_data_from_json(file_path: str) -> dict:
    """Load dataset data mapping from a .json file."""
    dataset_data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle if the JSON has a 'results' key containing the list
            if isinstance(data, dict) and "results" in data:
                entries = data["results"]
            elif isinstance(data, list):
                entries = data
            else:
                entries = [data]

            for entry in entries:
                dataset_id = entry.get("dataset_id")
                if dataset_id:
                    if dataset_id not in dataset_data:
                        dataset_data[dataset_id] = {
                            "models": set(),
                            "metric_names": set(),
                            "metric_values": set(),
                        }

                    # Add model name
                    if entry.get("model_id"):
                        dataset_data[dataset_id]["models"].add(entry["model_id"].lower())
                        model_id = entry["model_id"].split("/")[-1]
                        dataset_data[dataset_id]["models"].add(model_id.lower())

                    # Add metric names and values
                    if "metrics" in entry and isinstance(entry["metrics"], dict):
                        dataset_data[dataset_id]["metric_names"].update(
                            {k.lower() for k in entry["metrics"].keys()}
                        )
                        for value in entry["metrics"].values():
                            if isinstance(value, (int, float)):
                                dataset_data[dataset_id]["metric_values"].add(str(value))

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
    return dataset_data


def mask_text(text: str, phrases_to_mask: Set[str]) -> str:
    """Mask a set of phrases in a given text with [UNKNOWN]."""
    for phrase in phrases_to_mask:
        # Use regex to ensure we match whole words/phrases only
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", "[UNKNOWN]", text, flags=re.IGNORECASE)
    return text


def mask_metrics(text: str) -> str:
    """Mask metric names and values in a given text with [UNKNOWN]."""
    # Regex to find patterns like "accuracy: 85.5%", "F1-score of 0.92", "BLEU = 42.3"
    # This covers metric names (words with optional hyphens/underscores) followed by numbers.
    metric_patterns = [
        # metric_name: 12.34 or metric_name: 12
        r"\b([a-zA-Z_-]+)\s*:\s*\d+(\.\d+)?%?\b",
        # metric_name of 12.34 or metric_name of 12
        r"\b([a-zA-Z_-]+)\s+of\s+\d+(\.\d+)?%?\b",
        # metric_name = 12.34 or metric_name = 12
        r"\b([a-zA-Z_-]+)\s*=\s*\d+(\.\d+)?%?\b",
        # number followed by metric name
        r"\b\d+(\.\d+)?%?\s+([a-zA-Z_-]+)\b",
    ]

    for pattern in metric_patterns:
        text = re.sub(pattern, "[UNKNOWN]", text, flags=re.IGNORECASE)

    # Mask standalone numbers that look like metric values (e.g., 0.85, 92.3)
    # To avoid masking all numbers, we look for numbers between 0 and 1 (with decimals)
    # or between 1 and 100, often used for percentages.
    text = re.sub(r"\b(0\.\d{2,}|[1-9]\d(\.\d+)?)\b", "[UNKNOWN]", text)

    return text


def process_readme_for_model(readme_path: str, output_path: str, model_id: str, model_info: dict):
    """Load a README, mask model-specific data, and save to the output directory."""
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: README file not found at {readme_path}, skipping.")
        return

    content = content.lower()
    # Mask dataset names specific to this model
    masked_content = mask_text(content, model_info["datasets"])

    # Mask metric names specific to this model
    masked_content = mask_text(masked_content, model_info["metric_names"])

    # Mask metric values specific to this model
    masked_content = mask_text(masked_content, model_info["metric_values"])

    # Mask generic metric patterns
    masked_content = mask_metrics(masked_content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(masked_content)


def process_readme_for_dataset(
    readme_path: str, output_path: str, dataset_id: str, dataset_info: dict
):
    """Load a dataset README, mask dataset-specific data, and save to the output directory."""
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: README file not found at {readme_path}, skipping.")
        return

    # Mask model names specific to this dataset
    masked_content = mask_text(content, dataset_info["models"])

    # Mask metric names specific to this dataset
    masked_content = mask_text(masked_content, dataset_info["metric_names"])

    # Mask metric values specific to this dataset
    masked_content = mask_text(masked_content, dataset_info["metric_values"])

    # Mask generic metric patterns
    masked_content = mask_metrics(masked_content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(masked_content)


def main():
    parser = argparse.ArgumentParser(
        description="Mask dataset and metric information in README files."
    )
    parser.add_argument(
        "--readme_dir",
        type=str,
        default="output/models/readmes",
        help="Directory containing README files.",
    )
    parser.add_argument(
        "--dataset_readme_dir",
        type=str,
        default="output/datasets/readmes",
        help="Directory containing dataset README files.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="output/perfect_model_dataset_metrics.json",
        help="JSON file containing dataset and metric names.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/models/readmes_masked",
        help="Directory to save the masked README files.",
    )
    parser.add_argument(
        "--dataset_output_dir",
        type=str,
        default="output/datasets/readmes_masked",
        help="Directory to save the masked dataset README files.",
    )
    args = parser.parse_args()

    # Create output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.dataset_output_dir):
        os.makedirs(args.dataset_output_dir)

    # Load data
    model_data = load_model_data_from_json(args.input_file)
    dataset_data = load_dataset_data_from_json(args.input_file)

    if not model_data:
        print("Warning: No model data loaded.")
    if not dataset_data:
        print("Warning: No dataset data loaded.")

    # Process model README files
    if os.path.isdir(args.readme_dir):
        print("Processing model README files...")
        for filename in os.listdir(args.readme_dir):
            if filename.endswith(".md") or filename.endswith(".txt"):
                # Extract model_id from filename (assuming filename format like "model_id.md")
                model_id = os.path.splitext(filename)[0]
                model_id = model_id.replace("__", "/")

                if model_id in model_data:
                    readme_path = os.path.join(args.readme_dir, filename)
                    output_path = os.path.join(args.output_dir, filename)
                    print(f"Processing {readme_path} for model {model_id}...")
                    process_readme_for_model(
                        readme_path, output_path, model_id, model_data[model_id]
                    )
                else:
                    print(f"Warning: No data found for model {model_id}, skipping {filename}")
    else:
        print(f"Warning: Model README directory not found at {args.readme_dir}")

    # Process dataset README files
    if os.path.isdir(args.dataset_readme_dir):
        print("\nProcessing dataset README files...")
        for filename in os.listdir(args.dataset_readme_dir):
            if filename.endswith(".md") or filename.endswith(".txt"):
                # Extract dataset_id from filename (assuming filename format like "dataset_id.md")
                dataset_id = os.path.splitext(filename)[0]
                dataset_id = dataset_id.replace("__", "/")

                if dataset_id in dataset_data:
                    readme_path = os.path.join(args.dataset_readme_dir, filename)
                    output_path = os.path.join(args.dataset_output_dir, filename)
                    print(f"Processing {readme_path} for dataset {dataset_id}...")
                    process_readme_for_dataset(
                        readme_path, output_path, dataset_id, dataset_data[dataset_id]
                    )
                else:
                    print(f"Warning: No data found for dataset {dataset_id}, skipping {filename}")
    else:
        print(f"Warning: Dataset README directory not found at {args.dataset_readme_dir}")

    print("\nMasking complete.")
    print(f"Masked model files are saved in {args.output_dir}")
    print(f"Masked dataset files are saved in {args.dataset_output_dir}")


if __name__ == "__main__":
    main()
