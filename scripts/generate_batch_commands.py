#!/usr/bin/env python3
"""
Generate batch commands for single_eval.py based on prediction scores.
"""

import argparse
import json
import os


def generate_commands(
    json_file: str,
    min_score: float = 0.7,
    max_score: float = 0.8,
    output_file: str | None = None,
    script_name: str = "single_eval.py",
):
    """
    Generate batch commands for models with scores in the specified range.

    Args:
        json_file: Path to the predictions JSON file
        min_score: Minimum score threshold (inclusive)
        max_score: Maximum score threshold (exclusive)
        output_file: Output file for commands (if None, prints to stdout)
        script_name: Name of the script to run
    """

    # Load the JSON data
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file}")
        return

    # Extract dataset and metric from the data
    dataset = data.get("dataset", "rajpurkar/squad_v2")
    metric = data.get("metric", "exact_match")
    predictions = data.get("predictions", [])

    # Filter models with scores in the specified range
    filtered_models = []
    for pred in predictions:
        if (
            pred.get("status") == "Success"
            and "predicted_score" in pred
            and isinstance(pred["predicted_score"], (int, float))
        ):
            score = pred["predicted_score"]
            if min_score <= score < max_score:
                filtered_models.append({"model": pred["model"], "score": score})

    # Sort by score for easier review
    filtered_models.sort(key=lambda x: x["score"], reverse=True)

    # Generate commands
    commands = []
    for model_info in filtered_models:
        model = model_info["model"]
        score = model_info["score"]
        command = (
            f'python3 {script_name} --model "{model}" --dataset "{dataset}" --metric "{metric}"'
        )
        commands.append(f"# Score: {score:.3f}")
        commands.append(command)
        commands.append("")  # Empty line for readability

    # Output results
    if output_file:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Batch commands for models with scores {min_score} <= score < {max_score}\n")
            f.write(f"# Dataset: {dataset}\n")
            f.write(f"# Metric: {metric}\n")
            f.write(f"# Total models found: {len(filtered_models)}\n\n")
            f.write("\n".join(commands))
        print(f"Generated {len(filtered_models)} commands and saved to {output_file}")
    else:
        print(f"# Batch commands for models with scores {min_score} <= score < {max_score}")
        print(f"# Dataset: {dataset}")
        print(f"# Metric: {metric}")
        print(f"# Total models found: {len(filtered_models)}\n")
        for command in commands:
            print(command)

    return filtered_models


def generate_shell_script(
    json_file: str,
    min_score: float = 0.7,
    max_score: float = 0.8,
    output_file: str = "batch_eval.sh",
    script_name: str = "single_eval.py",
):
    """
    Generate a shell script to run all commands in batch.
    """

    # First generate the commands
    filtered_models = generate_commands(json_file, min_score, max_score, None, script_name)

    if not filtered_models:
        print("No models found in the specified score range.")
        return

    # Load the JSON data to get dataset and metric
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = data.get("dataset", "rajpurkar/squad_v2")
    metric = data.get("metric", "exact_match")

    # Generate shell script content
    script_content = [
        "#!/bin/bash",
        f"# Batch evaluation script for models with scores {min_score} <= score < {max_score}",
        f"# Dataset: {dataset}",
        f"# Metric: {metric}",
        f"# Total models: {len(filtered_models)}",
        "",
        "# Set error handling",
        "set -e",
        "",
        "# Create output directory",
        "mkdir -p batch_output",
        "",
        "# Log file",
        f'LOG_FILE="batch_output/batch_eval_{min_score}_{max_score}.log"',
        'echo "Starting batch evaluation at $(date)" > $LOG_FILE',
        "",
        "# Counter for progress",
        "counter=0",
        f"total={len(filtered_models)}",
        "",
    ]

    # Add commands for each model
    for model_info in filtered_models:
        model = model_info["model"]
        score = model_info["score"]
        safe_model_name = model.replace("/", "_").replace("-", "_")

        script_content.extend(
            [
                f"# Model: {model} (Score: {score:.3f})",
                "counter=$((counter + 1))",
                f'echo "[$counter/$total] Evaluating {model}..." | tee -a $LOG_FILE',
                "",
                f'python3 {script_name} --model "{model}" --dataset "{dataset}" --metric "{metric}" \\',
                f"  >> batch_output/{safe_model_name}_eval.log 2>&1 || \\",
                f'  echo "ERROR: Failed to evaluate {model}" | tee -a $LOG_FILE',
                "",
                f'echo "Completed {model}" | tee -a $LOG_FILE',
                'echo "" >> $LOG_FILE',
                "",
            ]
        )

    script_content.extend(
        [
            'echo "Batch evaluation completed at $(date)" | tee -a $LOG_FILE',
            'echo "Processed $total models" | tee -a $LOG_FILE',
        ]
    )

    # Write shell script
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(script_content))

    # Make script executable
    os.chmod(output_file, 0o755)

    print(f"Generated shell script {output_file} with {len(filtered_models)} models")
    print(f"Run with: ./{output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate batch commands for single_eval.py")
    parser.add_argument("json_file", help="Path to the predictions JSON file")
    parser.add_argument(
        "--min-score", type=float, default=0.7, help="Minimum score threshold (default: 0.7)"
    )
    parser.add_argument(
        "--max-score", type=float, default=0.8, help="Maximum score threshold (default: 0.8)"
    )
    parser.add_argument(
        "--output", "-o", help="Output file for commands (default: print to stdout)"
    )
    parser.add_argument(
        "--script", default="single_eval.py", help="Script name to use (default: single_eval.py)"
    )
    parser.add_argument("--shell-script", help="Generate executable shell script instead")
    parser.add_argument(
        "--format",
        choices=["commands", "shell"],
        default="commands",
        help="Output format: commands or shell script",
    )

    args = parser.parse_args()

    if args.format == "shell" or args.shell_script:
        output_file = args.shell_script or "batch_eval.sh"
        generate_shell_script(
            args.json_file, args.min_score, args.max_score, output_file, args.script
        )
    else:
        generate_commands(args.json_file, args.min_score, args.max_score, args.output, args.script)


if __name__ == "__main__":
    main()
