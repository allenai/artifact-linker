#!/usr/bin/env python3
"""
Script to validate edge metadata by cross-checking with README files.
Similar to summarize_readmes.py, this script verifies the correctness of each edge
in edge_metadata.json by reading the original README files for models and datasets
to check if the reported metrics match what's documented.
"""

import argparse
import concurrent.futures
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

# Import LLM functionality
from artifact_graph.utils.llm_client import call_llm


def load_edge_metadata(edge_metadata_file: Path) -> Dict:
    """Load edge metadata from JSON file."""
    print(f"Loading edge metadata from {edge_metadata_file}")
    with open(edge_metadata_file, "r") as f:
        return json.load(f)


def _owner_repo_to_filename(oid: str) -> str:
    """Convert owner/repo format to filename format."""
    return oid.replace("/", "__") + "_README.md"


def _find_readme(readme_dir: Path, oid: str) -> Optional[Path]:
    """Find README file for a given model or dataset ID."""
    # Try exact match first
    exact = readme_dir / _owner_repo_to_filename(oid)
    if exact.exists():
        return exact

    # Try prefix match
    prefix = oid.replace("/", "__")
    candidates = list(readme_dir.glob(f"{prefix}*.md"))
    if candidates:
        # Prefer files with _README in the name and shorter names
        candidates.sort(key=lambda p: ("_README" not in p.name, len(p.name)))
        return candidates[0]

    return None


def extract_metrics_from_readmes_with_llm(
    model_readme_path: Optional[Path],
    dataset_readme_path: Optional[Path],
    model_id: str,
    dataset_id: str,
    llm_model: str = "gpt-4o",
) -> Dict[str, Union[float, int]]:
    """Extract performance metrics from README files using LLM."""

    # Define metric ranges (same as in graph_builder.py)
    METRIC_RANGES = {
        "loss": [0.0, 10000000],
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
        "map": [0.0, 1.0],
        "ndcg": [0.0, 1.0],
        "mrr": [0.0, 1.0],
        "recall@k": [0.0, 1.0],
        "precision@k": [0.0, 1.0],
        "top-k_accuracy": [0.0, 1.0],
        "bleu": [0.0, 100.0],
        "rouge-1": [0.0, 100.0],
        "rouge-2": [0.0, 100.0],
        "rouge-l": [0.0, 100.0],
        "rouge-lsum": [0.0, 100.0],
        "meteor": [0.0, 1.0],
        "chrf": [0.0, 100.0],
        "wer": [0.0, 100.0],
        "cer": [0.0, 100.0],
        "eer": [0.0, 1.0],
        "perplexity": [1.0, 1000000],
        "bits_per_character": [0.0, 10.0],
        "bits_per_byte": [0.0, 10.0],
        "pass@k": [0.0, 1.0],
        "human_eval": [0.0, 1.0],
        "success_rate": [0.0, 1.0],
        "win_rate": [0.0, 1.0],
    }

    # Combine README contents
    combined_content = ""

    if model_readme_path and model_readme_path.exists():
        try:
            model_content = model_readme_path.read_text(encoding="utf-8", errors="ignore")
            combined_content += f"=== MODEL README ({model_id}) ===\n{model_content}\n\n"
        except Exception as e:
            print(f"Error reading model README {model_readme_path}: {e}")

    if dataset_readme_path and dataset_readme_path.exists():
        try:
            dataset_content = dataset_readme_path.read_text(encoding="utf-8", errors="ignore")
            combined_content += f"=== DATASET README ({dataset_id}) ===\n{dataset_content}\n\n"
        except Exception as e:
            print(f"Error reading dataset README {dataset_readme_path}: {e}")

    if not combined_content:
        print(f"No README content available for {model_id} + {dataset_id}")
        return {}

    # Truncate if too long (keep first 12000 chars to fit in context)
    if len(combined_content) > 12000:
        combined_content = combined_content[:12000] + "\n... (truncated)"

    # Build formatted metric ranges for the prompt
    metric_ranges_lines = []
    for metric, (min_val, max_val) in METRIC_RANGES.items():
        metric_ranges_lines.append(f'    "{metric}": [{min_val}, {max_val}]')
    metric_ranges_display = "{\n" + ",\n".join(metric_ranges_lines) + "\n}"

    # Build the prompt
    prompt = f"""You are a performance metric extraction specialist.

TASK: Extract specific performance metrics for the model-dataset combination from README content.

MODEL ID: {model_id}
DATASET ID: {dataset_id}

VALID METRICS AND RANGES (use EXACT metric names and ensure values are within ranges):
{metric_ranges_display}

README CONTENT:
{combined_content}

INSTRUCTIONS:
1. Look for performance results for this specific model-dataset combination
2. Extract ONLY metrics from the METRIC_RANGES above using EXACT metric names
3. Handle percentage conversions correctly:
   - For metrics with max range 1.0: convert percentages to decimals (85.3% → 0.853)
   - For metrics with max range 100.0: keep original values (BLEU: 24.5 → 24.5)
4. Ensure ALL extracted values are within the specified [min, max] ranges
5. Return results in JSON format with exact metric names as keys
6. If a metric is not found or value is outside valid range, omit it completely
7. If multiple values exist for the same metric, prefer the most prominent result

EXAMPLE OUTPUT:
{{
    "accuracy": 0.853,
    "f1": 0.762,
    "bleu": 24.5
}}

If nothing is found, return an empty dictionary.

Return ONLY valid JSON with no additional text or explanation."""

    try:
        # Call LLM
        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            agent_name="metric_extractor",
        )

        if not response.get("success"):
            print(f"LLM call failed: {response.get('error', 'Unknown error')}")
            return {}

        # Parse JSON response
        content_str = response.get("content", "").strip()

        # Remove code blocks if present
        if content_str.startswith("```"):
            lines = content_str.split("\n")
            content_str = "\n".join(lines[1:-1])

        metrics = json.loads(content_str)

        # Validate metrics against METRIC_RANGES
        validated_metrics = {}
        for key, value in metrics.items():
            try:
                float_value = float(value)

                # Check if metric is in valid list
                if key not in METRIC_RANGES:
                    print(f"Skipping invalid metric '{key}' (not in METRIC_RANGES)")
                    continue

                # Check if value is in valid range
                min_val, max_val = METRIC_RANGES[key]
                if min_val <= float_value <= max_val:
                    validated_metrics[key] = float_value
                else:
                    print(
                        f"Metric '{key}' value {float_value} outside valid range [{min_val}, {max_val}]"
                    )

            except (ValueError, TypeError):
                print(f"Invalid metric value for {key}: {value}")
                continue

        return validated_metrics

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response content: {response.get('content', '')[:200]}...")
        return {}
    except Exception as e:
        print(f"Error in LLM metric extraction: {e}")
        return {}


def validate_edge_structure(edge_key: str, edge_data: Dict) -> List[str]:
    """Validate the basic structure of an edge entry."""
    errors = []

    # Check required fields
    required_fields = ["model_id", "dataset_id", "metrics"]
    for field in required_fields:
        if field not in edge_data:
            errors.append(f"Missing required field: {field}")

    # Check edge key format
    if "," not in edge_key:
        errors.append(f"Invalid edge key format: {edge_key} (should be 'node1,node2')")

    # Check metrics structure
    if "metrics" in edge_data:
        if not isinstance(edge_data["metrics"], dict):
            errors.append("Metrics field should be a dictionary")
        elif len(edge_data["metrics"]) == 0:
            errors.append("Metrics dictionary is empty")
        else:
            # Check metric values
            for metric_name, metric_value in edge_data["metrics"].items():
                if not isinstance(metric_value, (int, float)):
                    errors.append(f"Metric '{metric_name}' has non-numeric value: {metric_value}")

    return errors


def validate_readme_availability(
    edge_data: Dict, model_readme_dir: Path, dataset_readme_dir: Path
) -> List[str]:
    """Check if README files are available for the model and dataset."""
    errors = []

    model_id = edge_data.get("model_id")
    dataset_id = edge_data.get("dataset_id")

    # Check model README
    model_readme = _find_readme(model_readme_dir, model_id)
    if not model_readme:
        errors.append(f"Model README not found for '{model_id}'")

    # Check dataset README
    dataset_readme = _find_readme(dataset_readme_dir, dataset_id)
    if not dataset_readme:
        errors.append(f"Dataset README not found for '{dataset_id}'")

    return errors


def validate_against_readme_files_with_llm(
    edge_data: Dict,
    model_readme_dir: Path,
    dataset_readme_dir: Path,
    llm_model: str = "gpt-4o-mini",
) -> Tuple[List[str], Dict[str, Union[float, int]]]:
    """Cross-validate edge metrics against README files using LLM."""
    errors = []
    warnings = []

    model_id = edge_data.get("model_id")
    dataset_id = edge_data.get("dataset_id")
    edge_metrics = edge_data.get("metrics", {})

    if not edge_metrics:
        warnings.append(f"No metrics to validate for '{model_id}' + '{dataset_id}'")
        return [f"WARNING: {w}" for w in warnings], {}

    # Find README files
    model_readme = _find_readme(model_readme_dir, model_id)
    dataset_readme = _find_readme(dataset_readme_dir, dataset_id)

    if not model_readme and not dataset_readme:
        errors.append(f"No README files found for model '{model_id}' or dataset '{dataset_id}'")
        return errors, {}

    # Prepare target metrics list
    target_metrics = list(edge_metrics.keys())

    # Extract metrics from combined README files using LLM
    print("  Extracting metrics from README files...")
    readme_metrics = extract_metrics_from_readmes_with_llm(
        model_readme, dataset_readme, model_id, dataset_id, llm_model
    )

    if not readme_metrics:
        warnings.append(f"No metrics extracted from README files for '{model_id}' + '{dataset_id}'")
        return [f"WARNING: {w}" for w in warnings], {}

    print(f"  Found README metrics: {readme_metrics}")
    print(f"  Edge metrics: {edge_metrics}")

    # Compare edge metrics with README metrics
    for metric_name, edge_value in edge_metrics.items():
        # Look for exact or similar metric in README
        matched_readme_value = None

        # First try exact match
        if metric_name in readme_metrics:
            matched_readme_value = readme_metrics[metric_name]
        else:
            # Try normalized matching
            edge_normalized = metric_name.lower().replace("_", "").replace("-", "")
            for readme_metric, readme_value in readme_metrics.items():
                readme_normalized = readme_metric.lower().replace("_", "").replace("-", "")
                if edge_normalized == readme_normalized:
                    matched_readme_value = readme_value
                    break

        if matched_readme_value is not None:
            # Compare values with tolerance
            if isinstance(matched_readme_value, (int, float)) and isinstance(
                edge_value, (int, float)
            ):
                diff = abs(float(matched_readme_value) - float(edge_value))
                tolerance = max(
                    0.02, 0.05 * abs(float(matched_readme_value))
                )  # 5% tolerance or 0.02, whichever is larger

                if diff > tolerance:
                    errors.append(
                        f"Metric '{metric_name}' value mismatch: "
                        f"edge={edge_value}, README={matched_readme_value}, diff={diff:.4f}"
                    )
                else:
                    print(
                        f"  ✓ Metric '{metric_name}' validated: edge={edge_value}, README={matched_readme_value}"
                    )
        else:
            warnings.append(f"Metric '{metric_name}' not found in README files")

    # Check for additional metrics in README that might be missing from edge data
    for readme_metric, readme_value in readme_metrics.items():
        readme_normalized = readme_metric.lower().replace("_", "").replace("-", "")
        found_in_edge = False

        for edge_metric in edge_metrics:
            edge_normalized = edge_metric.lower().replace("_", "").replace("-", "")
            if edge_normalized == readme_normalized:
                found_in_edge = True
                break

        if not found_in_edge:
            warnings.append(
                f"README metric '{readme_metric}' ({readme_value}) missing in edge data"
            )

    if warnings:
        errors.extend([f"WARNING: {w}" for w in warnings])

    return errors, readme_metrics


def validate_metric_ranges(edge_data: Dict) -> List[str]:
    """Validate that metric values are within expected ranges."""
    errors = []

    # Define expected ranges for common metrics
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
        "bleu": [0.0, 100.0],
        "rouge-1": [0.0, 100.0],
        "rouge-2": [0.0, 100.0],
        "rouge-l": [0.0, 100.0],
        "meteor": [0.0, 1.0],
        "pass@k": [0.0, 1.0],
        "success_rate": [0.0, 1.0],
        "win_rate": [0.0, 1.0],
    }

    metrics = edge_data.get("metrics", {})
    for metric_name, metric_value in metrics.items():
        # Check if we have range information for this metric
        for range_key, (min_val, max_val) in METRIC_RANGES.items():
            if range_key in metric_name.lower():
                if not (min_val <= metric_value <= max_val):
                    errors.append(
                        f"Metric '{metric_name}' value {metric_value} outside expected range [{min_val}, {max_val}]"
                    )
                break

    return errors


def process_single_edge(
    edge_key: str,
    edge_data: Dict,
    model_readme_dir: Path,
    dataset_readme_dir: Path,
    llm_model: str,
    detailed_output: bool = False,
) -> Dict:
    """Process a single edge for validation and metric extraction."""
    all_errors = []

    # 1. Validate basic structure
    structure_errors = validate_edge_structure(edge_key, edge_data)
    all_errors.extend(structure_errors)

    # 2. Validate README availability
    if model_readme_dir or dataset_readme_dir:
        readme_availability_errors = validate_readme_availability(
            edge_data, model_readme_dir or Path(), dataset_readme_dir or Path()
        )
        all_errors.extend(readme_availability_errors)

    # 3. Validate metric ranges
    range_errors = validate_metric_ranges(edge_data)
    all_errors.extend(range_errors)

    # 4. Cross-validate with README files using LLM (if available)
    llm_extracted_metrics = {}
    if model_readme_dir or dataset_readme_dir:
        print(
            f"  Processing edge {edge_key}: {edge_data.get('model_id')} + {edge_data.get('dataset_id')}"
        )
        readme_errors, llm_extracted_metrics = validate_against_readme_files_with_llm(
            edge_data, model_readme_dir or Path(), dataset_readme_dir or Path(), llm_model
        )
        all_errors.extend(readme_errors)

    # Categorize errors
    warnings = [e for e in all_errors if e.startswith("WARNING:")]
    errors = [e for e in all_errors if not e.startswith("WARNING:")]

    # Determine status
    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "valid"

    # Update edge metadata with LLM-extracted metrics if available
    updated_edge_data = edge_data.copy()
    if llm_extracted_metrics:
        updated_edge_data["metrics"] = llm_extracted_metrics
        print(f"  Updated metrics for {edge_key}: {llm_extracted_metrics}")

    result = {
        "edge_key": edge_key,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "updated_edge_data": updated_edge_data,
        "llm_extracted_metrics": llm_extracted_metrics,
        "detailed_errors": {
            "model_id": edge_data.get("model_id"),
            "dataset_id": edge_data.get("dataset_id"),
            "errors": errors,
            "warnings": warnings,
        }
        if (errors or warnings) and detailed_output
        else None,
    }

    return result


def run_validation(
    graph_data_dir: Path,
    model_readme_dir: Optional[Path] = None,
    dataset_readme_dir: Optional[Path] = None,
    detailed_output: bool = False,
    save_report: bool = True,
    llm_model: str = "gpt-4o-mini",
    max_edges: Optional[int] = None,
    update_edge_metadata: bool = False,
    max_workers: int = 4,
) -> Dict:
    """Run comprehensive validation of edge metadata."""

    # Load data files
    edge_metadata_file = graph_data_dir / "edge_metadata.json"

    if not edge_metadata_file.exists():
        raise FileNotFoundError(f"Edge metadata file not found: {edge_metadata_file}")

    edge_metadata = load_edge_metadata(edge_metadata_file)

    # Check README directories
    if not model_readme_dir or not model_readme_dir.exists():
        print("Warning: Model README directory not found or not specified")
        model_readme_dir = None

    if not dataset_readme_dir or not dataset_readme_dir.exists():
        print("Warning: Dataset README directory not found or not specified")
        dataset_readme_dir = None

    # Limit edges if specified
    edge_items = list(edge_metadata.items())
    if max_edges:
        edge_items = edge_items[:max_edges]
        print(f"\nValidating {len(edge_items)} edges (limited from {len(edge_metadata)})...")
    else:
        print(f"\nValidating {len(edge_items)} edges...")

    # Validation results
    validation_report = {
        "total_edges": len(edge_items),
        "valid_edges": 0,
        "edges_with_errors": 0,
        "edges_with_warnings": 0,
        "error_summary": {},
        "detailed_errors": {},
        "llm_model_used": llm_model,
    }

    # Updated edge metadata for saving
    updated_edge_metadata = {}

    # Process edges in parallel
    print(f"Using {max_workers} parallel workers for processing")

    process_func = partial(
        process_single_edge,
        model_readme_dir=model_readme_dir or Path(),
        dataset_readme_dir=dataset_readme_dir or Path(),
        llm_model=llm_model,
        detailed_output=detailed_output,
    )

    # Use ThreadPoolExecutor for I/O bound tasks (LLM calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_edge = {
            executor.submit(process_func, edge_key, edge_data): (edge_key, edge_data)
            for edge_key, edge_data in edge_items
        }

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_edge),
            total=len(edge_items),
            desc="Processing edges",
        ):
            try:
                result = future.result()
                edge_key = result["edge_key"]

                # Update validation statistics
                if result["status"] == "error":
                    validation_report["edges_with_errors"] += 1
                elif result["status"] == "warning":
                    validation_report["edges_with_warnings"] += 1
                else:
                    validation_report["valid_edges"] += 1

                # Track error types
                for error in result["errors"] + result["warnings"]:
                    error_type = error.split(":")[0] if ":" in error else error.split()[0]
                    validation_report["error_summary"][error_type] = (
                        validation_report["error_summary"].get(error_type, 0) + 1
                    )

                # Store detailed errors if requested
                if result["detailed_errors"] and detailed_output:
                    validation_report["detailed_errors"][edge_key] = result["detailed_errors"]

                # Store updated edge data
                updated_edge_metadata[edge_key] = result["updated_edge_data"]

            except Exception as e:
                edge_key, edge_data = future_to_edge[future]
                print(f"Error processing edge {edge_key}: {e}")
                validation_report["edges_with_errors"] += 1
                # Store original data if processing fails
                updated_edge_metadata[edge_key] = edge_data

    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total edges validated: {validation_report['total_edges']}")
    print(f"Valid edges: {validation_report['valid_edges']}")
    print(f"Edges with errors: {validation_report['edges_with_errors']}")
    print(f"Edges with warnings: {validation_report['edges_with_warnings']}")

    if validation_report["error_summary"]:
        print("\nError types:")
        for error_type, count in sorted(validation_report["error_summary"].items()):
            print(f"  {error_type}: {count}")

    # Save detailed report
    if save_report:
        report_file = graph_data_dir / "edge_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(validation_report, f, indent=2)
        print(f"\nDetailed validation report saved to: {report_file}")

    # Save updated edge metadata if requested
    if update_edge_metadata and updated_edge_metadata:
        updated_metadata_file = graph_data_dir / "edge_metadata_llm_updated.json"
        with open(updated_metadata_file, "w") as f:
            json.dump(updated_edge_metadata, f, indent=2)
        print(f"Updated edge metadata saved to: {updated_metadata_file}")

        # Add statistics to validation report
        total_edges_with_llm_metrics = sum(
            1 for data in updated_edge_metadata.values() if data.get("metrics")
        )
        validation_report["edges_with_llm_metrics"] = total_edges_with_llm_metrics
        print(f"Edges with LLM-extracted metrics: {total_edges_with_llm_metrics}")

    return validation_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate edge metadata against README files using LLM"
    )
    parser.add_argument(
        "--graph-data-dir",
        type=Path,
        default=Path("output/artifact_graph_data"),
        help="Directory containing edge_metadata.json",
    )
    parser.add_argument(
        "--model-readme-dir", type=Path, help="Directory containing model README files"
    )
    parser.add_argument(
        "--dataset-readme-dir", type=Path, help="Directory containing dataset README files"
    )
    parser.add_argument(
        "--llm-model", default="openai/gpt-4o", help="LLM model to use for metric extraction"
    )
    parser.add_argument(
        "--max-edges", type=int, help="Maximum number of edges to validate (for testing)"
    )
    parser.add_argument(
        "--detailed-output",
        action="store_true",
        help="Include detailed error information in the report",
    )
    parser.add_argument(
        "--no-save-report", action="store_true", help="Don't save validation report to file"
    )
    parser.add_argument(
        "--update-edge-metadata",
        action="store_true",
        help="Save updated edge metadata with LLM-extracted metrics",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for LLM calls (default: 4)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        validation_report = run_validation(
            graph_data_dir=args.graph_data_dir,
            model_readme_dir=args.model_readme_dir,
            dataset_readme_dir=args.dataset_readme_dir,
            detailed_output=args.detailed_output,
            save_report=not args.no_save_report,
            llm_model=args.llm_model,
            max_edges=args.max_edges,
            update_edge_metadata=args.update_edge_metadata,
            max_workers=args.max_workers,
        )

        # Exit with error code if there are validation errors
        if validation_report["edges_with_errors"] > 0:
            print(
                f"\n❌ Validation failed: {validation_report['edges_with_errors']} edges have errors"
            )
            sys.exit(1)
        else:
            print(f"\n✅ Validation passed: All {validation_report['total_edges']} edges are valid")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Validation script failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
