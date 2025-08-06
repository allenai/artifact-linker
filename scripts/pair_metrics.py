#!/usr/bin/env python3

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process
from tqdm import tqdm

from artifact_graph.collectors.dataset_collector import DatasetCollector
from artifact_graph.collectors.metric_collector import MetricCollector
from artifact_graph.collectors.model_collector import ModelCollector


def find_best_dataset_match(
    extracted_name: str, dataset_name_map: Dict[str, str], score_cutoff: int = 90
) -> Optional[str]:
    """
    Find the best match for an extracted dataset name from the list of known datasets.

    Args:
        extracted_name: The dataset name parsed from a README.
        dataset_name_map: A mapping from lowercase dataset name to canonical ID.
        score_cutoff: The minimum fuzz ratio to consider a match.

    Returns:
        The canonical dataset ID if a good match is found, otherwise None.
    """
    # First, check for an exact match
    if extracted_name.lower() in dataset_name_map:
        return dataset_name_map[extracted_name.lower()]

    # If no exact match, use fuzzy matching
    best_match = process.extractOne(
        extracted_name, dataset_name_map.keys(), scorer=fuzz.WRatio, score_cutoff=score_cutoff
    )

    if best_match:
        return dataset_name_map[best_match[0]]

    return None


def find_top_dataset_matches(
    extracted_name: str, dataset_name_map: Dict[str, List[Dict[str, Any]]], limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Find the top N fuzzy matches for an extracted dataset name.

    Args:
        extracted_name: The dataset name parsed from a README.
        dataset_name_map: A mapping from lowercase dataset name to canonical ID.
        limit: The maximum number of matches to return.

    Returns:
        A list of dictionaries, each containing the matched name, its score, and its canonical ID.
    """
    # Use process.extract with a more appropriate scorer
    matches = process.extract(
        extracted_name, dataset_name_map.keys(), scorer=fuzz.token_set_ratio, limit=limit
    )

    # Format the results into a more useful structure
    # For each match, we now retrieve a list of possible canonical IDs with their downloads
    return [
        {"match": match[0], "score": match[1], "ids": dataset_name_map[match[0]]}
        for match in matches
    ]


def process_single_model(
    model_id: str, readmes_dir: str, metric_collector: MetricCollector
) -> Tuple[str, List[Dict[str, Any]]]:
    """Worker function to process one model's README."""
    readme_bytes = ModelCollector.load_readme(model_id, readme_dir=readmes_dir)
    if not readme_bytes:
        return model_id, []

    try:
        readme_text = readme_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return model_id, []

    return model_id, metric_collector.extract_metrics_with_gpt(readme_text)


def pair_metrics(
    models_dir: str,
    datasets_dir: str,
    readmes_dir: str,
    min_model_downloads: int,
    output_file: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pair models to datasets based on metrics extracted from READMEs.

    Returns:
        A dictionary containing the results of the pairing process.
    """
    print("Loading all model and dataset metadata...")
    models = ModelCollector.load_all_metadata(models_dir, min_downloads=min_model_downloads)
    datasets = DatasetCollector.load_all_metadata(datasets_dir, min_downloads=0)

    print(f"Loaded {len(models)} models and {len(datasets)} datasets.")

    # Create a mapping from lowercase dataset names to a list of their canonical IDs and download counts
    dataset_name_map: Dict[str, List[Dict[str, Any]]] = {}
    for d_id, d_meta in datasets.items():
        name_only = d_id.split("/")[-1].lower()
        if name_only not in dataset_name_map:
            dataset_name_map[name_only] = []
        dataset_name_map[name_only].append({"id": d_id, "downloads": d_meta.get("downloads", 0)})

    metric_collector = MetricCollector()

    # --- Resumability: Load existing results if they exist ---
    if Path(output_file).exists():
        print(f"Loading existing results from {output_file} to resume.")
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f).get("pairings", [])
    else:
        all_results = []

    processed_model_ids = {p["model_id"] for p in all_results}
    print(f"Found {len(processed_model_ids)} previously processed models.")

    # --- Filter out already processed models ---
    model_ids = [mid for mid in list(models.keys()) if mid not in processed_model_ids]

    # Apply the limit if one was provided
    if limit:
        model_ids = model_ids[:limit]
        print(f"Processing a limit of {limit} models.")

    if not model_ids:
        print("All models have already been processed.")
        return {"pairings": all_results, "unmatched_dataset_names": [], "models_with_no_metrics": 0}

    print(f"Processing {len(model_ids)} new models...")

    unmatched_datasets = set()
    models_with_no_metrics = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_model = {
            executor.submit(process_single_model, model_id, readmes_dir, metric_collector): model_id
            for model_id in model_ids
        }

        with tqdm(total=len(model_ids), desc="Pairing Metrics with GPT") as pbar:
            for i, future in enumerate(as_completed(future_to_model)):
                model_id, eval_results = future.result()
                pbar.update(1)

                if not eval_results:
                    models_with_no_metrics += 1
                    continue

                # For each raw result from the LLM, find potential dataset matches
                for raw_result in eval_results:
                    dataset_name = raw_result.get("dataset")
                    metrics = raw_result.get("metrics")

                    if not dataset_name or not metrics:
                        continue

                    top_matches = find_top_dataset_matches(dataset_name, dataset_name_map)
                    best_match = (
                        top_matches[0] if top_matches and top_matches[0]["score"] >= 90 else None
                    )

                    pairing_result = {
                        "model_id": model_id,
                        "raw_llm_output": raw_result,
                        "top_fuzzy_matches": top_matches,
                        "best_match": best_match,
                    }
                    all_results.append(pairing_result)

                    if not best_match:
                        unmatched_datasets.add(dataset_name)

                # --- Batch Saving Logic ---
                if (i + 1) % 500 == 0:
                    pbar.set_description(f"Saving progress... (processed {i+1})")
                    temp_results = {
                        "pairings": all_results,
                        "unmatched_dataset_names": list(unmatched_datasets),
                        "models_with_no_metrics": models_with_no_metrics,
                    }
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(temp_results, f, indent=2, ensure_ascii=False)
                    pbar.set_description("Pairing Metrics with GPT")

                # Update progress bar with live stats
                pbar.set_postfix(
                    {
                        "paired": len(all_results),
                        "no_metrics": models_with_no_metrics,
                        "unmatched_ds": len(unmatched_datasets),
                    }
                )

    return {
        "pairings": all_results,
        "unmatched_dataset_names": list(unmatched_datasets),
        "models_with_no_metrics": models_with_no_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Pair models to datasets based on README metrics.")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="output/models/metadata",
        help="Directory containing model metadata.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="../data/output/datasets/metadata",
        help="Directory containing dataset metadata.",
    )
    parser.add_argument(
        "--readmes-dir",
        type=str,
        default="output/models/readmes",
        help="Directory containing model README files.",
    )
    parser.add_argument(
        "--min-model-downloads",
        type=int,
        default=100,
        help="Minimum downloads for a model to be considered.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of models to process. (default: no limit)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/metric_pairings.json",
        help="File to save the successful pairings.",
    )

    args = parser.parse_args()

    results = pair_metrics(
        models_dir=args.models_dir,
        datasets_dir=args.datasets_dir,
        readmes_dir=args.readmes_dir,
        min_model_downloads=args.min_model_downloads,
        output_file=args.output_file,
        limit=args.limit,
    )

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n✅ Metric pairing complete.")
    successful_count = sum(1 for p in results["pairings"] if p["best_match"])
    print(f"   - Found {successful_count} successful pairings.")
    print(f"   - Skipped {results['models_with_no_metrics']} models with no extractable metrics.")
    print(
        f"   - Found {len(results['unmatched_dataset_names'])} dataset names that could not be matched."
    )
    print(f"   - Results saved to {args.output_file}")

    if results["unmatched_dataset_names"]:
        print("\n🔍 Unmatched Dataset Names (Top 20):")
        for name in sorted(results["unmatched_dataset_names"])[:20]:
            print(f"   - {name}")


if __name__ == "__main__":
    main()
