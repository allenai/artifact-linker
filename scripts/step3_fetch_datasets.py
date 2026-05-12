#!/usr/bin/env python3
"""
Step 3: Fetch dataset metadata + READMEs for datasets found in websearch results.

Input:
  - data/model_eval_websearch.json  (step 2 output)

Output:
  - data/artifact_raw_data/datasets/metadata/*.json
  - data/artifact_raw_data/datasets/readmes/*.md
  - data/artifact_raw_data/filtered_eval_pairs.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from artifact_graph.collectors.dataset_collector import DatasetCollector


# ──────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────

def load_websearch_results(path: str) -> List[Dict[str, Any]]:
    """Load websearch JSON and return entries with successful evaluations."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    return [r for r in results if r.get("status") == "success" and r.get("evaluations")]


def parse_metric_value(raw: str) -> Optional[float]:
    """Parse a metric value string to float. Returns None if unparseable."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.endswith("%"):
        try:
            return float(raw[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


# ──────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────

def extract_dataset_ids(websearch_results: List[Dict[str, Any]]) -> Set[str]:
    """Extract all unique dataset IDs from websearch evaluations."""
    dataset_ids = set()
    for result in websearch_results:
        for ev in result.get("evaluations", []):
            ds_name = ev.get("dataset_name", "").strip()
            if ds_name and "/" in ds_name:
                dataset_ids.add(ds_name)
    return dataset_ids


def fetch_datasets(
    dataset_ids: Set[str],
    metadata_dir: str,
    readme_dir: str,
    hf_token: str,
    max_concurrent: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Download metadata + READMEs for datasets, return loaded metadata dict."""
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)
    Path(readme_dir).mkdir(parents=True, exist_ok=True)

    collector = DatasetCollector(hf_token=hf_token)

    datasets_to_fetch = [{"id": ds_id} for ds_id in sorted(dataset_ids)]
    print(f"Fetching metadata + READMEs for {len(datasets_to_fetch)} datasets...")

    results = collector.collect_all(
        datasets=datasets_to_fetch,
        metadata_dir=metadata_dir,
        readme_dir=readme_dir,
        max_concurrent=max_concurrent,
    )

    # Print error summary
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        # Categorize errors
        not_found = sum(1 for r in errors if "404" in r.get("reason", "") or "not found" in r.get("reason", "").lower())
        other = len(errors) - not_found
        print(f"  Errors: {not_found} not found (404), {other} other failures")
        if other > 0:
            print("  Sample other errors:")
            shown = 0
            for r in errors:
                reason = r.get("reason", "")
                if "404" not in reason and "not found" not in reason.lower():
                    print(f"    {r['dataset_id']}: {reason[:120]}")
                    shown += 1
                    if shown >= 5:
                        break

    return DatasetCollector.load_all_metadata(metadata_dir, min_downloads=0)


def build_cleaned_pairs(
    websearch_results: List[Dict[str, Any]],
    dataset_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build cleaned eval pairs from websearch results.

    Only filters:
      - dataset_name must contain '/' (valid HF ID)
      - dataset must have been successfully fetched (exists in dataset_meta)
      - metric value must be parseable as a float
    """
    pairs = []
    skipped_no_dataset = 0
    skipped_bad_metric = 0
    models_seen = set()
    datasets_seen = set()

    for result in tqdm(websearch_results, desc="Building pairs"):
        model_id = result["model_id"]

        for ev in result.get("evaluations", []):
            dataset_id = ev.get("dataset_name", "").strip()
            if not dataset_id or "/" not in dataset_id:
                skipped_bad_metric += 1
                continue

            if dataset_id not in dataset_meta:
                skipped_no_dataset += 1
                continue

            value = parse_metric_value(ev.get("result", ""))
            if value is None:
                skipped_bad_metric += 1
                continue

            pairs.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                "dataset_downloads": dataset_meta[dataset_id].get("downloads", 0),
                "metric": ev.get("metric", ""),
                "result": value,
                "source": ev.get("source", ""),
            })
            models_seen.add(model_id)
            datasets_seen.add(dataset_id)

    return {
        "results": pairs,
        "stats": {
            "total_pairs": len(pairs),
            "unique_models": len(models_seen),
            "unique_datasets": len(datasets_seen),
            "skipped_no_dataset_meta": skipped_no_dataset,
            "skipped_bad_metric": skipped_bad_metric,
        },
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Fetch dataset info and build cleaned eval pairs from websearch results."
    )
    parser.add_argument(
        "--websearch-file",
        default="../data/model_eval_websearch.json",
        help="Path to websearch results JSON.",
    )
    parser.add_argument(
        "--dataset-metadata-dir",
        default="../data/artifact_raw_data/datasets/metadata",
        help="Directory to save/load dataset metadata.",
    )
    parser.add_argument(
        "--dataset-readme-dir",
        default="../data/artifact_raw_data/datasets/readmes",
        help="Directory to save/load dataset READMEs.",
    )
    parser.add_argument(
        "--output-file",
        default="../data/artifact_raw_data/filtered_eval_pairs.json",
        help="Path to write cleaned eval pairs JSON.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent HF API requests (default: 5).",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip dataset fetching (use existing metadata only).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    websearch_file = (script_dir / args.websearch_file).resolve()
    ds_meta_dir = (script_dir / args.dataset_metadata_dir).resolve()
    ds_readme_dir = (script_dir / args.dataset_readme_dir).resolve()
    output_file = (script_dir / args.output_file).resolve()

    hf_token = os.getenv("HF_TOKEN", "")

    # 1. Load websearch results
    print(f"Loading websearch results from: {websearch_file}")
    ws_results = load_websearch_results(str(websearch_file))
    print(f"  {len(ws_results)} models with evaluations")

    # 2. Extract dataset IDs and fetch their info
    dataset_ids = extract_dataset_ids(ws_results)
    print(f"  {len(dataset_ids)} unique datasets referenced")

    if not args.skip_fetch:
        if not hf_token:
            raise EnvironmentError("HF_TOKEN is required for dataset fetching. Set it or use --skip-fetch.")
        dataset_meta = fetch_datasets(
            dataset_ids=dataset_ids,
            metadata_dir=str(ds_meta_dir),
            readme_dir=str(ds_readme_dir),
            hf_token=hf_token,
            max_concurrent=args.max_concurrent,
        )
    else:
        print("Skipping dataset fetch, loading existing metadata...")
        dataset_meta = DatasetCollector.load_all_metadata(str(ds_meta_dir), min_downloads=0)

    print(f"  {len(dataset_meta)} datasets with metadata loaded")

    # 3. Build cleaned pairs
    output = build_cleaned_pairs(ws_results, dataset_meta)

    # 4. Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    stats = output["stats"]
    print(f"\nDone. Saved to: {output_file}")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Unique models: {stats['unique_models']}")
    print(f"  Unique datasets: {stats['unique_datasets']}")
    print(f"  Skipped (no dataset meta): {stats['skipped_no_dataset_meta']}")
    print(f"  Skipped (bad metric): {stats['skipped_bad_metric']}")


if __name__ == "__main__":
    main()
