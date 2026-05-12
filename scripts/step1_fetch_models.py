#!/usr/bin/env python3
"""
Step 1: Fetch model metadata + READMEs from Hugging Face.

Modes:
  Default:        Fetch all models with downloads >= min_downloads.
  --from-websearch: Fetch only models referenced in step 2 websearch results.

Output:
  - data/artifact_raw_data/models/metadata/*.json
  - data/artifact_raw_data/models/readmes/*.md
"""

import argparse
import json
import os
from pathlib import Path

from artifact_graph.collectors.model_collector import ModelCollector


def load_model_ids_from_websearch(path: str) -> list[str]:
    """Extract unique model IDs from websearch results JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_ids = set()
    for r in data.get("results", []):
        if r.get("status") == "success" and r.get("evaluations"):
            model_ids.add(r["model_id"])
    return sorted(model_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Fetch model metadata and READMEs from Hugging Face."
    )
    parser.add_argument(
        "--from-websearch", type=str, default=None,
        help="Path to websearch JSON (step 2). Fetch only models in this file.",
    )
    parser.add_argument(
        "--min-downloads", type=int, default=100,
        help="Minimum download count for default mode (default: 100).",
    )
    parser.add_argument(
        "--metadata-dir", default="../data/artifact_raw_data/models/metadata",
        help="Directory to save model metadata JSON files.",
    )
    parser.add_argument(
        "--readme-dir", default="../data/artifact_raw_data/models/readmes",
        help="Directory to save model README files.",
    )
    parser.add_argument(
        "--hf-token", type=str, default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (default: $HF_TOKEN).",
    )
    parser.add_argument(
        "--cache-file", default="cached_models.json",
        help="Cache file for model list (default: cached_models.json).",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Force refresh the model list cache.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent downloads (default: 5).",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("HF_TOKEN is required. Set it via env or --hf-token.")

    script_dir = Path(__file__).parent
    metadata_dir = str((script_dir / args.metadata_dir).resolve())
    readme_dir = str((script_dir / args.readme_dir).resolve())

    collector = ModelCollector(hf_token=args.hf_token)

    if args.from_websearch:
        # Mode: fetch only models from websearch results
        ws_path = str((script_dir / args.from_websearch).resolve())
        print(f"Loading model IDs from websearch: {ws_path}")
        model_ids = load_model_ids_from_websearch(ws_path)
        print(f"  {len(model_ids)} unique models with evaluations")

        models = [{"id": mid} for mid in model_ids]
        print(f"Fetching metadata + READMEs for {len(models)} models...")
        results = collector.collect_all(
            models=models,
            metadata_dir=metadata_dir,
            readme_dir=readme_dir,
            max_concurrent=args.max_concurrent,
        )

        success = sum(1 for r in results if r["status"] == "success")
        errors = [r for r in results if r["status"] == "error"]
        skipped = sum(1 for r in results if r["status"] == "skipped")
        print(f"\nDone. Success: {success}, Errors: {len(errors)}, Skipped: {skipped}", flush=True)
        if errors:
            # Write errors to file so they don't get swallowed by sys.excepthook
            err_file = Path(metadata_dir) / "_fetch_errors.json"
            sample = errors[:20]
            with open(err_file, "w") as f:
                json.dump(sample, f, indent=2, default=str)
            print(f"\nSample errors written to: {err_file}", flush=True)
            for r in sample[:5]:
                print(f"  {r['model_id']}: {r.get('reason', 'unknown')[:150]}", flush=True)
    else:
        # Mode: fetch all models above download threshold
        collector.collect_top_models(
            min_downloads=args.min_downloads,
            metadata_dir=metadata_dir,
            readme_dir=readme_dir,
            max_concurrent=args.max_concurrent,
            cache_file=args.cache_file,
            force_refresh=args.force_refresh,
        )


if __name__ == "__main__":
    main()
