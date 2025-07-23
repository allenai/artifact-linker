#!/usr/bin/env python3

import argparse
import os

from artifact_graph.collectors.model_collector import ModelCollector


def main(
    limit: int,
    hf_token: str,
    cache_file: str,
    force_refresh: bool,
    max_concurrent: int,
) -> None:
    """
    Download metadata and READMEs for top Hugging Face models.
    """
    metadata_dir = "output/models/metadata"
    readme_dir = "output/models/readmes"

    collector = ModelCollector(hf_token=hf_token)
    collector.collect_top_models(
        limit=limit,
        metadata_dir=metadata_dir,
        readme_dir=readme_dir,
        max_concurrent=max_concurrent,
        cache_file=cache_file,
        force_refresh=force_refresh,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download metadata and READMEs for top Hugging Face models."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Number of top models to download, sorted by downloads (default: 100000).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="cached_models.json",
        help="Path to cache file for model list (default: cached_models.json).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh the model list cache.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent downloads (default: 10).",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set HF_TOKEN or pass --hf_token.")

    main(
        args.limit,
        args.hf_token,
        args.cache_file,
        args.force_refresh,
        args.max_concurrent,
    )
