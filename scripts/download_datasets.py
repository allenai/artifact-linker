import argparse
import os

from artifact_graph.collectors.dataset_collector import DatasetCollector


def main(limit: int, hf_token: str) -> None:
    """
    Download metadata and READMEs for top-N Hugging Face datasets.
    """
    print(f"Downloading top {limit} datasets...")
    metadata_dir = "output/datasets/metadata"
    readme_dir = "output/datasets/readmes"

    collector = DatasetCollector(hf_token=hf_token)
    collector.collect_all(limit=limit, metadata_dir=metadata_dir, readme_dir=readme_dir)

    print(f"\n✅ Download complete.")
    print(f"   - Metadata saved to: {metadata_dir}")
    print(f"   - READMEs saved to: {readme_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download metadata and READMEs for top-N Hugging Face datasets."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000000,
        help="Number of datasets to download, sorted by downloads.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token.",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set HF_TOKEN or pass --hf_token.")

    main(args.limit, args.hf_token) 