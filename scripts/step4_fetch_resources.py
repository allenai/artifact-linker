#!/usr/bin/env python3
"""
Step 4: Extract paper/codebase links from model and dataset READMEs,
        then fetch their content from arxiv and GitHub.

Scans model and dataset READMEs for arxiv IDs and GitHub repository URLs,
deduplicates them, fetches paper metadata from the arxiv API and GitHub
repository metadata / READMEs from the GitHub API, then saves everything
alongside a resource_links.json that maps each artifact to its resources.

Input:
  - data/artifact_raw_data/models/readmes/*.md    (step 1)
  - data/artifact_raw_data/datasets/readmes/*.md  (step 3)

Output:
  - data/artifact_raw_data/papers/metadata/*.json
  - data/artifact_raw_data/codebases/metadata/*.json
  - data/artifact_raw_data/codebases/readmes/*.md
  - data/artifact_raw_data/resource_links.json
"""

import argparse
import json
import os
from pathlib import Path

from artifact_graph.collectors.resource_collector import ResourceCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Fetch paper and codebase resources from model/dataset READMEs."
    )
    parser.add_argument(
        "--model-readme-dir",
        default="../data/artifact_raw_data/models/readmes",
        help="Directory containing model README files (step 1 output).",
    )
    parser.add_argument(
        "--dataset-readme-dir",
        default="../data/artifact_raw_data/datasets/readmes",
        help="Directory containing dataset README files (step 3 output).",
    )
    parser.add_argument(
        "--papers-dir",
        default="../data/artifact_raw_data/papers/metadata",
        help="Directory to save paper metadata JSON files.",
    )
    parser.add_argument(
        "--codebases-metadata-dir",
        default="../data/artifact_raw_data/codebases/metadata",
        help="Directory to save codebase metadata JSON files.",
    )
    parser.add_argument(
        "--codebases-readme-dir",
        default="../data/artifact_raw_data/codebases/readmes",
        help="Directory to save codebase README files.",
    )
    parser.add_argument(
        "--resource-links-out",
        default="../data/artifact_raw_data/resource_links.json",
        help="Output path for the artifact→resource mapping.",
    )
    parser.add_argument(
        "--model-metadata-dir",
        default="../data/artifact_raw_data/models/metadata",
        help="Directory containing model metadata JSON files (step 1 output).",
    )
    parser.add_argument(
        "--dataset-metadata-dir",
        default="../data/artifact_raw_data/datasets/metadata",
        help="Directory containing dataset metadata JSON files (step 3 output).",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub API token (default: $GITHUB_TOKEN).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    def resolve(p: str) -> Path:
        return (script_dir / p).resolve()

    model_readme_dir = resolve(args.model_readme_dir)
    dataset_readme_dir = resolve(args.dataset_readme_dir)
    model_metadata_dir = resolve(args.model_metadata_dir)
    dataset_metadata_dir = resolve(args.dataset_metadata_dir)
    papers_dir = resolve(args.papers_dir)
    codebases_metadata_dir = resolve(args.codebases_metadata_dir)
    codebases_readme_dir = resolve(args.codebases_readme_dir)
    resource_links_out = resolve(args.resource_links_out)

    collector = ResourceCollector(github_token=args.github_token)

    # ── 1. Extract links from READMEs ────────────────────────────────────
    print("Extracting links from model READMEs...")
    model_links = (
        collector.extract_links_from_dir(model_readme_dir)
        if model_readme_dir.exists()
        else {}
    )
    print("Extracting links from dataset READMEs...")
    dataset_links = (
        collector.extract_links_from_dir(dataset_readme_dir)
        if dataset_readme_dir.exists()
        else {}
    )
    print(f"  Models with links (README):   {len(model_links)}")
    print(f"  Datasets with links (README): {len(dataset_links)}")

    # ── 1b. Extract arxiv IDs from metadata tags ─────────────────────────
    print("Extracting arxiv IDs from model metadata tags...")
    model_meta_links = collector.extract_links_from_metadata_dir(model_metadata_dir)
    print("Extracting arxiv IDs from dataset metadata tags...")
    dataset_meta_links = collector.extract_links_from_metadata_dir(dataset_metadata_dir)
    print(f"  Models with arxiv tags (metadata):   {len(model_meta_links)}")
    print(f"  Datasets with arxiv tags (metadata): {len(dataset_meta_links)}")

    # Merge: metadata links into readme links (union)
    def _merge_links(
        readme_links: dict, meta_links: dict
    ) -> dict:
        merged = dict(readme_links)
        for aid, links in meta_links.items():
            if aid in merged:
                existing_arxiv = set(merged[aid].get("arxiv_ids", []))
                existing_arxiv.update(links.get("arxiv_ids", []))
                merged[aid]["arxiv_ids"] = sorted(existing_arxiv)
            else:
                merged[aid] = links
        return merged

    model_links = _merge_links(model_links, model_meta_links)
    dataset_links = _merge_links(dataset_links, dataset_meta_links)
    print(f"  Models with links (merged):   {len(model_links)}")
    print(f"  Datasets with links (merged): {len(dataset_links)}")

    # ── 2. Save resource_links.json ──────────────────────────────────────
    resource_links = {"models": model_links, "datasets": dataset_links}
    resource_links_out.parent.mkdir(parents=True, exist_ok=True)
    resource_links_out.write_text(
        json.dumps(resource_links, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved resource links → {resource_links_out}")

    # ── 3. Deduplicate across all artifacts ──────────────────────────────
    all_arxiv_ids: set[str] = set()
    all_github_repos: set[str] = set()
    for links in list(model_links.values()) + list(dataset_links.values()):
        all_arxiv_ids.update(links.get("arxiv_ids", []))
        all_github_repos.update(links.get("github_repos", []))

    print(f"\nUnique arxiv IDs:   {len(all_arxiv_ids)}")
    print(f"Unique GitHub repos: {len(all_github_repos)}")

    # ── 4. Fetch papers ──────────────────────────────────────────────────
    if all_arxiv_ids:
        print(f"\nFetching {len(all_arxiv_ids)} papers from arxiv...")
        paper_results = collector.collect_all_papers(
            sorted(all_arxiv_ids),
            papers_dir,
            max_concurrent=args.max_concurrent,
        )
        success = sum(1 for r in paper_results if r["status"] == "success")
        skipped = sum(1 for r in paper_results if r["status"] == "skipped")
        errors  = sum(1 for r in paper_results if r["status"] == "error")
        print(f"  {success} fetched, {skipped} skipped, {errors} errors")

    # ── 5. Fetch codebases ───────────────────────────────────────────────
    if all_github_repos:
        if not args.github_token:
            print(
                "\nWarning: GITHUB_TOKEN not set. "
                "Unauthenticated requests are rate-limited to 60/hr."
            )
        print(f"\nFetching {len(all_github_repos)} GitHub repositories...")
        codebase_results = collector.collect_all_codebases(
            sorted(all_github_repos),
            codebases_metadata_dir,
            codebases_readme_dir,
            max_concurrent=args.max_concurrent,
        )
        success = sum(1 for r in codebase_results if r["status"] == "success")
        skipped = sum(1 for r in codebase_results if r["status"] == "skipped")
        errors  = sum(1 for r in codebase_results if r["status"] == "error")
        print(f"  {success} fetched, {skipped} skipped, {errors} errors")

    print("\nStep 4 complete.")
    print(f"  Papers metadata  → {papers_dir}")
    print(f"  Codebases metadata → {codebases_metadata_dir}")
    print(f"  Codebases READMEs  → {codebases_readme_dir}")
    print(f"  Resource links     → {resource_links_out}")


if __name__ == "__main__":
    main()
