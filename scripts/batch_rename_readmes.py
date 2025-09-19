#!/usr/bin/env python3
"""
Batch rename files from xxx_README.md to xxx.md in specified directory.
"""

import argparse
import glob
import os


def batch_rename_readmes(directory: str, dry_run: bool = False):
    """
    Rename files from xxx_README.md to xxx.md in the given directory.

    Args:
        directory: Path to the directory containing files to rename
        dry_run: If True, only print what would be renamed without actually renaming
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all files ending with _README.md
    pattern = os.path.join(directory, "*_README.md")
    files_to_rename = glob.glob(pattern)

    if not files_to_rename:
        print(f"No files matching pattern '*_README.md' found in {directory}")
        return

    print(f"Found {len(files_to_rename)} files to rename:")

    renamed_count = 0
    for old_path in files_to_rename:
        # Extract filename without path
        old_filename = os.path.basename(old_path)

        # Remove _README.md suffix and add .md
        if old_filename.endswith("_README.md"):
            new_filename = old_filename[:-10] + ".md"  # Remove '_README.md' and add '.md'
            new_path = os.path.join(directory, new_filename)

            if dry_run:
                print(f"Would rename: {old_filename} -> {new_filename}")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_filename} -> {new_filename}")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming {old_filename}: {e}")

    if not dry_run:
        print(f"\nSuccessfully renamed {renamed_count} files")
    else:
        print(f"\nDry run complete. {len(files_to_rename)} files would be renamed")


def main():
    parser = argparse.ArgumentParser(
        description="Batch rename README files from xxx_README.md to xxx.md"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="./output/datasets/readmes/",
        help="Directory containing files to rename (default: ./output/datasets/readmes/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming files",
    )
    args = parser.parse_args()

    print(f"Processing directory: {args.directory}")
    if args.dry_run:
        print("DRY RUN MODE - No files will actually be renamed")

    batch_rename_readmes(args.directory, args.dry_run)


if __name__ == "__main__":
    main()
