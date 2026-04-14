#!/usr/bin/env python3
"""Shared runner helpers used across link and attribute pipelines."""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def detect_split_type(split_dir: str | Path) -> str:
    """Detect split type (trans/induc) from split_info.json.

    Returns 'trans' or 'induc' prefix for output file naming.
    Falls back to 'unknown' if detection fails.
    """
    import json

    info_path = Path(split_dir) / "split_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        split_type = info.get("type", "")
        if "transductive" in split_type:
            return "trans"
        elif "inductive" in split_type:
            return "induc"
    # Fallback: infer from directory name
    dir_name = str(split_dir).lower()
    if "transductive" in dir_name or "_trans" in dir_name:
        return "trans"
    elif "inductive" in dir_name or "_induc" in dir_name:
        return "induc"
    return "unknown"


def load_node_embeddings(split_dir: str | Path, mode: str):
    """Load node embeddings from split root."""
    import torch

    split_path = Path(split_dir)
    if mode == "embedding":
        emb_path = split_path / "node_embeddings_voyage.npy"
        arr = np.load(emb_path, allow_pickle=False)
        if hasattr(arr.dtype, "names") and arr.dtype.names and "embedding" in arr.dtype.names:
            x = torch.from_numpy(arr["embedding"]).float()
        else:
            x = torch.from_numpy(arr).float()
        print(f"Using real embeddings from {emb_path} (dim={x.size(1)})")
    else:
        emb_path = split_path / "node_embeddings_random.npy"
        arr = np.load(emb_path, allow_pickle=False)
        x = torch.from_numpy(arr).float()
        print(f"[Ablation] Using random embeddings from {emb_path} (dim={x.size(1)})")

    return x


def run_parallel(
    fn: Callable[..., Dict[str, Any]],
    items: Sequence[Tuple[Any, ...]],
    workers: int,
    include_failed_item: bool = False,
) -> List[Dict[str, Any]]:
    """Run a function in a thread pool and collect row-like dict outputs."""
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, *item): item for item in items}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                if include_failed_item:
                    results.append({"error": str(e), "item": futures[future]})
                else:
                    results.append({"error": str(e)})
    return results


def run_sequential(
    fn: Callable[..., Dict[str, Any]],
    items: Iterable[Tuple[Any, ...]],
) -> List[Dict[str, Any]]:
    """Run a function sequentially with progress reporting."""
    item_list = list(items)
    return [fn(*item) for item in tqdm(item_list, total=len(item_list))]
