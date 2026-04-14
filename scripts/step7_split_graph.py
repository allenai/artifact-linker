#!/usr/bin/env python3
"""
Step 6: Create train/val/test graph splits for link prediction.

Supports two split types:
  --type transductive  : All nodes visible, edges split (step7a logic)
  --type inductive     : Nodes split, new models/datasets in test (step7b logic)

Input:
  - data/artifact_graph_data_v3/  (step 5 output)

Output:
  - data/artifact_graph_splits_v3_transductive/
  - data/artifact_graph_splits_v3_inductive/
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.utils.graph_builder import load_nx_graph


# ──────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────

def load_normalized_edge_metadata(input_dir: Path) -> Dict[str, dict]:
    with open(input_dir / "edge_metadata_normalized.json") as f:
        return json.load(f)


def separate_edges_by_metrics(
    edge_metadata: Dict[str, dict],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    edges_with, edges_without = [], []
    for key, info in edge_metadata.items():
        u, v = map(int, key.split(","))
        if info.get("metrics", {}):
            edges_with.append((u, v))
        else:
            edges_without.append((u, v))
    return edges_with, edges_without


def normalize_edge(u: int, v: int, node_metadata: Dict) -> Tuple[int, int] | None:
    u_meta = node_metadata.get(u, node_metadata.get(str(u), {}))
    v_meta = node_metadata.get(v, node_metadata.get(str(v), {}))
    if u_meta.get("type") == "model" and v_meta.get("type") == "dataset":
        return u, v
    if u_meta.get("type") == "dataset" and v_meta.get("type") == "model":
        return v, u
    return None


def extract_split_edge_metadata(
    pos_edges: List[Tuple[int, int]], full_metadata: Dict[str, dict]
) -> Dict[str, dict]:
    split_meta = {}
    for u, v in pos_edges:
        key = f"{u},{v}"
        if key in full_metadata:
            split_meta[key] = full_metadata[key]
        else:
            rev = f"{v},{u}"
            if rev in full_metadata:
                split_meta[rev] = full_metadata[rev]
    return split_meta


def save_node_embeddings(input_dir: Path, output_dir: Path, num_nodes: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dim = 1024

    # Copy real embeddings
    for emb_name in ("node_embeddings_voyage.npy", "node_embeddings_random.npy"):
        src = input_dir / emb_name
        if src.exists():
            shutil.copy2(src, output_dir / emb_name)
            arr = np.load(src, allow_pickle=False)
            if hasattr(arr.dtype, "names") and arr.dtype.names and "embedding" in arr.dtype.names:
                emb_dim = arr["embedding"].shape[1]
            elif len(arr.shape) > 1:
                emb_dim = arr.shape[1]
            print(f"   Copied {emb_name} (dim={emb_dim})")

    # Always ensure random embeddings exist
    random_dst = output_dir / "node_embeddings_random.npy"
    if not random_dst.exists():
        rng = np.random.RandomState(seed)
        random_emb = rng.randn(num_nodes, emb_dim).astype(np.float32)
        np.save(random_dst, random_emb)
        print(f"   Generated random embeddings (dim={emb_dim})")


def save_split(
    split_dir: Path,
    msg_edges: np.ndarray,
    pos_edges: List[Tuple[int, int]],
    node_metadata: Dict,
    edge_metadata: Dict[str, dict],
) -> Dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / "edges.npz", edges=msg_edges)

    if pos_edges:
        np.savez(split_dir / "pos_edges.npz", edges=np.array(pos_edges, dtype=np.int32).T)
    else:
        np.savez(split_dir / "pos_edges.npz", edges=np.zeros((2, 0), dtype=np.int32))

    with open(split_dir / "node_metadata.json", "w") as f:
        json.dump({str(k): v for k, v in node_metadata.items()}, f, indent=2)

    with open(split_dir / "edge_metadata_normalized.json", "w") as f:
        json.dump(edge_metadata, f, indent=2)

    n_with = sum(1 for v in edge_metadata.values() if v.get("metrics"))
    return {"num_pos": len(pos_edges), "num_with_metrics": n_with}


def make_undirected(edges: List[Tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int32)
    arr = np.array(edges, dtype=np.int32)
    return np.concatenate([arr, arr[:, ::-1]], axis=0).T


# ──────────────────────────────────────────────
# Transductive splitting
# ──────────────────────────────────────────────

def _assign_by_ratio(items, val_ratio, test_ratio):
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return items[n_test + n_val:], items[n_test:n_test + n_val], items[:n_test]


def _build_dataset_degree_map(node_metadata, all_edges):
    degree = {}
    for u, v in all_edges:
        norm = normalize_edge(u, v, node_metadata)
        if norm:
            degree[norm[1]] = degree.get(norm[1], 0) + 1
    return degree


def _build_train_degree(train_edges, node_metadata):
    degree = Counter()
    for u, v in train_edges:
        norm = normalize_edge(u, v, node_metadata)
        if norm:
            degree[norm[0]] += 1
            degree[norm[1]] += 1
    return degree


def _enforce_both_seen(train_metric, train_no_metric, test_edges, node_metadata, seed):
    rng = random.Random(seed + 17)
    train_degree = _build_train_degree(train_metric + train_no_metric, node_metadata)

    invalid = []
    for e in test_edges:
        norm = normalize_edge(e[0], e[1], node_metadata)
        if not norm or train_degree[norm[0]] == 0 or train_degree[norm[1]] == 0:
            invalid.append(e)

    stats = {"initial_invalid": len(invalid), "repaired": 0, "dropped": 0}

    for bad in invalid:
        try:
            test_edges.remove(bad)
        except ValueError:
            continue
        train_metric.append(bad)
        norm = normalize_edge(bad[0], bad[1], node_metadata)
        if norm:
            train_degree[norm[0]] += 1
            train_degree[norm[1]] += 1

        indices = list(range(len(train_metric)))
        rng.shuffle(indices)
        swapped = False
        for idx in indices:
            c = train_metric[idx]
            if c == bad:
                continue
            cn = normalize_edge(c[0], c[1], node_metadata)
            if cn and train_degree[cn[0]] > 1 and train_degree[cn[1]] > 1:
                train_metric.pop(idx)
                train_degree[cn[0]] -= 1
                train_degree[cn[1]] -= 1
                test_edges.append(c)
                stats["repaired"] += 1
                swapped = True
                break
        if not swapped:
            stats["dropped"] += 1

    return stats


def run_transductive(
    node_metadata, full_edge_metadata, edges_with, edges_without,
    input_dir, output_dir, args,
):
    print("Transductive Graph Splitting")
    print("=" * 50)

    # Split edges
    edges_with_copy = list(edges_with)
    random.shuffle(edges_with_copy)
    train_metric, val_pos, test_pos = _assign_by_ratio(
        edges_with_copy, args.val_ratio, args.test_ratio
    )

    repair_stats = {}
    if args.enforce_both_seen:
        repair_stats = _enforce_both_seen(
            train_metric, list(edges_without), test_pos, node_metadata, args.seed
        )
        print(f"  Repair: {repair_stats}")

    train_pos = train_metric + list(edges_without)
    random.shuffle(train_pos)

    print(f"  Train: {len(train_pos)} ({len(edges_without)} without metrics)")
    print(f"  Val: {len(val_pos)}")
    print(f"  Test: {len(test_pos)}")

    # Build msg-passing edges
    train_undirected = make_undirected(train_pos)

    # Extract per-split edge metadata
    train_em = extract_split_edge_metadata(train_pos, full_edge_metadata)
    test_em = extract_split_edge_metadata(test_pos, full_edge_metadata)

    # Save
    output_dir = Path(output_dir)
    stats = {
        "train": save_split(output_dir / "train_split", train_undirected, train_pos, node_metadata, train_em),
        "test": save_split(output_dir / "test_split", train_undirected, test_pos, node_metadata, test_em),
    }

    if val_pos:
        val_em = extract_split_edge_metadata(val_pos, full_edge_metadata)
        stats["val"] = save_split(output_dir / "val_split", train_undirected, val_pos, node_metadata, val_em)
    else:
        val_dir = output_dir / "val_split"
        if val_dir.exists():
            shutil.rmtree(val_dir)

    save_node_embeddings(input_dir, output_dir, len(node_metadata), args.seed)

    split_info = {
        "type": "transductive",
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "enforce_both_seen_in_test": args.enforce_both_seen,
        **repair_stats,
        "source": "edge_metadata_normalized.json",
        "negative_sampling": "full",
        "edges_with_metrics": len(edges_with),
        "edges_no_metrics": len(edges_without),
        **{f"{k}_pos": v["num_pos"] for k, v in stats.items()},
        **{f"{k}_with_metrics": v["num_with_metrics"] for k, v in stats.items()},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDone: {output_dir}")
    for name, s in stats.items():
        print(f"  {name}: {s['num_pos']} pos ({s['num_with_metrics']} with metrics)")


# ──────────────────────────────────────────────
# Inductive splitting
# ──────────────────────────────────────────────

def split_nodes(node_ids, val_ratio, test_ratio, seed, stratify_degrees=None, degree_bins=4):
    random.seed(seed)
    nodes = list(node_ids)
    if not stratify_degrees:
        random.shuffle(nodes)
        n = len(nodes)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        return set(nodes[n_test + n_val:]), set(nodes[n_test:n_test + n_val]), set(nodes[:n_test])

    node_degrees = [stratify_degrees.get(nid, 0) for nid in nodes]
    boundaries = np.unique(np.quantile(
        np.array(node_degrees, dtype=float),
        np.linspace(0, 1, max(2, degree_bins + 1)),
    ))
    num_buckets = max(1, len(boundaries) - 1)
    buckets = [[] for _ in range(num_buckets)]

    for nid, deg in zip(nodes, node_degrees):
        idx = 0
        if len(boundaries) > 1:
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i + 1]
                if (i == len(boundaries) - 2 and lo <= deg <= hi) or (deg >= lo and deg < hi):
                    idx = i
                    break
        buckets[idx].append(nid)

    train, val, test = set(), set(), set()
    for b in buckets:
        random.shuffle(b)
        n = len(b)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test.update(b[:n_test])
        val.update(b[n_test:n_test + n_val])
        train.update(b[n_test + n_val:])
    return train, val, test


def _build_node_degrees(node_metadata, edges_with, edges_without):
    degree = {}
    for u, v in edges_with + edges_without:
        norm = normalize_edge(u, v, node_metadata)
        if norm:
            degree[norm[0]] = degree.get(norm[0], 0) + 1
            degree[norm[1]] = degree.get(norm[1], 0) + 1
    return degree


def _count_metric_edges_per_model(node_metadata, edges_with):
    count = {}
    for u, v in edges_with:
        norm = normalize_edge(u, v, node_metadata)
        if norm:
            count[norm[0]] = count.get(norm[0], 0) + 1
    return count


def _collect_metric_edges_per_model(node_metadata, edges):
    by_model = {}
    for u, v in edges:
        norm = normalize_edge(u, v, node_metadata)
        if norm:
            by_model.setdefault(norm[0], []).append((u, v))
    return by_model


def _get_edge_split(model_id, dataset_id, mode, train_m, val_m, test_m, train_d, val_d, test_d):
    if mode == "new_models":
        if model_id in train_m: return "train"
        elif model_id in val_m: return "val"
        return "test"
    elif mode == "new_datasets":
        if dataset_id in train_d: return "train"
        elif dataset_id in val_d: return "val"
        return "test"
    else:  # new_both
        if model_id in test_m or dataset_id in test_d: return "test"
        elif model_id in val_m or dataset_id in val_d: return "val"
        return "train"


def run_inductive(
    node_metadata, full_edge_metadata, edges_with, edges_without,
    input_dir, output_dir, args,
):
    mode = args.inductive_mode
    support_n = args.support_edges
    print(f"Inductive Graph Splitting (mode={mode}, support={support_n})")
    print("=" * 50)

    model_ids = [int(k) for k, v in node_metadata.items() if v.get("type") == "model"]
    dataset_ids = [int(k) for k, v in node_metadata.items() if v.get("type") == "dataset"]
    print(f"  Models: {len(model_ids)}, Datasets: {len(dataset_ids)}")

    node_degrees = _build_node_degrees(node_metadata, edges_with, edges_without)
    stratify = args.stratify_by_degree

    # Split nodes
    if mode == "new_models":
        deg_map = {n: node_degrees.get(n, 0) for n in model_ids} if stratify else None
        train_m, val_m, test_m = split_nodes(model_ids, args.val_ratio, args.test_ratio, args.seed, deg_map, args.degree_bins)
        train_d = val_d = test_d = set(dataset_ids)
    elif mode == "new_datasets":
        deg_map = {n: node_degrees.get(n, 0) for n in dataset_ids} if stratify else None
        train_d, val_d, test_d = split_nodes(dataset_ids, args.val_ratio, args.test_ratio, args.seed, deg_map, args.degree_bins)
        train_m = val_m = test_m = set(model_ids)
    else:
        m_deg = {n: node_degrees.get(n, 0) for n in model_ids} if stratify else None
        d_deg = {n: node_degrees.get(n, 0) for n in dataset_ids} if stratify else None
        train_m, val_m, test_m = split_nodes(model_ids, args.val_ratio, args.test_ratio, args.seed, m_deg, args.degree_bins)
        train_d, val_d, test_d = split_nodes(dataset_ids, args.val_ratio, args.test_ratio, args.seed + 1, d_deg, args.degree_bins)

    # Filter by support edges requirement
    if support_n > 0 and mode in ("new_models", "new_both"):
        metric_counts = _count_metric_edges_per_model(node_metadata, edges_with)
        min_req = support_n + 1
        demoted_test = {m for m in test_m if metric_counts.get(m, 0) < min_req}
        demoted_val = {m for m in val_m if metric_counts.get(m, 0) < min_req}
        test_m -= demoted_test
        val_m -= demoted_val
        train_m |= demoted_test | demoted_val
        print(f"  Demoted {len(demoted_test)} test + {len(demoted_val)} val models -> train")

    print(f"  Model split: train={len(train_m)}, val={len(val_m)}, test={len(test_m)}")
    if mode in ("new_datasets", "new_both"):
        print(f"  Dataset split: train={len(train_d)}, val={len(val_d)}, test={len(test_d)}")

    # Assign edges to splits
    splits = {"train": [], "val": [], "test": []}
    for u, v in edges_with:
        norm = normalize_edge(u, v, node_metadata)
        if not norm:
            continue
        s = _get_edge_split(norm[0], norm[1], mode, train_m, val_m, test_m, train_d, val_d, test_d)
        splits[s].append((u, v))

    # Add no-metric edges to train only
    train_no_metric = 0
    for u, v in edges_without:
        norm = normalize_edge(u, v, node_metadata)
        if not norm:
            continue
        s = _get_edge_split(norm[0], norm[1], mode, train_m, val_m, test_m, train_d, val_d, test_d)
        if s == "train":
            splits["train"].append((u, v))
            train_no_metric += 1

    # Support/query split
    support_list = []
    if support_n > 0:
        rng = random.Random(args.seed + 100)
        for split_name in ("test", "val"):
            by_model = _collect_metric_edges_per_model(node_metadata, splits[split_name])
            query = []
            for mid, m_edges in by_model.items():
                rng.shuffle(m_edges)
                support_list.extend(m_edges[:support_n])
                query.extend(m_edges[support_n:])
            splits[split_name] = query

    print(f"  Edge split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    if support_list:
        print(f"  Support edges: {len(support_list)}")

    # Build msg-passing edges
    train_ud = make_undirected(splits["train"])
    test_msg = splits["train"] + splits["val"] + support_list
    test_ud = make_undirected(test_msg)

    # Save
    output_dir = Path(output_dir)
    train_em = extract_split_edge_metadata(splits["train"], full_edge_metadata)
    test_em = extract_split_edge_metadata(splits["test"], full_edge_metadata)

    stats = {
        "train": save_split(output_dir / "train_split", train_ud, splits["train"], node_metadata, train_em),
        "test": save_split(output_dir / "test_split", test_ud, splits["test"], node_metadata, test_em),
    }
    if args.val_ratio > 0 and splits["val"]:
        val_em = extract_split_edge_metadata(splits["val"], full_edge_metadata)
        stats["val"] = save_split(output_dir / "val_split", train_ud, splits["val"], node_metadata, val_em)

    if support_list:
        np.savez(output_dir / "test_split" / "support_edges.npz",
                 edges=np.array(support_list, dtype=np.int32).T)
        sup_em = extract_split_edge_metadata(support_list, full_edge_metadata)
        with open(output_dir / "test_split" / "support_edge_metadata.json", "w") as f:
            json.dump(sup_em, f, indent=2)

    save_node_embeddings(input_dir, output_dir, len(node_metadata), args.seed)

    # Save node split
    with open(output_dir / "node_split.json", "w") as f:
        json.dump({
            "models": {"train": list(train_m), "val": list(val_m), "test": list(test_m)},
            "datasets": {"train": list(train_d), "val": list(val_d), "test": list(test_d)},
        }, f, indent=2)

    split_info = {
        "type": "inductive",
        "mode": mode,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "stratify_by_degree": args.stratify_by_degree,
        "degree_bins": args.degree_bins,
        "support_edges": support_n,
        "source": "edge_metadata_normalized.json",
        "negative_sampling": "full",
        "edges_with_metrics": len(edges_with),
        "edges_no_metrics": len(edges_without),
        "train_edges": len(splits["train"]),
        "train_edges_no_metrics": train_no_metric,
        "val_edges": len(splits["val"]),
        "test_edges": len(splits["test"]),
        "num_support_edges": len(support_list),
        **{f"{k}_with_metrics": v["num_with_metrics"] for k, v in stats.items()},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDone: {output_dir}")
    for name, s in stats.items():
        print(f"  {name}: {s['num_pos']} pos ({s['num_with_metrics']} with metrics)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 6: Create graph splits for link prediction.")
    parser.add_argument("--type", choices=["transductive", "inductive", "both"], default="both",
                        help="Split type to generate (default: both).")
    parser.add_argument("--input-dir", default="../data/artifact_graph_data_v3")
    parser.add_argument("--output-transductive", default="../data/artifact_graph_splits_v3_transductive")
    parser.add_argument("--output-inductive", default="../data/artifact_graph_splits_v3_inductive")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-both-seen", action="store_true",
                        help="[Transductive] Repair test so both endpoints seen in train.")
    parser.add_argument("--inductive-mode", choices=["new_models", "new_datasets", "new_both"],
                        default="new_models")
    parser.add_argument("--stratify-by-degree", action="store_true")
    parser.add_argument("--degree-bins", type=int, default=4)
    parser.add_argument("--support-edges", type=int, default=0,
                        help="[Inductive] Number of support edges per test model.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).parent
    input_dir = (script_dir / args.input_dir).resolve()

    # Load graph
    print("Loading graph...")
    G, node_metadata, _ = load_nx_graph(str(input_dir))
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Load normalized edge metadata
    full_edge_metadata = load_normalized_edge_metadata(input_dir)
    edges_with, edges_without = separate_edges_by_metrics(full_edge_metadata)
    print(f"  Edges with metrics: {len(edges_with)}")
    print(f"  Edges without metrics: {len(edges_without)}")

    if args.type in ("transductive", "both"):
        print(f"\n{'=' * 60}")
        out_trans = (script_dir / args.output_transductive).resolve()
        run_transductive(
            node_metadata, full_edge_metadata, edges_with, edges_without,
            input_dir, out_trans, args,
        )

    if args.type in ("inductive", "both"):
        print(f"\n{'=' * 60}")
        out_ind = (script_dir / args.output_inductive).resolve()
        run_inductive(
            node_metadata, full_edge_metadata, edges_with, edges_without,
            input_dir, out_ind, args,
        )

    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
