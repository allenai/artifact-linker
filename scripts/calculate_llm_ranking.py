#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------- I/O ----------


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            breakpoint()
            for k in ["data", "rows", "items", "records", "predictions"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
            raise ValueError(
                "JSON root is a dict; provide a list or a dict with key 'data'/similar."
            )
        else:
            raise ValueError("Unsupported JSON structure; expected list or jsonl.")
    else:
        raise ValueError("Unsupported file extension. Use .json or .jsonl")


# ---------- Metrics ----------


def dcg(relevances: List[float], k: Optional[int] = None, gain: str = "identity") -> float:
    if k is not None:
        relevances = relevances[:k]
    score = 0.0
    for i, rel in enumerate(relevances):
        if gain == "exp2":
            gain_val = 2.0**rel - 1.0
        elif gain == "identity":
            gain_val = rel
        else:
            raise ValueError("gain must be 'identity' or 'exp2'")
        score += gain_val / math.log2(i + 2.0)
    return score


def ndcg_at_k(
    true_vals_sorted_by_pred: List[float], k: Optional[int] = None, gain: str = "identity"
) -> float:
    if not true_vals_sorted_by_pred:
        return 0.0
    ideal = sorted(true_vals_sorted_by_pred, reverse=True)
    if k is not None:
        ideal = ideal[:k]
    idcg = dcg(ideal, k=None, gain=gain)
    if idcg == 0.0:
        return 0.0
    return dcg(true_vals_sorted_by_pred, k=k, gain=gain) / idcg


def mrr_top_n(pred_order_ids: List[Any], true_best_ids: List[Any]) -> float:
    pos = float("inf")
    index = {mid: i for i, mid in enumerate(pred_order_ids)}  # 0-based
    for tid in true_best_ids:
        if tid in index:
            pos = min(pos, index[tid] + 1)  # 1-based
    if pos == float("inf"):
        return 0.0
    return 1.0 / pos


# ---------- Core ----------


def autodetect_group_col(rows: List[Dict[str, Any]], fallback_name: str = "ALL") -> Optional[str]:
    """
    Try to find a sensible dataset grouping column.
    Preference order: dataset_id, dataset, task_id, group, corpus, collection.
    Return None if nothing found (caller can group everything into one).
    """
    candidates = ["dataset_id", "dataset", "task_id", "group", "corpus", "collection"]
    if not rows:
        return None
    keys = set()
    for r in rows:
        keys.update(r.keys())
    for c in candidates:
        if c in keys:
            return c
    return None  # no grouping column present


def compute_per_dataset_metrics(
    rows: List[Dict[str, Any]],
    group_col: Optional[str],
    id_col: str = "model_id",
    true_col: str = "true_metric",
    pred_col: str = "predicted_metric",
    k: Optional[int] = None,
    gain: str = "identity",
    mrr_topn: int = 1,
) -> pd.DataFrame:
    """
    If group_col is None, all rows are treated as a single group named 'ALL'.
    Otherwise, group by group_col.
    """
    by_group: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    if group_col is None:
        # Single bucket
        valid = [r for r in rows if id_col in r and true_col in r and pred_col in r]
        by_group["ALL"] = valid
    else:
        for r in rows:
            if group_col in r and id_col in r and true_col in r and pred_col in r:
                by_group[r[group_col]].append(r)

    results = []
    for ds, items in by_group.items():
        if not items:
            continue
        items_sorted_pred = sorted(items, key=lambda x: float(x[pred_col]), reverse=True)
        true_vals_in_pred_order = [float(x[true_col]) for x in items_sorted_pred]

        top_true_sorted = sorted(items, key=lambda x: float(x[true_col]), reverse=True)
        top_true_ids = [x[id_col] for x in top_true_sorted[: max(1, mrr_topn)]]

        nd = ndcg_at_k(true_vals_in_pred_order, k=k, gain=gain)
        mrr = mrr_top_n([x[id_col] for x in items_sorted_pred], top_true_ids)

        best_true_id = top_true_ids[0]
        pred_index = next(
            (i for i, x in enumerate(items_sorted_pred) if x[id_col] == best_true_id), None
        )
        best_true_rank = (pred_index + 1) if pred_index is not None else None

        results.append(
            {
                (group_col or "group"): ds,
                "n_items": len(items),
                f"NDCG@{k if k is not None else 'all'}({gain})": nd,
                f"MRR@top{mrr_topn}": mrr,
                "best_true_model": best_true_id,
                "best_true_pred_rank": best_true_rank,
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(df.columns[0])
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute NDCG/MRR per dataset and macro average.")
    ap.add_argument("--input", required=True, help="Path to input .json or .jsonl file.")
    ap.add_argument(
        "--group-col",
        default="auto",
        help="Grouping column (e.g., dataset_id). Use 'auto' (default) to autodetect; use '' to treat all as one group.",
    )
    ap.add_argument("--id-col", default="model_id")
    ap.add_argument("--true-col", default="true_metric")
    ap.add_argument("--pred-col", default="predicted_metric")
    ap.add_argument("--k", type=int, default=None, help="Cutoff for NDCG@k (default: all).")
    ap.add_argument("--gain", choices=["identity", "exp2"], default="identity")
    ap.add_argument("--mrr-topn", type=int, default=1)
    ap.add_argument(
        "--out",
        default="per_dataset_metrics.csv",
        help="Per-dataset CSV output (ignored if --macro-only).",
    )
    ap.add_argument(
        "--macro-only", action="store_true", help="Only print macro averages (no per-dataset CSV)."
    )
    args = ap.parse_args()

    rows = read_json_or_jsonl(args.input)

    if args.group_col == "auto":
        group_col = autodetect_group_col(rows)
    elif args.group_col == "":
        group_col = None
    else:
        group_col = args.group_col

    df = compute_per_dataset_metrics(
        rows,
        group_col=group_col,
        id_col=args.id_col,
        true_col=args.true_col,
        pred_col=args.pred_col,
        k=args.k,
        gain=args.gain,
        mrr_topn=args.mrr_topn,
    )

    # Macro averages: equal weight per dataset (unweighted mean), regardless of #models per dataset.
    ndcg_cols = [c for c in df.columns if c.startswith("NDCG@")]
    mrr_cols = [c for c in df.columns if c.startswith("MRR@")]
    macro = {}
    if not df.empty:
        for c in ndcg_cols + mrr_cols:
            macro[c + "_macro_avg"] = float(df[c].mean())

    # Output
    if args.macro_only:
        # 只打印宏平均
        print("Macro averages (equal weight across datasets):")
        if macro:
            for k, v in macro.items():
                print(f"  {k}: {v:.6f}")
        else:
            print("  (no data)")
    else:
        # 写出 per-dataset CSV，并打印宏平均
        df.to_csv(args.out, index=False)
        print("Wrote per-dataset metrics to:", args.out)
        print("Macro averages (equal weight across datasets):")
        if macro:
            for k, v in macro.items():
                print(f"  {k}: {v:.6f}")
        else:
            print("  (no data)")


if __name__ == "__main__":
    main()
