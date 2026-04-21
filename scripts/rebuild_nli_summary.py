#!/usr/bin/env python3
"""
Rebuild all_results_summary_fixed.json from raw predictions.

Fixes two inconsistencies in the original LLM-coder eval pipeline:

  1. Buggy 0-accuracy cells: 9 (model, dataset) pairs where the top-level
     `accuracy` field was overwritten with 0 but `previous_accuracy` holds
     the real value. We recover them.

  2. Inconsistent 3-way/binary GT on MNLI, SNLI, ANLI, NLI_FEVER:
     some per-cell eval scripts collapsed GT to {entailment, not_entailment}
     while others kept 3-way. We treat these 4 datasets as native 3-way.
     For a cell to have a valid 3-way score, BOTH predictions and GT in the
     stored predictions.json must be 3-way (>=3 distinct labels). Cells
     where either side was binarized (i.e. binary-output models, or eval
     scripts that collapsed GT) are masked — they cannot be compared
     apples-to-apples on the 3-way task.

Non-3-way datasets keep their original accuracy (with previous_accuracy
fallback for the 9 bug-fix cells).

Input:  /mnt/data_from_server1/.../smolagent_results_coding_agent_nli_sota_0112_full_shared_loader
Output: all_results_summary_fixed.json (in repo root)
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


SRC_DEFAULT = (
    "/mnt/data_from_server1/haofeiy2/artifact-graph/scripts/"
    "smolagent_results_coding_agent_nli_sota_0112_full_shared_loader"
)

DATASET_MAP = {
    "allenai_scitail": "allenai/scitail",
    "facebook_anli": "facebook/anli",
    "nyu-mll_multi_nli": "nyu-mll/multi_nli",
    "pietrolesci_nli_fever": "pietrolesci/nli_fever",
    "pietrolesci_robust_nli_ST_SE": "pietrolesci/robust_nli_ST_SE",
    "stanfordnlp_snli": "stanfordnlp/snli",
    "tasksource_babi_nli": "tasksource/babi_nli",
    "tasksource_defeasible-nli": "tasksource/defeasible-nli",
    "alisawuffles_WANLI": "alisawuffles/WANLI",
    "araag2_MedNLI": "araag2/MedNLI",
    "SetFit_qnli": "SetFit/qnli",
    "SetFit_rte": "SetFit/rte",
}

# Datasets whose native label space is 3-way. For these, we standardize
# evaluation to the binarized task so binary and 3-way output models are
# comparable.
THREE_WAY_DATASETS = {
    "nyu-mll/multi_nli",
    "stanfordnlp/snli",
    "facebook/anli",
    "pietrolesci/nli_fever",
}

# SOTA references (kept from the original summary)
SOTA_REFERENCE = {
    "allenai/scitail": 0.968,
    "facebook/anli": 0.702,
    "nyu-mll/multi_nli": 0.92,
    "pietrolesci/nli_fever": 0.777,
    "stanfordnlp/snli": 0.931,
    "tasksource/babi_nli": 0.97,
    "tasksource/defeasible-nli": 0.78,
    "alisawuffles/WANLI": 0.785,
    "araag2/MedNLI": 0.872,
    "SetFit/qnli": 0.96,
    "SetFit/rte": 0.92,
    "pietrolesci/robust_nli": 0.69,
}


def normalize(label):
    """Canonicalize label strings for comparison."""
    if label is None:
        return None
    s = str(label).strip().lower()
    # digit / label_N encodings (dataset convention: 0=entailment, 1=neutral, 2=contradiction)
    digit_map = {
        "0": "entailment", "label_0": "entailment",
        "1": "neutral", "label_1": "neutral",
        "2": "contradiction", "label_2": "contradiction",
    }
    if s in digit_map:
        return digit_map[s]
    if s in ("non_entailment", "not-entailment"):
        return "not_entailment"
    return s


VALID_3WAY = {"entailment", "neutral", "contradiction"}


def load_pred_pairs(preds_path):
    """Return (pairs, gt_labels, pr_labels) or (None, set(), set()) on failure."""
    try:
        preds = json.load(open(preds_path))
    except Exception:
        return None, set(), set()
    if not preds:
        return None, set(), set()
    pairs, gt_labels, pr_labels = [], set(), set()
    for r in preds:
        gt = normalize(r.get("ground_truth"))
        pr = normalize(r.get("prediction"))
        if gt is None or pr is None:
            continue
        gt_labels.add(gt)
        pr_labels.add(pr)
        pairs.append((gt, pr))
    return pairs, gt_labels, pr_labels


def compute_3way_acc(pairs):
    if not pairs:
        return None
    return sum(1 for gt, pr in pairs if gt == pr) / len(pairs)


def parse_dir(name):
    """Split a result dir name `{model}_{dataset}_accuracy` into model_id, dataset_id."""
    for dd, cid in DATASET_MAP.items():
        if name.endswith(f"_{dd}_accuracy"):
            model_part = name[: -(len(dd) + len("_accuracy") + 1)]
            return model_part.replace("_", "/", 1), cid
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC_DEFAULT,
                   help="Raw eval root with {model}_{dataset}_accuracy/ subdirs")
    p.add_argument("--out", default="all_results_summary_fixed.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = (root / args.out).resolve()
    src = Path(args.src)

    # -------- Pass 1: gather raw info per cell (no masking decision yet) --------
    cells = []  # list of dicts with full context
    for d in sorted(os.listdir(src)):
        full = src / d
        if not full.is_dir():
            continue
        model_id, ds_id = parse_dir(d)
        if model_id is None:
            continue
        res_path = full / "results.json"
        preds_path = full / "predictions.json"
        if not res_path.exists():
            continue
        try:
            j = json.load(open(res_path))
        except Exception:
            continue
        pairs, gt_lbls, pr_lbls = (None, set(), set())
        if preds_path.exists():
            pairs, gt_lbls, pr_lbls = load_pred_pairs(preds_path)
        cells.append({
            "model_id": model_id, "dataset_id": ds_id,
            "accuracy_raw": j.get("accuracy"),
            "previous_accuracy": j.get("previous_accuracy"),
            "pairs": pairs, "gt_labels": gt_lbls, "pr_labels": pr_lbls,
        })

    # -------- Pass 2: per-MODEL 3-way capability on 3-way datasets --------
    # A model is "3-way capable" iff on AT LEAST ONE 3-way dataset its
    # predictions cover all three NLI classes {entailment, neutral,
    # contradiction}. This signals the model can architecturally emit 3
    # classes. Mode-collapse on a specific test set (e.g. a generative LLM
    # only outputting "neutral" on SNLI) is an honest poor score, not a
    # reason to mask. Only architecturally-binary models — those that
    # NEVER produce 3 classes on any of the 4 datasets — are masked on all
    # 4 cells. Enforces per-model consistency: all-mask or all-unmask.
    from collections import defaultdict
    per_ds_pr = defaultdict(dict)  # model -> {ds: valid-3way pred labels}
    for c in cells:
        if c["dataset_id"] in THREE_WAY_DATASETS:
            per_ds_pr[c["model_id"]][c["dataset_id"]] = c["pr_labels"] & VALID_3WAY
    binary_only_models = set()
    for m, ds_map in per_ds_pr.items():
        # 3-way capable ⇔ ever emitted all 3 classes on at least one dataset
        if not any(pr == VALID_3WAY for pr in ds_map.values()):
            binary_only_models.add(m)

    # -------- Pass 3: build final rows --------
    results = []
    n_bug_fix = 0
    n_3way_recomputed = 0
    n_masked = 0
    n_still_zero = 0
    for c in cells:
        model_id, ds_id = c["model_id"], c["dataset_id"]
        a = c["accuracy_raw"]
        prev = c["previous_accuracy"]
        src_field = "accuracy"
        masked = False
        mask_reason = None

        # Step 1: previous_accuracy fallback for the 9 bug cells
        if (a == 0 or a is None) and prev and prev > 0:
            a = prev
            src_field = "previous_accuracy(bug_fix)"
            n_bug_fix += 1

        # Step 2: on 3-way datasets, decide per-MODEL
        if ds_id in THREE_WAY_DATASETS:
            if model_id in binary_only_models:
                masked = True
                mask_reason = (
                    "architecturally binary: never emitted all 3 NLI classes "
                    "on any of the 4 three-way datasets"
                )
                n_masked += 1
            elif c["pairs"] is not None:
                # 3-way capable: recompute honest 3-way accuracy on this cell
                acc_3way = compute_3way_acc(c["pairs"])
                if acc_3way is not None:
                    a = acc_3way
                    src_field = "3way_recomputed"
                    n_3way_recomputed += 1

        if a == 0 or a is None:
            n_still_zero += 1
            a = 0.0

        eval_type = "3way" if ds_id in THREE_WAY_DATASETS else "binary"
        row = {
            "model_id": model_id, "dataset_id": ds_id,
            "metric": "accuracy", "accuracy": a,
            "eval_type": eval_type, "source_field": src_field,
        }
        if masked:
            row["masked"] = True
            row["masked_reason"] = mask_reason
        results.append(row)

    models = sorted({r["model_id"] for r in results})
    datasets = sorted({r["dataset_id"] for r in results})
    print(f"Rows: {len(results)} | models: {len(models)} | datasets: {len(datasets)}")
    print(f"  bug-fixed via previous_accuracy: {n_bug_fix}")
    print(f"  3-way recomputed (MNLI/SNLI/ANLI/NLI_FEVER): {n_3way_recomputed}")
    print(f"  masked (cannot be scored 3-way): {n_masked}")
    print(f"  still zero (true failures): {n_still_zero}")

    # Best per dataset (exclude zeros and masked)
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["dataset_id"]].append(r)
    best = {}
    for ds, rs in by_ds.items():
        rs = [r for r in rs if r["accuracy"] > 0 and not r.get("masked")]
        if not rs:
            continue
        b = max(rs, key=lambda r: r["accuracy"])
        best[ds] = {"best_accuracy": b["accuracy"], "best_model": b["model_id"]}

    out = {
        "total_evaluated": len(results),
        "total_entries": len(results),
        "num_models": len(models),
        "num_datasets": len(datasets),
        "is_rectangular": len(results) == len(models) * len(datasets),
        "sota_reference": SOTA_REFERENCE,
        "best_per_dataset": best,
        "three_way_datasets_binarized": sorted(THREE_WAY_DATASETS),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Show best per dataset
    print("\n== Best per dataset ==")
    for ds in sorted(best):
        b = best[ds]
        sota_v = SOTA_REFERENCE.get(ds) or SOTA_REFERENCE.get("pietrolesci/robust_nli" if ds == "pietrolesci/robust_nli_ST_SE" else ds)
        gap = f" (SOTA {sota_v:.3f}, gap {b['best_accuracy'] - sota_v:+.3f})" if sota_v else ""
        print(f"  {ds}: {b['best_accuracy']:.4f}  {b['best_model']}{gap}")


if __name__ == "__main__":
    main()
