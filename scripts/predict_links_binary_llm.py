#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from artifact_graph.utils.graph_builder import load_artifact_graph_from_json
from artifact_graph.models.llm_link_predictor import LLMBinaryLinkPredictor  # same directory or adjust import


Edge = Tuple[str, str]

def _load_summaries(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
        return data if isinstance(data, dict) else {}


def _extract_md_nodes(G) -> Tuple[List[str], List[str]]:
    models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
    datasets = [n for n, d in G.nodes(data=True) if d.get("type") == "dataset"]
    return models, datasets


def _positive_edges(G) -> List[Edge]:
    pos = []
    for u, v in G.edges():
        ut, vt = G.nodes[u].get("type"), G.nodes[v].get("type")
        if ut == "model" and vt == "dataset":
            pos.append((u, v))
        elif vt == "model" and ut == "dataset":
            pos.append((v, u))
    # make unique
    return list({(m, d) for (m, d) in pos})


def _sample_negatives(
    models: Sequence[str],
    datasets: Sequence[str],
    existing: set[Edge],
    n: int,
    rng: random.Random,
) -> List[Edge]:
    neg: List[Edge] = []
    mlist, dlist = list(models), list(datasets)
    while len(neg) < n:
        pair = (rng.choice(mlist), rng.choice(dlist))
        if pair not in existing:
            neg.append(pair)
    return neg


def _shuffle_pairs(
    edges: List[Edge],
    labels: List[int],
    seed: int,
) -> Tuple[List[Edge], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(edges))
    rng.shuffle(idx)
    edges_shuf = [edges[i] for i in idx]
    labels_shuf = [labels[i] for i in idx]
    return edges_shuf, labels_shuf


def evaluate(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision": float(precision_score(true_labels, pred_labels)),
        "recall": float(recall_score(true_labels, pred_labels)),
        "f1": float(f1_score(true_labels, pred_labels)),
    }


def run(
    graph_file: Path,
    summaries_file: Path,
    model_name: str,
    mode: str,
    metric_name: str,
    seed: int,
    max_pairs: int,
):
    rng = random.Random(seed)

    # Create a valid filename from the model name
    safe_model_name = model_name.replace("/", "_")
    output_file = Path(f"output/llm_binary_predictions_{mode}_{safe_model_name}.json")

    # 1) graph + summaries
    G = load_artifact_graph_from_json(json_file=str(graph_file), min_downloads=1)
    summaries = _load_summaries(summaries_file)

    # 2) positives & negatives
    pos = _positive_edges(G)
    models, datasets = _extract_md_nodes(G)
    neg = _sample_negatives(models, datasets, set(pos), n=len(pos), rng=rng)

    edges = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    edges, labels = _shuffle_pairs(edges, labels, seed)

    if max_pairs > 0:
        edges, labels = edges[:max_pairs], labels[:max_pairs]

    print(f"Total pairs to predict: {len(edges)}  (mode={mode}, metric={metric_name})")

    # 3) predictor
    predictor = LLMBinaryLinkPredictor(model_name=model_name)
    pred_objs = predictor.predict(
        edges,
        G=G,
        mode=mode,
        summaries=summaries,
    )

    # 4) collect results + metrics
    out_rows = []
    y_true, y_pred = [], []
    for (m, d), y, obj in zip(edges, labels, pred_objs):
        row = {"model_id": m, "dataset_id": d, "true_label": y, "predicted_label": None, "reason": "", "status": "Failed"}
        if obj and (obj.get("prediction") is not None):
            pred_label = 1 if bool(obj["prediction"]) else 0
            y_true.append(y)
            y_pred.append(pred_label)
            row.update({"predicted_label": pred_label, "reason": obj.get("reason", ""), "status": "Success"})
        out_rows.append(row)

    if y_pred:
        metrics = evaluate(y_true, y_pred)
        print("\n--- Binary Classification Metrics ---")
        for k, v in metrics.items():
            print(f"  - {k.capitalize()}: {v:.4f}")
        print("------------------------------------")
    else:
        print("No valid predictions produced.")

    # 5) save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(out_rows, f, indent=2)
    print(f"\nPredictions saved to {output_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-file", default="output/perfect_model_dataset_metrics.json")
    p.add_argument("--summaries-file", default="output/readme_summaries.json")
    p.add_argument("--model", choices=["openai/gpt-4o", "openai/o3", "Qwen/Qwen2.5-72B-Instruct-Turbo"], default="Qwen/Qwen2.5-72B-Instruct-Turbo")
    p.add_argument("--mode", choices=["zero-shot", "simple", "neighborhood"], default="neighborhood")
    p.add_argument("--metric", default="accuracy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs", type=int, default=100000)  # cap like your original
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        graph_file=Path(a.graph_file),
        summaries_file=Path(a.summaries_file),
        model_name=a.model,
        mode=a.mode,
        metric_name=a.metric,
        seed=a.seed,
        max_pairs=a.max_pairs,
    )
