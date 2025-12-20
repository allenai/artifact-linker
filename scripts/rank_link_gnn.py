#!/usr/bin/env python3
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from types import SimpleNamespace

import numpy as np
import torch

# local imports
sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.utils.evaluation_utils import calculate_precision_at_k, calculate_recall_at_k
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.models.gnn_link_predictor import GNNLinkPredictor


# -------------------------
# Loading helpers
# -------------------------
def _load_embeddings(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.load(path, allow_pickle=False)
    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None) and "embedding" in arr.dtype.names:
        arr = arr["embedding"]
    return torch.from_numpy(arr).float().to(device)


def load_model_and_train_graph(model_path: str, data_dir: str, split_dir: str):
    """Load checkpoint, training graph (for message passing), and precompute node reps z."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_config' or 'model_state_dict'.")

    model = GNNLinkPredictor(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    droot = Path(data_dir)
    emb_path = droot / "node_embeddings.npy"
    train_edges_path = Path(split_dir) / "train_split" / "edges.npz"
    if not emb_path.exists() or not train_edges_path.exists():
        raise FileNotFoundError(f"Missing required files:\n  {emb_path}\n  {train_edges_path}")

    x = _load_embeddings(emb_path, device)

    # TODO: random embedding ablation
    x = torch.randn(x.size(0), x.size(1))

    edge_index = torch.from_numpy(np.load(train_edges_path)["edges"]).long().to(device)

    with torch.no_grad():
        z = model.encode(x, edge_index)

    print(f"[model] loaded: {ckpt['model_config']}")
    print(f"[data] train graph: nodes={x.shape[0]}, edges={edge_index.shape[1]}, feat_dim={x.shape[1]}")
    return model, device, z


def load_test_edges(split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    test_pos = Path(split_dir) / "test_split" / "pos_edges.npz"
    test_neg = Path(split_dir) / "test_split" / "neg_edges.npz"
    if not test_pos.exists() or not test_neg.exists():
        raise FileNotFoundError("Test split positive/negative edges not found.")
    pos = np.load(test_pos)["edges"]
    neg = np.load(test_neg)["edges"]
    print(f"[split] test edges: pos={pos.shape[1]}, neg={neg.shape[1]}")
    return pos, neg


# -------------------------
# Task construction & scoring
# -------------------------
def build_tasks(G, all_model_ids: Set[int], test_pos_edges: np.ndarray, max_datasets: int, num_neg_samples: int) -> List[Dict[str, Any]]:
    """Build ranking tasks by sampling N negative candidates for each dataset's positive items."""
    
    # Bucket all positive edges (train/val/test) by dataset to find true negatives later
    all_pos_by_dataset: Dict[int, Set[int]] = {}
    for u, v in G.edges():
        ut, vt = G.nodes[u].get("type"), G.nodes[v].get("type")
        if ut == "dataset" and vt == "model":
            all_pos_by_dataset.setdefault(u, set()).add(v)
        elif ut == "model" and vt == "dataset":
            all_pos_by_dataset.setdefault(v, set()).add(u)
            
    # Bucket test positive edges
    test_pos_by_dataset: Dict[int, Set[int]] = {}
    for i in range(test_pos_edges.shape[1]):
        u, v = int(test_pos_edges[0, i]), int(test_pos_edges[1, i])
        ut, vt = G.nodes.get(u, {}).get("type"), G.nodes.get(v, {}).get("type")
        if ut == "dataset" and vt == "model":
            test_pos_by_dataset.setdefault(u, set()).add(v)
        elif ut == "model" and vt == "dataset":
            test_pos_by_dataset.setdefault(v, set()).add(u)

    valid_dids = list(test_pos_by_dataset.keys())
    if len(valid_dids) > max_datasets:
        valid_dids = random.sample(valid_dids, max_datasets)

    tasks = []
    for did in valid_dids:
        test_pos = test_pos_by_dataset[did]
        all_pos = all_pos_by_dataset.get(did, set())
        
        # True negatives are all models MINUS any model that has ever been linked to this dataset
        true_neg_pool = list(all_model_ids - all_pos)
        
        if not true_neg_pool:
            continue
            
        k = min(num_neg_samples, len(true_neg_pool))
        neg_candidates = random.sample(true_neg_pool, k)

        tasks.append({
            "dataset_id": did,
            "positive_models": list(test_pos),
            "negative_candidates": neg_candidates,
            "relevant_items": test_pos,  # The ground truth for this test task
        })
    return tasks


@torch.no_grad()
def rank_for_dataset(model: GNNLinkPredictor, device: torch.device, z: torch.Tensor,
                     dataset_id: int, candidates: List[int]) -> List[Tuple[int, float]]:
    if not candidates:
        return []
    pairs = torch.tensor([[dataset_id, mid] for mid in candidates], dtype=torch.long, device=device).t()
    probs = torch.sigmoid(model.decode(z, pairs)).cpu().tolist()
    
    # Return list of (model_id, predicted_prob) sorted by probability
    return sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)


def eval_ranking(ranked_with_probs: List[Tuple[int, float]], relevant: Set[int], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    ranked_ids = [mid for mid, _ in ranked_with_probs] # Extract just the IDs for metric calculation
    
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = calculate_recall_at_k(ranked_ids, relevant, k)
        out[f"precision@{k}"] = calculate_precision_at_k(ranked_ids, relevant, k)
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="GNN link ranking on test set")
    ap.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    ap.add_argument("--data_dir", type=str, default="scripts/output/artifact_graph_data", help="Graph data directory")
    ap.add_argument("--split_dir", type=str, default="scripts/output/artifact_graph_splits", help="Dir with train/val/test splits")
    ap.add_argument("--max_datasets", type=int, default=1000000, help="Max datasets to sample for ranking")
    ap.add_argument("--output_file", type=str, default="scripts/output/final_results/gnn_link_rankings.json", help="Output JSON file")
    ap.add_argument("--num_neg_samples", type=int, default=10, help="Number of negative samples per dataset for ranking.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # graph (for node types) + test edges (for tasks)
    G, _, _ = load_nx_graph(args.data_dir)
    pos_edges, neg_edges = load_test_edges(args.split_dir)

    # Get all model IDs from the graph
    all_model_ids = {node_id for node_id, data in G.nodes(data=True) if data.get("type") == "model"}

    tasks = build_tasks(G, all_model_ids, pos_edges, args.max_datasets, args.num_neg_samples)
    print(f"[tasks] built from test set: {len(tasks)}")

    # model + train graph for message passing; fallback to random if it fails
    try:
        model, device, z = load_model_and_train_graph(args.model_path, args.data_dir, args.split_dir)
        use_gnn = True
        print("✅ GNN ready")
    except Exception as e:
        print(f"⚠️  GNN load/encode failed: {e}\n🔄 Using random ranking baseline")
        model = z = device = None
        use_gnn = False

    # inference
    results: List[Dict[str, Any]] = []
    detailed_rankings: List[Dict[str, Any]] = []  # New list for detailed, grouped outputs
    for i, t in enumerate(tasks):
        if i % 10 == 0:
            print(f"[progress] {i}/{len(tasks)}")
        all_models = t["positive_models"] + t["negative_candidates"]
        if use_gnn:
            ranked_with_probs = rank_for_dataset(model, device, z, t["dataset_id"], all_models)
        else:
            ranked_with_probs = sorted([(m, random.random()) for m in all_models], key=lambda x: x[1], reverse=True)
        
        # Store detailed ranking for this dataset
        candidates_details = []
        for model_id, prob in ranked_with_probs:
            candidates_details.append({
                "model_id": model_id,
                "predicted_probability": prob,
                "ground_truth_label": 1 if model_id in t["relevant_items"] else 0
            })
        detailed_rankings.append({
            "dataset_id": t["dataset_id"],
            "ranked_candidates": candidates_details
        })

        metrics = eval_ranking(ranked_with_probs, t["relevant_items"])
        metrics["dataset_id"] = t["dataset_id"]
        results.append(metrics)

    if not results:
        print("No valid rankings produced.")
        return

    # aggregate + print
    keys = [f"{m}@{k}" for m in ("recall", "precision") for k in (1, 3, 5, 10)]
    avgs = {k: float(np.mean([r[k] for r in results if k in r])) for k in keys}

    print("\n=== Link Ranking Results (Test Set) ===")
    for k in keys:
        if k in avgs:
            print(f"{k.upper()}: {avgs[k]:.4f}")
    print(f"Valid rankings: {len(results)}/{len(tasks)}")

    # save
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "model_used": use_gnn,
                "ranking_method": "GNN_inference" if use_gnn else "random_baseline",
                "num_ranking_tasks": len(tasks),
                "avg_metrics": {f"avg_{k}": v for k, v in avgs.items()},
                "individual_evaluations": results,
                "detailed_rankings_by_dataset": detailed_rankings
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
