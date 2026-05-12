#!/usr/bin/env python3
"""
Joint GNN training (link + attribute) with CLI arguments.

Usage:
    python scripts/run_joint_gnn.py \
        --split-dir data/artifact_graph_splits_v3_0314_transductive \
        --output-dir data/final_results_ablation_layers/L3 \
        --backbone gatv2 --embedding-mode embedding --num-layers 3
"""
import argparse
import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, matthews_corrcoef

sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.training.gnn_joint_trainer import (
    JointModelConfig, JointTrainingConfig, GNNJointTrainer,
    build_joint_model, set_seed,
)
from artifact_graph.training.gnn_attribute_trainer import load_attribute_split
from artifact_graph.runners.runner_utils import load_node_embeddings, detect_split_type
from artifact_graph.utils.attribute_ranking_utils import load_attribute_ranking_data
from artifact_graph.utils.link_ranking_utils import load_link_ranking_data, compute_link_ranking_metrics
from artifact_graph.utils.link_prediction_utils import load_link_prediction_data
from artifact_graph.utils.evaluation_utils import calculate_ndcg


def full_eval(model, split_dir, emb_mode_str, device):
    """Evaluate on all tasks: attr pred/rank, link pred/rank."""
    forced_x = load_node_embeddings(str(split_dir), emb_mode_str)
    G_te, S_te = load_attribute_split(str(split_dir / "test_split"), forced_x)
    G_te.x, G_te.edge_index = G_te.x.to(device), G_te.edge_index.to(device)
    S_te.edge_label_index = S_te.edge_label_index.to(device)
    S_te.edge_label = S_te.edge_label.to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(G_te.x, G_te.edge_index)

    # --- Attr prediction ---
    with torch.no_grad():
        logits = model.decode_attr(z, S_te.edge_label_index).squeeze(-1)
        preds = torch.sigmoid(torch.clamp(logits, -10, 10)).cpu().numpy()
    y_true = S_te.edge_label.cpu().numpy()
    mae = float(np.mean(np.abs(preds - y_true)))
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    ss_res = np.sum((preds - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- Attr ranking ---
    _, _, _, ranking_data, _ = load_attribute_ranking_data(str(split_dir))
    sp_list, hit1_list, mrr_list = [], [], []
    for did, mvs in ranking_data.items():
        if len(mvs) < 2:
            continue
        mids = [m for m, v in mvs]
        tv = [float(v) for m, v in mvs]
        if all(t == tv[0] for t in tv):
            continue
        pairs = torch.tensor([[m, did] for m in mids], dtype=torch.long, device=device).t()
        with torch.no_grad():
            sc = torch.sigmoid(torch.clamp(model.decode_attr(z, pairs), -10, 10)).cpu().numpy().flatten()
        rho, _ = spearmanr(sc, tv)
        if not np.isnan(rho):
            sp_list.append(rho)
        # hit@1 / mrr
        true_best = mids[np.argmax(tv)]
        ranked_mids = [m for m, _ in sorted(zip(mids, sc), key=lambda x: x[1], reverse=True)]
        for i, m in enumerate(ranked_mids):
            if m == true_best:
                mrr_list.append(1.0 / (i + 1))
                hit1_list.append(1.0 if i == 0 else 0.0)
                break

    # --- Link prediction ---
    _, _, edges_lp_full, labels_lp_full = load_link_prediction_data(str(split_dir), seed=42)
    random.seed(42)
    idx = list(range(len(edges_lp_full)))
    random.shuffle(idx)
    idx = idx[:50000]
    edges_lp = [edges_lp_full[i] for i in idx]
    labels_lp = [labels_lp_full[i] for i in idx]
    pairs_lp = torch.tensor([[m, d] for m, d in edges_lp], dtype=torch.long, device=device).t()
    with torch.no_grad():
        lp_scores = torch.sigmoid(model.decode_link(z, pairs_lp)).cpu().numpy().flatten()
    lp_ap = average_precision_score(labels_lp, lp_scores)
    lp_mcc = matthews_corrcoef(labels_lp, (lp_scores > 0.5).astype(int))

    # --- Link ranking ---
    _, _, lr_data = load_link_ranking_data(str(split_dir))
    lr_results = []
    for did, (pos, neg) in lr_data.items():
        cands = pos + neg
        pairs = torch.tensor([[m, did] for m in cands], dtype=torch.long, device=device).t()
        with torch.no_grad():
            sc = torch.sigmoid(model.decode_link(z, pairs)).cpu().numpy().flatten()
        ranked = sorted(zip(cands, sc), key=lambda x: x[1], reverse=True)
        lr_results.append({"positive_models": pos, "ranked_model_ids": [m for m, _ in ranked]})
    lr_m = compute_link_ranking_metrics(lr_results)

    results = {
        "attr_mae": mae, "attr_rmse": rmse, "attr_r2": r2,
        "attr_spearman": float(np.mean(sp_list)) if sp_list else 0.0,
        "attr_hit1": float(np.mean(hit1_list)) if hit1_list else 0.0,
        "attr_mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "lp_ap_auc": float(lp_ap), "lp_mcc": float(lp_mcc),
        "lr_mrr": lr_m.get("mrr", 0), "lr_hit1": lr_m.get("hit@1", 0),
        "lr_ndcg10": lr_m.get("ndcg@10", 0),
        "num_attr_rankings": len(sp_list),
    }
    return results


def main():
    p = argparse.ArgumentParser(description="Joint GNN (link + attr) Training & Evaluation")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", default="gatv2",
                   choices=["gatv2", "gcn", "ncn", "ncnc", "neognn", "buddy"])
    p.add_argument("--embedding-mode", default="embedding",
                   choices=["random", "embedding"])
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--attr-weight", type=float, default=5.0)
    p.add_argument("--neg-ratio", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)

    split_prefix = detect_split_type(str(split_dir))
    emb_tag = "random" if args.embedding_mode == "random" else "emb"

    set_seed(args.seed)

    # Load data
    forced_x = load_node_embeddings(str(split_dir), args.embedding_mode)
    G_tr, S_tr = load_attribute_split(str(split_dir / "train_split"), forced_x)
    G_te, S_te = load_attribute_split(str(split_dir / "test_split"), forced_x)
    for G in (G_tr, G_te):
        G.x, G.edge_index = G.x.to(device), G.edge_index.to(device)
    for S in (S_tr, S_te):
        S.edge_label_index = S.edge_label_index.to(device)
        S.edge_label = S.edge_label.to(device)

    # Build & train
    cfg = JointModelConfig(
        in_channels=G_tr.x.size(1),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        backbone=args.backbone,
        model_type=args.backbone,
    )
    tcfg = JointTrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        neg_ratio=args.neg_ratio,
        attr_weight=args.attr_weight,
    )
    model = build_joint_model(cfg, device)
    print(f"Joint Model: {args.backbone} | layers={args.num_layers} | params={sum(p.numel() for p in model.parameters()):,}")

    trainer = GNNJointTrainer(model, device, tcfg)
    trainer.train(G_tr, S_tr, G_tr, S_te, verbose=False)

    # Evaluate
    results = full_eval(model, split_dir, args.embedding_mode, device)

    # Print
    print(f"\n--- Results ({split_prefix}, {args.backbone}, L={args.num_layers}, {emb_tag}) ---")
    for k, v in results.items():
        print(f"  {k:>20}: {v:.4f}")

    # Save results
    result_path = out_dir / f"{split_prefix}_joint_{args.backbone}_results_{emb_tag}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    model_path = out_dir / f"{split_prefix}_joint_{args.backbone}_model_{emb_tag}.pth"
    trainer.save_model(model_path, cfg)

    print(f"\nSaved: {result_path}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
