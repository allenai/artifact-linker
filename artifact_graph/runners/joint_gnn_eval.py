#!/usr/bin/env python3
"""Shared loader + per-task evaluators for the joint GNN (link + attr heads).

The joint model is trained once (scripts/train_joint_gnn.py) and saved as a
checkpoint; the four task scripts load that checkpoint and call exactly one of
the eval_* functions below. Logic is extracted verbatim from the original
run_joint_gnn.full_eval so numbers are identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score, matthews_corrcoef

from ..training.gnn_joint_trainer import JointModelConfig, build_joint_model
from ..training.gnn_attribute_trainer import load_attribute_split
from .runner_utils import load_node_embeddings, detect_split_type
from ..utils.attribute_ranking_utils import load_attribute_ranking_data
from ..utils.link_ranking_utils import load_link_ranking_data, compute_link_ranking_metrics
from ..utils.link_prediction_utils import load_link_prediction_data


def load_joint_model(ckpt_path: str | Path, device: torch.device):
    """Rebuild a trained joint model from a checkpoint saved by GNNJointTrainer.save_model."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mc = dict(ckpt["model_config"])
    cfg = JointModelConfig(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        num_layers=mc["num_layers"],
        heads=mc["heads"],
        dropout=mc["dropout"],
        backbone=mc["backbone"],
        model_type=mc.get("gnn_model_type", mc.get("model_type", "gatv2")),
    )
    model = build_joint_model(cfg, device, use_heckman=ckpt.get("use_heckman", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _encode(model, split_dir, emb_mode_str, device):
    """Encode test-graph node embeddings z (shared by every task)."""
    forced_x = load_node_embeddings(str(split_dir), emb_mode_str)
    G_te, S_te = load_attribute_split(str(Path(split_dir) / "test_split"), forced_x)
    G_te.x, G_te.edge_index = G_te.x.to(device), G_te.edge_index.to(device)
    S_te.edge_label_index = S_te.edge_label_index.to(device)
    S_te.edge_label = S_te.edge_label.to(device)
    model.eval()
    with torch.no_grad():
        z = model.encode(G_te.x, G_te.edge_index)
    return z, S_te


def eval_attr_prediction(model, split_dir, emb_mode_str, device):
    z, S_te = _encode(model, split_dir, emb_mode_str, device)
    with torch.no_grad():
        logits = model.decode_attr(z, S_te.edge_label_index).squeeze(-1)
        preds = torch.sigmoid(torch.clamp(logits, -10, 10)).cpu().numpy()
    y_true = S_te.edge_label.cpu().numpy()
    mae = float(np.mean(np.abs(preds - y_true)))
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    ss_res = np.sum((preds - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"attr_mae": mae, "attr_rmse": rmse, "attr_r2": r2}


def eval_attr_ranking(model, split_dir, emb_mode_str, device):
    z, _ = _encode(model, split_dir, emb_mode_str, device)
    _, _, _, ranking_data, _ = load_attribute_ranking_data(str(split_dir))
    sp_list, tau_list, hit1_list, mrr_list, ndcg1_list = [], [], [], [], []
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
        tau, _ = kendalltau(sc, tv)
        if not np.isnan(tau):
            tau_list.append(tau)
        true_best = mids[np.argmax(tv)]
        ranked_mids = [m for m, _ in sorted(zip(mids, sc), key=lambda x: x[1], reverse=True)]
        top1_mid = ranked_mids[0]
        top1_rel = tv[mids.index(top1_mid)]
        ideal_rel = max(tv)
        ndcg1_list.append(top1_rel / ideal_rel if ideal_rel > 0 else 0.0)
        for i, m in enumerate(ranked_mids):
            if m == true_best:
                mrr_list.append(1.0 / (i + 1))
                hit1_list.append(1.0 if i == 0 else 0.0)
                break
    avg = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "attr_kendall_tau": avg(tau_list),
        "attr_spearman": avg(sp_list),
        "attr_hit1": avg(hit1_list),
        "attr_ndcg1": avg(ndcg1_list),
        "attr_mrr": avg(mrr_list),
        "num_attr_rankings": len(sp_list),
    }


def eval_link_prediction(model, split_dir, emb_mode_str, device):
    """Full-set AP/MCC (~5.3M pairs, same protocol as the baseline pipeline)."""
    z, _ = _encode(model, split_dir, emb_mode_str, device)
    _, _, edges_lp, labels_lp = load_link_prediction_data(str(split_dir), seed=42)
    pairs_lp = torch.tensor([[m, d] for m, d in edges_lp], dtype=torch.long, device=device).t()
    with torch.no_grad():
        chunks = []
        bs = 1_000_000  # bound peak memory on ~5.3M pairs
        for i in range(0, pairs_lp.size(1), bs):
            chunks.append(model.decode_link(z, pairs_lp[:, i:i + bs]).cpu())
        lp_scores = torch.sigmoid(torch.cat(chunks)).numpy().flatten()
    return {
        "lp_ap_auc": float(average_precision_score(labels_lp, lp_scores)),
        "lp_mcc": float(matthews_corrcoef(labels_lp, (lp_scores > 0.5).astype(int))),
        "num_pairs": int(len(labels_lp)),
    }


def eval_link_ranking(model, split_dir, emb_mode_str, device):
    z, _ = _encode(model, split_dir, emb_mode_str, device)
    _, _, lr_data = load_link_ranking_data(str(split_dir))
    lr_results = []
    for did, (pos, neg) in lr_data.items():
        cands = pos + neg
        pairs = torch.tensor([[m, did] for m in cands], dtype=torch.long, device=device).t()
        with torch.no_grad():
            sc = torch.sigmoid(model.decode_link(z, pairs)).cpu().numpy().flatten()
        ranked = sorted(zip(cands, sc), key=lambda x: x[1], reverse=True)
        lr_results.append({"positive_models": pos, "ranked_model_ids": [m for m, _ in ranked]})
    m = compute_link_ranking_metrics(lr_results)
    return {
        "lr_mrr": m.get("mrr", 0), "lr_hit1": m.get("hit@1", 0),
        "lr_hit5": m.get("hit@5", 0), "lr_recall5": m.get("recall@5", 0),
        "lr_ndcg5": m.get("ndcg@5", 0), "lr_ndcg10": m.get("ndcg@10", 0),
        "num_queries": len(lr_results),
    }
