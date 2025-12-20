#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 你的模型
from artifact_graph.models.gnn_link_predictor import GNNLinkPredictor


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_edge_metadata(graph_data_dir: str) -> dict:
    """读取 graph_data_dir/edge_metadata.json，规范化为 {(u,v): {metric: val, ...}}"""
    f = Path(graph_data_dir) / "edge_metadata.json"
    if not f.exists():
        raise FileNotFoundError(f"edge metadata not found: {f}")
    meta = json.loads(f.read_text(encoding="utf-8"))

    mapping = {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                u, w = int(k[0]), int(k[1])
            else:
                s = str(k).strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                sep = "," if "," in s else ("|" if "|" in s else " ")
                parts = [t for t in s.split(sep) if t.strip()]
                if len(parts) < 2:
                    continue
                u, w = int(parts[0]), int(parts[1])
            metrics = v.get("metrics", v)
            mapping[(u, w)] = metrics if isinstance(metrics, dict) else {}
    elif isinstance(meta, list):
        for item in meta:
            u = int(item.get("u", item.get("src", item.get("from", -1))))
            w = int(item.get("v", item.get("dst", item.get("to", -1))))
            if u < 0 or w < 0:
                continue
            metrics = item.get("metrics", {})
            mapping[(u, w)] = metrics if isinstance(metrics, dict) else {}
    else:
        raise ValueError("edge_metadata.json format not recognized")
    return mapping


def _pick_metric_key(mapping: dict, metric_key: str | None) -> str:
    keys = set()
    for m in mapping.values():
        keys.update(m.keys())
    if not keys:
        raise ValueError("No metrics found in edge metadata.")
    if metric_key is None:
        mk = sorted(keys)[0]
        print(f"[info] auto-picked metric_key='{mk}' from {sorted(keys)}")
        return mk
    if metric_key not in keys:
        raise KeyError(f"metric_key '{metric_key}' not in available {sorted(keys)}")
    return metric_key


def _values_from_metadata_for_pos(
    pos_edges_np: np.ndarray,
    mapping: dict,
    metric_key: str,
    undirected: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 pos_edges 顺序取值；仅保留有真值的样本，返回过滤后的 edge_index 与 values。"""
    kept_edges, kept_vals = [], []
    for u, v in pos_edges_np.T:
        u, v = int(u), int(v)
        val = None
        if (u, v) in mapping and metric_key in mapping[(u, v)]:
            val = mapping[(u, v)][metric_key]
        elif undirected and (v, u) in mapping and metric_key in mapping[(v, u)]:
            val = mapping[(v, u)][metric_key]
        if val is None:
            continue
        val = float(val)
        if val > 1.0:  # 允许百分比，归一化到 [0,1]
            val /= 100.0
        kept_edges.append([u, v])
        kept_vals.append(val)

    if not kept_edges:
        raise ValueError(f"No positive edges have metric '{metric_key}'.")
    edge_label_index = torch.tensor(kept_edges, dtype=torch.long).t().contiguous()  # [2, K]
    edge_label = torch.tensor(kept_vals, dtype=torch.float)                          # [K]
    return edge_label_index, edge_label


def _load_embeddings(artifact_graph_data_dir: Path) -> torch.Tensor:
    """支持 .npy（plain 或含 'embedding' 字段）或 .npz（尝试常见键）。"""
    npy = artifact_graph_data_dir / "node_embeddings.npy"
    npz = artifact_graph_data_dir / "node_embeddings.npz"
    if npy.exists():
        arr = np.load(npy, allow_pickle=False)
        if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None) and "embedding" in arr.dtype.names:
            emb = arr["embedding"]
        else:
            emb = arr
        print(f"[emb] loaded {npy}")
        return torch.from_numpy(emb).float()
    if npz.exists():
        arr = np.load(npz)
        for k in ("embeddings", "embedding", "X", "x"):
            if k in arr:
                print(f"[emb] loaded {npz}:{k}")
                return torch.from_numpy(arr[k]).float()
        raise KeyError(f"No embeddings array found in {npz}. Tried keys: embeddings/embedding/X/x")
    raise FileNotFoundError(f"No embeddings file found in {artifact_graph_data_dir}")


# =========================
# Data Loader (PyG Data)
# =========================
def load_split(split_dir: str, graph_data_dir: str, metric_key: str | None = None, forced_x: torch.Tensor | None = None):
    """
    返回:
      G: Data(x, edge_index, num_nodes) —— 结构图（消息传递）
      split: Data(edge_label_index, edge_label) —— 仅包含有真标签的正边及连续值
    """
    p = Path(split_dir)
    # 结构图
    edges = torch.from_numpy(np.load(p / "edges.npz")["edges"]).long()
    node_meta = json.loads((p / "node_metadata.json").read_text(encoding="utf-8"))
    
    if forced_x is not None:
        x = forced_x
    else:
        x = _load_embeddings(p.parent.parent / "artifact_graph_data")

    G = Data(x=x, edge_index=edges, num_nodes=len(node_meta))

    # 候选正边
    pos_all = np.load(p / "pos_edges.npz")["edges"]  # [2, N_pos] numpy

    # 从 metadata 取 metric
    mapping = _load_edge_metadata(graph_data_dir)
    metric_key = _pick_metric_key(mapping, metric_key)
    edge_label_index, edge_label = _values_from_metadata_for_pos(pos_all, mapping, metric_key, undirected=True)

    split = Data()
    split.edge_label_index = edge_label_index
    split.edge_label = edge_label
    print(f"[split] kept {split.edge_label.numel()} / {pos_all.shape[1]} pos edges with '{metric_key}'")
    return G, split


# =========================
# Train / Eval
# =========================
def train_epoch(model, G: Data, split: Data, opt: torch.optim.Optimizer) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    z = model.encode(G.x, G.edge_index)
    logits = model.decode(z, split.edge_label_index).squeeze(-1)  # [K]
    
    # 检查logits范围，避免数值爆炸
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"⚠️  警告：logits包含NaN或Inf值")
        return float('inf')
    
    # 不使用sigmoid，直接回归logits！
    # 将目标转换到logit空间：logit(y) = log(y / (1-y))
    y = split.edge_label
    # 避免log(0)和log(inf)
    y_clipped = torch.clamp(y, min=1e-7, max=1-1e-7)
    y_logits = torch.log(y_clipped / (1 - y_clipped))
    
    # 在logit空间做MSE
    loss = F.mse_loss(logits, y_logits)
    
    loss.backward()
    
    # 检查梯度
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if total_norm < 1e-6:
        print(f"⚠️  梯度太小: {total_norm:.8f}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate(model, G: Data, split: Data, return_preds: bool = False):
    model.eval()
    z = model.encode(G.x, G.edge_index)
    logits = model.decode(z, split.edge_label_index).squeeze(-1)
    
    # 将logits转换回概率空间
    logits_clipped = torch.clamp(logits, min=-10.0, max=10.0)
    y_pred = torch.sigmoid(logits_clipped).cpu().numpy()
    y_true = split.edge_label.cpu().numpy()
    
    # 打印一些调试信息
    if return_preds:
        print(f"🔍 调试信息:")
        print(f"   原始logits范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"   裁剪logits范围: [{logits_clipped.min().item():.3f}, {logits_clipped.max().item():.3f}]")
        print(f"   预测范围: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
        print(f"   真实范围: [{y_true.min():.3f}, {y_true.max():.3f}]")
        print(f"   预测均值: {y_pred.mean():.3f}, 真实均值: {y_true.mean():.3f}")
        
        # 显示目标logits范围
        y_clipped = np.clip(y_true, 1e-7, 1-1e-7)
        target_logits = np.log(y_clipped / (1 - y_clipped))
        print(f"   目标logits范围: [{target_logits.min():.3f}, {target_logits.max():.3f}]")

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    nz = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else float("inf")

    metrics = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

    # --- Degree-Controlled Evaluation ---
    # Compute degrees from the message-passing graph (G)
    # Use in-degree + out-degree (undirected logic for GNN)
    node_degrees = degree(G.edge_index[0], G.num_nodes)
    
    edges = split.edge_label_index # [2, K]
    u_deg = node_degrees[edges[0]].cpu().numpy()
    v_deg = node_degrees[edges[1]].cpu().numpy()
    edge_min_deg = np.minimum(u_deg, v_deg)

    buckets = {
        "Tail (deg<=5)": edge_min_deg <= 5,
        "Medium (5<deg<=20)": (edge_min_deg > 5) & (edge_min_deg <= 20),
        "Head (deg>20)": edge_min_deg > 20
    }

    for name, mask in buckets.items():
        if mask.sum() > 0:
            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            sub_mse = mean_squared_error(sub_true, sub_pred)
            sub_mae = mean_absolute_error(sub_true, sub_pred)
            sub_r2 = r2_score(sub_true, sub_pred)
            
            metrics[f"mse_{name}"] = float(sub_mse)
            metrics[f"mae_{name}"] = float(sub_mae)
            metrics[f"r2_{name}"] = float(sub_r2)
            
            if return_preds:
                print(f"   [{name}] N={mask.sum()} | MSE: {sub_mse:.6f} | R2: {sub_r2:.4f}")
    # ------------------------------------

    if not return_preds:
        return metrics

    edges_list = split.edge_label_index.t().cpu().numpy().tolist()
    records = [{"input": {"edge": [int(u), int(v)]}, "prediction": float(p), "ground_truth": float(t)}
               for (u, v), p, t in zip(edges_list, y_pred.tolist(), y_true.tolist())]
    return metrics, records


# =========================
# Main
# =========================
def build_argparser():
    ap = argparse.ArgumentParser("Pos-only edge-attribute regression from metadata (simple)")
    ap.add_argument("--split_dir", type=str, default="scripts/output/artifact_graph_splits")
    ap.add_argument("--graph_data_dir", type=str, default="scripts/output/artifact_graph_data",
                    help="Directory containing edge_metadata.json")
    ap.add_argument("--metric_key", type=str, default=None, help="Metric name to use; auto-pick if omitted")

    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.005)  # 增加学习率

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_out", type=str, default="scripts/output/final_results/gnn_attribute_prediction_model.pth")
    ap.add_argument("--pred_json", type=str, default="scripts/output/final_results/gnn_attribute_predictions.json")
    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    # 预加载一次 Embedding (用于获取维度)
    # 或者更简单的：先读取 node_metadata 获取节点数，直接生成随机 embedding
    # 为了兼容性，我们还是先读原始 embedding 的维度，然后生成一致的随机噪声
    
    # 临时读取一个 split 的 meta 来获取节点数和 embedding 维度
    temp_p = Path(f"{args.split_dir}/train_split")
    temp_meta = json.loads((temp_p / "node_metadata.json").read_text(encoding="utf-8"))
    num_nodes = len(temp_meta)
    
    # 原始逻辑是读文件，现在我们做 ablation，生成一致的随机 Embedding
    # 假设维度与 BERT embedding 一致 (768) 或者自定义
    emb_dim = 768 
    
    # ⚠️ 关键修正：只生成一次随机矩阵，保证 Train/Val/Test 使用相同的随机特征
    print(f"[Ablation] Using consistent RANDOM embeddings (dim={emb_dim})")
    global_random_x = torch.randn(num_nodes, emb_dim)

    # 加载三份 split，传入固定的 x
    G_tr, S_tr = load_split(f"{args.split_dir}/train_split", args.graph_data_dir, args.metric_key, forced_x=global_random_x)
    G_va, S_va = load_split(f"{args.split_dir}/val_split",   args.graph_data_dir, args.metric_key, forced_x=global_random_x)
    G_te, S_te = load_split(f"{args.split_dir}/test_split",  args.graph_data_dir, args.metric_key, forced_x=global_random_x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for G in (G_tr, G_va, G_te):
        G.x = G.x.to(device)
        G.edge_index = G.edge_index.to(device)
    for S in (S_tr, S_va, S_te):
        S.edge_label_index = S.edge_label_index.to(device)
        S.edge_label = S.edge_label.to(device)

    model = GNNLinkPredictor(
        in_channels=G_tr.x.size(1),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    
    # 改进权重初始化 - 特别是最后的边预测器
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    print(f"[model] 参数总数: {sum(p.numel() for p in model.parameters())}")
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_mse, best_state = None, None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, G_tr, S_tr, opt)
        
        # 更频繁的评估，特别是前几个epoch
        eval_freq = 10 if epoch <= 50 else 25
        if epoch % eval_freq == 0 or epoch == args.epochs:
            # CORRECT: Use training graph (G_tr) for message passing during validation
            val = evaluate(model, G_tr, S_va)
            is_best = best_mse is None or val["mse"] < best_mse
            if is_best:
                best_mse, best_state = val["mse"], {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"epoch {epoch:04d} | loss {loss:.6f} | val_mse {val['mse']:.6f} | val_r2 {val['r2']:.4f} ⭐ 新最佳!")
            else:
                bucket_info = ""
                if "r2_Tail (deg<=5)" in val:
                    bucket_info = f" | Tail_R2 {val['r2_Tail (deg<=5)']:.4f} | Head_R2 {val['r2_Head (deg>20)']:.4f}"
                print(f"epoch {epoch:04d} | loss {loss:.6f} | val_mse {val['mse']:.6f} | val_r2 {val['r2']:.4f}{bucket_info}")
                
            # 早期调试：打印预测统计
            if epoch <= 30:
                with torch.no_grad():
                    # CORRECT: Use training graph (G_tr) for message passing
                    z = model.encode(G_tr.x, G_tr.edge_index)
                    raw_logits = model.decode(z, S_va.edge_label_index).squeeze(-1)
                    logits_clamped = torch.clamp(raw_logits, min=-10.0, max=10.0)
                    pred_sample = torch.sigmoid(logits_clamped)[:5]
                    true_sample = S_va.edge_label[:5]
                    print(f"      样本预测: {pred_sample.cpu().numpy()}")
                    print(f"      样本真实: {true_sample.cpu().numpy()}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"restored best (val_mse={best_mse:.6f})")
        
    # 测试并保存 JSON 预测
    # CORRECT: Use training graph (G_tr) for message passing, and test split (S_te) for evaluation
    test_metrics, test_records = evaluate(model, G_tr, S_te, return_preds=True)
    print(f"test_mse {test_metrics['mse']:.6f} | test_mae {test_metrics['mae']:.6f} | "
          f"test_rmse {test_metrics['rmse']:.6f} | test_r2 {test_metrics['r2']:.4f}")
    
    print("\n--- Degree-Controlled Performance ---")
    for k, v in test_metrics.items():
        if "Tail" in k or "Medium" in k or "Head" in k:
             print(f"  {k}: {v:.4f}")
    print("-------------------------------------")

    outp = Path(args.pred_json); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump({"split": "test", "num_records": len(test_records), "records": test_records}, f, indent=2, ensure_ascii=False)
    print(f"saved predictions -> {outp}")
    
    mp = Path(args.model_out); mp.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model with its configuration
    model_config = {
        "in_channels": G_tr.x.size(1),
        "hidden_channels": args.hidden,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict()
    }, mp)
    print(f"saved model -> {mp}")


if __name__ == "__main__":
    main()
