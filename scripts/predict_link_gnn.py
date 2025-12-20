#!/usr/bin/env python3
import sys, json, argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.utils import degree

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.models.gnn_link_predictor import GNNLinkPredictor


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_data(split_dir, forced_x=None):
    p = Path(split_dir)
    edges = torch.from_numpy(np.load(p / "edges.npz")["edges"]).long()
    with open(p / "node_metadata.json", "r") as f:
        node_meta = json.load(f)
    num_nodes = len(node_meta)

    if forced_x is not None:
        x = forced_x
    else:
        emb_path = p.parent.parent / "artifact_graph_data" / "node_embeddings.npy"
        arr = np.load(emb_path, allow_pickle=False)
        if getattr(arr, "dtype", None) is not None and arr.dtype.names and "embedding" in arr.dtype.names:
            x = torch.from_numpy(arr["embedding"]).float()
        else:
            x = torch.from_numpy(arr).float()

    pos = torch.from_numpy(np.load(p / "pos_edges.npz")["edges"]).long()
    neg = torch.from_numpy(np.load(p / "neg_edges.npz")["edges"]).long()

    data = SimpleNamespace(x=x, edge_index=edges, num_nodes=num_nodes)
    split = SimpleNamespace(edge_index=edges, pos_edge_label_index=pos, neg_edge_label_index=neg)
    return data, split


def prepare(args, train_data=None, val_data=None, test_data=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ablation: Generate consistent random embeddings across splits
    # We need to know num_nodes and emb_dim first.
    # Let's peek at train_split metadata.
    temp_p = Path(f"{args.split_dir}/train_split")
    with open(temp_p / "node_metadata.json", "r") as f:
        temp_meta = json.load(f)
    num_nodes = len(temp_meta)
    emb_dim = 768 # Assuming standard embedding size, or could be args.hidden
    
    print(f"[Ablation] Using consistent RANDOM embeddings (dim={emb_dim})")
    global_random_x = torch.randn(num_nodes, emb_dim)

    if train_data and val_data and test_data:
        train_full, train_split = train_data
        val_full, val_split = val_data
        test_full, test_split = test_data
    else:
        train_full, train_split = load_split_data(f"{args.split_dir}/train_split", forced_x=global_random_x)
        val_full, val_split = load_split_data(f"{args.split_dir}/val_split", forced_x=global_random_x)
        test_full, test_split = load_split_data(f"{args.split_dir}/test_split", forced_x=global_random_x)

    for d in (train_full, val_full, test_full):
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)

    for s in (train_split, val_split, test_split):
        s.edge_index = s.edge_index.to(device)
        s.pos_edge_label_index = s.pos_edge_label_index.to(device)
        s.neg_edge_label_index = s.neg_edge_label_index.to(device)

    return device, train_full, train_split, val_full, val_split, test_full, test_split


def build_model(args, in_channels, device):
    model = GNNLinkPredictor(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.8, patience=args.lr_patience, min_lr=1e-6, verbose=False)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    return model, opt, sch, scaler


def bce_logits_loss(pos_logits, neg_logits, pos_weight=None):
    y = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits, y)
    w = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=w)


@torch.no_grad()
def evaluate(model, z, split, node_degrees=None, return_predictions=False):
    model.eval()
    pos = model.decode(z, split.pos_edge_label_index)
    neg = model.decode(z, split.neg_edge_label_index)
    y_true = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], 0).detach().cpu().numpy()
    y_prob = torch.sigmoid(torch.cat([pos, neg], 0)).detach().cpu().numpy()
    y_bin = (y_prob > 0.5).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_bin)),
        "precision": float(precision_score(y_true, y_bin, zero_division=0)),
        "recall": float(recall_score(y_true, y_bin, zero_division=0)),
        "f1": float(f1_score(y_true, y_bin, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }

    # --- Degree-controlled evaluation ---
    if node_degrees is not None:
        # Combine pos and neg edges to match y_true order
        all_edges = torch.cat([split.pos_edge_label_index, split.neg_edge_label_index], dim=1) # [2, N]
        
        # Calculate min degree for each edge (e.g., if (u,v) has deg(u)=2, deg(v)=100, it's still a "Tail" task involved)
        u_deg = node_degrees[all_edges[0]].cpu().numpy()
        v_deg = node_degrees[all_edges[1]].cpu().numpy()
        edge_min_deg = np.minimum(u_deg, v_deg)

        buckets = {
            "Tail (deg<=5)": edge_min_deg <= 5,
            "Medium (5<deg<=20)": (edge_min_deg > 5) & (edge_min_deg <= 20),
            "Head (deg>20)": edge_min_deg > 20
        }

        for name, mask in buckets.items():
            if mask.sum() > 0:
                sub_true = y_true[mask]
                sub_pred = y_bin[mask]
                # Only calc F1/AUC for valid subsets
                sub_f1 = f1_score(sub_true, sub_pred, zero_division=0)
                sub_acc = accuracy_score(sub_true, sub_pred)
                try:
                    sub_auc = roc_auc_score(sub_true, y_prob[mask])
                except ValueError:
                    sub_auc = 0.0 # Handle case with only one class present
                
                metrics[f"f1_{name}"] = float(sub_f1)
                metrics[f"acc_{name}"] = float(sub_acc)
                metrics[f"auc_{name}"] = float(sub_auc)
    # ------------------------------------

    if return_predictions:
        # 收集边信息用于保存结果
        edge_index = torch.cat([split.pos_edge_label_index, split.neg_edge_label_index], dim=1)
        edge_indices_np = edge_index.detach().cpu().numpy()

        # 为每条边创建详细记录
        edge_predictions = []
        for i in range(len(y_true)):
            edge_predictions.append({
                "v_id": int(edge_indices_np[0, i]),  # 源节点ID
                "u_id": int(edge_indices_np[1, i]),  # 目标节点ID
                "ground_truth": float(y_true[i]),   # 真实标签 (1 or 0)
                "prediction_prob": float(y_prob[i]), # 预测概率 [0,1]
                "prediction_binary": int(y_bin[i])   # 二元预测 (1 or 0)
            })

        predictions_data = {
            "edges": edge_predictions,
            "summary": {
                "total_edges": len(y_true),
                "positive_edges": int(sum(y_true)),
                "negative_edges": int(len(y_true) - sum(y_true)),
                "correct_predictions": int(sum(y_true == y_bin)),
                "accuracy": float(sum(y_true == y_bin) / len(y_true))
            }
        }
        return metrics, predictions_data

    return metrics


def train_epoch(model, data, split, opt, scaler, pos_weight=None):
    model.train()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        z = model.encode(data.x, split.edge_index)
        pos = model.decode(z, split.pos_edge_label_index)
        neg = model.decode(z, split.neg_edge_label_index)
        loss = bce_logits_loss(pos, neg, pos_weight)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return float(loss.detach().cpu())


def save_model(path, model, cfg):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cfg | {"model_state_dict": model.state_dict()}, p)
    return p


def train(args, model_cls):
    set_seed(args.seed)
    device, train_full, train_split, val_full, val_split, test_full, test_split = prepare(args)

    model, opt, sch, scaler = build_model(args, train_full.x.size(1), device)

    pos_n = train_split.pos_edge_label_index.size(1)
    neg_n = train_split.neg_edge_label_index.size(1)
    pos_weight = neg_n / max(1, pos_n)

    # Pre-calculate node degrees from training graph (the "known" structure)
    # Note: train_full.edge_index is usually undirected (contains u->v and v->u), 
    # so out-degree using index[0] is sufficient for total degree.
    node_degrees = degree(train_full.edge_index[0], train_full.num_nodes)

    best, best_state = None, None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_full, train_split, opt, scaler, pos_weight)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                z = model.encode(train_full.x, train_full.edge_index)
            val = evaluate(model, z, val_split, node_degrees=node_degrees)
            sch.step(val["auc"])
            improved = best is None or val["auc"] > best["auc"]
            if improved:
                best, best_state, wait = val, {k: v.cpu() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1

            # Format bucket metrics for printing
            bucket_info = ""
            tail_auc = val.get('auc_Tail (deg<=5)')
            head_auc = val.get('auc_Head (deg>20)')

            if tail_auc is not None:
                bucket_info += f" | Tail_AUC {tail_auc:.4f}"
            if head_auc is not None:
                bucket_info += f" | Head_AUC {head_auc:.4f}"

            print(f"epoch {epoch:04d} | loss {loss:.4f} | val_auc {val['auc']:.4f} | val_f1 {val['f1']:.4f}{bucket_info}")

            if wait >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"restored best checkpoint (val_auc={best['auc']:.4f})")

    # 获取测试结果和详细预测信息
    with torch.no_grad():
        z = model.encode(train_full.x, train_full.edge_index)
    test, test_predictions = evaluate(model, z, test_split, node_degrees=node_degrees, return_predictions=True)
    print(f"test_auc {test['auc']:.4f} | test_f1 {test['f1']:.4f} | test_acc {test['accuracy']:.4f}")

    # Print final bucket breakdown
    print("\n--- Degree-Controlled Performance ---")
    for k, v in test.items():
        if "Tail" in k or "Medium" in k or "Head" in k:
            print(f"  {k}: {v:.4f}")
    print("-------------------------------------")

    # 也获取验证集的详细预测信息用于分析
    val_final, val_predictions = evaluate(model, z, val_split, return_predictions=True)

    # 保存详细的测试结果到JSON文件
    results_path = args.pred_json
    detailed_results = {
        "test_metrics": test,
        "test_predictions": test_predictions,
        "val_metrics": val_final,
        "val_predictions": val_predictions,
        "model_config": {
            "in_channels": train_full.x.size(1),
            "hidden_channels": args.hidden,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
        },
        "training_args": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        }
    }

    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"💾 测试结果已保存到: {results_path}")

    payload = {
        "model_config": {
            "in_channels": train_full.x.size(1),
            "hidden_channels": args.hidden,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
        },
        "task_type": "classification",
        "best_val_metrics": best,
        "test_metrics": test,
    }
    
    # Save model with its configuration
    model_config = {
        "in_channels": train_full.x.size(1),
        "hidden_channels": args.hidden,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        **payload  # Include other payload info like metrics
    }, Path(args.model_save_path))
    
    print(f"saved model to: {args.model_save_path}")
    return args.model_save_path


def build_argparser():
    p = argparse.ArgumentParser(description="GNN Link Prediction (Classification)")
    p.add_argument("--split_dir", type=str, default="scripts/output/artifact_graph_splits")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_patience", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_save_path", type=str, default="scripts/output/final_results/gnn_link_prediction_model.pth")
    p.add_argument("--pred_json", type=str, default="scripts/output/final_results/gnn_link_predictions.json")
    return p


def main():
    args = build_argparser().parse_args()
    path = train(args, GNNLinkPredictor)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
