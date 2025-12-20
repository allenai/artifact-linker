#!/usr/bin/env python3
"""
GNN训练和推理的共享工具函数（分类/回归彻底拆分版）
避免 predict_link_gnn.py 和 rank_link_gnn.py 之间的代码重复
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================
# 公共：数据加载
# =========================
def load_split_data(split_dir):
    """加载分割后的数据"""
    import numpy as np
    split_path = Path(split_dir)

    # 1. 加载可见图结构
    edges_data = torch.from_numpy(np.load(split_path / "edges.npz")["edges"]).long()

    # 2. 加载节点特征
    with open(split_path / "node_metadata.json", "r") as f:
        node_metadata = json.load(f)

    # 3. 尝试加载真实的节点嵌入
    num_nodes = len(node_metadata)
    try:
        embedding_path = split_path.parent.parent / "artifact_graph_data" / "node_embeddings.npy"
        if embedding_path.exists():
            print(f"  加载节点嵌入: {embedding_path}")
            embeddings_raw = np.load(embedding_path)

            if embeddings_raw.dtype.names is not None and "embedding" in embeddings_raw.dtype.names:
                embeddings = embeddings_raw["embedding"]
                x = torch.from_numpy(embeddings).float()
                print(f"  ✅ 使用真实嵌入: {x.shape}")
            else:
                x = torch.from_numpy(embeddings_raw).float()
                print(f"  ✅ 使用真实嵌入: {x.shape}")
        else:
            raise FileNotFoundError("嵌入文件不存在")
    except Exception as e:
        print(f"  ❌ 加载嵌入失败: {e}")
        print("  🔄 使用随机特征作为备选")
        x = torch.randn(num_nodes, 128)

    # 4. 加载评估边对
    try:
        pos_edges = torch.from_numpy(np.load(split_path / "pos_edges.npz")["edges"]).long()
        neg_edges = torch.from_numpy(np.load(split_path / "neg_edges.npz")["edges"]).long()
        print(f"✅ 加载边对: pos={pos_edges.size(1)}, neg={neg_edges.size(1)}")
    except:
        # 备选：从edge_labels.npz加载
        edge_labels_data = np.load(split_path / "edge_labels.npz")
        edge_label_index = torch.from_numpy(edge_labels_data["edge_label_index"]).long()
        edge_labels = torch.from_numpy(edge_labels_data["edge_label"]).long()

        pos_mask = edge_labels == 1
        neg_mask = edge_labels == 0

        pos_edges = edge_label_index[:, pos_mask]
        neg_edges = edge_label_index[:, neg_mask]
        print(f"✅ 从edge_labels加载: pos={pos_edges.size(1)}, neg={neg_edges.size(1)}")

    # 5. 创建数据对象
    data = SimpleNamespace(
        x=x,
        edge_index=edges_data,
        num_nodes=num_nodes,
    )

    split_data = SimpleNamespace(
        edge_index=edges_data,
        pos_edge_label_index=pos_edges,
        neg_edge_label_index=neg_edges,
    )

    return data, split_data


# =========================
# 分类：训练与评估
# =========================
def train_one_epoch_classification(model, data, train_split, optimizer):
    """分类任务：训练一个epoch"""
    model.train()
    optimizer.zero_grad()

    # 编码
    z = model.encode(data.x, train_split.edge_index)

    # 解码正负边
    pos_pred = model.decode(z, train_split.pos_edge_label_index)
    neg_pred = model.decode(z, train_split.neg_edge_label_index)

    # 二元交叉熵损失
    pos_loss = -torch.log(torch.sigmoid(pos_pred) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_pred) + 1e-15).mean()
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model_classification(model, data, split_data):
    """分类任务：评估模型性能"""
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, split_data.edge_index)
        pos_pred = model.decode(z, split_data.pos_edge_label_index)
        neg_pred = model.decode(z, split_data.neg_edge_label_index)

        y_true = torch.cat(
            [torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0
        )
        y_pred_prob = torch.cat(
            [torch.sigmoid(pos_pred), torch.sigmoid(neg_pred)], dim=0
        )
        y_pred_binary = (y_pred_prob > 0.5).int()

        try:
            metrics = {
                "accuracy": float(accuracy_score(y_true.cpu(), y_pred_binary.cpu())),
                "precision": float(
                    precision_score(y_true.cpu(), y_pred_binary.cpu(), zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true.cpu(), y_pred_binary.cpu(), zero_division=0)
                ),
                "f1": float(
                    f1_score(y_true.cpu(), y_pred_binary.cpu(), zero_division=0)
                ),
                "auc": float(roc_auc_score(y_true.cpu(), y_pred_prob.cpu())),
                "average_precision": float(
                    average_precision_score(y_true.cpu(), y_pred_prob.cpu())
                ),
            }
            return metrics
        except Exception as e:
            print(f"分类评估错误: {e}")
            return {
                "accuracy": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.5,
                "average_precision": 0.5,
            }


# =========================
# 回归：训练与评估
# =========================
def train_one_epoch_regression(model, data, train_split, optimizer):
    """回归任务：训练一个epoch（当前以正负边目标1/0为占位标签）"""
    model.train()
    optimizer.zero_grad()

    # 编码
    z = model.encode(data.x, train_split.edge_index)

    # 解码正负边
    pos_pred = model.decode(z, train_split.pos_edge_label_index)
    neg_pred = model.decode(z, train_split.neg_edge_label_index)

    # 回归（占位）：正样本目标1，负样本目标0
    pos_labels = torch.ones(pos_pred.size(0), device=pos_pred.device)
    neg_labels = torch.zeros(neg_pred.size(0), device=neg_pred.device)

    pos_loss = F.mse_loss(pos_pred, pos_labels)
    neg_loss = F.mse_loss(neg_pred, neg_labels)
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model_regression(model, data, split_data):
    """回归任务：评估模型性能（当前用1/0占位标签；如有真实连续标签，请在此替换）"""
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, split_data.edge_index)
        pos_pred = model.decode(z, split_data.pos_edge_label_index)
        neg_pred = model.decode(z, split_data.neg_edge_label_index)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        y_true = torch.cat(
            [torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0
        )

        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np

            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()

            mse = float(mean_squared_error(y_true_np, y_pred_np))
            mae = float(mean_absolute_error(y_true_np, y_pred_np))
            rmse = float((mse) ** 0.5)

            try:
                r2 = float(r2_score(y_true_np, y_pred_np))
            except Exception:
                r2 = 0.0

            # MAPE
            try:
                non_zero_mask = (np.abs(y_true_np) > 1e-8)
                if np.sum(non_zero_mask) > 0:
                    mape = float(
                        np.mean(
                            np.abs(
                                (y_true_np[non_zero_mask] - y_pred_np[non_zero_mask])
                                / y_true_np[non_zero_mask]
                            )
                        )
                        * 100
                    )
                else:
                    mape = float("inf")
            except Exception:
                mape = float("inf")

            mean_abs_diff = float(np.mean(np.abs(y_pred_np - y_true_np)))

            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
                "mean_abs_diff": mean_abs_diff,
            }
        except Exception as e:
            print(f"回归评估错误: {e}")
            return {
                "mse": float("inf"),
                "mae": float("inf"),
                "rmse": float("inf"),
                "mape": float("inf"),
                "r2": 0.0,
                "mean_abs_diff": float("inf"),
            }


# =========================
# 公共：训练主流程的通用骨架
# =========================
def _prepare_data_and_device(args, train_data, val_data, test_data):
    """准备设备与（可能的）磁盘加载数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")

    if train_data and val_data and test_data:
        print("1. 使用预加载的数据...")
        train_data_full, train_split = train_data
        val_data_full, val_split = val_data
        test_data_full, test_split = test_data
    else:
        print("1. 加载分割数据...")
        train_data_full, train_split = load_split_data(f"{args.split_dir}/train_split")
        val_data_full, val_split = load_split_data(f"{args.split_dir}/val_split")
        test_data_full, test_split = load_split_data(f"{args.split_dir}/test_split")

    # 移到设备
    for data_obj in [train_data_full, val_data_full, test_data_full]:
        data_obj.x = data_obj.x.to(device)
        data_obj.edge_index = data_obj.edge_index.to(device)

    for split_obj in [train_split, val_split, test_split]:
        split_obj.edge_index = split_obj.edge_index.to(device)
        split_obj.pos_edge_label_index = split_obj.pos_edge_label_index.to(device)
        split_obj.neg_edge_label_index = split_obj.neg_edge_label_index.to(device)

    print(f"✅ 训练图: {train_data_full.num_nodes} 节点, {train_data_full.edge_index.size(1)} 边")
    return device, train_data_full, train_split, val_data_full, val_split, test_data_full, test_split


def _build_model(args, model_class, in_channels, device):
    print("2. 创建MinimalGNN模型...")
    model = model_class(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"✅ 模型参数: {sum(p.numel() for p in model.parameters())}")
    return model, optimizer


def _save_model(args, model, in_channels, extra_payload):
    output_path = Path(args.model_save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": in_channels,
                "hidden_channels": args.hidden,
                "num_layers": args.num_layers,
                "heads": args.heads,
                "dropout": args.dropout,
            },
            **extra_payload,
        },
        output_path,
    )
    print(f"💾 模型已保存: {output_path}")
    return output_path


# =========================
# 分类：完整训练流程
# =========================
def train_gnn_model_classification(args, model_class, train_data=None, val_data=None, test_data=None):
    """分类任务的统一训练入口"""
    print("🚀 GNN模型训练（分类）")
    print("=" * 50)

    # 1) 数据 & 设备
    (
        device,
        train_data_full,
        train_split,
        val_data_full,
        val_split,
        test_data_full,
        test_split,
    ) = _prepare_data_and_device(args, train_data, val_data, test_data)

    # 2) 模型
    model, optimizer = _build_model(args, model_class, train_data_full.x.size(1), device)

    # 3) 训练
    print("3. 开始训练...")
    best_val_metrics = None
    best_model_state = None

    for epoch in range(args.epochs):
        loss = train_one_epoch_classification(model, train_data_full, train_split, optimizer)

        if epoch % 10 == 0:
            val_metrics = evaluate_model_classification(model, val_data_full, val_split)
            is_best = best_val_metrics is None or val_metrics["auc"] > best_val_metrics["auc"]

            if is_best:
                best_val_metrics = val_metrics
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_metrics['auc']:.4f} ⭐ 新最佳!")
            else:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")

    # 4) 恢复最佳并测试
    if best_model_state is not None:
        print("4. 恢复最佳验证检查点...")
        best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state)
        print(f"   ✅ 已恢复最佳验证AUC: {best_val_metrics['auc']:.4f} 时的模型")

    test_metrics = evaluate_model_classification(model, test_data_full, test_split)
    print("\n✅ 最终结果:")
    print(f"   测试 AUC: {test_metrics['auc']:.4f}")
    print(f"   测试 F1: {test_metrics['f1']:.4f}")
    print(f"   测试 Accuracy: {test_metrics['accuracy']:.4f}")

    # 5) 保存
    return _save_model(
        args,
        model,
        in_channels=train_data_full.x.size(1),
        extra_payload={
            "task_type": "classification",
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
    )


# =========================
# 回归：完整训练流程
# =========================
def train_gnn_model_regression(args, model_class, train_data=None, val_data=None, test_data=None):
    """回归任务的统一训练入口"""
    print("🚀 GNN模型训练（回归）")
    print("=" * 50)

    # 1) 数据 & 设备
    (
        device,
        train_data_full,
        train_split,
        val_data_full,
        val_split,
        test_data_full,
        test_split,
    ) = _prepare_data_and_device(args, train_data, val_data, test_data)

    # 2) 模型
    model, optimizer = _build_model(args, model_class, train_data_full.x.size(1), device)

    # 3) 训练
    print("3. 开始训练...")
    best_val_metrics = None
    best_model_state = None

    for epoch in range(args.epochs):
        loss = train_one_epoch_regression(model, train_data_full, train_split, optimizer)

        if epoch % 10 == 0:
            val_metrics = evaluate_model_regression(model, val_data_full, val_split)
            is_best = best_val_metrics is None or val_metrics["mse"] < best_val_metrics["mse"]

            if is_best:
                best_val_metrics = val_metrics
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val MSE: {val_metrics['mse']:.4f} ⭐ 新最佳!")
            else:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val MSE: {val_metrics['mse']:.4f}")

    # 4) 恢复最佳并测试
    if best_model_state is not None:
        print("4. 恢复最佳验证检查点...")
        best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state)
        print(f"   ✅ 已恢复最佳验证MSE: {best_val_metrics['mse']:.4f} 时的模型")

    test_metrics = evaluate_model_regression(model, test_data_full, test_split)
    print("\n✅ 最终结果:")
    print(f"   测试 MSE: {test_metrics['mse']:.4f}")
    print(f"   测试 MAE: {test_metrics['mae']:.4f}")
    print(f"   测试 RMSE: {test_metrics['rmse']:.4f}")
    if test_metrics["mape"] != float("inf"):
        print(f"   测试 MAPE: {test_metrics['mape']:.2f}%")
    else:
        print("   测试 MAPE: Undefined (zero true values)")
    print(f"   测试 R²: {test_metrics['r2']:.4f}")
    print(f"   测试 Mean Absolute Difference: {test_metrics['mean_abs_diff']:.4f}")

    # 5) 保存
    return _save_model(
        args,
        model,
        in_channels=train_data_full.x.size(1),
        extra_payload={
            "task_type": "regression",
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
    )


# =========================
# 公共：CLI 参数
# =========================
def get_common_args(parser):
    """添加通用参数"""
    # 数据路径
    parser.add_argument(
        "--split_dir",
        type=str,
        default="scripts/output/artifact_graph_splits",
        help="分割数据目录",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--hidden", type=int, default=128, help="隐藏维度")
    parser.add_argument("--num_layers", type=int, default=4, help="GAT层数")
    parser.add_argument("--heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")

    # 输出
    parser.add_argument("--model_save_path", type=str, default="outputs/gnn.pt", help="模型保存路径")

    return parser
