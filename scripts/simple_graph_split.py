#!/usr/bin/env python3
"""
超简单的图分割脚本（改进版）

- 统一用 .npz 保存评估边对：edge_label_index + edge_label
- 可选导出小样本 JSON 便于肉眼检查
- 在 split_info.json 里记录每个 split 的 pos/neg 统计
"""

import sys
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.utils.graph_builder import load_pyg_graph
from torch_geometric.transforms import RandomLinkSplit


def _to_numpy_int(arr: torch.Tensor, dtype=np.int32):
    return arr.detach().cpu().numpy().astype(dtype)


def _save_edge_labels_npz(data, split_dir: Path, sample_json: bool = True, sample_k: int = 100):
    """
    保存 edge_label_index (2, M) 和 edge_label (M,) 为 .npz
    并（可选）导出前 sample_k 条的 JSON 便于检查
    返回 (num_pairs, num_pos, num_neg)
    """
    if not hasattr(data, "edge_label_index") or not hasattr(data, "edge_label"):
        # 某些设置下 train 也可能没有负样本或标签
        return 0, 0, 0

    lbl_idx = _to_numpy_int(data.edge_label_index, np.int32)  # shape: (2, M)
    labels = _to_numpy_int(data.edge_label, np.int8)          # shape: (M,)

    np.savez(split_dir / "edge_labels.npz",
             edge_label_index=lbl_idx,
             edge_label=labels)

    # 可选：另存正/负各自的 npz（便于快速读取）
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    pos_edges = lbl_idx[:, pos_mask]
    neg_edges = lbl_idx[:, neg_mask]
    np.savez(split_dir / "pos_edges.npz", edges=pos_edges)
    np.savez(split_dir / "neg_edges.npz", edges=neg_edges)

    # 可选：导出小样本 JSON（仅用于人眼查看）
    if sample_json:
        m = labels.shape[0]
        k = min(sample_k, m)
        sample = [[int(lbl_idx[0, i]), int(lbl_idx[1, i]), int(labels[i])] for i in range(k)]
        with open(split_dir / "edge_labels_sample.json", "w") as f:
            json.dump(sample, f, indent=2)

    num_pairs = int(labels.shape[0])
    num_pos = int(labels.sum())
    num_neg = int(num_pairs - num_pos)
    return num_pairs, num_pos, num_neg


def save_pyg_split(data, output_dir: str, split_name: str, original_data_dir: str):
    """
    保存 PyG 分割结果到标准格式
    - edges.npz: 当前阶段可见图的边 (edge_index)
    - edge_labels.npz: 评估用候选边对 (edge_label_index + edge_label)
    - pos_edges.npz / neg_edges.npz: 便捷拆分
    - edge_labels_sample.json: 可选小样本便于查看
    """
    split_dir = Path(output_dir) / f"{split_name}_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    # 保存可见图结构（用于消息传递）
    edges = _to_numpy_int(data.edge_index, np.int32)  # shape: (2, E_visible)
    np.savez(split_dir / "edges.npz", edges=edges)

    # 复制原始元数据文件
    source_dir = Path(original_data_dir)
    for filename in ["node_metadata.json", "edge_metadata.json",
                     "node_embeddings.npz", "node_mappings.json"]:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, split_dir / filename)

    # 保存评估边对（统一 .npz）
    num_pairs, num_pos, num_neg = _save_edge_labels_npz(data, split_dir, sample_json=True, sample_k=100)

    print(f"✅ {split_name}: 可见边 {data.edge_index.size(1)} | 评估边对 {num_pairs} (pos={num_pos}, neg={num_neg})")
    return {
        "visible_edges": int(data.edge_index.size(1)),
        "num_label_pairs": int(num_pairs),
        "num_pos": int(num_pos),
        "num_neg": int(num_neg),
    }


def main():
    parser = argparse.ArgumentParser(description="超简单PyG图分割（改进版）")
    parser.add_argument("--input_dir", type=str, default="./output/artifact_graph_data_v2_1125",
                        help="输入图数据目录")
    parser.add_argument("--output_dir", type=str, default="./output/artifact_graph_splits_5_neg_ratio_v2_1125",
                        help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--undirected", action="store_true", help="图是否按无向处理（影响 split 和可见边）")
    parser.add_argument("--neg_ratio", type=float, default=5.0, help="负采样比例（每个正样本对应多少负样本）")
    parser.add_argument("--no_json_sample", action="store_true", help="不导出 JSON 小样本")

    args = parser.parse_args()

    print("🚀 超简单PyG图分割（改进版）")
    print("=" * 40)

    try:
        # 1. 加载 PyG 图
        print("1. 加载PyG图...")
        data, _ = load_pyg_graph(args.input_dir)
        print(f"✅ 图: {data.num_nodes} 节点, {data.edge_index.size(1)} 边")

        # 2. PyG 分割
        print("2. PyG分割...")
        transform = RandomLinkSplit(
            num_val=args.val_ratio,
            num_test=args.test_ratio,
            is_undirected=True,
            neg_sampling_ratio=args.neg_ratio,
            add_negative_train_samples=True,  # 保持默认训练有负样本
        )
        # breakpoint()  # 如需调试可开启
        train_data, val_data, test_data = transform(data)
        print("✅ 分割完成")
        print(f"   训练可见边: {train_data.edge_index.size(1)}")
        print(f"   验证可见边: {val_data.edge_index.size(1)}")
        print(f"   测试可见边: {test_data.edge_index.size(1)}")

        # 3. 保存
        print("3. 保存...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_train = save_pyg_split(train_data, args.output_dir, "train", args.input_dir)
        # 覆盖 JSON 小样本导出开关
        if args.no_json_sample:
            (Path(args.output_dir) / "train_split" / "edge_labels_sample.json").unlink(missing_ok=True)

        stats_val = save_pyg_split(val_data, args.output_dir, "val", args.input_dir)
        if args.no_json_sample:
            (Path(args.output_dir) / "val_split" / "edge_labels_sample.json").unlink(missing_ok=True)

        stats_test = save_pyg_split(test_data, args.output_dir, "test", args.input_dir)
        if args.no_json_sample:
            (Path(args.output_dir) / "test_split" / "edge_labels_sample.json").unlink(missing_ok=True)

        # 4. 保存信息
        split_info = {
            "method": "RandomLinkSplit",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "is_undirected": True,
            "neg_sampling_ratio": args.neg_ratio,
            "train": stats_train,
            "val": stats_val,
            "test": stats_test,
        }

        with open(output_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"\n✅ 完成！输出: {args.output_dir}")

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
