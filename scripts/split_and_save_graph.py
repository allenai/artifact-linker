#!/usr/bin/env python3
"""
图数据分割和保存脚本

将原始图数据分割为训练/验证/测试集，并保存为与artifact_graph_data相同的格式。
这样可以避免数据泄漏问题，并允许直接加载分割后的数据进行训练和测试。
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import networkx as nx

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_splitter import GraphSplitter, extract_node_types


def save_graph_split(graph: nx.Graph, 
                    node_metadata: Dict,
                    edge_metadata: Dict,
                    edges_to_include: List[Tuple],
                    output_dir: str,
                    split_name: str):
    """
    保存分割后的图数据到指定目录
    
    Args:
        graph: NetworkX图
        node_metadata: 节点元数据
        edge_metadata: 边元数据  
        edges_to_include: 要包含的边列表
        output_dir: 输出目录
        split_name: 分割名称 (train/val/test)
    """
    split_dir = Path(output_dir) / f"{split_name}_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"保存 {split_name} split 到 {split_dir}...")
    
    # 1. 保存节点元数据 (所有split共享相同的节点)
    node_metadata_path = split_dir / "node_metadata.json"
    with open(node_metadata_path, 'w') as f:
        # 转换键为字符串以便JSON序列化
        serializable_metadata = {str(k): v for k, v in node_metadata.items()}
        json.dump(serializable_metadata, f, indent=2)
    
    # 2. 过滤边元数据，只保留当前split的边
    edges_to_include_set = set(edges_to_include) | set((v, u) for u, v in edges_to_include)
    filtered_edge_metadata = {}
    
    for edge_key, metadata in edge_metadata.items():
        if edge_key in edges_to_include_set:
            # 转换边键为字符串格式
            if isinstance(edge_key, tuple):
                str_key = f"{edge_key[0]},{edge_key[1]}"
            else:
                str_key = str(edge_key)
            filtered_edge_metadata[str_key] = metadata
    
    # 保存边元数据
    edge_metadata_path = split_dir / "edge_metadata.json"
    with open(edge_metadata_path, 'w') as f:
        json.dump(filtered_edge_metadata, f, indent=2)
    
    # 3. 创建边数组并保存
    edges_array = np.array(edges_to_include, dtype=np.int32)
    edges_path = split_dir / "edges.npz"
    np.savez(edges_path, edges=edges_array)
    
    # 4. 复制节点嵌入 (所有split共享相同的节点嵌入)
    source_embeddings = Path(output_dir).parent / "artifact_graph_data" / "node_embeddings.npz"
    if source_embeddings.exists():
        target_embeddings = split_dir / "node_embeddings.npz"
        shutil.copy2(source_embeddings, target_embeddings)
    else:
        print(f"警告: 未找到节点嵌入文件 {source_embeddings}")
    
    # 5. 复制节点映射 (可选)
    source_mappings = Path(output_dir).parent / "artifact_graph_data" / "node_mappings.json"
    if source_mappings.exists():
        target_mappings = split_dir / "node_mappings.json"
        shutil.copy2(source_mappings, target_mappings)
    
    print(f"✅ {split_name} split 已保存:")
    print(f"   - 节点: {len(node_metadata)} 个")
    print(f"   - 边: {len(edges_to_include)} 条")
    print(f"   - 目录: {split_dir}")


def create_negative_samples_file(splitter: GraphSplitter,
                                graph: nx.Graph,
                                positive_edges: List[Tuple],
                                node_types: Dict,
                                output_dir: str,
                                split_name: str):
    """
    创建负样本文件
    """
    print(f"生成 {split_name} split 的负样本...")
    
    # 生成负样本
    negative_edges = splitter.generate_negative_edges(
        graph, positive_edges, len(positive_edges), node_types
    )
    
    # 保存负样本
    split_dir = Path(output_dir) / f"{split_name}_split"
    negative_edges_path = split_dir / "negative_edges.json"
    
    with open(negative_edges_path, 'w') as f:
        json.dump(negative_edges, f, indent=2)
    
    print(f"✅ {split_name} 负样本已保存: {len(negative_edges)} 条边")


def create_split_info_file(split_result: Dict[str, Any], output_dir: str):
    """
    创建分割信息文件
    """
    split_info = {
        "split_method": "edge_split",
        "train_edges": len(split_result['train_edges']),
        "val_edges": len(split_result['val_edges']),
        "test_edges": len(split_result['test_edges']),
        "total_edges": len(split_result['train_edges']) + len(split_result['val_edges']) + len(split_result['test_edges']),
        "train_ratio": len(split_result['train_edges']) / (len(split_result['train_edges']) + len(split_result['val_edges']) + len(split_result['test_edges'])),
        "val_ratio": len(split_result['val_edges']) / (len(split_result['train_edges']) + len(split_result['val_edges']) + len(split_result['test_edges'])),
        "test_ratio": len(split_result['test_edges']) / (len(split_result['train_edges']) + len(split_result['val_edges']) + len(split_result['test_edges'])),
        "preserve_connectivity": True,
        "created_by": "split_and_save_graph.py"
    }
    
    info_path = Path(output_dir) / "split_info.json"
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ 分割信息已保存到: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="分割图数据并保存为标准格式")
    parser.add_argument("--input_dir", type=str, default="scripts/output/artifact_graph_data",
                       help="输入图数据目录")
    parser.add_argument("--output_dir", type=str, default="scripts/output/graph_splits",
                       help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="测试集比例")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--preserve_connectivity", action="store_true", default=True,
                       help="保持图的连通性")
    parser.add_argument("--generate_negatives", action="store_true", default=True,
                       help="生成负样本")
    
    args = parser.parse_args()
    
    # 验证比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("错误: 训练、验证、测试比例之和必须为1.0")
        return
    
    print("🔬 图数据分割和保存")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"分割比例: 训练({args.train_ratio}) / 验证({args.val_ratio}) / 测试({args.test_ratio})")
    print(f"随机种子: {args.seed}")
    print(f"保持连通性: {args.preserve_connectivity}")
    
    try:
        # 1. 加载原始图数据
        print("\n1. 加载原始图数据...")
        G, node_metadata, edge_metadata = load_nx_graph(args.input_dir)
        print(f"✅ 加载完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
        # 2. 分析节点类型
        print("\n2. 分析节点类型...")
        node_types = extract_node_types(G)
        type_counts = {}
        for node_type in node_types.values():
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        for node_type, count in type_counts.items():
            print(f"   {node_type}: {count} 个节点")
        
        # 3. 执行图分割
        print("\n3. 执行图分割...")
        splitter = GraphSplitter(seed=args.seed)
        split_result = splitter.edge_split(
            G,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            preserve_connectivity=args.preserve_connectivity
        )
        
        train_edges = split_result['train_edges']
        val_edges = split_result['val_edges']
        test_edges = split_result['test_edges']
        
        print(f"✅ 分割完成:")
        print(f"   训练边: {len(train_edges)} 条")
        print(f"   验证边: {len(val_edges)} 条")
        print(f"   测试边: {len(test_edges)} 条")
        
        # 4. 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. 保存各个分割
        print("\n4. 保存分割数据...")
        
        # 保存训练集
        save_graph_split(G, node_metadata, edge_metadata, train_edges, 
                        args.output_dir, "train")
        
        # 保存验证集 (包含训练+验证边)
        val_edges_cumulative = train_edges + val_edges
        save_graph_split(G, node_metadata, edge_metadata, val_edges_cumulative,
                        args.output_dir, "val")
        
        # 保存测试集 (包含所有边，用于最终评估)
        test_edges_cumulative = train_edges + val_edges + test_edges
        save_graph_split(G, node_metadata, edge_metadata, test_edges_cumulative,
                        args.output_dir, "test")
        
        # 6. 生成负样本 (如果需要)
        if args.generate_negatives:
            print("\n5. 生成负样本...")
            
            train_graph = split_result['train_graph']
            val_graph = split_result['val_graph']
            full_graph = split_result['full_graph']
            
            # 为每个split生成负样本
            create_negative_samples_file(splitter, train_graph, train_edges, 
                                       node_types, args.output_dir, "train")
            create_negative_samples_file(splitter, val_graph, val_edges,
                                       node_types, args.output_dir, "val")
            create_negative_samples_file(splitter, full_graph, test_edges,
                                       node_types, args.output_dir, "test")
        
        # 7. 保存分割信息
        print("\n6. 保存分割信息...")
        create_split_info_file(split_result, args.output_dir)
        
        # 8. 创建加载示例脚本
        print("\n7. 创建加载示例...")
        create_loading_example(args.output_dir)
        
        print(f"\n✅ 图数据分割完成！")
        print(f"输出目录结构:")
        print(f"  {args.output_dir}/")
        print(f"  ├── train_split/")
        print(f"  │   ├── node_metadata.json")
        print(f"  │   ├── edge_metadata.json")
        print(f"  │   ├── edges.npz")
        print(f"  │   ├── node_embeddings.npz")
        print(f"  │   └── negative_edges.json")
        print(f"  ├── val_split/")
        print(f"  ├── test_split/")
        print(f"  ├── split_info.json")
        print(f"  └── load_example.py")
        print(f"\n🚀 使用方法:")
        print(f"  # 加载训练数据")
        print(f"  from artifact_graph.utils.graph_builder import load_nx_graph")
        print(f"  G_train, meta, edge_meta = load_nx_graph('{args.output_dir}/train_split')")
        print(f"  ")
        print(f"  # 加载验证数据")
        print(f"  G_val, _, _ = load_nx_graph('{args.output_dir}/val_split')")
        print(f"  ")
        print(f"  # 加载测试数据")
        print(f"  G_test, _, _ = load_nx_graph('{args.output_dir}/test_split')")
        
    except Exception as e:
        print(f"❌ 分割过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def create_loading_example(output_dir: str):
    """
    创建加载示例脚本
    """
    example_code = f'''#!/usr/bin/env python3
"""
图分割数据加载示例

演示如何加载分割后的图数据进行训练和测试
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from artifact_graph.utils.graph_builder import load_nx_graph


def load_split_data(split_dir: str):
    """加载指定split的数据"""
    print(f"加载 {{split_dir}} 的数据...")
    
    # 加载图数据
    G, node_metadata, edge_metadata = load_nx_graph(split_dir)
    
    # 加载负样本 (如果存在)
    negative_edges = None
    neg_path = Path(split_dir) / "negative_edges.json"
    if neg_path.exists():
        with open(neg_path, 'r') as f:
            negative_edges = json.load(f)
    
    print(f"  图: {{G.number_of_nodes()}} 个节点, {{G.number_of_edges()}} 条边")
    if negative_edges:
        print(f"  负样本: {{len(negative_edges)}} 条边")
    
    return G, node_metadata, edge_metadata, negative_edges


def main():
    # 定义分割目录
    base_dir = "{output_dir}"
    train_dir = f"{{base_dir}}/train_split"
    val_dir = f"{{base_dir}}/val_split"  
    test_dir = f"{{base_dir}}/test_split"
    
    print("📊 加载分割后的图数据")
    print("=" * 40)
    
    # 加载分割信息
    split_info_path = f"{{base_dir}}/split_info.json"
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)
    
    print("分割信息:")
    print(f"  训练边: {{split_info['train_edges']}} 条 ({{split_info['train_ratio']:.1%}})")
    print(f"  验证边: {{split_info['val_edges']}} 条 ({{split_info['val_ratio']:.1%}})")
    print(f"  测试边: {{split_info['test_edges']}} 条 ({{split_info['test_ratio']:.1%}})")
    print()
    
    # 加载各个split
    try:
        # 训练数据 - 用于模型训练
        G_train, train_node_meta, train_edge_meta, train_neg = load_split_data(train_dir)
        
        # 验证数据 - 用于超参数调优和模型选择
        G_val, val_node_meta, val_edge_meta, val_neg = load_split_data(val_dir)
        
        # 测试数据 - 用于最终性能评估
        G_test, test_node_meta, test_edge_meta, test_neg = load_split_data(test_dir)
        
        print("✅ 所有数据加载成功！")
        print()
        print("💡 使用建议:")
        print("  1. 在 G_train 上训练模型")
        print("  2. 在 G_val 上进行超参数调优")
        print("  3. 在 G_test 上进行最终评估")
        print("  4. 使用对应的负样本进行链接预测任务")
        
    except Exception as e:
        print(f"❌ 加载失败: {{e}}")


if __name__ == "__main__":
    main()
'''
    
    example_path = Path(output_dir) / "load_example.py"
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"✅ 加载示例已保存到: {example_path}")


if __name__ == "__main__":
    main()
