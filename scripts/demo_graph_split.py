#!/usr/bin/env python3
"""
图分割演示脚本

展示如何正确地分割图数据以避免数据泄漏问题。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.utils.graph_splitter import GraphSplitter, extract_node_types


def main():
    print("🔬 图分割演示")
    print("=" * 50)

    # 加载原始图
    print("1. 加载原始图数据...")
    try:
        data_dir = "scripts/output/artifact_graph_data"
        G, node_metadata, edge_metadata = load_nx_graph(data_dir)
        print(f"✅ 原始图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    except Exception as e:
        print(f"❌ 加载图数据失败: {e}")
        return

    # 分析图结构
    print("\n2. 分析图结构...")
    node_types = extract_node_types(G)
    type_counts = {}
    for node_type in node_types.values():
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    for node_type, count in type_counts.items():
        print(f"   {node_type}: {count} 个节点")
    
    # 演示边分割
    print("\n3. 演示边分割策略...")
    splitter = GraphSplitter(seed=42)

    try:
        split_result = splitter.edge_split(
            G, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15,
            preserve_connectivity=True
        )
        
        train_graph = split_result['train_graph']
        val_graph = split_result['val_graph']
        full_graph = split_result['full_graph']
        
        print(f"✅ 训练图: {train_graph.number_of_nodes()} 个节点, {train_graph.number_of_edges()} 条边")
        print(f"✅ 验证图: {val_graph.number_of_nodes()} 个节点, {val_graph.number_of_edges()} 条边")
        print(f"✅ 完整图: {full_graph.number_of_nodes()} 个节点, {full_graph.number_of_edges()} 条边")
        
        # 验证分割的正确性
        print("\n4. 验证分割正确性...")
        train_edges = set(split_result['train_edges'])
        val_edges = set(split_result['val_edges'])
        test_edges = set(split_result['test_edges'])
        
        print(f"   训练边: {len(train_edges)} 条")
        print(f"   验证边: {len(val_edges)} 条")  
        print(f"   测试边: {len(test_edges)} 条")
        print(f"   总边数: {len(train_edges) + len(val_edges) + len(test_edges)}")
        
        # 检查重叠
        train_val_overlap = len(train_edges & val_edges)
        train_test_overlap = len(train_edges & test_edges)
        val_test_overlap = len(val_edges & test_edges)
        
        print(f"   训练-验证重叠: {train_val_overlap} 条边")
        print(f"   训练-测试重叠: {train_test_overlap} 条边")
        print(f"   验证-测试重叠: {val_test_overlap} 条边")
        
        if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
            print("✅ 分割正确，没有数据泄漏")
        else:
            print("⚠️  存在数据泄漏！")
            
    except Exception as e:
        print(f"❌ 边分割失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 演示负样本生成
    print("\n5. 演示负样本生成...")
    try:
        train_pos_edges = split_result['train_edges']
        train_neg_edges = splitter.generate_negative_edges(
            train_graph, train_pos_edges, len(train_pos_edges), node_types
        )
        
        print(f"✅ 正样本边: {len(train_pos_edges)} 条")
        print(f"✅ 负样本边: {len(train_neg_edges)} 条")
        
        # 验证负样本的正确性
        train_pos_set = set(train_pos_edges) | set((v, u) for u, v in train_pos_edges)
        overlap_count = sum(1 for edge in train_neg_edges if edge in train_pos_set)
        
        if overlap_count == 0:
            print("✅ 负样本生成正确，与正样本无重叠")
        else:
            print(f"⚠️  负样本中有 {overlap_count} 条边与正样本重叠")
            
    except Exception as e:
        print(f"❌ 负样本生成失败: {e}")
    
    # 显示使用建议
    print("\n6. 使用建议")
    print("=" * 30)
    print("✅ 推荐做法:")
    print("   - 使用 edge_split 进行边分割")
    print("   - 设置 preserve_connectivity=True 保持连通性")
    print("   - 在训练图上训练，验证图上调参，测试图上最终评估")
    print("   - 生成负样本时使用相应的分割图")
    print()
    print("❌ 避免做法:")
    print("   - 在完整图上训练然后在子图上测试")
    print("   - 使用测试边的信息进行模型选择")
    print("   - 忽视图的连通性")
    print()
    print("🚀 训练命令示例:")
    print("   python scripts/train_gnn_unified.py --use_graph_split --split_type edge")


if __name__ == "__main__":
    main()
