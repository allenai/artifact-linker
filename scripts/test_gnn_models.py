#!/usr/bin/env python3
"""
测试GNN模型的基本功能

这个脚本用于验证所有GNN模型组件都能正常工作。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from artifact_graph.models import (
        GNN_AVAILABLE, GNNUnifiedModel, GNNLinkPredictor, 
        GNNAttributePredictor, GNNLinkRanker, GNNAttributeRanker
    )
    from artifact_graph.utils.graph_builder import load_nx_graph
    print("✅ 成功导入所有GNN模型组件")
except ImportError as e:
    print(f"❌ 导入GNN模型失败: {e}")
    print("请确保安装了PyTorch Geometric: pip install torch torch-geometric")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 开始基本功能测试...")
    
    if not GNN_AVAILABLE:
        print("❌ GNN模型不可用")
        return False
    
    try:
        # 测试统一模型创建
        print("测试统一模型创建...")
        model = GNNUnifiedModel(in_feats=4, hidden_feats=64, num_layers=2)
        print(f"✅ 统一模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试各个预测器创建
        print("测试各个预测器创建...")
        link_predictor = GNNLinkPredictor(hidden_feats=64)
        print("✅ 链接预测器创建成功")
        
        attr_predictor = GNNAttributePredictor(hidden_feats=64)
        print("✅ 属性预测器创建成功")
        
        link_ranker = GNNLinkRanker(hidden_feats=64)
        print("✅ 链接排名器创建成功")
        
        attr_ranker = GNNAttributeRanker(hidden_feats=64)
        print("✅ 属性排名器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_sample_data():
    """使用示例数据测试"""
    print("\n📊 开始示例数据测试...")
    
    try:
        # 检查数据目录是否存在
        data_dir = "scripts/output/artifact_graph_data"
        if not os.path.exists(data_dir):
            print(f"⚠️  数据目录不存在: {data_dir}")
            print("请先运行数据准备脚本")
            return False
        
        # 尝试加载图数据
        print("加载图数据...")
        G, node_metadata, edge_metadata = load_nx_graph(data_dir)
        print(f"✅ 成功加载图数据: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
        # 获取一些示例节点
        models = [node for node, data in G.nodes(data=True) if data.get("type") == "model"][:3]
        datasets = [node for node, data in G.nodes(data=True) if data.get("type") == "dataset"][:3]
        
        if not models or not datasets:
            print("⚠️  没有找到足够的模型或数据集节点")
            return False
        
        print(f"找到 {len(models)} 个模型节点, {len(datasets)} 个数据集节点")
        
        # 测试链接预测
        print("测试链接预测...")
        link_predictor = GNNLinkPredictor(hidden_feats=64)
        result = link_predictor.predict(models[0], datasets[0], G, node_metadata)
        if result:
            print(f"✅ 链接预测成功: {result['prediction']} (置信度: {result.get('confidence', 0):.3f})")
        else:
            print("⚠️  链接预测返回空结果")
        
        # 测试属性预测
        print("测试属性预测...")
        attr_predictor = GNNAttributePredictor(hidden_feats=64)
        result = attr_predictor.predict(models[0], datasets[0], G, node_metadata, edge_metadata, "accuracy")
        if result:
            print(f"✅ 属性预测成功: {result['prediction']:.3f}")
        else:
            print("⚠️  属性预测返回空结果")
        
        # 测试链接排名
        print("测试链接排名...")
        link_ranker = GNNLinkRanker(hidden_feats=64)
        result = link_ranker.rank(datasets[0], models[:2], models[2:], G, node_metadata)
        if result and result.get("ranked_model_ids"):
            print(f"✅ 链接排名成功: 排名了 {len(result['ranked_model_ids'])} 个模型")
        else:
            print("⚠️  链接排名返回空结果")
        
        # 测试属性排名（需要真实的指标值）
        print("测试属性排名...")
        models_to_rank = [(m, 0.5 + i * 0.1) for i, m in enumerate(models)]  # 模拟指标值
        attr_ranker = GNNAttributeRanker(hidden_feats=64)
        result = attr_ranker.rank(datasets[0], models_to_rank, G, node_metadata, edge_metadata, "accuracy")
        if result and result.get("ranked_models"):
            print(f"✅ 属性排名成功: 排名了 {len(result['ranked_models'])} 个模型")
        else:
            print("⚠️  属性排名返回空结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 示例数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripts_availability():
    """测试脚本文件是否可用"""
    print("\n📄 检查脚本文件...")
    
    script_dir = Path(__file__).parent
    required_scripts = [
        "train_gnn_unified.py",
        "predict_link_gnn.py", 
        "predict_attribute_gnn.py",
        "rank_link_gnn.py",
        "rank_attribute_gnn.py",
        "run_gnn_experiments.py"
    ]
    
    all_available = True
    for script in required_scripts:
        script_path = script_dir / script
        if script_path.exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - 文件不存在")
            all_available = False
    
    return all_available


def main():
    print("🚀 GNN模型测试开始...")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("基本功能", test_basic_functionality),
        ("脚本文件", test_scripts_availability),
        ("示例数据", test_with_sample_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("🎯 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！GNN模型系统可以正常使用。")
        print("\n📝 下一步:")
        print("1. 运行 python scripts/run_gnn_experiments.py 来执行完整实验")
        print("2. 或者单独运行各个任务的脚本")
        print("3. 查看 GNN_MODELS_README.md 了解详细使用方法")
    else:
        print("⚠️  部分测试失败，请检查依赖项和配置")
        if passed == 0:
            print("💡 建议:")
            print("1. 确保安装了 PyTorch Geometric: pip install torch torch-geometric")
            print("2. 检查图数据是否已准备好")
            print("3. 查看错误信息并修复相关问题")


if __name__ == "__main__":
    main()
