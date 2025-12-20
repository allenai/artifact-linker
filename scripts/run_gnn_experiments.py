#!/usr/bin/env python3
"""
运行完整的GNN实验

这个脚本将运行所有四个GNN任务：
1. 链接预测 (Link Prediction)
2. 链接排名 (Link Ranking)
3. 属性预测 (Attribute Prediction)
4. 属性排名 (Attribute Ranking)

并生成综合的实验报告。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.models import GNN_AVAILABLE


def run_command(cmd: List[str], description: str) -> Dict[str, Any]:
    """
    运行命令并返回结果
    """
    print(f"\n{'='*50}")
    print(f"正在运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        return {
            "success": True,
            "description": description,
            "command": " ".join(cmd),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": end_time - start_time,
            "return_code": result.returncode,
        }
    except subprocess.CalledProcessError as e:
        end_time = time.time()

        return {
            "success": False,
            "description": description,
            "command": " ".join(cmd),
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else "",
            "duration": end_time - start_time,
            "return_code": e.returncode,
            "error": str(e),
        }


def load_experiment_results(file_path: str) -> Dict[str, Any]:
    """
    加载实验结果文件
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"警告：无法加载结果文件 {file_path}: {e}")
        return {}


def generate_summary_report(results: Dict[str, Any], output_dir: Path) -> str:
    """
    生成综合实验报告
    """
    report_lines = [
        "# GNN模型实验综合报告",
        "",
        f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验概览",
        "",
    ]

    # 添加每个任务的结果摘要
    tasks = [
        ("链接预测", "gnn_link_predictions.json", ["accuracy"]),
        ("链接排名", "gnn_link_rankings.json", ["ndcg@10", "ndcg@20", "map@10", "map@20"]),
        ("属性预测", "gnn_attribute_predictions.json", ["mse", "mae", "mean_relative_error"]),
        (
            "属性排名",
            "gnn_attribute_rankings.json",
            ["ndcg@10", "ndcg@20", "map@10", "map@20", "spearman_correlation"],
        ),
    ]

    successful_tasks = 0
    total_tasks = len(tasks)

    for task_name, result_file, metrics in tasks:
        report_lines.append(f"### {task_name}")

        result_path = output_dir / result_file
        if result_path.exists():
            try:
                task_results = load_experiment_results(str(result_path))

                if task_results:
                    successful_tasks += 1
                    report_lines.append("✅ **状态**: 成功")

                    # 添加关键指标
                    if task_name == "链接预测":
                        accuracy = task_results.get("accuracy", 0.0)
                        total_pred = task_results.get("total_predictions", 0)
                        report_lines.extend([f"- 总预测数量: {total_pred}", f"- 准确率: {accuracy:.4f}"])

                    elif task_name == "链接排名":
                        avg_metrics = task_results.get("average_metrics", {})
                        total_tasks_count = task_results.get("total_ranking_tasks", 0)
                        report_lines.extend(
                            [
                                f"- 排名任务数量: {total_tasks_count}",
                                f"- 平均 NDCG@10: {avg_metrics.get('ndcg@10', 0.0):.4f}",
                                f"- 平均 MAP@10: {avg_metrics.get('map@10', 0.0):.4f}",
                            ]
                        )

                    elif task_name == "属性预测":
                        mse = task_results.get("mse", float("inf"))
                        mae = task_results.get("mae", float("inf"))
                        total_pred = task_results.get("total_predictions", 0)
                        metric_name = task_results.get("metric_name", "未知")
                        report_lines.extend(
                            [
                                f"- 预测指标: {metric_name}",
                                f"- 总预测数量: {total_pred}",
                                f"- 均方误差 (MSE): {mse:.6f}",
                                f"- 平均绝对误差 (MAE): {mae:.6f}",
                            ]
                        )

                    elif task_name == "属性排名":
                        avg_metrics = task_results.get("average_metrics", {})
                        total_tasks_count = task_results.get("total_ranking_tasks", 0)
                        metric_name = task_results.get("metric_name", "未知")
                        report_lines.extend(
                            [
                                f"- 排名指标: {metric_name}",
                                f"- 排名任务数量: {total_tasks_count}",
                                f"- 平均 NDCG@10: {avg_metrics.get('ndcg@10', 0.0):.4f}",
                                f"- 平均 Spearman 相关性: {avg_metrics.get('spearman_correlation', 0.0):.4f}",
                            ]
                        )
                else:
                    report_lines.append("❌ **状态**: 结果文件为空")

            except Exception as e:
                report_lines.append(f"❌ **状态**: 加载结果失败 - {str(e)}")
        else:
            report_lines.append("❌ **状态**: 结果文件不存在")

        report_lines.append("")

    # 添加总体摘要
    report_lines.extend(
        [
            "## 总体摘要",
            "",
            f"- 成功完成的任务: {successful_tasks}/{total_tasks}",
            f"- 成功率: {successful_tasks/total_tasks*100:.1f}%",
            "",
        ]
    )

    # 添加执行日志摘要
    if "execution_log" in results:
        report_lines.extend(["## 执行日志摘要", ""])

        for log_entry in results["execution_log"]:
            status = "✅" if log_entry["success"] else "❌"
            duration = log_entry["duration"]
            description = log_entry["description"]
            report_lines.append(f"{status} {description} (耗时: {duration:.1f}s)")

        report_lines.append("")

    # 保存报告
    report_content = "\n".join(report_lines)
    report_path = output_dir / "gnn_experiment_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="运行完整的GNN实验")
    parser.add_argument(
        "--data_dir", type=str, default="scripts/output/artifact_graph_data", help="图数据目录"
    )
    parser.add_argument("--model_path", type=str, help="预训练GNN模型路径（可选）")
    parser.add_argument("--output_dir", type=str, default="scripts/output", help="输出目录")
    parser.add_argument("--metric_name", type=str, default="accuracy", help="属性预测/排名的指标名称")
    parser.add_argument("--max_predictions", type=int, default=100, help="最大预测数量")
    parser.add_argument("--max_datasets", type=int, default=20, help="最大测试数据集数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练，直接运行预测/排名")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["link_prediction", "link_ranking", "attribute_prediction", "attribute_ranking"],
        default=["link_prediction", "link_ranking", "attribute_prediction", "attribute_ranking"],
        help="要运行的任务",
    )

    args = parser.parse_args()

    if not GNN_AVAILABLE:
        print("错误：GNN模型不可用。请确保安装了PyTorch Geometric及其依赖项。")
        print("\n安装命令示例:")
        print("pip install torch torch-geometric")
        return

    print("开始GNN实验...")
    print(f"输出目录: {args.output_dir}")
    print(f"选择的任务: {', '.join(args.tasks)}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 实验脚本路径
    script_dir = Path(__file__).parent

    # 构建基础参数
    base_args = ["--data_dir", args.data_dir, "--seed", str(args.seed)]

    if args.model_path:
        base_args.extend(["--model_path", args.model_path])

    # 定义实验命令
    experiments = []

    # 1. 训练统一模型（如果需要）
    if not args.skip_training:
        train_cmd = (
            [sys.executable, str(script_dir / "train_gnn_unified.py")]
            + base_args
            + ["--output_dir", str(output_dir / "gnn_models"), "--tasks"]
            + args.tasks
            + ["--metric_name", args.metric_name]
        )
        experiments.append((train_cmd, "训练统一GNN模型"))

    # 2. 链接预测
    if "link_prediction" in args.tasks:
        link_pred_cmd = (
            [sys.executable, str(script_dir / "predict_link_gnn.py")]
            + base_args
            + [
                "--output_file",
                str(output_dir / "gnn_link_predictions.json"),
                "--max_predictions",
                str(args.max_predictions),
            ]
        )
        experiments.append((link_pred_cmd, "GNN链接预测"))

    # 3. 链接排名
    if "link_ranking" in args.tasks:
        link_rank_cmd = (
            [sys.executable, str(script_dir / "rank_link_gnn.py")]
            + base_args
            + [
                "--output_file",
                str(output_dir / "gnn_link_rankings.json"),
                "--max_datasets",
                str(args.max_datasets),
            ]
        )
        experiments.append((link_rank_cmd, "GNN链接排名"))

    # 4. 属性预测
    if "attribute_prediction" in args.tasks:
        attr_pred_cmd = (
            [sys.executable, str(script_dir / "predict_attribute_gnn.py")]
            + base_args
            + [
                "--output_file",
                str(output_dir / "gnn_attribute_predictions.json"),
                "--metric_name",
                args.metric_name,
                "--max_predictions",
                str(args.max_predictions),
            ]
        )
        experiments.append((attr_pred_cmd, f"GNN属性预测 ({args.metric_name})"))

    # 5. 属性排名
    if "attribute_ranking" in args.tasks:
        attr_rank_cmd = (
            [sys.executable, str(script_dir / "rank_attribute_gnn.py")]
            + base_args
            + [
                "--output_file",
                str(output_dir / "gnn_attribute_rankings.json"),
                "--metric_name",
                args.metric_name,
                "--max_datasets",
                str(args.max_datasets),
            ]
        )
        experiments.append((attr_rank_cmd, f"GNN属性排名 ({args.metric_name})"))

    # 执行实验
    results = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "execution_log": [],
    }

    total_experiments = len(experiments)
    successful_experiments = 0

    for i, (cmd, description) in enumerate(experiments, 1):
        print(f"\n进度: {i}/{total_experiments}")

        # 运行实验
        result = run_command(cmd, description)
        results["execution_log"].append(result)

        if result["success"]:
            successful_experiments += 1
            print(f"✅ {description} 完成")
        else:
            print(f"❌ {description} 失败")
            print(f"错误信息: {result.get('error', '未知错误')}")
            if result.get("stderr"):
                print(f"标准错误: {result['stderr']}")

    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["total_experiments"] = total_experiments
    results["successful_experiments"] = successful_experiments
    results["success_rate"] = (
        successful_experiments / total_experiments if total_experiments > 0 else 0.0
    )

    # 保存执行日志
    log_file = output_dir / "gnn_experiment_log.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 生成综合报告
    report_path = generate_summary_report(results, output_dir)

    print(f"\n{'='*60}")
    print("实验完成!")
    print(f"{'='*60}")
    print(f"成功完成的实验: {successful_experiments}/{total_experiments}")
    print(f"成功率: {results['success_rate']*100:.1f}%")
    print(f"执行日志: {log_file}")
    print(f"综合报告: {report_path}")
    print(f"输出目录: {output_dir}")

    if successful_experiments == total_experiments:
        print("🎉 所有实验都成功完成!")
    elif successful_experiments > 0:
        print("⚠️  部分实验成功完成，请检查失败的实验。")
    else:
        print("❌ 所有实验都失败了，请检查配置和依赖项。")


if __name__ == "__main__":
    main()
