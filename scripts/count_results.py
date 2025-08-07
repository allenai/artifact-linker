#!/usr/bin/env python3
"""
统计指定目录中metadata和results文件的数量
Usage: python count_results.py [directory_path] [file_type]
       file_type: results (default), dataset, model, all, directories
"""

import glob
import os
import sys


def count_directories(directory="simple_results_0801"):
    """统计目录中的子目录数量"""
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return 0

    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    print(f"📁 目录: {directory}")
    print(f"📊 子目录数量: {len(subdirs)}")

    if subdirs and len(subdirs) <= 20:
        print("📝 子目录列表:")
        for i, subdir in enumerate(sorted(subdirs), 1):
            print(f"  {i:2d}. {subdir}")
    elif len(subdirs) > 20:
        print("📝 子目录列表 (前10个):")
        for i, subdir in enumerate(sorted(subdirs)[:10], 1):
            print(f"  {i:2d}. {subdir}")
        print(f"  ... (省略 {len(subdirs)-10} 个目录)")

    return len(subdirs)


def count_metadata_files(directory="simple_results_0801", file_type="results"):
    """统计指定类型的metadata文件数量"""

    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return {}

    print(f"📁 目录: {directory}")
    print("=" * 60)

    # 定义文件类型和对应的搜索模式
    file_patterns = {
        "results": "**/results.json",
        "dataset": "**/dataset_metadata.json",
        "model": "**/model_metadata.json",
    }

    # 如果文件类型是directories，统计目录数量
    if file_type == "directories" or file_type == "dirs":
        return {"directories": count_directories(directory)}

    results = {}

    if file_type == "all":
        # 统计所有类型的文件
        for ftype, pattern in file_patterns.items():
            full_pattern = os.path.join(directory, pattern)
            files = glob.glob(full_pattern, recursive=True)
            results[ftype] = files

            print(
                f"📊 {ftype}_metadata.json 文件数量: {len(files)}"
                if ftype != "results"
                else f"📊 {ftype}.json 文件数量: {len(files)}"
            )

            if files and len(files) <= 20:  # 只显示前20个文件
                for i, file_path in enumerate(sorted(files), 1):
                    relative_path = os.path.relpath(file_path, directory)
                    print(f"  {i:2d}. {relative_path}")
            elif len(files) > 20:
                for i, file_path in enumerate(sorted(files)[:10], 1):
                    relative_path = os.path.relpath(file_path, directory)
                    print(f"  {i:2d}. {relative_path}")
                print(f"  ... (省略 {len(files)-10} 个文件)")
            print()

    else:
        # 统计指定类型的文件
        if file_type not in file_patterns:
            print(f"❌ 不支持的文件类型: {file_type}")
            print(f"支持的类型: {', '.join(file_patterns.keys())}, all, directories")
            return {}

        pattern = file_patterns[file_type]
        full_pattern = os.path.join(directory, pattern)
        files = glob.glob(full_pattern, recursive=True)
        results[file_type] = files

        file_desc = f"{file_type}_metadata.json" if file_type != "results" else f"{file_type}.json"
        print(f"📊 {file_desc} 文件数量: {len(files)}")

        if files:
            print("📝 文件列表:")
            for i, file_path in enumerate(sorted(files), 1):
                relative_path = os.path.relpath(file_path, directory)
                print(f"  {i:2d}. {relative_path}")

    return results


if __name__ == "__main__":
    # 获取目录和文件类型参数
    directory = sys.argv[1] if len(sys.argv) > 1 else "simple_results_0801"
    file_type = sys.argv[2] if len(sys.argv) > 2 else "results"

    results = count_metadata_files(directory, file_type)

    # 显示总计
    if file_type == "all":
        print("🎯 统计总计:")
        total_files = 0
        for ftype, files in results.items():
            file_desc = f"{ftype}_metadata.json" if ftype != "results" else f"{ftype}.json"
            print(f"  📊 {file_desc}: {len(files)} 个")
            total_files += len(files)
        print(f"  📦 总计: {total_files} 个文件")
    else:
        if results:
            if file_type == "directories" or file_type == "dirs":
                dir_count = results["directories"]
                print(f"\n🎯 总计: {dir_count} 个子目录")
            else:
                file_count = len(results[file_type])
                file_desc = (
                    f"{file_type}_metadata.json" if file_type != "results" else f"{file_type}.json"
                )
                print(f"\n🎯 总计: {file_count} 个 {file_desc} 文件")
