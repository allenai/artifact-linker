"""
重构后的DockerCoder - 使用分离的组件
"""

import os
from typing import Dict

from .dataset_checker import DatasetCheckGenerator
from .dependency_parser import DependencyParser
from .docker_manager import DockerManager
from .evaluator import EvaluationGenerator
from .json_fixer import JSONFixer
from .model_checker import ModelCheckGenerator
from .utils.llm import create_client


class DockerCoder:
    """重构后的Docker代码生成器，使用分离的组件"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        output_dir: str = "results",
        memory_limit: str = "8g",
        enable_gpu: bool = True,
    ):
        self.model = model
        self.output_dir = output_dir
        self.memory_limit = memory_limit
        self.enable_gpu = enable_gpu

        # 初始化LLM客户端
        self.client, self.actual_model = create_client(model)

        # 初始化Docker管理器
        self.docker_manager = DockerManager(memory_limit=memory_limit, enable_gpu=enable_gpu)

        # 初始化各组件
        self.dataset_generator = DatasetCheckGenerator(self)
        self.model_generator = ModelCheckGenerator(self)
        self.evaluation_generator = EvaluationGenerator(self)

    def evaluate(
        self,
        model_name: str,
        dataset_name: str,
        metric: str = "accuracy",
        max_runs: int = 3,
        max_fixes: int = 3,
    ) -> Dict:
        """运行评估流程"""

        print(f"\n{'='*80}")
        print(f" ML MODEL EVALUATION: {model_name} on {dataset_name}")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Metric: {metric}")
        print(f"Max Runs: {max_runs}")
        print(f"Max Fixes: {max_fixes}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Memory Limit: {self.memory_limit}")

        try:
            print("\n\nCONTAINER SETUP")

            # 创建Docker容器
            container = self.docker_manager.create_container(self.output_dir)
            if not container:
                print("❌ Failed to create Docker container")
                return {}

            if self.client is None:
                print("❌ No LLM client available")
                self.docker_manager.cleanup()
                return {}
            print("✅ LLM client ready")

            os.makedirs(self.output_dir, exist_ok=True)

            successful_runs = 0
            total_runs = 0

            # 获取模型README
            model_readme = self.model_generator.get_model_readme(model_name)

            for run in range(max_runs):
                total_runs += 1

                print(f"\n{'='*80}")
                print(f" RUN {run + 1}/{max_runs}")
                print(f"{'='*80}")

                if self._execute_single_run(
                    run + 1, max_fixes, model_name, dataset_name, metric, model_readme, container
                ):
                    successful_runs += 1
                    print(f"✅ Run {run + 1} completed successfully")
                else:
                    print(f"❌ Run {run + 1} failed")

            self.docker_manager.cleanup()

            success_rate = successful_runs / total_runs if total_runs > 0 else 0

            result = {
                "success": successful_runs > 0,
                "successful_runs": successful_runs,
                "total_runs": total_runs,
                "success_rate": success_rate,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metric": metric,
                "output_dir": self.output_dir,
            }

            print("\n\nFINAL RESULTS")
            print(f"Successful runs: {successful_runs}/{total_runs}")
            print(f"Success rate: {success_rate:.1%}")

            return result

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metric": metric,
            }

    def _execute_single_run(
        self,
        run: int,
        max_fixes: int,
        model_name: str,
        dataset_name: str,
        metric: str,
        model_readme: str,
        container,
    ) -> bool:
        """执行单次运行"""
        try:
            # 检查是否已有结果
            if self._check_results_exist(run):
                print(f"⚡ Results for run {run} already exist, skipping...")
                return True

            # 顺序执行：获取元信息后生成评估脚本
            return self._run_sequential_experiment(
                container, run, max_fixes, model_name, dataset_name, metric, model_readme
            )

        except Exception as e:
            print(f"❌ Run {run} failed with error: {e}")
            return False

    def _run_sequential_experiment(
        self,
        container,
        run_num: int,
        max_fixes: int,
        model_name: str,
        dataset_name: str,
        metric: str,
        model_readme: str,
    ) -> bool:
        """顺序执行实验：先获取元信息，再生成评估脚本"""

        # 阶段1: 生成并运行 dataset checker，获取数据集元信息
        print("\n\nPHASE 1: DATASET ANALYSIS")

        if not self.dataset_generator.generate_dataset_check(dataset_name):
            print("❌ Failed to generate dataset_check.py")
            return False

        # 解析并安装脚本依赖
        self._install_script_dependencies("dataset_check.py")

        # 运行 dataset checker 获取元信息
        dataset_metadata = self._run_and_get_metadata(
            container, "dataset_check.py", "dataset_analysis.json", max_fixes
        )
        if not dataset_metadata:
            print("❌ Failed to get dataset metadata")
            return False

        # 阶段2: 生成并运行 model checker，获取模型元信息
        print("\n\nPHASE 2: MODEL ANALYSIS")

        if not self.model_generator.generate_model_check(model_name, model_readme):
            print("❌ Failed to generate model_check.py")
            return False

        # 解析并安装脚本依赖
        self._install_script_dependencies("model_check.py")

        # 运行 model checker 获取元信息
        model_metadata = self._run_and_get_metadata(
            container, "model_check.py", "model_analysis.json", max_fixes
        )
        print(f"🔍 Model metadata result: {type(model_metadata)} - {model_metadata}")  # 调试信息
        if not model_metadata:
            print("❌ Failed to get model metadata - returned empty or None")
            return False

        # 阶段3: 基于元信息生成并运行评估脚本
        print("\n\nPHASE 3: EVALUATION (with metadata)")

        if not self.evaluation_generator.generate_evaluate_script_with_metadata(
            model_name, dataset_name, metric, model_readme, model_metadata, dataset_metadata
        ):
            print("❌ Failed to generate evaluate.py with metadata")
            return False

        # 解析并安装脚本依赖
        self._install_script_dependencies("evaluate.py")

        # 运行最终评估
        return self._run_final_evaluation(container, max_fixes)

    def _run_and_get_metadata(
        self, container, script_name: str, output_file: str, max_fixes: int
    ) -> dict:
        """运行脚本并获取元数据"""

        # 直接运行脚本，简化版本
        print(f"🔄 Running {script_name} to get metadata...")

        for attempt in range(max_fixes + 1):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_fixes + 1}: Running {script_name}...")

                # 运行脚本
                exit_code, output = self.docker_manager.execute_script(script_name)
                print(f"Script output: {output[:500]}...")

                # 首先尝试读取元数据文件（即使脚本失败，也可能生成了有用的元数据）
                metadata_exit_code, metadata_content = self.docker_manager.read_file(
                    f"/workspace/{output_file}"
                )
                if metadata_exit_code == 0:
                    try:
                        import json

                        metadata = json.loads(metadata_content)
                        print(f"✅ Retrieved metadata from {output_file}")
                        print(f"📊 Metadata content: {metadata}")  # 调试信息
                        # 即使脚本exit_code != 0，如果有有效元数据就返回
                        if exit_code != 0:
                            print(
                                f"⚠️ Script had exit code {exit_code} but produced valid metadata"
                            )
                        # 确保返回的不是空字典
                        if metadata:
                            return metadata
                        else:
                            print(f"⚠️ Metadata file {output_file} is empty or null")
                            output += f"\nEmpty metadata in {output_file}"
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON in {output_file}: {e}")
                        output += f"\nJSON decode error: {e}"
                else:
                    print(f"❌ Metadata file {output_file} not found")
                    output += f"\nMissing output file: {output_file}"

                # 如果没有有效元数据且脚本失败，记录错误
                if exit_code != 0:
                    print(f"❌ Script failed with exit code {exit_code}")
                    output += f"\nScript exit code: {exit_code}"

                # 如果失败且还有尝试次数，使用Aider修复
                if attempt < max_fixes:
                    print(f"🔧 Attempt {attempt + 1}: Using Aider to fix {script_name}...")
                    if self.docker_manager.run_aider_fix(script_name, output):
                        print("✅ Aider fix completed, retrying...")
                        continue
                    else:
                        print("❌ Aider fix failed, stopping retries")
                        break
                else:
                    print("💀 Maximum attempts reached")
                    break

            except Exception as e:
                print(f"❌ Error running {script_name}: {e}")
                if attempt < max_fixes:
                    print("🔧 Attempting Aider fix for exception...")
                    self.docker_manager.run_aider_fix(script_name, str(e))
                    continue
                break

        return {}

    def _run_final_evaluation(self, container, max_fixes: int) -> bool:
        """运行最终评估"""

        print("🔄 Running final evaluation...")

        for attempt in range(max_fixes + 1):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_fixes + 1}: Running evaluate.py...")

                # 运行评估脚本
                exit_code, output = self.docker_manager.execute_script("evaluate.py")
                print(f"Evaluation output: {output[:500]}...")

                if exit_code == 0:
                    # 检查结果文件是否存在
                    if self.docker_manager.check_file_exists("/workspace/results.json"):
                        # 验证JSON文件完整性
                        json_exit_code, json_content = self.docker_manager.read_file(
                            "/workspace/results.json"
                        )
                        if json_exit_code == 0:
                            try:
                                import json

                                json_data = json.loads(json_content)
                                print(
                                    f"✅ Evaluation completed successfully - JSON size: {len(json_content)} chars"
                                )
                                print(f"📊 Results keys: {list(json_data.keys())}")
                                return True
                            except json.JSONDecodeError as e:
                                print(f"❌ Invalid JSON in results.json: {e}")
                                print(f"🔍 JSON content preview: {json_content[:200]}...")

                                # 尝试修复损坏的JSON
                                print("🔧 Attempting to fix corrupted JSON...")
                                fixed_json = JSONFixer.fix_truncated_json(json_content)

                                if fixed_json:
                                    print("✅ JSON successfully repaired!")
                                    print(f"📊 Repaired results keys: {list(fixed_json.keys())}")

                                    # 将修复后的JSON写回文件
                                    try:
                                        import json as json_module

                                        fixed_content = json_module.dumps(fixed_json, indent=2)

                                        # 通过Docker执行写入命令
                                        write_cmd = f"python3 -c \"import json; data={repr(fixed_json)}; f=open('/workspace/results.json', 'w'); json.dump(data, f, indent=2); f.close(); print('JSON repaired and saved')\""
                                        self.docker_manager.execute_command(write_cmd)

                                        print("💾 Repaired JSON saved back to results.json")
                                        return True
                                    except Exception as write_error:
                                        print(f"⚠️ Could not write repaired JSON: {write_error}")
                                        output += f"\nJSON repaired but write failed: {write_error}"
                                else:
                                    print("❌ JSON repair failed")
                                    output += f"\nCorrupted JSON file: {e}"
                        else:
                            print("❌ Cannot read results.json content")
                            output += "\nCannot read results.json file"
                    else:
                        print("❌ Results file not generated")
                        output += "\nMissing results.json file"
                else:
                    print(f"❌ Evaluation failed with exit code {exit_code}")

                # 如果失败且还有尝试次数，使用Aider修复
                if attempt < max_fixes:
                    print(f"🔧 Attempt {attempt + 1}: Using Aider to fix evaluate.py...")
                    if self.docker_manager.run_aider_fix("evaluate.py", output):
                        print("✅ Aider fix completed, retrying...")
                        continue
                    else:
                        print("❌ Aider fix failed, stopping retries")
                        break
                else:
                    print("💀 Maximum attempts reached")
                    break

            except Exception as e:
                print(f"❌ Error running evaluation: {e}")
                if attempt < max_fixes:
                    print("🔧 Attempting Aider fix for exception...")
                    self.docker_manager.run_aider_fix("evaluate.py", str(e))
                    continue
                break

        print("❌ Final evaluation failed")
        return False

    def _run_experiment_with_fixes(self, container, run_num: int, max_fixes: int) -> bool:
        """运行实验流程，支持自动修复"""

        # 创建处理器实例
        dataset_handler = DatasetCheckHandler(self, container, run_num, max_fixes)
        model_handler = ModelCheckHandler(self, container, run_num, max_fixes)
        evaluation_handler = EvaluationHandler(self, container, run_num, max_fixes)

        # 执行各阶段
        handlers = [dataset_handler, model_handler, evaluation_handler]
        results = []

        for handler in handlers:
            result = handler.execute_phase()
            results.append(result)

            if not result.success:
                print(f"❌ {handler.phase_name} failed after {result.attempts} attempts")
                return False

        print("\n✅ All phases completed successfully!")
        return True

    def _install_script_dependencies(self, script_name: str) -> bool:
        """解析脚本并安装其依赖"""
        script_path = os.path.join(self.output_dir, script_name)

        if not os.path.exists(script_path):
            print(f"⚠️ Script {script_name} not found, skipping dependency installation")
            return False

        print(f"🔍 Analyzing dependencies for {script_name}...")
        dependencies = DependencyParser.parse_script(script_path)

        if dependencies:
            print(f"📦 Found {len(dependencies)} dependencies: {', '.join(dependencies)}")
            self.docker_manager.install_packages(dependencies)
            return True
        else:
            print(f"📦 No external dependencies found for {script_name}")
            return True

    def _check_results_exist(self, run_num: int) -> bool:
        """检查结果文件是否已存在"""
        results_file = os.path.join(self.output_dir, f"run_{run_num}_results.json")
        return os.path.exists(results_file)
