"""
重构后的DockerCoder - 使用分离的组件
"""

import json
import os
import tempfile
from typing import Dict, Optional
import docker
import docker.types

from .utils.llm import create_client
from .base import ExperimentPhase, PhaseResult
from .dataset_checker import DatasetCheckHandler, DatasetCheckGenerator
from .model_checker import ModelCheckHandler, ModelCheckGenerator
from .evaluator import EvaluationHandler, EvaluationGenerator


class DockerCoder:
    """重构后的Docker代码生成器，使用分离的组件"""
    
    def __init__(self, model: str = "gpt-4o-mini", output_dir: str = "results", memory_limit: str = "8g"):
        self.model = model
        self.output_dir = output_dir
        self.memory_limit = memory_limit
        # create_client returns (client, model) tuple
        self.client, self.actual_model = create_client(model)
        
        # 初始化各组件
        self.dataset_generator = DatasetCheckGenerator(self)
        self.model_generator = ModelCheckGenerator(self)
        self.evaluation_generator = EvaluationGenerator(self)
        
    def evaluate(self, model_name: str, dataset_name: str, metric: str = "accuracy", max_runs: int = 3, max_fixes: int = 3) -> Dict:
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
            print(f"\n{'='*60}")
            print(" CONTAINER SETUP")
            print(f"{'='*60}")
            
            print("Creating Docker container...")
            container = self._create_container()
            
            if not container:
                print("❌ Failed to create Docker container")
                return {}
            print("✅ Docker container created")
            
            if self.client is None:
                print("❌ No LLM client available")
                container.stop()
                container.remove()
                return {}
            print("✅ LLM client ready")
            
            # 预安装包
            self._preinstall_missing_packages(container)
            
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
                
                if self._execute_single_run(run + 1, max_fixes, model_name, dataset_name, metric, model_readme, container):
                    successful_runs += 1
                    print(f"✅ Run {run + 1} completed successfully")
                else:
                    print(f"❌ Run {run + 1} failed")
            
            container.stop()
            container.remove()
            print("🧹 Container cleaned up")
            
            success_rate = successful_runs / total_runs if total_runs > 0 else 0
            
            result = {
                "success": successful_runs > 0,
                "successful_runs": successful_runs,
                "total_runs": total_runs,
                "success_rate": success_rate,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metric": metric,
                "output_dir": self.output_dir
            }
            
            print(f"\n{'='*60}")
            print(" FINAL RESULTS")
            print(f"{'='*60}")
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
                "metric": metric
            }
    
    def _execute_single_run(self, run: int, max_fixes: int, model_name: str, dataset_name: str, metric: str, model_readme: str, container) -> bool:
        """执行单次运行"""
        try:
            # 检查是否已有结果
            if self._check_results_exist(run):
                print(f"⚡ Results for run {run} already exist, skipping...")
                return True
            
            # 阶段1: 生成脚本
            print(f"\n{'='*60}")
            print(" SCRIPT GENERATION")
            print(f"{'='*60}")
            
            if not self.dataset_generator.generate_dataset_check(dataset_name):
                print("❌ Failed to generate dataset_check.py")
                return False
            
            if not self.model_generator.generate_model_check(model_name, model_readme):
                print("❌ Failed to generate model_check.py")
                return False
                
            if not self.evaluation_generator.generate_evaluate_script(model_name, dataset_name, metric, model_readme):
                print("❌ Failed to generate evaluate.py")
                return False
            
            # 阶段2: 执行实验流程
            return self._run_experiment_with_fixes(container, run, max_fixes)
            
        except Exception as e:
            print(f"❌ Run {run} failed with error: {e}")
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
        
        print(f"\n✅ All phases completed successfully!")
        return True
    
    def _create_container(self):
        """创建Docker容器"""
        try:
            # 先创建输出目录，确保是当前用户权限
            os.makedirs(self.output_dir, exist_ok=True)
            
            docker_client = docker.from_env()
            
            container = docker_client.containers.run(
                "simple-coder:latest",
                command="sleep infinity",
                detach=True,
                remove=False,
                working_dir="/workspace",
                volumes={
                    os.path.abspath(self.output_dir): {"bind": "/workspace", "mode": "rw"}
                },
                mem_limit=self.memory_limit,
                # Enable GPU support
                device_requests=[
                    docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
                ],
                environment={
                    "PYTHONPATH": "/workspace",
                    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                    "CUDA_VISIBLE_DEVICES": "all"
                }
            )
            
            return container
            
        except Exception as e:
            print(f"Failed to create container: {e}")
            return None
    
    def _preinstall_missing_packages(self, container) -> None:
        """预安装可能缺失的包"""
        packages = [
            "evaluate",
            "scikit-learn", 
            "scipy",
            "accelerate"
        ]
        
        print(f"📦 Installing packages: {', '.join(packages)}")
        
        for package in packages:
            try:
                result = container.exec_run(f"pip install {package}", workdir="/workspace")
                if result.exit_code == 0:
                    print(f"✅ {package} installed")
                else:
                    print(f"⚠️  {package} installation failed (might already exist)")
            except Exception as e:
                print(f"⚠️  Failed to install {package}: {e}")
    
    def _check_results_exist(self, run_num: int) -> bool:
        """检查结果文件是否已存在"""
        results_file = os.path.join(self.output_dir, f"run_{run_num}_results.json")
        return os.path.exists(results_file) 