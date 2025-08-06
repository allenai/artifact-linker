"""
Refactored DockerCoder - using separated components
"""

import json
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
    """Refactored Docker code generator using separated components"""

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

        # Initialize Docker manager
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
        max_fixes: int = 3,
    ) -> Dict:
        """Run evaluation process"""

        print(f"\n{'='*80}")
        print(f" ML MODEL EVALUATION: {model_name} on {dataset_name}")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Metric: {metric}")
        print(f"Max Fixes: {max_fixes}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Memory Limit: {self.memory_limit}")

        try:
            print("\n\nCONTAINER SETUP")

            # Create Docker container
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

            # Get model README
            model_readme = self.model_generator.get_model_readme(model_name)

            total_runs = 1

            print(f"\n{'='*80}")
            print(" RUN")
            print(f"{'='*80}")

            run_success = self._execute_single_run(
                1, max_fixes, model_name, dataset_name, metric, model_readme, container
            )

            print(f"\n{'='*80}")
            print(" SUMMARY")
            print(f"{'='*80}")
            
            # Read and display the result.json content
            result_file = os.path.join(self.output_dir, "run_1", "final_info.json")
            experiment_results = {}
            
            if run_success and os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        experiment_results = json.load(f)
                    print("📊 EXPERIMENT RESULTS:")
                    print(json.dumps(experiment_results, indent=2))
                except Exception as e:
                    print(f"❌ Error reading result file: {e}")
                    experiment_results = {}
            else:
                print("❌ No results available - experiment failed or result file not found")

            self.docker_manager.cleanup()

            result = {
                "success": run_success,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metric": metric,
                "output_dir": self.output_dir,
                "experiment_results": experiment_results
            }

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
        """Execute single run"""
        try:
            # Check if results already exist
            if self._check_results_exist(run):
                print(f"⚡ Results for run {run} already exist, skipping...")
                return True

            # Sequential execution: get metadata then generate evaluation script
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
        """Execute experiment sequentially: get metadata first, then generate evaluation script"""

        print("\n\nPHASE 1: DATASET ANALYSIS")

        # Get dataset README
        dataset_readme = self.dataset_generator.get_dataset_readme(dataset_name)

        try:
            script_content = self.dataset_generator.generate_script(dataset_name, dataset_readme)
            script_path = os.path.join(self.output_dir, "dataset_check.py")
            with open(script_path, 'w') as f:
                f.write(script_content)
            print("✅ Generated dataset_check.py with README")
        except Exception as e:
            print(f"❌ Failed to generate dataset_check.py: {e}")
            return False

        self._install_script_dependencies("dataset_check.py")

        dataset_metadata = self._run_and_get_metadata(
            container, "dataset_check.py", "dataset_metadata.json", max_fixes
        )
        if not dataset_metadata:
            print("❌ Failed to get dataset metadata")
            return False

        print("\n\nPHASE 2: MODEL ANALYSIS")

        try:
            script_content = self.model_generator.generate_script(model_name, dataset_name, dataset_metadata, model_readme)
            script_path = os.path.join(self.output_dir, "model_check.py")
            with open(script_path, 'w') as f:
                f.write(script_content)
            print("✅ Generated model_check.py with README and dataset context")
        except Exception as e:
            print(f"❌ Failed to generate model_check.py: {e}")
            return False

        self._install_script_dependencies("model_check.py")

        model_metadata = self._run_and_get_metadata(
            container, "model_check.py", "model_metadata.json", max_fixes
        )
        print(f"🔍 Model metadata result: {type(model_metadata)} - {model_metadata}")  # 调试信息
        if not model_metadata:
            print("❌ Failed to get model metadata - returned empty or None")
            return False

        print("\n\nPHASE 3: EVALUATION (with metadata)")

        if not self.evaluation_generator.generate_evaluate_script_with_metadata(
            model_name, dataset_name, metric, model_readme, model_metadata, dataset_metadata
        ):
            print("❌ Failed to generate metric_check.py with metadata")
            return False

        self._install_script_dependencies("metric_check.py")

        return self._run_final_evaluation(container, max_fixes)

    def _run_and_get_metadata(self, container, script_name: str, output_file: str, max_fixes: int) -> dict:
        """Run script and get metadata with simplified logic"""
        print(f"🔄 Running {script_name} to get metadata...")

        for attempt in range(max_fixes + 1):
            print(f"🔄 Attempt {attempt + 1}/{max_fixes + 1}: Running {script_name}...")
            
            # Step 1: Execute script
            exit_code, output = self.docker_manager.execute_script(script_name)
            print(f"Script output: {output[:500]}...")
            
            # Step 2: Try to get metadata regardless of exit code
            metadata = self._extract_metadata(output_file)
            if metadata:
                if exit_code != 0:
                    print(f"⚠️ Script had exit code {exit_code} but produced valid metadata")
                return metadata
            
            # Step 3: Handle failure
            print(f"❌ Failed to get valid metadata from {script_name}")
            if exit_code != 0:
                print(f"Exit code: {exit_code}")
            
            # Step 4: Try to fix if we have attempts left
            if attempt < max_fixes:
                print(f"🔧 Attempting to fix {script_name} (attempt {attempt + 1})...")
                if self.docker_manager.run_aider_fix(script_name, output):
                    print("✅ Fix applied, retrying...")
                    continue
                else:
                    print("❌ Fix failed, stopping retries")
                    break
            else:
                print("💀 Maximum attempts reached")
                break

        print(f"❌ Failed to get metadata from {script_name} after all attempts")
        return {}

    def _extract_metadata(self, output_file: str) -> dict:
        """Extract metadata from output file"""
        # Check if metadata file exists
        if not self.docker_manager.check_file_exists(f"/workspace/{output_file}"):
            print(f"❌ Metadata file {output_file} not found")
            return {}
            
        # Read the metadata file
        exit_code, content = self.docker_manager.read_file(f"/workspace/{output_file}")
        if exit_code != 0:
            print(f"❌ Cannot read {output_file}")
            return {}
            
        # Parse JSON
        try:
            import json
            metadata = json.loads(content)
            if metadata:
                print(f"✅ Retrieved metadata from {output_file}")
                print(f"📊 Metadata keys: {list(metadata.keys())}")
                return metadata
            else:
                print(f"⚠️ Metadata file {output_file} is empty")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {output_file}: {e}")
            return {}

    def _run_final_evaluation(self, container, max_fixes: int) -> bool:
        """Run final evaluation with simplified logic"""
        print("🔄 Running final evaluation...")

        for attempt in range(max_fixes + 1):
            print(f"🔄 Attempt {attempt + 1}/{max_fixes + 1}: Running metric_check.py...")
            
            # Step 1: Execute the script
            exit_code, output = self.docker_manager.execute_script("metric_check.py")
            print(f"Evaluation output: {output[:500]}...")
            
            # Step 2: Check if script succeeded
            if exit_code == 0:
                # Step 3: Validate results
                if self._validate_evaluation_results():
                    print("✅ Evaluation completed successfully")
                    return True
                else:
                    print("⚠️ Script succeeded but results are invalid")
                    output += "\nInvalid or missing results"
            else:
                print(f"❌ Evaluation failed with exit code {exit_code}")
                print(f"❌ Full output:\n{output}")
                
            # Step 4: Try to fix if we have attempts left
            if attempt < max_fixes:
                print(f"🔧 Attempting to fix metric_check.py (attempt {attempt + 1})...")
                if self.docker_manager.run_aider_fix("metric_check.py", output):
                    print("✅ Fix applied, retrying...")
                    continue
                else:
                    print("❌ Fix failed, stopping retries")
                    break
            else:
                print("💀 Maximum attempts reached")
                break

        print("❌ Final evaluation failed after all attempts")
        return False

    def _validate_evaluation_results(self) -> bool:
        """Validate evaluation results file"""
        # Check if results.json exists
        if not self.docker_manager.check_file_exists("/workspace/results.json"):
            print("❌ results.json not found")
            return False
            
        # Try to read and parse the results
        json_exit_code, json_content = self.docker_manager.read_file("/workspace/results.json")
        if json_exit_code != 0:
            print("❌ Cannot read results.json")
            return False
            
        try:
            import json
            json_data = json.loads(json_content)
            print(f"📊 Results keys: {list(json_data.keys())}")
            print(f"📏 JSON size: {len(json_content)} chars")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            # Try to fix corrupted JSON
            return self._try_fix_json(json_content)
    
    def _try_fix_json(self, json_content: str) -> bool:
        """Try to fix corrupted JSON"""
        print("🔧 Attempting to fix corrupted JSON...")
        
        try:
            from .json_fixer import JSONFixer
            fixed_json = JSONFixer.fix_json_content(json_content)
            
            if fixed_json:
                print("✅ JSON successfully repaired!")
                print(f"📊 Repaired results keys: {list(fixed_json.keys())}")
                
                # Save the fixed JSON back
                import json
                fixed_content = json.dumps(fixed_json, indent=2)
                write_cmd = f"python3 -c \"import json; data={repr(fixed_json)}; json.dump(data, open('/workspace/results.json', 'w'), indent=2)\""
                exit_code, _ = self.docker_manager.execute_command([write_cmd])
                
                if exit_code == 0:
                    print("💾 Repaired JSON saved successfully")
                    return True
                else:
                    print("⚠️ Could not save repaired JSON")
                    
        except Exception as e:
            print(f"❌ JSON repair failed: {e}")
            
        return False



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
