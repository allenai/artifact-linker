"""
数据集检查器 - 负责生成和执行数据集检查脚本
"""

import os

from .base import ExperimentPhase, ExperimentPhaseHandler
from .utils.llm import get_response_from_llm


class DatasetCheckHandler(ExperimentPhaseHandler):
    """数据集检查处理器"""

    def __init__(self, coder, container, run_num: int, max_attempts: int):
        super().__init__(coder, container, run_num, max_attempts)
        self.phase_name = "DATASET CHECK"
        self.script_name = "dataset_check.py"
        self.expected_outputs = ["/workspace/dataset_analysis.json"]

    def _generate_fix_prompt(self, error_output: str) -> str:
        """生成数据集检查修复提示"""
        return f"""
Fix the dataset_check.py script. The error is:
{error_output}

Common issues and solutions:
1. If datasets library fails - try pip install datasets
2. If authentication fails - check HF_TOKEN environment variable
3. If download fails - try smaller subset or different dataset split
4. If memory issues - process data in smaller batches
5. Always save results to dataset_analysis.json

Please fix the script to handle these issues robustly.
"""

    def _get_phase_enum(self) -> ExperimentPhase:
        return ExperimentPhase.DATASET_CHECK


class DatasetCheckGenerator:
    """数据集检查脚本生成器"""

    def __init__(self, coder):
        self.coder = coder

    def generate_dataset_check(self, dataset_name: str) -> bool:
        """生成数据集检查脚本"""
        prompt = f"""
Create a Python script to download and analyze the dataset: {dataset_name}

The script should:
1. Import necessary libraries (datasets, json, os, etc.)
2. Download the dataset using datasets.load_dataset()
3. Analyze basic properties:
   - Number of samples in each split
   - Column names and types
   - Sample data examples (first 3-5 items)
   - Data size information
4. Handle authentication with HF_TOKEN if needed
5. Save comprehensive analysis to dataset_analysis.json
6. Include error handling for common issues:
   - Network connectivity
   - Authentication
   - Memory constraints
   - Dataset availability

Key requirements:
- Use robust error handling and fallbacks
- Print progress information
- Create dataset_analysis.json with complete information
- Handle both public and gated datasets gracefully

Return only the complete Python code for dataset_check.py.
"""

        try:
            print("🤖 Generating dataset_check.py...")
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.coder.client,
                model=self.coder.actual_model,
                system_message="You are an expert data engineer. Generate a robust dataset download and analysis script that can be improved with Aider.",
            )

            code = self._extract_code(response)

            file_path = os.path.join(self.coder.output_dir, "dataset_check.py")
            with open(file_path, "w") as f:
                f.write(code)

            print(f"dataset_check.py generated ({len(code)} characters)")
            return True

        except Exception as e:
            print(f"Failed to generate dataset_check.py: {e}")
            return False

    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        # 查找代码块
        patterns = [
            r"```python\n(.*?)\n```",
            r"```\n(.*?)\n```",
            r"```python(.*?)```",
            r"```(.*?)```",
        ]

        import re

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # 如果没找到代码块，返回整个响应
        return response.strip()
