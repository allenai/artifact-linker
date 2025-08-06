"""
评估器 - 负责生成和执行评估脚本
"""

import os

from .utils.llm import get_response_from_llm


class EvaluationGenerator:
    """评估脚本生成器"""

    def __init__(self, coder):
        self.coder = coder

    def generate_evaluate_script_with_metadata(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        model_readme: str,
        model_metadata: dict,
        dataset_metadata: dict,
    ) -> bool:
        model_check_content = self._read_script_content("model_check.py")
        dataset_check_content = self._read_script_content("dataset_check.py")

        prompt = f"""
Create a comprehensive Python evaluation script for:
- Model: {model_name}
- Dataset: {dataset_name}
- Metric: {metric}

GENERATED MODEL CHECK SCRIPT:
```python
{model_check_content}
```

GENERATED DATASET CHECK SCRIPT:
```python
{dataset_check_content}
```

Use the above scripts as reference for understanding how to properly load and process the model and dataset.

Based on the metadata above, the script should:

1. Import all necessary libraries (transformers, datasets, torch, evaluate, sklearn, json, etc.)
2. Load the pre-trained model and tokenizer with optimal configuration (you should check the GENERATED MODEL CHECK SCRIPT)
3. Load and preprocess the dataset with proper configuration (you should check the GENERATED DATASET CHECK SCRIPT)
4. Implement evaluation logic for {metric}:
   - Use the correct preprocessing pipeline for the model architecture
   - Calculate {metric} accurately for the task type

5. Save comprehensive results to results.json with robust writing:
   - Include the calculated {metric}, model_name, dataset_name, total_samples, processing_time, device_used
   - CRITICAL: Use proper JSON format:
     results = {{
         "{metric}": calculated_metric_value,
         "model_name": "{model_name}",
         "dataset_name": "{dataset_name}",
         "total_samples": total_samples,
         "processing_time": processing_time,
         "device_used": device_used
     }}

The script should be optimized for the specific model-dataset combination based on the metadata analysis.

Return only the complete Python code for metric_check.py.
"""

        try:
            print("🤖 Generating evaluation script with metadata...")
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.coder.client,
                model=self.coder.actual_model,
                system_message="You are an expert ML engineer. Generate a robust, metadata-informed evaluation script.",
            )
            code = self._extract_code(response)

            file_path = os.path.join(self.coder.output_dir, "metric_check.py")
            with open(file_path, "w") as f:
                f.write(code)

            print(f"✅ Metadata-informed metric_check.py generated ({len(code)} characters)")
            return True

        except Exception as e:
            print(f"❌ Failed to generate metric_check.py with metadata: {e}")
            return False

    def _read_script_content(self, script_name: str) -> str:
        """读取生成的脚本内容"""
        try:
            script_path = os.path.join(self.coder.output_dir, script_name)
            if os.path.exists(script_path):
                with open(script_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"# {script_name} not found"
        except Exception as e:
            return f"# Error reading {script_name}: {e}"

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
