"""
评估器 - 负责生成和执行评估脚本
"""

import os

from .base import ExperimentPhase, ExperimentPhaseHandler
from .utils.llm import get_response_from_llm


class EvaluationHandler(ExperimentPhaseHandler):
    """评估处理器"""

    def __init__(self, coder, container, run_num: int, max_attempts: int):
        super().__init__(coder, container, run_num, max_attempts)
        self.phase_name = "EVALUATION"
        self.script_name = "evaluate.py"
        self.expected_outputs = ["/workspace/results.json"]

    def _generate_fix_prompt(self, error_output: str) -> str:
        """生成评估修复提示"""
        return f"""
Fix the evaluate.py script. The error is:
{error_output}

Common issues and solutions:
1. If model loading fails - ensure model compatibility and try CPU fallback
2. If dataset loading fails - check dataset ID and authentication
3. If GPU memory issues - reduce batch size or use CPU
4. If metric calculation fails - implement fallback metrics
5. If evaluation takes too long - use smaller dataset subset
6. Always save results to results.json

Please fix the script to handle these issues robustly and complete the evaluation.
"""

    def _get_phase_enum(self) -> ExperimentPhase:
        return ExperimentPhase.EVALUATION


class EvaluationGenerator:
    """评估脚本生成器"""

    def __init__(self, coder):
        self.coder = coder

    def generate_evaluate_script(
        self, model_name: str, dataset_name: str, metric: str, model_readme: str
    ) -> bool:
        """生成评估脚本"""
        prompt = f"""
Create a comprehensive Python evaluation script for:
- Model: {model_name}
- Dataset: {dataset_name}
- Metric: {metric}

Model information:
{model_readme[:500] if model_readme else "No README available"}

The script should:
1. Import all necessary libraries (transformers, datasets, torch, evaluate, sklearn, json, etc.)
2. Load the pre-trained model and tokenizer with error handling
3. Load and preprocess the dataset appropriately
4. Implement proper evaluation logic for {metric}:
   - Handle different data formats and task types
   - Use appropriate preprocessing for the model
   - Calculate {metric} accurately
   - Include standard metrics as fallbacks

4. Performance optimizations:
   - Use model analysis to configure model properly
   - Handle device placement (CPU/GPU) intelligently
   - Process data in appropriate batch sizes

5. Save comprehensive results to results.json:
   - Include the calculated {metric}
   - Add processing statistics
   - Include any relevant metadata

6. Error handling:
   - Graceful fallbacks for device/memory issues
   - Robust data processing
   - Clear error messages

Return only the complete Python code for evaluate.py.
"""

        try:
            print("Generating evaluation script...")
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.coder.client,
                model=self.coder.actual_model,
                system_message="You are an expert ML engineer. Generate a robust evaluation script that can be improved with Aider.",
            )
            code = self._extract_code(response)

            file_path = os.path.join(self.coder.output_dir, "evaluate.py")
            with open(file_path, "w") as f:
                f.write(code)

            print(f"evaluate.py generated ({len(code)} characters)")
            return True

        except Exception as e:
            print(f"Failed to generate evaluate.py: {e}")
            return False

    def generate_evaluate_script_with_metadata(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        model_readme: str,
        model_metadata: dict,
        dataset_metadata: dict,
    ) -> bool:
        """基于模型和数据集元信息生成评估脚本"""

        # 提取关键信息
        model_info = self._extract_model_info(model_metadata)
        dataset_info = self._extract_dataset_info(dataset_metadata)

        prompt = f"""
Create a comprehensive Python evaluation script for:
- Model: {model_name}
- Dataset: {dataset_name}
- Metric: {metric}

MODEL METADATA:
{model_info}

DATASET METADATA:
{dataset_info}

Model README information:
{model_readme[:500] if model_readme else "No README available"}

Based on the metadata above, the script should:

1. Import all necessary libraries (transformers, datasets, torch, evaluate, sklearn, json, etc.)

2. Load the pre-trained model and tokenizer with optimal configuration:
   - Use the detected architecture: {model_metadata.get('architecture', 'auto')}
   - Apply appropriate device placement based on model size
   - Use the correct task type: {model_metadata.get('task_type', 'auto')}

3. Load and preprocess the dataset with proper configuration:
   - Dataset has {dataset_metadata.get('total_samples', 'unknown')} samples
   - Use appropriate split: {dataset_metadata.get('available_splits', ['train', 'test'])}
   - Handle the data format: {dataset_metadata.get('data_format', 'auto')}
   - Input columns: {dataset_metadata.get('input_columns', [])}
   - Target columns: {dataset_metadata.get('target_columns', [])}

4. Implement evaluation logic for {metric}:
   - Process data in appropriate batch sizes based on model memory requirements
   - Use the correct preprocessing pipeline for the model architecture
   - Calculate {metric} accurately for the task type
   - Include confidence intervals and error analysis

5. Performance optimizations:
   - Batch size: optimize based on model size ({model_metadata.get('parameters', 'unknown')} parameters)
   - Memory management: {model_metadata.get('memory_usage', 'optimize automatically')}
   - Device placement: {model_metadata.get('device_recommendation', 'auto')}

6. Save comprehensive results to results.json with robust writing:
   - Include the calculated {metric}: value and confidence interval
   - Add processing statistics and performance metrics
   - Include model and dataset metadata used
   - Add any relevant error analysis
   - CRITICAL: Use proper JSON writing with error handling:
     ```python
     import json

     results = {{
         "{metric}": calculated_metric_value,
         "confidence_interval": confidence_interval,
         "model_name": "{model_name}",
         "dataset_name": "{dataset_name}",
         "total_samples": total_samples,
         "correct_predictions": correct_predictions,
         "error_analysis": error_analysis_dict,
         "processing_time": processing_time,
         "device_used": device_used
     }}

     # Write JSON with atomic operation and validation
     import tempfile
     import os
     import shutil

     # Write to temporary file first
     temp_path = "/workspace/results_temp.json"
     final_path = "/workspace/results.json"

     try:
         with open(temp_path, 'w', encoding='utf-8') as f:
             json.dump(results, f, indent=2, ensure_ascii=False)
             f.flush()
             os.fsync(f.fileno())  # Force write to disk

         # Verify the temp file is valid JSON
         with open(temp_path, 'r', encoding='utf-8') as f:
             json.load(f)  # Validate JSON

         # Atomically move to final location
         shutil.move(temp_path, final_path)
         print(f"✅ Results saved to {{final_path}}")

     except Exception as e:
         print(f"❌ Error saving results: {{e}}")
         # Fallback: simple write
         with open(final_path, 'w', encoding='utf-8') as f:
             json.dump(results, f, indent=2)
     ```

7. Robust error handling:
   - Graceful fallbacks for device/memory issues
   - Data type conversion and format handling
   - Clear progress reporting and error messages
   - Ensure all file operations complete successfully

The script should be optimized for the specific model-dataset combination based on the metadata analysis.

Return only the complete Python code for evaluate.py.
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

            file_path = os.path.join(self.coder.output_dir, "evaluate.py")
            with open(file_path, "w") as f:
                f.write(code)

            print(f"✅ Metadata-informed evaluate.py generated ({len(code)} characters)")
            return True

        except Exception as e:
            print(f"❌ Failed to generate evaluate.py with metadata: {e}")
            return False

    def _extract_model_info(self, model_metadata: dict) -> str:
        """提取关键模型信息"""
        info_parts = []

        if "architecture" in model_metadata:
            info_parts.append(f"Architecture: {model_metadata['architecture']}")
        if "parameters" in model_metadata:
            info_parts.append(f"Parameters: {model_metadata['parameters']}")
        if "task_type" in model_metadata:
            info_parts.append(f"Task Type: {model_metadata['task_type']}")
        if "device_recommendation" in model_metadata:
            info_parts.append(f"Recommended Device: {model_metadata['device_recommendation']}")
        if "memory_usage" in model_metadata:
            info_parts.append(f"Memory Usage: {model_metadata['memory_usage']}")
        if "input_format" in model_metadata:
            info_parts.append(f"Input Format: {model_metadata['input_format']}")
        if "output_format" in model_metadata:
            info_parts.append(f"Output Format: {model_metadata['output_format']}")

        return "\n".join(info_parts) if info_parts else "No detailed model metadata available"

    def _extract_dataset_info(self, dataset_metadata: dict) -> str:
        """提取关键数据集信息"""
        info_parts = []

        if "total_samples" in dataset_metadata:
            info_parts.append(f"Total Samples: {dataset_metadata['total_samples']}")
        if "available_splits" in dataset_metadata:
            info_parts.append(f"Available Splits: {dataset_metadata['available_splits']}")
        if "data_format" in dataset_metadata:
            info_parts.append(f"Data Format: {dataset_metadata['data_format']}")
        if "input_columns" in dataset_metadata:
            info_parts.append(f"Input Columns: {dataset_metadata['input_columns']}")
        if "target_columns" in dataset_metadata:
            info_parts.append(f"Target Columns: {dataset_metadata['target_columns']}")
        if "sample_data" in dataset_metadata:
            info_parts.append(f"Sample Data: {dataset_metadata['sample_data']}")
        if "data_types" in dataset_metadata:
            info_parts.append(f"Data Types: {dataset_metadata['data_types']}")

        return "\n".join(info_parts) if info_parts else "No detailed dataset metadata available"

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
