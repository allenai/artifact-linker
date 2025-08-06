"""
评估器 - 负责生成和执行评估脚本
"""

import os
from .base import ExperimentPhaseHandler, ExperimentPhase
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
    
    def generate_evaluate_script(self, model_name: str, dataset_name: str, metric: str, model_readme: str) -> bool:
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
                system_message="You are an expert ML engineer. Generate a robust evaluation script that can be improved with Aider."
            )
            code = self._extract_code(response)
            
            file_path = os.path.join(self.coder.output_dir, "evaluate.py")
            with open(file_path, 'w') as f:
                f.write(code)
            
            print(f"evaluate.py generated ({len(code)} characters)")
            return True
            
        except Exception as e:
            print(f"Failed to generate evaluate.py: {e}")
            return False
    
    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        # 查找代码块
        patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'```python(.*?)```',
            r'```(.*?)```'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 如果没找到代码块，返回整个响应
        return response.strip() 