"""
模型检查器 - 负责生成和执行模型检查脚本
"""

import os
import requests
import time
from .base import ExperimentPhaseHandler, ExperimentPhase
from .utils.llm import get_response_from_llm


class ModelCheckHandler(ExperimentPhaseHandler):
    """模型检查处理器"""
    
    def __init__(self, coder, container, run_num: int, max_attempts: int):
        super().__init__(coder, container, run_num, max_attempts)
        self.phase_name = "MODEL CHECK"
        self.script_name = "model_check.py"
        self.expected_outputs = ["/workspace/model_analysis.json"]
    
    def _generate_fix_prompt(self, error_output: str) -> str:
        """生成模型检查修复提示"""
        return f"""
Fix the model_check.py script. The error is:
{error_output}

Common issues and solutions:
1. If transformers library fails - try updating transformers
2. If CUDA/GPU issues - fallback to CPU mode
3. If model is too large - use smaller model variant or quantization
4. If authentication fails - check HF_TOKEN environment variable
5. If download fails - try different model or check model ID
6. Always save results to model_analysis.json

Please fix the script to handle these issues robustly, with CPU fallbacks.
"""
    
    def _get_phase_enum(self) -> ExperimentPhase:
        return ExperimentPhase.MODEL_CHECK


class ModelCheckGenerator:
    """模型检查脚本生成器"""
    
    def __init__(self, coder):
        self.coder = coder
    
    def generate_model_check(self, model_name: str, model_readme: str) -> bool:
        """生成模型检查脚本"""
        prompt = f"""
Create a Python script to analyze the model: {model_name}

Model information from README:
{model_readme[:1000] if model_readme else "No README available"}

The script should:
1. Import necessary libraries (transformers, torch, json, os, etc.)
2. Load model and tokenizer with robust error handling
3. Analyze model properties:
   - Model architecture and size
   - Input/output specifications
   - Device capabilities (GPU/CPU)
   - Memory requirements
   - Supported tasks and formats
4. Test model inference with sample inputs
5. Handle authentication with HF_TOKEN if needed
6. Save comprehensive analysis to model_analysis.json
7. Include fallbacks for:
   - GPU unavailable -> use CPU
   - Large models -> use smaller precision
   - Authentication issues
   - Memory constraints

Key requirements:
- Graceful degradation from GPU to CPU
- Robust error handling and retries
- Save model_analysis.json with complete information
- Test actual model functionality

Return only the complete Python code for model_check.py.
"""

        try:
            print("🤖 Generating model_check.py...")
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.coder.client,
                model=self.coder.actual_model,
                system_message="You are an expert ML engineer. Generate a robust model analysis script that can be improved with Aider."
            )
            
            code = self._extract_code(response)
            
            file_path = os.path.join(self.coder.output_dir, "model_check.py")
            with open(file_path, 'w') as f:
                f.write(code)
            
            print(f"model_check.py generated ({len(code)} characters)")
            return True
            
        except Exception as e:
            print(f"Failed to generate model_check.py: {e}")
            return False
    
    def get_model_readme(self, model_name: str) -> str:
        """获取模型的README信息"""
        try:
            print(f"Fetching README for {model_name}...")
            
            # 尝试通过HuggingFace API获取模型信息
            url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                return model_info.get('cardData', {}).get('readme', '')
            else:
                print(f"Failed to fetch model info: {response.status_code}")
            
            # 如果API失败，尝试直接获取README
            readme_url = f"https://huggingface.co/{model_name}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            
            if response.status_code == 200:
                return response.text[:2000]  # 限制长度
            else:
                print(f"Failed to fetch README: {response.status_code}")
                return ""
        
        except Exception as e:
            print(f"Error fetching model README: {e}")
            return ""
    
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