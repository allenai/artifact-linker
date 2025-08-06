"""
依赖解析器 - 从Python脚本中自动检测所需的依赖包
"""

import ast
import re
from typing import List, Set, Dict
from pathlib import Path


class DependencyParser:
    """Python脚本依赖解析器"""
    
    # 常见的第三方包映射 (import名 -> pip包名)
    PACKAGE_MAPPING = {
        # ML/AI packages
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'datasets': 'datasets',
        'accelerate': 'accelerate',
        'tokenizers': 'tokenizers',
        'huggingface_hub': 'huggingface-hub',
        
        # Data science
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        
        # Evaluation
        'evaluate': 'evaluate',
        'rouge': 'rouge-score',
        'bleu': 'bleu',
        'sacrebleu': 'sacrebleu',
        
        # Utilities
        'requests': 'requests',
        'tqdm': 'tqdm',
        'wandb': 'wandb',
        'tensorboard': 'tensorboard',
        'yaml': 'PyYAML',
        'toml': 'toml',
        'click': 'click',
        'argparse': None,  # Built-in
        'json': None,      # Built-in
        'os': None,        # Built-in
        'sys': None,       # Built-in
        're': None,        # Built-in
        'time': None,      # Built-in
        'datetime': None,  # Built-in
        'math': None,      # Built-in
        'random': None,    # Built-in
        'itertools': None, # Built-in
        'collections': None, # Built-in
        'pathlib': None,   # Built-in (Python 3.4+)
        'typing': None,    # Built-in (Python 3.5+)
    }
    
    @classmethod
    def parse_script(cls, script_path: str) -> List[str]:
        """解析Python脚本，返回所需的pip包列表"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return cls.parse_code(code)
            
        except Exception as e:
            print(f"⚠️ Error parsing {script_path}: {e}")
            return []
    
    @classmethod
    def parse_code(cls, code: str) -> List[str]:
        """解析Python代码，返回所需的pip包列表"""
        dependencies = set()
        
        try:
            # 方法1: AST解析 (更准确)
            ast_deps = cls._parse_with_ast(code)
            dependencies.update(ast_deps)
            
        except Exception as e:
            print(f"⚠️ AST parsing failed: {e}")
            
        try:
            # 方法2: 正则表达式解析 (备用)
            regex_deps = cls._parse_with_regex(code)
            dependencies.update(regex_deps)
            
        except Exception as e:
            print(f"⚠️ Regex parsing failed: {e}")
        
        # 过滤掉内置模块和无效包
        pip_packages = []
        for dep in dependencies:
            if dep in cls.PACKAGE_MAPPING:
                pip_pkg = cls.PACKAGE_MAPPING[dep]
                if pip_pkg and pip_pkg not in pip_packages:
                    pip_packages.append(pip_pkg)
            elif dep and not cls._is_builtin_module(dep):
                # 未知包，直接使用import名
                if dep not in pip_packages:
                    pip_packages.append(dep)
        
        return sorted(pip_packages)
    
    @classmethod
    def _parse_with_ast(cls, code: str) -> Set[str]:
        """使用AST解析import语句"""
        dependencies = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # import numpy, pandas
                        root_module = alias.name.split('.')[0]
                        dependencies.add(root_module)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # from transformers import AutoModel
                        root_module = node.module.split('.')[0]
                        dependencies.add(root_module)
                    
        except SyntaxError as e:
            print(f"⚠️ Syntax error in code: {e}")
            
        return dependencies
    
    @classmethod 
    def _parse_with_regex(cls, code: str) -> Set[str]:
        """使用正则表达式解析import语句（备用方法）"""
        dependencies = set()
        
        # 匹配 import xxx
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            module = match.group(1).split('.')[0]
            dependencies.add(module)
        
        # 匹配 from xxx import yyy
        from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        for match in re.finditer(from_pattern, code, re.MULTILINE):
            module = match.group(1).split('.')[0]
            dependencies.add(module)
        
        return dependencies
    
    @classmethod
    def _is_builtin_module(cls, module_name: str) -> bool:
        """检查是否为内置模块"""
        import sys
        
        # Python内置模块列表
        builtin_modules = set(sys.builtin_module_names)
        
        # 标准库模块（部分）
        stdlib_modules = {
            'os', 'sys', 'json', 'math', 'random', 'time', 'datetime',
            'collections', 'itertools', 'functools', 'operator',
            'pathlib', 'typing', 're', 'string', 'io', 'tempfile',
            'shutil', 'glob', 'fnmatch', 'argparse', 'configparser',
            'logging', 'threading', 'multiprocessing', 'subprocess',
            'urllib', 'http', 'email', 'xml', 'html', 'sqlite3',
            'csv', 'gzip', 'zipfile', 'tarfile', 'pickle', 'base64',
            'hashlib', 'hmac', 'secrets', 'uuid', 'decimal', 'fractions'
        }
        
        return module_name in builtin_modules or module_name in stdlib_modules
    
    @classmethod
    def analyze_script_requirements(cls, script_path: str) -> Dict[str, any]:
        """分析脚本的完整需求信息"""
        dependencies = cls.parse_script(script_path)
        
        # 按类别分组
        categories = {
            'ml_ai': [],
            'data_science': [],
            'evaluation': [],
            'utilities': [],
            'unknown': []
        }
        
        ml_ai_packages = {'torch', 'transformers', 'datasets', 'accelerate', 'tokenizers', 'huggingface-hub'}
        data_packages = {'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn'}
        eval_packages = {'evaluate', 'rouge-score', 'bleu', 'sacrebleu'}
        util_packages = {'requests', 'tqdm', 'wandb', 'tensorboard', 'PyYAML', 'toml', 'click'}
        
        for dep in dependencies:
            if dep in ml_ai_packages:
                categories['ml_ai'].append(dep)
            elif dep in data_packages:
                categories['data_science'].append(dep)
            elif dep in eval_packages:
                categories['evaluation'].append(dep)
            elif dep in util_packages:
                categories['utilities'].append(dep)
            else:
                categories['unknown'].append(dep)
        
        return {
            'total_dependencies': len(dependencies),
            'dependencies': dependencies,
            'categories': categories,
            'script_path': script_path
        }


def test_dependency_parser():
    """测试依赖解析器"""
    test_code = """
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from datasets import load_dataset
import evaluate
import json
import os
from typing import Dict, List
"""
    
    deps = DependencyParser.parse_code(test_code)
    print("🧪 Test dependencies found:")
    for dep in deps:
        print(f"  📦 {dep}")


if __name__ == "__main__":
    test_dependency_parser() 