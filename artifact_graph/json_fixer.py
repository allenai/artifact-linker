"""
JSON修复工具 - 修复截断或损坏的JSON文件
"""

import json
import re
from typing import Dict, Any, Optional


class JSONFixer:
    """JSON文件修复器"""
    
    @classmethod
    def fix_truncated_json(cls, json_content: str) -> Optional[Dict[str, Any]]:
        """修复截断的JSON内容"""
        if not json_content.strip():
            return None
            
        try:
            # 首先尝试直接解析
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"🔧 Attempting to fix JSON error: {e}")
            
        # 尝试各种修复策略
        fixed_content = cls._try_fix_strategies(json_content)
        
        if fixed_content:
            try:
                return json.loads(fixed_content)
            except json.JSONDecodeError:
                print("❌ All JSON fix attempts failed")
                return None
        
        return None
    
    @classmethod
    def _try_fix_strategies(cls, content: str) -> Optional[str]:
        """尝试多种修复策略"""
        
        # 策略1: 补全缺失的结束括号/引号
        fixed = cls._fix_missing_brackets(content)
        if cls._is_valid_json(fixed):
            print("✅ Fixed with missing brackets strategy")
            return fixed
            
        # 策略2: 移除最后一个不完整的键值对
        fixed = cls._remove_incomplete_pair(content)
        if cls._is_valid_json(fixed):
            print("✅ Fixed by removing incomplete pair")
            return fixed
            
        # 策略3: 尝试找到最后一个完整的对象结束点
        fixed = cls._find_last_complete_object(content)
        if cls._is_valid_json(fixed):
            print("✅ Fixed by truncating to last complete object")
            return fixed
            
        # 策略4: 构建最小可用的JSON
        fixed = cls._build_minimal_json(content)
        if cls._is_valid_json(fixed):
            print("✅ Fixed with minimal JSON strategy")
            return fixed
            
        return None
    
    @classmethod
    def _fix_missing_brackets(cls, content: str) -> str:
        """修复缺失的括号和引号"""
        content = content.strip()
        
        # 计算括号平衡
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # 补全缺失的括号
        if open_braces > close_braces:
            content += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            content += ']' * (open_brackets - close_brackets)
            
        # 处理未闭合的字符串
        if content.count('"') % 2 == 1:
            # 找到最后一个引号
            last_quote = content.rfind('"')
            if last_quote != -1:
                # 检查是否是未闭合的值
                after_quote = content[last_quote+1:].strip()
                if after_quote and not after_quote.startswith(',') and not after_quote.startswith('}'):
                    content = content[:last_quote+1] + '"' + content[last_quote+1:]
        
        return content
    
    @classmethod
    def _remove_incomplete_pair(cls, content: str) -> str:
        """移除最后一个不完整的键值对"""
        content = content.strip()
        
        # 找到最后一个逗号
        last_comma = content.rfind(',')
        if last_comma != -1:
            # 检查逗号后是否有完整的键值对
            after_comma = content[last_comma+1:].strip()
            if after_comma and not after_comma.endswith('}'):
                # 移除最后一个不完整的部分
                content = content[:last_comma] + content[content.rfind('}', last_comma):]
                
        return content
    
    @classmethod
    def _find_last_complete_object(cls, content: str) -> str:
        """找到最后一个完整的对象结束点"""
        content = content.strip()
        
        # 从后往前查找完整的JSON结构
        for i in range(len(content) - 1, -1, -1):
            if content[i] == '}':
                candidate = content[:i+1]
                if cls._is_valid_json(candidate):
                    return candidate
                    
        return content
    
    @classmethod
    def _build_minimal_json(cls, content: str) -> str:
        """构建最小可用的JSON"""
        # 尝试提取可识别的键值对
        minimal = {}
        
        # 查找常见的评估结果字段
        patterns = {
            'accuracy': r'"accuracy"\s*:\s*([\d.]+)',
            'model_name': r'"model_name"\s*:\s*"([^"]*)"',
            'dataset_name': r'"dataset_name"\s*:\s*"([^"]*)"',
            'total_samples': r'"total_samples"\s*:\s*(\d+)',
            'correct_predictions': r'"correct_predictions"\s*:\s*(\d+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                try:
                    # 尝试转换为数字
                    if key in ['accuracy']:
                        minimal[key] = float(value)
                    elif key in ['total_samples', 'correct_predictions']:
                        minimal[key] = int(value)
                    else:
                        minimal[key] = value
                except ValueError:
                    minimal[key] = value
        
        if minimal:
            return json.dumps(minimal, indent=2)
        else:
            # 返回最基本的结构
            return '{"error": "JSON parsing failed", "status": "incomplete"}'
    
    @classmethod
    def _is_valid_json(cls, content: str) -> bool:
        """检查JSON是否有效"""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    @classmethod
    def diagnose_json_issue(cls, content: str) -> Dict[str, Any]:
        """诊断JSON问题"""
        diagnosis = {
            "length": len(content),
            "ends_with_brace": content.strip().endswith('}'),
            "brace_balance": content.count('{') - content.count('}'),
            "quote_balance": content.count('"') % 2 == 0,
            "last_char": content.strip()[-1] if content.strip() else None,
            "has_comma_at_end": content.strip().endswith(','),
        }
        
        try:
            json.loads(content)
            diagnosis["valid"] = True
            diagnosis["error"] = None
        except json.JSONDecodeError as e:
            diagnosis["valid"] = False
            diagnosis["error"] = str(e)
            diagnosis["error_line"] = getattr(e, 'lineno', None)
            diagnosis["error_col"] = getattr(e, 'colno', None)
        
        return diagnosis


def test_json_fixer():
    """测试JSON修复功能"""
    test_cases = [
        # 截断的JSON
        '{"accuracy": 0.6256880733944954, "model_name": "test", "total_samples": 3270, "correct_predictions": ',
        # 缺少结束括号
        '{"accuracy": 0.85, "model_name": "test"',
        # 未闭合的字符串
        '{"accuracy": 0.85, "model_name": "test}',
    ]
    
    for i, test_json in enumerate(test_cases):
        print(f"\n🧪 Test case {i+1}:")
        print(f"Original: {test_json}")
        
        diagnosis = JSONFixer.diagnose_json_issue(test_json)
        print(f"Diagnosis: {diagnosis}")
        
        fixed = JSONFixer.fix_truncated_json(test_json)
        if fixed:
            print(f"Fixed: {json.dumps(fixed, indent=2)}")
        else:
            print("❌ Could not fix")


if __name__ == "__main__":
    test_json_fixer() 