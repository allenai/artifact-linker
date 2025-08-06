"""
JSON repair tool - fixes truncated or corrupted JSON files
"""

import json
import re
from typing import Dict, Any, Optional


class JSONFixer:
    """JSON file fixer"""
    
    @staticmethod
    def fix_json_content(content: str) -> Optional[Dict[str, Any]]:
        """Fix truncated JSON content"""
        if not content.strip():
            return None
            
        # Try to parse directly first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try various repair strategies
        for strategy in [
            JSONFixer._fix_missing_brackets,
            JSONFixer._fix_truncated_content,
            JSONFixer._extract_partial_data
        ]:
            try:
                fixed_content = strategy(content)
                if fixed_content:
                    return json.loads(fixed_content)
            except (json.JSONDecodeError, Exception):
                continue
        
        return None
    
    @staticmethod
    def _fix_missing_brackets(content: str) -> str:
        """Try multiple repair strategies"""
        content = content.strip()
        
        # Strategy 1: Add missing closing braces
        open_braces = content.count('{') - content.count('}')
        if open_braces > 0:
            content += '}' * open_braces
        
        # Strategy 2: Remove trailing commas
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Strategy 3: Try to find the last complete object end point
        last_brace = content.rfind('}')
        if last_brace > 0:
            # Try content up to the last complete brace
            test_content = content[:last_brace + 1]
            try:
                json.loads(test_content)
                return test_content
            except json.JSONDecodeError:
                pass
        
        return content
    
    @staticmethod
    def _fix_truncated_content(content: str) -> str:
        """Fix missing brackets and quotes"""
        content = content.strip()
        
        # Remove incomplete lines at the end
        lines = content.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line or line.endswith(','):
                continue
            
            # Check if it's an unclosed value
            if ':' in line and not (line.endswith('"') or line.endswith('}') or line.endswith(']')):
                # Try to close the line properly
                if '"' in line and line.count('"') % 2 == 1:
                    lines[i] += '"'
                break
        
        content = '\n'.join(lines[:i+1])
        
        # Add missing closing brackets
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Check if there's a complete key-value pair after comma
        if content.endswith(','):
            content = content[:-1]  # Remove trailing comma
        
        content += '}' * open_braces
        content += ']' * open_brackets
        
        return content
    
    @staticmethod
    def _extract_partial_data(content: str) -> str:
        """Extract recognizable key-value pairs"""
        result = {}
        
        # Look for common evaluation result fields
        patterns = {
            'accuracy': r'"accuracy"\s*:\s*([0-9.]+)',
            'loss': r'"loss"\s*:\s*([0-9.]+)', 
            'f1': r'"f1"\s*:\s*([0-9.]+)',
            'precision': r'"precision"\s*:\s*([0-9.]+)',
            'recall': r'"recall"\s*:\s*([0-9.]+)',
            'eval_accuracy': r'"eval_accuracy"\s*:\s*([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                # Try to convert to number
                try:
                    value = float(match.group(1))
                    result[key] = value
                except ValueError:
                    result[key] = match.group(1)
        
        # If we found some data, return it
        if result:
            return json.dumps(result)
        
        # Last resort: try to find any JSON-like structure
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            return json_match.group(0)
        
        return content

    @staticmethod 
    def is_valid_json(content: str) -> bool:
        """Check if JSON is valid"""
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def fix_json_file(file_path: str) -> bool:
        """Fix JSON file in place"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if JSONFixer.is_valid_json(content):
                return True
                
            fixed_data = JSONFixer.fix_json_content(content)
            if fixed_data is not None:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(fixed_data, f, indent=2, ensure_ascii=False)
                return True
            
        except Exception as e:
            print(f"Error fixing JSON file {file_path}: {e}")
        
        return False


def test_json_fixer():
    """Test JSON repair functionality"""
    test_cases = [
        '{"accuracy": 0.95, "loss": 0.1',  # Missing closing brace
        '{"accuracy": 0.95, "loss": 0.1,',  # Trailing comma
        '{"accuracy": 0.95, "los',  # Truncated
        '{"accuracy": 0.95\n"loss": 0.1}',  # Missing comma
    ]
    
    for i, case in enumerate(test_cases):
        print(f"Test case {i+1}: {case}")
        result = JSONFixer.fix_json_content(case)
        print(f"Result: {result}")
        print()


if __name__ == "__main__":
    test_json_fixer() 