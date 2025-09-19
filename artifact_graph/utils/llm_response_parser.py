from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional


def parse_llm_response_to_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response content to JSON, handling various response formats.
    
    This function can handle:
    - JSON wrapped in ```json code blocks
    - Raw JSON objects in the response
    - Responses with <think>...</think> blocks that need to be removed
    
    Args:
        content: The raw LLM response content
        
    Returns:
        Parsed JSON dictionary if successful, None otherwise
    """
    try:
        # Remove <think>...</think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        if "```json" in content:
            # Extract JSON from code blocks
            json_str = content.split("```json")[1].split("```")[0].strip()
        else:
            # Find JSON object in the content
            content = content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            json_str = content[start:end]
        
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None
