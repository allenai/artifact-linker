from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def _resolve_provider_and_model(model: str) -> Tuple[Optional[str], str, Optional[str]]:
    raw = model.strip()

    if "/" in raw and not raw.startswith("http"):
        prefix, rest = raw.split("/", 1)
        pfx = prefix.strip().lower()
        if pfx == "together":
            return "together_ai", rest.strip(), "TOGETHER_API_KEY"
        if pfx == "openai":
            if rest == ("gpt-oss-120b"):
                return "together_ai", raw, "TOGETHER_API_KEY"
            return "openai", rest.strip(), "OPENAI_API_KEY"
        if pfx == "anthropic":
            return "anthropic", rest.strip(), "ANTHROPIC_API_KEY"
        if pfx == "gemini":
            return "gemini", rest.strip(), "GEMINI_API_KEY"
        if pfx == "x-ai" or pfx == "grok" or pfx == "xai":
            return "xai", rest.strip(), "X_AI_API_KEY"

    return "together_ai", raw, "TOGETHER_API_KEY"


def call_llm(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    agent_name: str = "default_agent",
) -> Dict[str, Any]:
    try:
        import litellm

        litellm.drop_params = True
        provider, normalized_model, api_key_env = _resolve_provider_and_model(model)

        completion_params: Dict[str, Any] = {
            "model": normalized_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 16000,
        }

        if "gpt-5" in normalized_model.lower() or "o3-2025-04-16" in normalized_model.lower():
            del completion_params["temperature"]
            del completion_params["max_tokens"]
        if provider:
            completion_params["custom_llm_provider"] = provider
        if api_key_env and os.getenv(api_key_env):
            completion_params["api_key"] = os.getenv(api_key_env)

        response = litellm.completion(**completion_params)
        content = response.choices[0].message.content
        print(f"✅ LLM ({agent_name}) call successful")
        return {"success": True, "content": content}

    except Exception as e:
        print(f"❌ LLM ({agent_name}) call failed: {e}")
        # Log the exception for debugging purposes
        import traceback

        traceback.print_exc()
        return {"success": False, "content": None, "error": str(e)}
