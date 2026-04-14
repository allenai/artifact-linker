from __future__ import annotations

import os
import time
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
        if pfx == "gemma":
            # Route Gemma models through Google GenAI SDK directly
            return "google_genai", rest.strip(), "GOOGLE_API_KEY"
        if pfx == "x-ai" or pfx == "grok" or pfx == "xai":
            return "xai", rest.strip(), "X_AI_API_KEY"
        if pfx == "vllm":
            return "vllm", rest.strip(), None

    return "together_ai", raw, "TOGETHER_API_KEY"


def _call_google_genai(
    messages: List[Dict[str, str]],
    model_name: str,
    agent_name: str,
    api_key: Optional[str],
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Dict[str, Any]:
    """Call Google GenAI SDK directly for Gemma (and other Google-hosted) models."""
    try:
        from google import genai

        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            return {
                "success": False,
                "content": None,
                "error": "GOOGLE_API_KEY (or GEMINI_API_KEY) not set",
            }

        client = genai.Client(api_key=key)

        # Flatten messages into a single prompt string for generate_content
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System] {content}")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant] {content}")
            else:
                prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    print(f"✅ LLM ({agent_name}) call successful [google_genai: {model_name}]")
                    return {"success": True, "content": response.text}
                else:
                    raise ValueError("Empty response from Google GenAI API")
            except Exception as e:
                print(
                    f"❌ LLM ({agent_name}) call failed on attempt "
                    f"{attempt + 1}/{max_retries} [google_genai: {model_name}]: {e}"
                )
                if attempt + 1 == max_retries:
                    import traceback
                    traceback.print_exc()
                    return {"success": False, "content": None, "error": str(e)}
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    except ImportError:
        return {
            "success": False,
            "content": None,
            "error": "google-genai package not installed. Run: pip install google-genai",
        }
    except Exception as e:
        print(f"❌ LLM ({agent_name}) setup failed [google_genai]: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "content": None, "error": str(e)}


def call_llm(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    agent_name: str = "default_agent",
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Dict[str, Any]:
    provider, normalized_model, api_key_env = _resolve_provider_and_model(model)

    # Google GenAI SDK path (for gemma/ prefix models)
    if provider == "google_genai":
        api_key = os.getenv(api_key_env) if api_key_env else None
        return _call_google_genai(
            messages, normalized_model, agent_name, api_key, max_retries, retry_delay,
        )

    # Default path: litellm
    try:
        import litellm

        litellm.drop_params = True

        completion_params: Dict[str, Any] = {
            "model": normalized_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 16000,
        }

        if "gpt-5" in normalized_model.lower() or "o3-2025-04-16" in normalized_model.lower():
            del completion_params["temperature"]
            del completion_params["max_tokens"]
        if provider == "vllm":
            completion_params["custom_llm_provider"] = "openai"
            completion_params["api_base"] = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
            completion_params["api_key"] = "dummy"
            del completion_params["max_tokens"]
            completion_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            # Override litellm's context window check for vLLM-served models
            # (vLLM may be configured with a larger --max-model-len than the
            # model's default, but litellm pre-checks against its own DB)
            try:
                litellm.register_model({
                    normalized_model: {
                        "max_tokens": 131072,
                        "max_input_tokens": 131072,
                        "max_output_tokens": 16384,
                        "input_cost_per_token": 0,
                        "output_cost_per_token": 0,
                    }
                })
            except Exception:
                pass
        elif provider:
            completion_params["custom_llm_provider"] = provider
        if api_key_env and os.getenv(api_key_env):
            completion_params["api_key"] = os.getenv(api_key_env)

        for attempt in range(max_retries):
            try:
                response = litellm.completion(**completion_params)
                if response and response.choices:
                    content = response.choices[0].message.content
                    print(f"✅ LLM ({agent_name}) call successful")
                    return {"success": True, "content": content}
                else:
                    raise ValueError("Invalid response from LLM API")
            except Exception as e:
                print(
                    f"❌ LLM ({agent_name}) call failed on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt + 1 == max_retries:
                    # Log the final exception for debugging purposes
                    import traceback

                    traceback.print_exc()
                    return {"success": False, "content": None, "error": str(e)}
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    except Exception as e:
        print(f"❌ LLM ({agent_name}) setup failed: {e}")
        # Log the exception for debugging purposes
        import traceback

        traceback.print_exc()
        return {"success": False, "content": None, "error": str(e)}
