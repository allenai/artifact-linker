#!/usr/bin/env python3
"""LLM runner for attribute ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .attribute_runner import AttributeConfig, _run_attribute_ranking_with_ranker


def run(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from ..models import LLMAttributeRanker

    ranker = LLMAttributeRanker(
        model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info
    )
    method_name = f"LLM ({config.llm_model})"
    return _run_attribute_ranking_with_ranker(
        config, output, ranker, method_name, use_parallel=True, include_failed_item=True
    )
