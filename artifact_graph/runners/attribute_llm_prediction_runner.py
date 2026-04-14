#!/usr/bin/env python3
"""LLM runner for attribute prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .attribute_runner import AttributeConfig, _run_attribute_prediction_with_predictor


def run(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from ..models import LLMAttributePredictor

    predictor = LLMAttributePredictor(
        model_name=config.llm_model, hop_number=config.hops, use_info=config.use_info
    )
    method_name = f"LLM ({config.llm_model})"
    return _run_attribute_prediction_with_predictor(
        config, output, predictor, method_name, use_parallel=True, include_failed_item=True
    )
