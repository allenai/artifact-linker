#!/usr/bin/env python3
"""Baseline runner for attribute prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .attribute_runner import AttributeConfig, _run_attribute_prediction_with_predictor


def run(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from ..models import BaselineAttributePredictor

    predictor = BaselineAttributePredictor(mode=config.baseline_mode)
    method_name = f"Baseline ({config.baseline_mode})"
    return _run_attribute_prediction_with_predictor(
        config, output, predictor, method_name, use_parallel=False
    )
