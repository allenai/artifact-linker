#!/usr/bin/env python3
"""Baseline runner for attribute ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .attribute_runner import AttributeConfig, _run_attribute_ranking_with_ranker


def run(config: AttributeConfig, output: Path) -> Dict[str, Any]:
    from ..models import BaselineAttributeRanker

    ranker = BaselineAttributeRanker(mode=config.baseline_mode)
    method_name = f"Baseline ({config.baseline_mode})"
    return _run_attribute_ranking_with_ranker(
        config, output, ranker, method_name, use_parallel=False
    )
