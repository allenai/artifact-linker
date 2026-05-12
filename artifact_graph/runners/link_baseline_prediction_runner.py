#!/usr/bin/env python3
"""Baseline runner for link prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .link_runner import LinkConfig, _run_link_prediction_with_predictor


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from ..models import BaselineLinkPredictor

    kwargs = {}
    if config.baseline_mode == "katz":
        kwargs["beta"] = 0.1
    if config.threshold is not None:
        kwargs["threshold"] = config.threshold
    predictor = BaselineLinkPredictor(mode=config.baseline_mode, **kwargs)
    method_name = f"Baseline ({config.baseline_mode})"
    return _run_link_prediction_with_predictor(
        config,
        output,
        predictor,
        method_name,
        use_parallel=False,
        degree_metric_name=config.baseline_mode,
    )
