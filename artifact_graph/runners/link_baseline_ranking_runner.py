#!/usr/bin/env python3
"""Baseline runner for link ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .link_runner import LinkConfig, _run_link_ranking_with_ranker


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from ..models import BaselineLinkRanker

    ranker = BaselineLinkRanker(mode=config.baseline_mode)
    method_name = f"Baseline ({config.baseline_mode})"
    return _run_link_ranking_with_ranker(
        config, output, ranker, method_name, use_parallel=False
    )
