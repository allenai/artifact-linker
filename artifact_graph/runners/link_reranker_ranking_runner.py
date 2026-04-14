#!/usr/bin/env python3
"""Reranker runner for link ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .link_runner import LinkConfig, _run_link_ranking_with_ranker


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from ..models import RerankerLinkRanker

    ranker = RerankerLinkRanker(
        reranker_model=config.reranker_model,
        hop_number=config.hops,
        use_info=config.use_info,
    )
    method_name = f"Reranker ({config.reranker_model})"
    # All reranker models are local; run sequentially to avoid GPU contention
    use_parallel = False
    return _run_link_ranking_with_ranker(
        config, output, ranker, method_name, use_parallel=use_parallel, include_failed_item=True,
    )
