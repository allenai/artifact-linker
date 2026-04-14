#!/usr/bin/env python3
"""LLM runner for link prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .link_runner import LinkConfig, _run_link_prediction_with_predictor


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from ..models import LLMLinkPredictor

    predictor = LLMLinkPredictor(
        model_name=config.llm_model,
        hop_number=config.hops,
        use_info=config.use_info,
        use_rag=config.use_rag,
        rag_top_k=config.rag_top_k,
        rag_strategy=config.rag_strategy,
        data_dir=config.data_dir,
    )
    method_name = f"LLM ({config.llm_model})"
    return _run_link_prediction_with_predictor(
        config,
        output,
        predictor,
        method_name,
        use_parallel=True,
        include_failed_item=True,
    )
