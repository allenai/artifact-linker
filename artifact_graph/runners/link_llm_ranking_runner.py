#!/usr/bin/env python3
"""LLM runner for link ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .link_runner import LinkConfig, _run_link_ranking_with_ranker


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from ..models import LLMLinkRanker

    retriever = None
    if config.use_rag:
        from ..utils.retriever import CandidateRetriever

        retriever = CandidateRetriever.from_data_dir(
            config.data_dir, strategy=config.rag_strategy, top_k=config.rag_top_k
        )

    ranker = LLMLinkRanker(
        model_name=config.llm_model,
        hop_number=config.hops,
        use_info=config.use_info,
        use_rag=config.use_rag,
        rag_top_k=config.rag_top_k,
        data_dir=config.data_dir,
        retriever=retriever,
    )
    method_name = f"LLM ({config.llm_model})"
    return _run_link_ranking_with_ranker(
        config, output, ranker, method_name, use_parallel=True, include_failed_item=True
    )
