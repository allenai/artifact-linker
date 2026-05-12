#!/usr/bin/env python3
"""Reranker runner for link prediction — **batched** scoring.

Instead of calling ``score_single`` 300 k+ times, this runner:
1. Builds all (query, document) text pairs up-front.
2. Calls ``reranker.score_pairs`` once (internally mini-batched on GPU).
3. Assembles the prediction rows from the returned scores.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from .link_runner import LinkConfig

from ..utils.link_prediction_utils import (
    collect_link_predictions,
    create_link_prediction_row,
    load_link_prediction_data,
    print_link_prediction_metrics,
    save_link_predictions,
)


def run(config: LinkConfig, output: Path) -> Dict[str, Any]:
    from ..models import RerankerLinkPredictor

    predictor = RerankerLinkPredictor(
        reranker_model=config.reranker_model,
        hop_number=config.hops,
        use_info=config.use_info,
    )

    method_name = f"Reranker ({config.reranker_model})"

    # Load data
    G, meta, edges, labels = load_link_prediction_data(config.split_dir, config.seed)
    if not edges:
        print("No edges to predict.")
        return {"status": "empty"}

    print(f"Building text pairs for {len(edges)} edges [{method_name}] ...")

    # 1. Build all (query, document) pairs
    pairs = []
    for (m, d), _label in tqdm(zip(edges, labels), total=len(edges), desc="Building pairs"):
        q, doc = predictor.build_pair(m, d, G, meta)
        pairs.append((q, doc))

    # 2. Batch score
    print(f"Scoring {len(pairs)} pairs in batch ...")
    scores = predictor.reranker.score_pairs(pairs, batch_size=256)

    # 3. Assemble prediction rows
    predictions = []
    for idx, ((m, d), label) in enumerate(zip(edges, labels)):
        row = create_link_prediction_row(m, d, label, meta)
        s = scores[idx]
        row.update(
            predicted_label=1 if s >= predictor.threshold else 0,
            reason=f"reranker score={s:.4f}",
            status="Success",
            score=float(s),
        )
        predictions.append(row)

    y_true, y_pred, y_score = collect_link_predictions(predictions)
    print_link_prediction_metrics(y_true, y_pred, y_score, method_name)
    save_link_predictions(predictions, output)

    return {"predictions": predictions, "output": str(output)}
