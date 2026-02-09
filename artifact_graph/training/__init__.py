"""GNN training and evaluation utilities."""
from __future__ import annotations

try:
    from .gnn_link_trainer import (
        GNNLinkTrainer, LinkTrainingConfig, LinkModelConfig,
        build_link_model, load_link_model, LINK_MODEL_TYPES,
    )
    from .gnn_link_evaluator import GNNLinkEvaluator
    from .gnn_attribute_trainer import (
        GNNAttributeTrainer, AttributeTrainingConfig, AttributeModelConfig,
        build_attribute_model, load_attribute_split, load_split_edge_metadata,
    )
    from .gnn_attribute_evaluator import GNNAttributeEvaluator

    GNN_TRAINING_AVAILABLE = True
    __all__ = [
        "GNNLinkTrainer",
        "GNNLinkEvaluator",
        "LinkTrainingConfig",
        "LinkModelConfig",
        "build_link_model",
        "load_link_model",
        "LINK_MODEL_TYPES",
        "GNNAttributeTrainer",
        "GNNAttributeEvaluator",
        "AttributeTrainingConfig",
        "AttributeModelConfig",
        "build_attribute_model",
        "load_attribute_split",
        "load_split_edge_metadata",
    ]
except ImportError as e:
    print(f"Warning: GNN training not available: {e}")
    GNN_TRAINING_AVAILABLE = False
    __all__ = []
