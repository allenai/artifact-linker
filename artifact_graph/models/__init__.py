from .baseline_attribute_predictor import BaselineAttributePredictor
from .baseline_attribute_ranker import BaselineAttributeRanker
from .baseline_link_ranker import BaselineLinkRanker
from .llm_attribute_predictor import LLMAttributePredictor
from .llm_attribute_ranker import LLMAttributeRanker
from .llm_link_predictor import LLMLinkPredictor
from .llm_link_ranker import LLMLinkRanker
from .random_baseline import RandomBaseline

# GNN models
try:
    from .gnn_link_predictor import GNNLinkPredictor

    GNN_AVAILABLE = True
    __all__ = [
        "LLMLinkPredictor",
        "LLMAttributePredictor", 
        "LLMLinkRanker",
        "LLMAttributeRanker",
        "BaselineLinkRanker",
        "BaselineAttributePredictor",
        "BaselineAttributeRanker",
        "RandomBaseline",
        "GNNLinkPredictor",
    ]
except ImportError as e:
    print(f"Warning: GNN models not available: {e}")
    GNN_AVAILABLE = False
    __all__ = [
        "LLMLinkPredictor",
        "LLMAttributePredictor",
        "LLMLinkRanker", 
        "LLMAttributeRanker",
        "BaselineLinkRanker",
        "BaselineAttributePredictor",
        "BaselineAttributeRanker",
        "RandomBaseline",
    ]
