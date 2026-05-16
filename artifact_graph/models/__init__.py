from .baseline_attribute_predictor import BaselineAttributePredictor
from .baseline_attribute_ranker import BaselineAttributeRanker
from .baseline_link_predictor import BaselineLinkPredictor
from .baseline_link_ranker import BaselineLinkRanker
from .llm_attribute_predictor import LLMAttributePredictor
from .llm_attribute_ranker import LLMAttributeRanker
from .llm_link_predictor import LLMLinkPredictor
from .llm_link_ranker import LLMLinkRanker
from .reranker_link_predictor import RerankerLinkPredictor
from .reranker_link_ranker import RerankerLinkRanker
from .random_baseline import RandomBaseline

# GNN models (architecture only – trainers/evaluators are in artifact_graph.training)
try:
    from .gnn_link_predictor import GNNLinkPredictor
    from .ncn_link_predictor import NCNLinkPredictor
    from .neognn_link_predictor import NeoGNNLinkPredictor
    from .buddy_link_predictor import BUDDYLinkPredictor

    GNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GNN models not available: {e}")
    GNN_AVAILABLE = False

__all__ = [
    "LLMLinkPredictor",
    "LLMAttributePredictor",
    "LLMLinkRanker",
    "LLMAttributeRanker",
    "RerankerLinkPredictor",
    "RerankerLinkRanker",
    "BaselineLinkPredictor",
    "BaselineLinkRanker",
    "BaselineAttributePredictor",
    "BaselineAttributeRanker",
    "RandomBaseline",
]

if GNN_AVAILABLE:
    __all__.extend([
        "GNNLinkPredictor",
        "NCNLinkPredictor",
        "NeoGNNLinkPredictor",
        "BUDDYLinkPredictor",
    ])
