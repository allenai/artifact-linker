from .version import VERSION, VERSION_SHORT

# Import from data.py for backward compatibility
from .data import build_bipartite_graph, nx_to_pyg_data, prepare_link_pred_splits

# Import new modular components
from . import collectors, processors, utils

# For convenience, also export the main classes directly
from .collectors import ModelCollector, DatasetCollector, MetricCollector
from .processors import GraphBuilder, CardProcessor

__all__ = [
    "VERSION",
    "VERSION_SHORT", 
    "build_bipartite_graph",
    "nx_to_pyg_data", 
    "prepare_link_pred_splits",
    "collectors",
    "processors", 
    "utils",
    "ModelCollector",
    "DatasetCollector",
    "MetricCollector",
    "GraphBuilder",
    "CardProcessor",
]
