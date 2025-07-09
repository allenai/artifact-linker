# Import new modular components
from . import collectors, utils

# For convenience, also export the main classes directly
from .collectors import DatasetCollector, MetricCollector, ModelCollector

# Import from data.py for backward compatibility
from .data import build_bipartite_graph, nx_to_pyg_data, prepare_link_pred_splits
from .version import VERSION, VERSION_SHORT

__all__ = [
    "VERSION",
    "VERSION_SHORT",
    "build_bipartite_graph",
    "nx_to_pyg_data",
    "prepare_link_pred_splits",
    "collectors",
    "utils",
    "ModelCollector",
    "DatasetCollector",
    "MetricCollector",
    "GraphBuilder",
    "CardProcessor",
]
