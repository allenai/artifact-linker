# Minimal imports to avoid dependency issues
try:
    from . import utils
except ImportError:
    # Skip if dependencies are missing
    pass

try:
    from . import collectors
    from .collectors import DatasetCollector, MetricCollector, ModelCollector
except ImportError:
    # Skip if huggingface_hub is missing
    pass

# Import from data.py for backward compatibility
try:
    from .data import build_bipartite_graph, nx_to_pyg_data, prepare_link_pred_splits
except ImportError:
    # Skip if dependencies are missing
    pass

from .version import VERSION, VERSION_SHORT

__all__ = [
    "VERSION",
    "VERSION_SHORT",
]

# Add available modules to __all__
try:
    build_bipartite_graph
    __all__.extend(["build_bipartite_graph", "nx_to_pyg_data", "prepare_link_pred_splits"])
except NameError:
    pass

try:
    collectors
    __all__.extend(["collectors", "utils", "ModelCollector", "DatasetCollector", "MetricCollector"])
except NameError:
    pass
