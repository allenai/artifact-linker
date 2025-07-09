# Artifact Graph - Refactored Code Structure

## Overview

This project has been refactored into a modular code structure that separates data collection, processing, model training and other functionalities, improving code maintainability and extensibility.

## New Directory Structure

```
artifact_graph/
├── collectors/              # Data collectors  
│   ├── model_collector.py         # Model information collection (HF + local loading)
│   ├── dataset_collector.py       # Dataset information collection (HF + local loading)
│   └── metric_collector.py        # Metric collection and computation
├── processors/              # Data processors
│   ├── graph_builder.py           # Graph builder
│   └── card_processor.py          # Card processor
├── utils/                   # Utilities
│   ├── data_utils.py              # Data utility functions
│   ├── graph_builder.py           # Graph building utilities
│   ├── graph_visualizer.py        # Graph visualization
│   ├── data_parser.py             # Data parsing utilities
│   ├── arxiv_downloader.py        # ArXiv paper downloader
│   └── hf_collector.py            # HuggingFace collector
├── models/
│   ├── gnn_link_predictor.py      # GNN link predictor
│   ├── gnn_edge_predictor.py      # GNN edge predictor
│   └── llm_link_predictor.py      # LLM link predictor

└── scripts/                 # Scripts (in project root)
    ├── data_collection.py         # Data collection script
    └── llm_inference.py           # LLM inference script
```

## Main Components

### 1. Data Collectors

#### ModelCollector
- **Function**: Collect model metadata information
- **Main methods**: 
  - `collect_model_info(model_id)`: Collect single model information
  - `collect_all_models(min_downloads)`: Collect all model information

#### DatasetCollector
- **Function**: Collect dataset information
- **Main methods**:
  - `collect_dataset_info(min_downloads)`: Collect dataset information

#### MetricCollector
- **Function**: Collect and compute evaluation metrics
- **Main methods**:
  - `collect_evaluation_metrics()`: Compute classification metrics
  - `collect_regression_metrics()`: Compute regression metrics
  - `load_model_dataset_accuracies()`: Load saved accuracy data

### 2. Data Processors

#### GraphBuilder
- **Function**: Build and process graph data
- **Main methods**:
  - `build_bipartite_graph(min_downloads)`: Build bipartite graph
  - `nx_to_pyg_data(G)`: Convert to PyG data
  - `prepare_link_pred_splits(data, val_ratio, test_ratio)`: Prepare link prediction data splits

#### CardProcessor
- **Function**: Process model and dataset card information
- **Main methods**:
  - `prepare_cards()`: Prepare card information

### 3. Models

#### LLM Models
- **File**: `models/llm/llm_link_predictor.py`
- **Function**: Use LLM for link prediction
- **Supported modes**: 
  - `simple`: Simple mode, only use model name and dataset name
  - `neighborhood`: Neighborhood mode, include neighbor information

#### GNN Models
- **File**: `models/gnn/gnn_link_predictor.py`
- **Function**: Use GNN for link prediction

## Usage

### 1. Data Collection

```python
from artifact_graph.collectors import ModelCollector, DatasetCollector, MetricCollector
from artifact_graph.processors import GraphBuilder, CardProcessor

# Initialize collectors
model_collector = ModelCollector("path/to/metadata")
dataset_collector = DatasetCollector("path/to/dataset.json")
metric_collector = MetricCollector("path/to/data")

# Build graph
graph_builder = GraphBuilder(model_collector, dataset_collector, metric_collector)
G = graph_builder.build_bipartite_graph(min_downloads=1000)

# Process card information
card_processor = CardProcessor("path/to/data", "path/to/metadata", "path/to/dataset.json")
model_cards, dataset_cards = card_processor.prepare_cards()
```

### 2. LLM Inference

```python
from artifact_graph.models.llm_link_predictor import OpenAIGPTLinkPredictor

# Initialize LLM predictor
predictor = OpenAIGPTLinkPredictor(model_name="gpt-4o")

# Make predictions
results = predictor.predict(
    edge_pairs=edge_pairs,
    node_names=node_names,
    G=G,
    model_cards=model_cards,
    dataset_cards=dataset_cards,
    mode="neighborhood"
)
```

### 3. Running Scripts

#### Data Collection Script
```bash
python scripts/data_collection.py \
    --data_dir ../data/eval_datasets_json_download_ranks \
    --dataset_json ../data/dataset_info.json \
    --metadata_dir ../data/model_metadata_download_ranks \
    --min_downloads 1000 \
    --output_dir ./processed_data
```

#### LLM Inference Script
```bash
python scripts/llm_inference.py \
    --llm_model gpt-4o \
    --data_dir ../data/eval_datasets_json_download_ranks \
    --dataset_json ../data/dataset_info.json \
    --metadata_dir ../data/model_metadata_download_ranks \
    --min_downloads 1000
```

## Backward Compatibility

To maintain backward compatibility, the original function interfaces are still available:

```python
from artifact_graph import build_bipartite_graph, nx_to_pyg_data, prepare_link_pred_splits

# These functions now use the new modular structure, but the interface remains unchanged
G = build_bipartite_graph(data_dir, dataset_json, metadata_dir, min_downloads)
data = nx_to_pyg_data(G)
data = prepare_link_pred_splits(data, val_ratio, test_ratio)
```

## Advantages

1. **Modular**: Each component has clear responsibilities
2. **Reusable**: Collectors can be reused in different scenarios
3. **Testable**: Each component can be tested independently
4. **Extensible**: Easy to add new data sources or processing methods
5. **Maintainable**: Clear code structure, easy to understand and modify

## Extension Guide

### Adding New Data Collectors

1. Create new collector class in `collectors/` directory
2. Implement necessary methods
3. Export in `collectors/__init__.py`

### Adding New Data Processors

1. Create new processor class in `processors/` directory
2. Implement necessary methods
3. Export in `processors/__init__.py`

### Adding New Models

1. Create new model file in `models/` directory
2. Implement necessary interfaces
3. Export in corresponding `__init__.py` 