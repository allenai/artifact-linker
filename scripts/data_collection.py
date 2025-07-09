#!/usr/bin/env python3
"""
Data collection script - demonstrates how to use the new modular data collection structure
"""

import argparse
import sys
import os

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from artifact_graph.data.collectors import ModelCollector, DatasetCollector, AccuracyCollector
from artifact_graph.data.processors import GraphBuilder, CardProcessor

def main():
    parser = argparse.ArgumentParser(description="Data collection and preprocessing")
    parser.add_argument('--data_dir', default='../data/eval_datasets_json_download_ranks',
                       help='Directory containing model-dataset mappings')
    parser.add_argument('--dataset_json', default='../data/dataset_info.json',
                       help='Dataset information JSON file')
    parser.add_argument('--metadata_dir', default='../data/model_metadata_download_ranks',
                       help='Model metadata directory')
    parser.add_argument('--min_downloads', type=int, default=1000,
                       help='Minimum download threshold')
    parser.add_argument('--output_dir', default='./processed_data',
                       help='Output directory')
    args = parser.parse_args()
    
    print("=== Starting data collection ===")
    
    # 1. Initialize collectors
    print("Initializing data collectors...")
    model_collector = ModelCollector(args.metadata_dir)
    dataset_collector = DatasetCollector(args.dataset_json)
    accuracy_collector = AccuracyCollector(args.data_dir)
    
    # 2. Collect data
    print("Collecting model information...")
    models = model_collector.collect_all_models(args.min_downloads)
    print(f"Collected {len(models)} models")
    
    print("Collecting dataset information...")
    datasets = dataset_collector.collect_dataset_info(args.min_downloads)
    print(f"Collected {len(datasets)} datasets")
    
    print("Collecting accuracy information...")
    accuracies = accuracy_collector.collect_model_dataset_accuracies()
    print(f"Collected {len(accuracies)} accuracy records")
    
    # 3. Build graph
    print("Building bipartite graph...")
    graph_builder = GraphBuilder(model_collector, dataset_collector, accuracy_collector)
    G = graph_builder.build_bipartite_graph(args.min_downloads)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # 4. Process card information
    print("Processing card information...")
    card_processor = CardProcessor(args.data_dir, args.metadata_dir, args.dataset_json)
    model_cards, dataset_cards = card_processor.prepare_cards()
    print(f"Processed {len(model_cards)} model cards and {len(dataset_cards)} dataset cards")
    
    # 5. Convert to PyG data
    print("Converting to PyG data...")
    data = graph_builder.nx_to_pyg_data(G)
    print(f"PyG data contains {data.x.size(0)} node features with dimension {data.x.size(1)}")
    
    # 6. Prepare link prediction splits
    print("Preparing link prediction data splits...")
    data = graph_builder.prepare_link_pred_splits(data, val_ratio=0.1, test_ratio=0.1)
    print("Data splitting completed")
    
    # 7. Save processed data
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    import torch
    torch.save(data, os.path.join(args.output_dir, 'processed_data.pt'))
    torch.save(model_cards, os.path.join(args.output_dir, 'model_cards.pt'))
    torch.save(dataset_cards, os.path.join(args.output_dir, 'dataset_cards.pt'))
    
    print(f"Data saved to {args.output_dir}")
    print("=== Data collection completed ===")
    
    # 8. Output statistics
    print("\n=== Statistics ===")
    print(f"Number of models: {len(models)}")
    print(f"Number of datasets: {len(datasets)}")
    print(f"Number of accuracy records: {len(accuracies)}")
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    
    # Count model and dataset types
    model_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'model']
    dataset_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'dataset']
    print(f"Model nodes: {len(model_nodes)}")
    print(f"Dataset nodes: {len(dataset_nodes)}")

if __name__ == "__main__":
    main() 