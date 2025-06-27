import os
import json
import random
import networkx as nx
from .graph_builder import visualize_graph_interactive

MODEL, DATASET = "model", "dataset"

def load_model_dataset_graph(data_dir: str) -> nx.Graph:
    """Load model-dataset JSON files and construct bipartite graph."""
    with open('dataset_info.json', 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    dataset_names = [dataset['id'].split('/')[-1].lower() for dataset in dataset_info if dataset['downloads'] > 1000]

    G = nx.Graph()
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        model_id = fname[:-5]  # remove ".json"
        path = os.path.join(data_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            with open('model_metadata_download_ranks/{}.json'.format(model_id), 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)

            has_paper_or_not = False
            tags = model_metadata.get('tags', [])
            for tag in tags:
                if tag.startswith('arxiv:'):
                    has_paper_or_not = True
                    break
            if not has_paper_or_not:
                print(f"✗ {model_id} does not have a paper.")
                continue

            if model_metadata['downloads'] < 1000:
                continue
            if not isinstance(data, dict):
                continue
        except Exception as e:
            print(f"✗ Failed to load {model_id}: {e}")
            continue

        for dataset in data.keys():
            dataset_name = dataset.split("/")[-1].lower() if "/" in dataset else dataset.lower()
            if dataset_name not in dataset_names:
                continue
            else:
                index = dataset_names.index(dataset_name)

            G.add_node(model_id, type=MODEL)
            G.add_node(dataset_names[index], type=DATASET)
            G.add_edge(model_id, dataset_names[index])
    return G

def get_subgraph(G: nx.Graph, models_per_dataset=3) -> nx.Graph:
    """
    Keep all dataset nodes, and for each dataset node, sample a few connected model nodes.
    This ensures all data nodes are included and visualization is meaningful.
    """
    dataset_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == DATASET]
    selected_nodes = set(dataset_nodes)

    for dataset in dataset_nodes:
        neighbors = list(G.neighbors(dataset))
        sampled_models = random.sample(neighbors, min(models_per_dataset, len(neighbors)))
        selected_nodes.update(sampled_models)

    return G.subgraph(selected_nodes).copy()


def main():
    data_dir = "eval_datasets_json_download_ranks"
    G = load_model_dataset_graph(data_dir)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Count model and dataset nodes
    types = nx.get_node_attributes(G, "type")
    model_nodes = [n for n, t in types.items() if t == MODEL]
    dataset_nodes = [n for n, t in types.items() if t == DATASET]
    print(f"✓ Model nodes: {len(model_nodes)}")
    print(f"✓ Dataset nodes: {len(dataset_nodes)}")

    # G_sub = get_subgraph(G)
    visualize_graph_interactive(G)


if __name__ == "__main__":
    main()
