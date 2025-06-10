import os
import json
import random
import networkx as nx
from pyvis.network import Network

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
            
            with open('model_metadata/{}.json'.format(model_id), 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
            
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


def visualize_graph_interactive(G: nx.Graph, output_file="model_dataset_graph.html"):
    """Visualize the bipartite graph using PyVis (interactive HTML)."""
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type")
        color = "#87ceeb" if node_type == MODEL else "#90ee90"
        shape = "box"
        net.add_node(node, label=node, color=color, shape=shape)
    
    for source, target in G.edges():
        net.add_edge(source, target)
    
    net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 18
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
    """)

    try:
        from jinja2 import Template
        import pkg_resources
        template_str = pkg_resources.resource_string('pyvis', 'templates/template.html').decode('utf-8')
        net.template = Template(template_str)
    except:
        pass
    
    net.show(output_file)
    print(f"✅ Graph saved to {output_file}")

def main():
    data_dir = "eval_datasets_json"
    G = load_model_dataset_graph(data_dir)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    #G_sub = get_subgraph(G)
    print(f"Subgraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    visualize_graph_interactive(G)

if __name__ == "__main__":
    main()
