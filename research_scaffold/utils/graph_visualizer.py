import os
import json
import random
import networkx as nx
from pyvis.network import Network

MODEL, DATASET = "model", "dataset"

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