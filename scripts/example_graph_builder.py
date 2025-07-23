import os
from pathlib import Path

from artifact_graph.utils.graph_builder import load_artifact_graph
from artifact_graph.utils.graph_visualizer import visualize_graph_interactive

MODEL_NODE = "model"
DATASET_NODE = "dataset"


def main():
    base = Path("output")
    (base / "models/metadata").mkdir(parents=True, exist_ok=True)
    (base / "datasets/metadata").mkdir(parents=True, exist_ok=True)
    (base / "metrics").mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    G = load_artifact_graph(
        models_dir="output/models/metadata",
        datasets_dir="output/datasets/metadata",
        metrics_dir="output/metrics",
        hf_token=hf_token,
    )

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    model_count = sum(1 for _, attr in G.nodes(data=True) if attr.get("type") == MODEL_NODE)
    dataset_count = total_nodes - model_count

    print(f"Graph loaded with {total_nodes} nodes and {total_edges} edges.")
    print(f"✓ Model nodes: {model_count}")
    print(f"✓ Dataset nodes: {dataset_count}")

    visualize_graph_interactive(G)


if __name__ == "__main__":
    main()
