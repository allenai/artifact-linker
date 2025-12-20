#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_data():
    """Load source data files."""
    with open("./output/artifact_graph_raw_data/readme_summaries.json", "r") as f:
        summaries = json.load(f)

    with open("./output/artifact_graph_raw_data/perfect_model_dataset_metrics_v2_1125.json", "r") as f:
        model_dataset_metrics = json.load(f)

    filtered_results = []
    for result in model_dataset_metrics["results"]:
        dataset_name = result["dataset_id"].split("/")[-1]
        dataset_id = result["dataset_id"]
        model_id = result["model_id"]
        # if dataset_id is a number or "unknown", skip
        if (
            dataset_name.isdigit()
            or dataset_name.lower() == "unknown"
            or model_id.lower() == "unknown"
        ):
            continue
        if dataset_id not in summaries["datasets"]:
            continue
        if "model_info" not in summaries["datasets"][dataset_id]:
            continue
        if summaries["datasets"][dataset_id]["model_info"] is None:
            continue
        if model_id not in summaries["models"]:
            continue
        if "model_info" not in summaries["models"][model_id]:
            continue
        if summaries["models"][model_id]["model_info"] is None:
            continue
        filtered_results.append(result)
    model_dataset_metrics["results"] = filtered_results
    return summaries, model_dataset_metrics


def create_node_mappings(results: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create mappings from model/dataset IDs to node IDs."""
    # Collect unique model and dataset IDs
    model_ids = set()
    dataset_ids = set()

    for result in results:
        model_ids.add(result["model_id"])
        dataset_ids.add(result["dataset_id"])

    # Create node ID mappings (starting from 0)
    model_to_node = {model_id: i for i, model_id in enumerate(sorted(model_ids))}
    dataset_to_node = {
        dataset_id: i + len(model_ids) for i, dataset_id in enumerate(sorted(dataset_ids))
    }

    print(f"Found {len(model_ids)} unique models and {len(dataset_ids)} unique datasets")
    print(f"Total nodes: {len(model_ids) + len(dataset_ids)}")

    return model_to_node, dataset_to_node


def create_node_metadata(
    summaries: Dict,
    results: List[Dict],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
) -> Dict[int, Dict[str, Any]]:
    """Create node metadata dictionary."""
    node_metadata = {}

    # Create model node metadata
    model_downloads = {}
    for result in results:
        model_id = result["model_id"]
        model_downloads[model_id] = result["model_downloads"]

    for model_id, node_id in model_to_node.items():
        model_summary = summaries.get("models", {}).get(model_id, {})
        node_metadata[node_id] = {
            "type": "model",
            "name": model_id,
            "downloads": model_downloads.get(model_id, 0),
            "info": model_summary.get("model_info", ""),
            "evaluation_results": model_summary.get("evaluation_results", ""),
            "code_example": model_summary.get("code_example", ""),
        }

    # Create dataset node metadata
    dataset_downloads = {}
    for result in results:
        dataset_id = result["dataset_id"]
        dataset_downloads[dataset_id] = result["dataset_downloads"]

    for dataset_id, node_id in dataset_to_node.items():
        dataset_summary = summaries.get("datasets", {}).get(dataset_id, {})
        node_metadata[node_id] = {
            "type": "dataset",
            "name": dataset_id,
            "downloads": dataset_downloads.get(dataset_id, 0),
            "info": dataset_summary.get(
                "model_info", ""
            ),  # Note: using "model_info" key as in original
            "evaluation_results": dataset_summary.get("evaluation_results", ""),
            "code_example": dataset_summary.get("code_example", ""),
        }

    return node_metadata


def create_edges_and_metadata(
    results: List[Dict], model_to_node: Dict[str, int], dataset_to_node: Dict[str, int]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, Any]]]:
    """Create edge list and edge metadata."""
    edges = []
    edge_metadata = {}

    for result in results:
        model_id = result["model_id"]
        dataset_id = result["dataset_id"]

        model_node = model_to_node[model_id]
        dataset_node = dataset_to_node[dataset_id]

        edge = (model_node, dataset_node)
        edges.append(edge)

        # Store edge metadata with edge tuple as key
        filtered_metrics = {}
        for metric_name, metric_value in result["metrics"].items():
            # metric_value should be float or int should not be None or string
            if (
                metric_value is None
                or isinstance(metric_value, str)
                or isinstance(metric_value, dict)
            ):
                continue
            filtered_metrics[metric_name] = metric_value
        edge_meta = {"model_id": model_id, "dataset_id": dataset_id, "metrics": filtered_metrics}
        edge_metadata[edge] = edge_meta

    print(f"Created {len(edges)} edges")
    return edges, edge_metadata


def generate_node_embeddings(
    num_nodes: int, embedding_dim: int = 128, seed: int = 42
) -> np.ndarray:
    """Generate random node embeddings."""
    np.random.seed(seed)

    # Generate random embeddings with normal distribution
    embeddings = np.random.normal(0, 1, (num_nodes, embedding_dim)).astype(np.float32)

    # Normalize embeddings to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

    print(f"Generated random embeddings: shape {embeddings.shape}, normalized to unit length")
    return embeddings


def save_graph_data(
    node_metadata: Dict[int, Dict[str, Any]],
    edges: List[Tuple[int, int]],
    edge_metadata: Dict[Tuple[int, int], Dict[str, Any]],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
    node_embeddings: np.ndarray = None,
    output_dir: str = "./output/artifact_graph_data_v2_1125",
):
    """Save graph data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save node metadata as JSON (keys will be converted to strings by JSON)
    with open(output_path / "node_metadata.json", "w") as f:
        json.dump(node_metadata, f, indent=2)
    print(f"Saved node metadata to {output_path / 'node_metadata.json'}")
    print("Note: JSON converts integer keys to strings automatically")

    # Save edges as NPZ
    edges_array = np.array(edges, dtype=np.int32)
    np.savez_compressed(output_path / "edges.npz", edges=edges_array)
    print(f"Saved edges to {output_path / 'edges.npz'} (shape: {edges_array.shape})")

    # Save edge metadata as JSON (convert tuple keys to strings)
    edge_metadata_serializable = {f"{k[0]},{k[1]}": v for k, v in edge_metadata.items()}
    with open(output_path / "edge_metadata.json", "w") as f:
        json.dump(edge_metadata_serializable, f, indent=2)
    print(f"Saved edge metadata to {output_path / 'edge_metadata.json'}")

    # Save node embeddings if provided
    if node_embeddings is not None:
        np.savez_compressed(output_path / "node_embeddings.npz", embeddings=node_embeddings)
        print(
            f"Saved node embeddings to {output_path / 'node_embeddings.npz'} (shape: {node_embeddings.shape})"
        )

    # Save node ID mappings for reference
    node_mappings = {
        "model_to_node": {model_id: int(node_id) for model_id, node_id in model_to_node.items()},
        "dataset_to_node": {
            dataset_id: int(node_id) for dataset_id, node_id in dataset_to_node.items()
        },
    }
    with open(output_path / "node_mappings.json", "w") as f:
        json.dump(node_mappings, f, indent=2)
    print(f"Saved node mappings to {output_path / 'node_mappings.json'}")


def main(embedding_dim: int = 128, generate_embeddings: bool = True):
    """Main conversion function."""
    print("Loading data...")
    summaries, model_dataset_metrics = load_data()

    results = model_dataset_metrics["results"]
    print(f"Processing {len(results)} model-dataset pairs...")

    # Create node mappings
    model_to_node, dataset_to_node = create_node_mappings(results)
    total_nodes = len(model_to_node) + len(dataset_to_node)

    # Create node metadata
    print("Creating node metadata...")
    node_metadata = create_node_metadata(summaries, results, model_to_node, dataset_to_node)

    # Create edges and edge metadata
    print("Creating edges and edge metadata...")
    edges, edge_metadata = create_edges_and_metadata(results, model_to_node, dataset_to_node)

    # Generate node embeddings
    node_embeddings = None
    if generate_embeddings:
        print("Generating random node embeddings...")
        node_embeddings = generate_node_embeddings(total_nodes, embedding_dim)

    # Save all data
    print("Saving graph data...")
    save_graph_data(
        node_metadata, edges, edge_metadata, model_to_node, dataset_to_node, node_embeddings
    )

    print("\nConversion completed!")
    print("Generated files:")
    print("  - node_metadata.json: Node information indexed by node_id")
    print("  - edges.npz: Edge list as numpy array")
    print("  - edge_metadata.json: Edge metadata (metrics) indexed by edge position")
    print("  - node_mappings.json: Mapping from original IDs to node IDs")
    if generate_embeddings:
        print(f"  - node_embeddings.npz: Random node embeddings ({embedding_dim}D)")


if __name__ == "__main__":
    main()
