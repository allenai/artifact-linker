#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def load_data(summaries_path: str, metrics_path: str, skip_filter: bool = False):
    """Load source data files."""
    with open(summaries_path, "r") as f:
        summaries = json.load(f)

    with open(metrics_path, "r") as f:
        model_dataset_metrics = json.load(f)

    if skip_filter:
        # Skip filtering, keep all results
        return summaries, model_dataset_metrics

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
    model_ids = set()
    dataset_ids = set()

    for result in results:
        model_ids.add(result["model_id"])
        dataset_ids.add(result["dataset_id"])

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
            "info": dataset_summary.get("model_info", ""),
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

        filtered_metrics = {}
        for metric_name, metric_value in result["metrics"].items():
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


# =============================================================================
# Embedding Generation Functions
# =============================================================================

def generate_random_embeddings(num_nodes: int, embedding_dim: int = 768, seed: int = 42) -> np.ndarray:
    """Generate random node embeddings."""
    np.random.seed(seed)
    embeddings = np.random.normal(0, 1, (num_nodes, embedding_dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    print(f"Generated random embeddings: shape {embeddings.shape}")
    return embeddings


def generate_voyage_embeddings(
    node_metadata: Dict[int, Dict[str, Any]],
    model_name: str = "voyage-3",
    batch_size: int = 128,
) -> np.ndarray:
    """Generate embeddings using Voyage AI API."""
    try:
        import voyageai
    except ImportError:
        raise ImportError("Please install voyageai: pip install voyageai")
    
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    
    client = voyageai.Client(api_key=api_key)
    
    # Prepare texts in node order
    num_nodes = len(node_metadata)
    texts = []
    for node_id in range(num_nodes):
        meta = node_metadata.get(node_id, {})
        text = meta.get("info", "") or meta.get("name", f"node_{node_id}")
        # Truncate to avoid token limits
        texts.append(text[:8000])
    
    print(f"Generating Voyage embeddings for {num_nodes} nodes using {model_name}...")
    
    all_embeddings = []
    for i in tqdm(range(0, num_nodes, batch_size), desc="Voyage API"):
        batch_texts = texts[i:i + batch_size]
        result = client.embed(batch_texts, model=model_name, input_type="document")
        all_embeddings.extend(result.embeddings)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Generated Voyage embeddings: shape {embeddings.shape}")
    return embeddings


def generate_google_embeddings(
    node_metadata: Dict[int, Dict[str, Any]],
    model_name: str = "text-embedding-004",
    batch_size: int = 100,
) -> np.ndarray:
    """Generate embeddings using Google Generative AI API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Prepare texts in node order
    num_nodes = len(node_metadata)
    texts = []
    for node_id in range(num_nodes):
        meta = node_metadata.get(node_id, {})
        text = meta.get("info", "") or meta.get("name", f"node_{node_id}")
        texts.append(text[:8000])
    
    print(f"Generating Google embeddings for {num_nodes} nodes using {model_name}...")
    
    all_embeddings = []
    for i in tqdm(range(0, num_nodes, batch_size), desc="Google API"):
        batch_texts = texts[i:i + batch_size]
        result = genai.embed_content(
            model=f"models/{model_name}",
            content=batch_texts,
            task_type="RETRIEVAL_DOCUMENT",
        )
        all_embeddings.extend(result["embedding"])
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Generated Google embeddings: shape {embeddings.shape}")
    return embeddings


def generate_openai_embeddings(
    node_metadata: Dict[int, Dict[str, Any]],
    model_name: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> np.ndarray:
    """Generate embeddings using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    num_nodes = len(node_metadata)
    texts = []
    for node_id in range(num_nodes):
        meta = node_metadata.get(node_id, {})
        text = meta.get("info", "") or meta.get("name", f"node_{node_id}")
        texts.append(text[:8000])
    
    print(f"Generating OpenAI embeddings for {num_nodes} nodes using {model_name}...")
    
    all_embeddings = []
    for i in tqdm(range(0, num_nodes, batch_size), desc="OpenAI API"):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch_texts, model=model_name)
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Generated OpenAI embeddings: shape {embeddings.shape}")
    return embeddings


def generate_embeddings(
    node_metadata: Dict[int, Dict[str, Any]],
    embedding_type: str = "random",
    embedding_dim: int = 768,
) -> np.ndarray:
    """Generate embeddings based on the specified type."""
    num_nodes = len(node_metadata)
    
    if embedding_type == "random":
        return generate_random_embeddings(num_nodes, embedding_dim)
    elif embedding_type == "voyage":
        return generate_voyage_embeddings(node_metadata)
    elif embedding_type == "google":
        return generate_google_embeddings(node_metadata)
    elif embedding_type == "openai":
        return generate_openai_embeddings(node_metadata)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# =============================================================================
# Save Functions
# =============================================================================

def save_graph_data(
    node_metadata: Dict[int, Dict[str, Any]],
    edges: List[Tuple[int, int]],
    edge_metadata: Dict[Tuple[int, int], Dict[str, Any]],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
    node_embeddings: Optional[np.ndarray] = None,
    output_dir: str = "./output/artifact_graph_data_v2_1125",
    embedding_type: str = "random",
):
    """Save graph data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / "node_metadata.json", "w") as f:
        json.dump(node_metadata, f, indent=2)
    print(f"Saved node metadata to {output_path / 'node_metadata.json'}")

    edges_array = np.array(edges, dtype=np.int32)
    np.savez_compressed(output_path / "edges.npz", edges=edges_array)
    print(f"Saved edges to {output_path / 'edges.npz'} (shape: {edges_array.shape})")

    edge_metadata_serializable = {f"{k[0]},{k[1]}": v for k, v in edge_metadata.items()}
    with open(output_path / "edge_metadata.json", "w") as f:
        json.dump(edge_metadata_serializable, f, indent=2)
    print(f"Saved edge metadata to {output_path / 'edge_metadata.json'}")

    if node_embeddings is not None:
        # Save with embedding type in filename for clarity
        emb_filename = f"node_embeddings_{embedding_type}.npy"
        # Save as structured array with node_id and embedding
        dtype = np.dtype([('node_id', np.int32), ('embedding', np.float32, (node_embeddings.shape[1],))])
        structured_arr = np.zeros(len(node_embeddings), dtype=dtype)
        for i in range(len(node_embeddings)):
            structured_arr[i]['node_id'] = i
            structured_arr[i]['embedding'] = node_embeddings[i]
        np.save(output_path / emb_filename, structured_arr)
        print(f"Saved node embeddings to {output_path / emb_filename} (shape: {node_embeddings.shape})")
        
        # Also save as default node_embeddings.npy for compatibility
        np.save(output_path / "node_embeddings.npy", structured_arr)
        print(f"Saved copy to {output_path / 'node_embeddings.npy'}")

    node_mappings = {
        "model_to_node": {model_id: int(node_id) for model_id, node_id in model_to_node.items()},
        "dataset_to_node": {
            dataset_id: int(node_id) for dataset_id, node_id in dataset_to_node.items()
        },
    }
    with open(output_path / "node_mappings.json", "w") as f:
        json.dump(node_mappings, f, indent=2)
    print(f"Saved node mappings to {output_path / 'node_mappings.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert raw data to graph format")
    parser.add_argument(
        "--summaries-json",
        type=str,
        default="./output/artifact_graph_raw_data/readme_summaries_v2_1125.json",
        help="Path to readme_summaries.json",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="./output/artifact_graph_raw_data/perfect_model_dataset_metrics_v2_1125.json",
        help="Path to perfect_model_dataset_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/artifact_graph_data_v2_1125",
        help="Output directory for graph data",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["random", "voyage", "google", "openai", "none"],
        default="random",
        help="Type of embeddings to generate",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (only for random embeddings)",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip filtering (keep all results even without summaries)",
    )
    args = parser.parse_args()

    print("Loading data...")
    summaries, model_dataset_metrics = load_data(
        args.summaries_json, args.metrics_json, args.skip_filter
    )

    results = model_dataset_metrics["results"]
    print(f"Processing {len(results)} model-dataset pairs...")

    model_to_node, dataset_to_node = create_node_mappings(results)

    print("Creating node metadata...")
    node_metadata = create_node_metadata(summaries, results, model_to_node, dataset_to_node)

    print("Creating edges and edge metadata...")
    edges, edge_metadata = create_edges_and_metadata(results, model_to_node, dataset_to_node)

    node_embeddings = None
    if args.embedding_type != "none":
        print(f"Generating {args.embedding_type} embeddings...")
        node_embeddings = generate_embeddings(
            node_metadata, args.embedding_type, args.embedding_dim
        )

    print("Saving graph data...")
    save_graph_data(
        node_metadata,
        edges,
        edge_metadata,
        model_to_node,
        dataset_to_node,
        node_embeddings,
        args.output_dir,
        args.embedding_type,
    )

    print("\nConversion completed!")
    print("Generated files:")
    print("  - node_metadata.json: Node information indexed by node_id")
    print("  - edges.npz: Edge list as numpy array")
    print("  - edge_metadata.json: Edge metadata (metrics)")
    print("  - node_mappings.json: Mapping from original IDs to node IDs")
    if node_embeddings is not None:
        print(f"  - node_embeddings_{args.embedding_type}.npy: {args.embedding_type.capitalize()} embeddings")
        print(f"  - node_embeddings.npy: Copy for compatibility")


if __name__ == "__main__":
    main()
