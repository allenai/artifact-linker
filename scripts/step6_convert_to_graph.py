#!/usr/bin/env python3
"""
Step 6: Convert normalized eval pairs + summaries into graph format for GNN training.

Node types (ordered):
  0 .. N-1          : models
  N .. N+M-1        : datasets
  N+M .. N+M+P-1    : papers
  N+M+P .. N+M+P+C-1: codebases

Edge files (separate to support heterogeneous GNNs):
  edges_eval.npz          — model → dataset, with eval metrics
  edges_resource.npz      — model/dataset → paper/codebase (has_paper, has_codebase)

Input:
  - data/artifact_raw_data/normalized_eval_pairs.json  (step 5)
  - data/artifact_raw_data/readme_summaries.json       (step 5)
  - data/artifact_raw_data/node_embeddings_*.npy       (step 5)
  - data/artifact_raw_data/filtered_eval_pairs.json    (step 3)
  - data/artifact_raw_data/resource_links.json         (step 4)
  - data/artifact_raw_data/papers/metadata/            (step 4)
  - data/artifact_raw_data/codebases/metadata/         (step 4)

Output (data/artifact_graph_data_v3/):
  - node_metadata.json
  - node_mappings.json
  - edges_eval.npz  /  edge_metadata_eval.json
  - edges_resource.npz  /  edge_metadata_resource.json
  - node_embeddings_*.npy
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_paper_metadata(papers_metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all fetched paper metadata keyed by arxiv_id."""
    result: Dict[str, Dict[str, Any]] = {}
    if not papers_metadata_dir.exists():
        return result
    for f in papers_metadata_dir.glob("*.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            arxiv_id = meta.get("arxiv_id") or f.stem.replace("__", "/")
            result[arxiv_id] = meta
        except Exception:
            continue
    return result


def load_base_model_map(models_metadata_dir: Path) -> Dict[str, List[str]]:
    """Load base_model relationships from model metadata. Returns {child_model: [parent_models]}."""
    result: Dict[str, List[str]] = {}
    if not models_metadata_dir.exists():
        return result
    for f in models_metadata_dir.glob("*.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            model_id = meta.get("modelId") or f.stem.replace("__", "/")
            base = meta.get("baseModel")
            if base:
                bases = base if isinstance(base, list) else [base]
                result[model_id] = [b for b in bases if b]
        except Exception:
            continue
    return result


def load_codebase_metadata(codebases_metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all fetched codebase metadata keyed by owner/repo."""
    result: Dict[str, Dict[str, Any]] = {}
    if not codebases_metadata_dir.exists():
        return result
    for f in codebases_metadata_dir.glob("*.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            repo = meta.get("full_name") or f.stem.replace("__", "/")
            result[repo] = meta
        except Exception:
            continue
    return result


def build_download_maps(
    filtered_pairs_path: Path,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    data = load_json(filtered_pairs_path)
    model_dl: Dict[str, int] = {}
    dataset_dl: Dict[str, int] = {}
    for p in data["results"]:
        mid, did = p["model_id"], p["dataset_id"]
        if mid not in model_dl:
            model_dl[mid] = 0
        if did not in dataset_dl:
            dataset_dl[did] = p.get("dataset_downloads", 0)
    return model_dl, dataset_dl


# ──────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────

def _has_readme_and_metadata(
    artifact_id: str,
    readme_dir: Optional[Path],
    metadata_dir: Optional[Path],
) -> bool:
    """Check if an artifact has both a README and metadata file."""
    if readme_dir is None or metadata_dir is None:
        return True  # No filtering if dirs not provided
    safe_name = artifact_id.replace("/", "__")
    has_readme = (readme_dir / f"{safe_name}.md").exists()
    has_meta = (metadata_dir / f"{safe_name}.json").exists()
    return has_readme and has_meta


def collect_eval_ids(
    edges: Dict[str, Dict[str, Any]],
    model_readme_dir: Optional[Path] = None,
    model_metadata_dir: Optional[Path] = None,
    dataset_readme_dir: Optional[Path] = None,
    dataset_metadata_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Collect sorted unique model/dataset IDs from normalized eval edges.

    Filters out models/datasets that lack both a README and metadata file.
    """
    model_ids: Set[str] = set()
    dataset_ids: Set[str] = set()
    for edge_data in edges.values():
        model_ids.add(edge_data["model_id"])
        dataset_ids.add(edge_data["dataset_id"])

    # Filter: keep only models/datasets with both README and metadata
    if model_readme_dir and model_metadata_dir:
        before = len(model_ids)
        model_ids = {
            mid for mid in model_ids
            if _has_readme_and_metadata(mid, model_readme_dir, model_metadata_dir)
        }
        print(f"  Filtered models: {before} → {len(model_ids)} (removed {before - len(model_ids)} without README+metadata)")

    if dataset_readme_dir and dataset_metadata_dir:
        before = len(dataset_ids)
        dataset_ids = {
            did for did in dataset_ids
            if _has_readme_and_metadata(did, dataset_readme_dir, dataset_metadata_dir)
        }
        print(f"  Filtered datasets: {before} → {len(dataset_ids)} (removed {before - len(dataset_ids)} without README+metadata)")

    return sorted(model_ids), sorted(dataset_ids)


def collect_resource_ids(
    resource_links: Dict[str, Any],
    graph_model_ids: Set[str],
    graph_dataset_ids: Set[str],
    paper_metadata: Dict[str, Any],
    codebase_metadata: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Collect paper/codebase IDs that are referenced by models/datasets
    already in the eval graph. Papers must have title (from metadata).
    Codebases must have description (from metadata).
    """
    paper_ids: Set[str] = set()
    codebase_ids: Set[str] = set()

    # Papers with valid title
    valid_papers = {pid for pid, m in paper_metadata.items() if m.get("title", "").strip()}

    # Codebases with valid description
    valid_codebases = {cid for cid, m in codebase_metadata.items() if m.get("description", "").strip()}

    for artifact_id, links in resource_links.get("models", {}).items():
        if artifact_id not in graph_model_ids:
            continue
        for pid in links.get("arxiv_ids", []):
            if pid in valid_papers:
                paper_ids.add(pid)
        for cid in links.get("github_repos", []):
            if cid in valid_codebases:
                codebase_ids.add(cid)

    for artifact_id, links in resource_links.get("datasets", {}).items():
        if artifact_id not in graph_dataset_ids:
            continue
        for pid in links.get("arxiv_ids", []):
            if pid in valid_papers:
                paper_ids.add(pid)
        for cid in links.get("github_repos", []):
            if cid in valid_codebases:
                codebase_ids.add(cid)

    return sorted(paper_ids), sorted(codebase_ids)


def create_node_mappings(
    model_ids: List[str],
    dataset_ids: List[str],
    paper_ids: List[str],
    codebase_ids: List[str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Assign global node indices.
    Order: models | datasets | papers | codebases
    """
    offset = 0
    model_to_node = {mid: i + offset for i, mid in enumerate(model_ids)}
    offset += len(model_ids)
    dataset_to_node = {did: i + offset for i, did in enumerate(dataset_ids)}
    offset += len(dataset_ids)
    paper_to_node = {pid: i + offset for i, pid in enumerate(paper_ids)}
    offset += len(paper_ids)
    codebase_to_node = {cid: i + offset for i, cid in enumerate(codebase_ids)}
    return model_to_node, dataset_to_node, paper_to_node, codebase_to_node


def create_node_metadata(
    model_ids: List[str],
    dataset_ids: List[str],
    paper_ids: List[str],
    codebase_ids: List[str],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
    paper_to_node: Dict[str, int],
    codebase_to_node: Dict[str, int],
    summaries: Dict[str, Dict[str, Any]],
    model_downloads: Dict[str, int],
    dataset_downloads: Dict[str, int],
    paper_metadata: Dict[str, Dict[str, Any]],
    codebase_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}

    for mid in model_ids:
        nid = model_to_node[mid]
        summary = summaries.get("models", {}).get(mid, {})
        info = summary.get("info", "") if isinstance(summary, dict) else str(summary)
        metadata[str(nid)] = {
            "type": "model",
            "name": mid,
            "downloads": model_downloads.get(mid, 0),
            "info": info,
        }

    for did in dataset_ids:
        nid = dataset_to_node[did]
        summary = summaries.get("datasets", {}).get(did, {})
        info = summary.get("info", "") if isinstance(summary, dict) else str(summary)
        metadata[str(nid)] = {
            "type": "dataset",
            "name": did,
            "downloads": dataset_downloads.get(did, 0),
            "info": info,
        }

    for pid in paper_ids:
        nid = paper_to_node[pid]
        summary = summaries.get("papers", {}).get(pid, {})
        info = summary.get("info", "") if isinstance(summary, dict) else str(summary)
        pmeta = paper_metadata.get(pid, {})
        # Fallback: build info from title + abstract when summary is missing
        if not info.strip() and pmeta:
            title = pmeta.get("title", "").strip()
            abstract = pmeta.get("abstract", "").strip()
            cats = pmeta.get("categories", [])
            parts = []
            if title:
                parts.append(title)
            if cats:
                parts.append(f"[{', '.join(cats)}]")
            if abstract:
                parts.append(abstract)
            info = " ".join(parts)
        metadata[str(nid)] = {
            "type": "paper",
            "name": pid,
            "title": pmeta.get("title", ""),
            "authors": pmeta.get("authors", []),
            "published": pmeta.get("published", ""),
            "categories": pmeta.get("categories", []),
            "info": info,
        }

    for cid in codebase_ids:
        nid = codebase_to_node[cid]
        summary = summaries.get("codebases", {}).get(cid, {})
        info = summary.get("info", "") if isinstance(summary, dict) else str(summary)
        cmeta = codebase_metadata.get(cid, {})
        # Fallback: build info from GitHub metadata when summary is missing
        if not info.strip() and cmeta:
            desc = cmeta.get("description", "").strip()
            lang = cmeta.get("language", "").strip()
            topics = cmeta.get("topics", [])
            parts = []
            if desc:
                parts.append(desc)
            if lang:
                parts.append(f"Language: {lang}.")
            if topics:
                parts.append(f"Topics: {', '.join(topics)}.")
            info = " ".join(parts) if parts else cid
        metadata[str(nid)] = {
            "type": "codebase",
            "name": cid,
            "description": cmeta.get("description", ""),
            "stars": cmeta.get("stars", 0),
            "language": cmeta.get("language", ""),
            "topics": cmeta.get("topics", []),
            "info": info,
        }

    return metadata


def create_eval_edges(
    norm_edges: Dict[str, Dict[str, Any]],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Create model→dataset eval edge array and metadata."""
    edge_list = []
    edge_meta: Dict[str, Dict[str, Any]] = {}

    for data in norm_edges.values():
        mid, did = data["model_id"], data["dataset_id"]
        if mid not in model_to_node or did not in dataset_to_node:
            continue
        u, v = model_to_node[mid], dataset_to_node[did]
        edge_list.append((u, v))
        edge_meta[f"{u},{v}"] = {
            "model_id": mid,
            "dataset_id": did,
            "metrics": data.get("metrics", {}),
            "edge_type": "eval",
        }

    edges_array = (
        np.array(edge_list, dtype=np.int32) if edge_list else np.zeros((0, 2), dtype=np.int32)
    )
    return edges_array, edge_meta


def create_resource_edges(
    resource_links: Dict[str, Any],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
    paper_to_node: Dict[str, int],
    codebase_to_node: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Create artifact→paper and artifact→codebase edges.
    Both models and datasets can reference papers and codebases.
    """
    edge_list = []
    edge_meta: Dict[str, Dict[str, Any]] = {}

    def _add_edges(artifact_id: str, links: Dict[str, Any], artifact_node: int) -> None:
        for pid in links.get("arxiv_ids", []):
            if pid not in paper_to_node:
                continue
            u, v = artifact_node, paper_to_node[pid]
            key = f"{u},{v}"
            if key not in edge_meta:
                edge_list.append((u, v))
                edge_meta[key] = {
                    "source_id": artifact_id,
                    "target_id": pid,
                    "edge_type": "has_paper",
                }

        for cid in links.get("github_repos", []):
            if cid not in codebase_to_node:
                continue
            u, v = artifact_node, codebase_to_node[cid]
            key = f"{u},{v}"
            if key not in edge_meta:
                edge_list.append((u, v))
                edge_meta[key] = {
                    "source_id": artifact_id,
                    "target_id": cid,
                    "edge_type": "has_codebase",
                }

    for artifact_id, links in resource_links.get("models", {}).items():
        if artifact_id in model_to_node:
            _add_edges(artifact_id, links, model_to_node[artifact_id])

    for artifact_id, links in resource_links.get("datasets", {}).items():
        if artifact_id in dataset_to_node:
            _add_edges(artifact_id, links, dataset_to_node[artifact_id])

    edges_array = (
        np.array(edge_list, dtype=np.int32) if edge_list else np.zeros((0, 2), dtype=np.int32)
    )
    return edges_array, edge_meta


def create_base_model_edges(
    base_model_map: Dict[str, List[str]],
    model_to_node: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Create child_model → parent_model edges from base_model metadata."""
    edge_list = []
    edge_meta: Dict[str, Dict[str, Any]] = {}

    for child_id, parent_ids in base_model_map.items():
        if child_id not in model_to_node:
            continue
        for parent_id in parent_ids:
            if parent_id not in model_to_node:
                continue
            u, v = model_to_node[child_id], model_to_node[parent_id]
            if u == v:
                continue
            key = f"{u},{v}"
            if key not in edge_meta:
                edge_list.append((u, v))
                edge_meta[key] = {
                    "source_id": child_id,
                    "target_id": parent_id,
                    "edge_type": "base_model",
                }

    edges_array = (
        np.array(edge_list, dtype=np.int32) if edge_list else np.zeros((0, 2), dtype=np.int32)
    )
    return edges_array, edge_meta


# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────

def _edges_to_npz(path: Path, edges_array: np.ndarray) -> None:
    edges_t = edges_array.T if edges_array.shape[0] > 0 else np.zeros((2, 0), dtype=np.int32)
    np.savez_compressed(path, edges=edges_t)


def _generate_random(num_nodes: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.normal(0, 1, (num_nodes, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-8)


def save_graph(
    output_dir: Path,
    node_metadata: Dict[str, Dict[str, Any]],
    eval_edges: np.ndarray,
    eval_edge_meta: Dict[str, Dict[str, Any]],
    resource_edges: np.ndarray,
    resource_edge_meta: Dict[str, Dict[str, Any]],
    base_model_edges: np.ndarray,
    base_model_edge_meta: Dict[str, Dict[str, Any]],
    model_to_node: Dict[str, int],
    dataset_to_node: Dict[str, int],
    paper_to_node: Dict[str, int],
    codebase_to_node: Dict[str, int],
    embedding_paths: List[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = len(node_metadata)

    # Node metadata
    with open(output_dir / "node_metadata.json", "w", encoding="utf-8") as f:
        json.dump(node_metadata, f, indent=2, ensure_ascii=False)

    # Node mappings (all four types)
    node_mappings = {
        "model_to_node":    {k: int(v) for k, v in model_to_node.items()},
        "dataset_to_node":  {k: int(v) for k, v in dataset_to_node.items()},
        "paper_to_node":    {k: int(v) for k, v in paper_to_node.items()},
        "codebase_to_node": {k: int(v) for k, v in codebase_to_node.items()},
    }
    with open(output_dir / "node_mappings.json", "w", encoding="utf-8") as f:
        json.dump(node_mappings, f, indent=2, ensure_ascii=False)

    # Eval edges
    _edges_to_npz(output_dir / "edges_eval.npz", eval_edges)
    with open(output_dir / "edge_metadata_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_edge_meta, f, indent=2, ensure_ascii=False)

    # Resource edges (paper + codebase)
    _edges_to_npz(output_dir / "edges_resource.npz", resource_edges)
    with open(output_dir / "edge_metadata_resource.json", "w", encoding="utf-8") as f:
        json.dump(resource_edge_meta, f, indent=2, ensure_ascii=False)

    # Base model edges (model → model)
    _edges_to_npz(output_dir / "edges_base_model.npz", base_model_edges)
    with open(output_dir / "edge_metadata_base_model.json", "w", encoding="utf-8") as f:
        json.dump(base_model_edge_meta, f, indent=2, ensure_ascii=False)

    # edges.npz = eval + resource + base_model edges (GNN message-passing graph)
    parts = [eval_edges]
    if resource_edges.shape[0] > 0:
        parts.append(resource_edges)
    if base_model_edges.shape[0] > 0:
        parts.append(base_model_edges)
    all_edges = np.concatenate(parts, axis=0) if len(parts) > 1 else eval_edges
    _edges_to_npz(output_dir / "edges.npz", all_edges)
    # edge_metadata.json = eval-only (used for metric lookups)
    with open(output_dir / "edge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(eval_edge_meta, f, indent=2, ensure_ascii=False)
    with open(output_dir / "edge_metadata_normalized.json", "w", encoding="utf-8") as f:
        json.dump(eval_edge_meta, f, indent=2, ensure_ascii=False)

    # Embeddings
    for emb_path in embedding_paths:
        if emb_path.exists():
            emb = np.load(emb_path)
            if emb.shape[0] == num_nodes:
                dest = output_dir / emb_path.name
                np.save(dest, emb)
                print(f"  Copied embeddings: {dest.name} (shape: {emb.shape})")
            else:
                print(f"  Warning: embedding shape {emb.shape[0]} != num_nodes {num_nodes}, regenerating")
                dim = emb.shape[1] if emb.ndim > 1 else 1024
                np.save(output_dir / emb_path.name, _generate_random(num_nodes, dim))
        else:
            print(f"  Embedding not found: {emb_path.name}, generating random")
            np.save(output_dir / emb_path.name, _generate_random(num_nodes, 1024))

    # Always ensure random embedding exists for ablation
    random_path = output_dir / "node_embeddings_random.npy"
    if not random_path.exists():
        emb = _generate_random(num_nodes, 1024)
        np.save(random_path, emb)
        print(f"  Generated random embeddings: {random_path.name} (shape: {emb.shape})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Convert to graph format.")
    parser.add_argument("--normalized-pairs",
                        default="../data/artifact_raw_data/normalized_eval_pairs.json")
    parser.add_argument("--summaries",
                        default="../data/artifact_raw_data/readme_summaries.json")
    parser.add_argument("--filtered-pairs",
                        default="../data/artifact_raw_data/filtered_eval_pairs.json")
    parser.add_argument("--resource-links",
                        default="../data/artifact_raw_data/resource_links.json",
                        help="Artifact→resource mapping from step 4.")
    parser.add_argument("--papers-metadata-dir",
                        default="../data/artifact_raw_data/papers/metadata",
                        help="Directory of paper metadata JSON files (step 4).")
    parser.add_argument("--codebases-metadata-dir",
                        default="../data/artifact_raw_data/codebases/metadata",
                        help="Directory of codebase metadata JSON files (step 4).")
    parser.add_argument("--models-metadata-dir",
                        default="../data/artifact_raw_data/models/metadata",
                        help="Directory of model metadata JSON files (step 1, for base_model).")
    parser.add_argument("--model-readme-dir",
                        default="../data/artifact_raw_data/models/readmes",
                        help="Directory of model README files (for filtering).")
    parser.add_argument("--dataset-readme-dir",
                        default="../data/artifact_raw_data/datasets/readmes",
                        help="Directory of dataset README files (for filtering).")
    parser.add_argument("--dataset-metadata-dir",
                        default="../data/artifact_raw_data/datasets/metadata",
                        help="Directory of dataset metadata JSON files (for filtering).")
    parser.add_argument("--embeddings-dir",
                        default="../data/artifact_raw_data",
                        help="Directory containing node_embeddings_*.npy from step 5.")
    parser.add_argument("--output-dir", default="../data/artifact_graph_data_v3")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    norm_path       = (script_dir / args.normalized_pairs).resolve()
    summ_path       = (script_dir / args.summaries).resolve()
    filt_path       = (script_dir / args.filtered_pairs).resolve()
    rl_path         = (script_dir / args.resource_links).resolve()
    papers_dir      = (script_dir / args.papers_metadata_dir).resolve()
    codebases_dir   = (script_dir / args.codebases_metadata_dir).resolve()
    models_meta_dir   = (script_dir / args.models_metadata_dir).resolve()
    model_readme_dir  = (script_dir / args.model_readme_dir).resolve()
    dataset_readme_dir = (script_dir / args.dataset_readme_dir).resolve()
    dataset_meta_dir  = (script_dir / args.dataset_metadata_dir).resolve()
    emb_dir           = (script_dir / args.embeddings_dir).resolve()
    output_dir        = (script_dir / args.output_dir).resolve()

    # Load
    print("Loading normalized pairs...")
    norm_data = load_json(norm_path)
    edges_dict = norm_data["edges"]
    print(f"  {len(edges_dict)} eval edges")

    print("Loading summaries...")
    summaries = (
        load_json(summ_path)
        if summ_path.exists()
        else {"models": {}, "datasets": {}, "papers": {}, "codebases": {}}
    )

    print("Loading download counts...")
    model_dl, dataset_dl = build_download_maps(filt_path)

    print("Loading resource links and metadata...")
    resource_links = load_json(rl_path) if rl_path.exists() else {}
    paper_metadata = load_paper_metadata(papers_dir)
    codebase_metadata = load_codebase_metadata(codebases_dir)
    base_model_map = load_base_model_map(models_meta_dir)
    print(f"  {len(paper_metadata)} papers, {len(codebase_metadata)} codebases, {len(base_model_map)} models with base_model")

    # Build node sets
    print("\nBuilding graph...")
    model_ids, dataset_ids = collect_eval_ids(
        edges_dict,
        model_readme_dir=model_readme_dir,
        model_metadata_dir=models_meta_dir,
        dataset_readme_dir=dataset_readme_dir,
        dataset_metadata_dir=dataset_meta_dir,
    )
    paper_ids, codebase_ids = collect_resource_ids(
        resource_links,
        set(model_ids),
        set(dataset_ids),
        paper_metadata,
        codebase_metadata,
    )
    total_nodes = len(model_ids) + len(dataset_ids) + len(paper_ids) + len(codebase_ids)
    print(
        f"  {len(model_ids)} models + {len(dataset_ids)} datasets"
        f" + {len(paper_ids)} papers + {len(codebase_ids)} codebases"
        f" = {total_nodes} nodes"
    )

    model_to_node, dataset_to_node, paper_to_node, codebase_to_node = create_node_mappings(
        model_ids, dataset_ids, paper_ids, codebase_ids
    )

    node_metadata = create_node_metadata(
        model_ids, dataset_ids, paper_ids, codebase_ids,
        model_to_node, dataset_to_node, paper_to_node, codebase_to_node,
        summaries, model_dl, dataset_dl, paper_metadata, codebase_metadata,
    )

    eval_edges, eval_edge_meta = create_eval_edges(edges_dict, model_to_node, dataset_to_node)
    resource_edges, resource_edge_meta = create_resource_edges(
        resource_links, model_to_node, dataset_to_node, paper_to_node, codebase_to_node
    )
    has_paper_count = sum(1 for e in resource_edge_meta.values() if e["edge_type"] == "has_paper")
    has_codebase_count = sum(1 for e in resource_edge_meta.values() if e["edge_type"] == "has_codebase")

    base_model_edges, base_model_edge_meta = create_base_model_edges(base_model_map, model_to_node)

    print(
        f"  {eval_edges.shape[0]} eval edges,"
        f" {has_paper_count} has_paper edges,"
        f" {has_codebase_count} has_codebase edges,"
        f" {base_model_edges.shape[0]} base_model edges"
    )

    emb_paths = list(emb_dir.glob("node_embeddings_*.npy"))
    print(f"  Found {len(emb_paths)} embedding file(s): {[p.name for p in emb_paths]}")

    print(f"\nSaving to {output_dir}/")
    save_graph(
        output_dir, node_metadata,
        eval_edges, eval_edge_meta,
        resource_edges, resource_edge_meta,
        base_model_edges, base_model_edge_meta,
        model_to_node, dataset_to_node, paper_to_node, codebase_to_node,
        emb_paths,
    )

    print(f"\nStep 6 complete.")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Eval edges: {eval_edges.shape[0]}")
    print(f"  Resource edges: {resource_edges.shape[0]}")
    print(f"  Base model edges: {base_model_edges.shape[0]}")


if __name__ == "__main__":
    main()
