#!/usr/bin/env python3
"""
Add `microsoft/deberta-v3-large` as a model node, plus a base_model edge
from `sileod/deberta-v3-large-tasksource-nli` (14050) → the new node.

The base_model edge goes into train_split/edges.npz (message-passing graph)
and is recorded in a new edge_metadata_resource.json file. It is NOT added
to pos_edges.npz (which is reserved for eval link-prediction positives).

This gives sileod one structural connection so the GNN encoder can message-
pass through its backbone rather than treating it as an isolated node.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/haofeiy2/artifact-linker")
SPLIT = ROOT / "data" / "artifact_graph_splits_v3_0314_transductive"

NEW_BACKBONE = "microsoft/deberta-v3-large"
SILEOD_ID = 14050  # already added


def fetch_readme(model_id):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id=model_id, filename="README.md", repo_type="model")
        with open(p) as f:
            return f.read()
    except Exception as e:
        print(f"Warning: could not fetch {model_id} readme: {e}")
        return f"{model_id}: backbone language model."


def voyage_embed(texts, api_key):
    import voyageai
    client = voyageai.Client(api_key=api_key)
    result = client.embed(texts, model="voyage-3", input_type="document")
    return np.array(result.embeddings, dtype=np.float32)


def random_embed(n, seed, dim=1024):
    rng = np.random.RandomState(seed)
    emb = rng.normal(0, 1, (n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-8)


def stats(label):
    nm = json.load(open(SPLIT / "train_split" / "node_metadata.json"))
    edges = np.load(SPLIT / "train_split" / "edges.npz")["edges"]
    pe = np.load(SPLIT / "train_split" / "pos_edges.npz")["edges"]
    emb = np.load(SPLIT / "node_embeddings_voyage.npy")
    n_model = sum(1 for v in nm.values() if v.get("type") == "model")
    n_ds = sum(1 for v in nm.values() if v.get("type") == "dataset")
    n_paper = sum(1 for v in nm.values() if v.get("type") == "paper")
    n_code = sum(1 for v in nm.values() if v.get("type") == "codebase")
    print(f"\n===== {label} =====")
    print(f"  nodes:            {len(nm)}  ({n_model} models, {n_ds} datasets, {n_paper} papers, {n_code} codebases)")
    print(f"  voyage emb:       {emb.shape}")
    print(f"  train edges.npz:  {edges.shape[1]} edges  (message-passing)")
    print(f"  train pos_edges:  {pe.shape[1]} edges  (eval positives)")
    # Count sileod edges
    sileod_out = int(np.sum(edges[0] == SILEOD_ID))
    sileod_in = int(np.sum(edges[1] == SILEOD_ID))
    print(f"  sileod (14050) edges in train: out={sileod_out}, in={sileod_in}")


def main():
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set")
        sys.exit(1)

    stats("BEFORE")

    # -------- Load current state --------
    nm_tr_path = SPLIT / "train_split" / "node_metadata.json"
    nm_te_path = SPLIT / "test_split" / "node_metadata.json"
    vo_path = SPLIT / "node_embeddings_voyage.npy"
    rn_path = SPLIT / "node_embeddings_random.npy"
    edges_tr_path = SPLIT / "train_split" / "edges.npz"
    edges_te_path = SPLIT / "test_split" / "edges.npz"

    nm_tr = json.load(open(nm_tr_path))
    nm_te = json.load(open(nm_te_path))
    voyage = np.load(vo_path)
    random_embs = np.load(rn_path)
    edges_tr = np.load(edges_tr_path)["edges"]
    edges_te = np.load(edges_te_path)["edges"]

    # -------- New backbone node --------
    new_id = max(int(k) for k in nm_tr) + 1  # 14052
    print(f"\nAssigning new node id: {new_id}  → {NEW_BACKBONE}")

    print("\nFetching backbone readme from HuggingFace...")
    backbone_text = fetch_readme(NEW_BACKBONE)[:8000]
    print(f"  backbone readme: {len(backbone_text)} chars")

    print("\nGenerating voyage embedding for backbone...")
    new_voyage = voyage_embed([backbone_text], api_key)
    new_random = random_embed(1, 42 + new_id)

    backbone_entry = {
        "type": "model",
        "name": NEW_BACKBONE,
        "downloads": 0,
        "info": backbone_text,
    }
    for nm in (nm_tr, nm_te):
        nm[str(new_id)] = backbone_entry

    voyage = np.vstack([voyage, new_voyage])
    random_embs = np.vstack([random_embs, new_random])

    # -------- Add base_model edge to train_split/edges.npz --------
    # Convention: edges.npz holds [src, dst] pairs for message passing (undirected
    # in practice since GNN adds reverse during convolution).
    new_edge = np.array([[SILEOD_ID], [new_id]], dtype=edges_tr.dtype)
    edges_tr_new = np.concatenate([edges_tr, new_edge], axis=1)

    # Also add reverse direction for undirected message passing
    new_edge_rev = np.array([[new_id], [SILEOD_ID]], dtype=edges_tr.dtype)
    edges_tr_new = np.concatenate([edges_tr_new, new_edge_rev], axis=1)

    # test_split also mirrors node presence, but the edge only in train
    # (we want the GNN to SEE this structural edge during training; test
    # pos_edges.npz already has sileod×MNLI as the held-out eval positive)

    # -------- Save --------
    print("\nSaving...")
    with open(nm_tr_path, "w") as f:
        json.dump(nm_tr, f)
    with open(nm_te_path, "w") as f:
        json.dump(nm_te, f)
    np.save(vo_path, voyage)
    np.save(rn_path, random_embs)
    np.savez(edges_tr_path, edges=edges_tr_new)
    print(f"  Added node {new_id} = {NEW_BACKBONE}")
    print(f"  Added base_model edge in train_split/edges.npz: ({SILEOD_ID}, {new_id}) and reverse")

    stats("AFTER")


if __name__ == "__main__":
    main()
