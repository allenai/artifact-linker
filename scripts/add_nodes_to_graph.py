#!/usr/bin/env python3
"""
Append 2 new nodes (MNLI dataset + sileod model) and 1 new edge to the
transductive graph, into the test split. Leaves train split untouched so
the GNN can predict the new edge as a genuinely unseen held-out case.

Nodes added (IDs assigned as max_id + 1, max_id + 2):
    14050: model    sileod/deberta-v3-large-tasksource-nli
    14051: dataset  nyu-mll/multi_nli

Edge added (test_split only):
    (14050, 14051) with accuracy = 0.9300
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/haofeiy2/artifact-linker")
SPLIT = ROOT / "data" / "artifact_graph_splits_v3_0314_transductive"
RAW = ROOT / "data" / "artifact_raw_data"

NEW_MODEL = "sileod/deberta-v3-large-tasksource-nli"
NEW_DATASET = "nyu-mll/multi_nli"
EDGE_ACCURACY = 0.9300


def fetch_sileod_readme():
    """Fetch sileod model readme from HF."""
    from huggingface_hub import hf_hub_download
    try:
        readme_path = hf_hub_download(
            repo_id=NEW_MODEL,
            filename="README.md",
            repo_type="model",
        )
        with open(readme_path) as f:
            return f.read()
    except Exception as e:
        print(f"Warning: could not fetch sileod readme: {e}")
        # Fallback: use model name as text
        return f"{NEW_MODEL}: A DeBERTa-v3-large model fine-tuned on 600 NLI tasks from the tasksource collection. Strong multi-task NLI performance."


def voyage_embed(texts, api_key):
    """Embed texts via voyage-3 (matches existing embeddings: dim=1024)."""
    import voyageai
    client = voyageai.Client(api_key=api_key)
    result = client.embed(texts, model="voyage-3", input_type="document")
    return np.array(result.embeddings, dtype=np.float32)


def random_embed(n, dim=1024, seed=42):
    """L2-normalized random embeddings, matching the existing convention."""
    rng = np.random.RandomState(seed)
    emb = rng.normal(0, 1, (n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-8)


def print_stats(label):
    """Print graph statistics for a given state (before/after)."""
    nm_tr = json.load(open(SPLIT / "train_split" / "node_metadata.json"))
    nm_te = json.load(open(SPLIT / "test_split" / "node_metadata.json"))
    em_tr = json.load(open(SPLIT / "train_split" / "edge_metadata_normalized.json"))
    em_te = json.load(open(SPLIT / "test_split" / "edge_metadata_normalized.json"))
    emb = np.load(SPLIT / "node_embeddings_voyage.npy")
    rnd = np.load(SPLIT / "node_embeddings_random.npy")
    pe_tr = np.load(SPLIT / "train_split" / "pos_edges.npz")["edges"]
    pe_te = np.load(SPLIT / "test_split" / "pos_edges.npz")["edges"]
    n_model_tr = sum(1 for v in nm_tr.values() if v.get("type") == "model")
    n_ds_tr = sum(1 for v in nm_tr.values() if v.get("type") == "dataset")
    n_model_te = sum(1 for v in nm_te.values() if v.get("type") == "model")
    n_ds_te = sum(1 for v in nm_te.values() if v.get("type") == "dataset")
    print(f"\n===== {label} =====")
    print(f"  node_metadata train: {len(nm_tr)} total ({n_model_tr} models, {n_ds_tr} datasets)")
    print(f"  node_metadata test:  {len(nm_te)} total ({n_model_te} models, {n_ds_te} datasets)")
    print(f"  voyage embeddings:   shape={emb.shape}")
    print(f"  random embeddings:   shape={rnd.shape}")
    print(f"  edge_metadata train: {len(em_tr)} edges")
    print(f"  edge_metadata test:  {len(em_te)} edges")
    print(f"  pos_edges train:     {pe_tr.shape} (shape)")
    print(f"  pos_edges test:      {pe_te.shape} (shape)")
    print(f"  ID range train:      {min(int(k) for k in nm_tr)}–{max(int(k) for k in nm_tr)}")
    return {
        "n_nodes_train": len(nm_tr),
        "n_nodes_test": len(nm_te),
        "emb_shape": emb.shape,
        "n_edges_train": len(em_tr),
        "n_edges_test": len(em_te),
    }


def main():
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set")
        sys.exit(1)

    print_stats("BEFORE")

    # -------- Prepare texts for voyage --------
    summaries = json.load(open(RAW / "readme_summaries.json"))
    mnli_info = summaries["datasets"].get(NEW_DATASET, {})
    mnli_text = mnli_info.get("info") if isinstance(mnli_info, dict) else None
    if not mnli_text:
        print("MNLI summary not found; using raw readme")
        with open(RAW / "datasets" / "readmes" / "nyu-mll__multi_nli.md") as f:
            mnli_text = f.read()
    mnli_text = mnli_text[:8000]

    print("\nFetching sileod readme from HuggingFace...")
    sileod_text = fetch_sileod_readme()[:8000]
    print(f"  sileod readme: {len(sileod_text)} chars")
    print(f"  mnli summary:  {len(mnli_text)} chars")

    # -------- Voyage embed --------
    print("\nGenerating voyage embeddings...")
    new_voyage = voyage_embed([sileod_text, mnli_text], api_key)
    print(f"  voyage embeddings: shape={new_voyage.shape}")

    # Random embeddings for the 2 new nodes (deterministic via seed offset)
    # Existing convention: L2-normalized N(0,1). Use seed=42 + num_existing for stability.
    new_random = random_embed(2, dim=1024, seed=42 + 14050)

    # -------- Load existing data --------
    tr_nm_path = SPLIT / "train_split" / "node_metadata.json"
    te_nm_path = SPLIT / "test_split" / "node_metadata.json"
    tr_em_path = SPLIT / "train_split" / "edge_metadata_normalized.json"
    te_em_path = SPLIT / "test_split" / "edge_metadata_normalized.json"
    tr_pe_path = SPLIT / "train_split" / "pos_edges.npz"
    te_pe_path = SPLIT / "test_split" / "pos_edges.npz"
    vo_path = SPLIT / "node_embeddings_voyage.npy"
    rn_path = SPLIT / "node_embeddings_random.npy"

    nm_tr = json.load(open(tr_nm_path))
    nm_te = json.load(open(te_nm_path))
    em_tr = json.load(open(te_em_path))  # init from test side; actually from its own
    em_tr = json.load(open(tr_em_path))
    em_te = json.load(open(te_em_path))
    voyage = np.load(vo_path)
    random_embs = np.load(rn_path)
    pe_tr = np.load(tr_pe_path)["edges"]
    pe_te = np.load(te_pe_path)["edges"]

    # -------- Assign new IDs --------
    max_id = max(int(k) for k in nm_tr)
    model_id = max_id + 1  # 14050
    dataset_id = max_id + 2  # 14051

    # -------- Add nodes to both splits' node_metadata --------
    model_entry = {
        "type": "model",
        "name": NEW_MODEL,
        "downloads": 0,  # not tracking
        "info": sileod_text,
    }
    dataset_entry = {
        "type": "dataset",
        "name": NEW_DATASET,
        "downloads": mnli_info.get("downloads", 0) if isinstance(mnli_info, dict) else 0,
        "info": mnli_text,
    }
    for nm in (nm_tr, nm_te):
        nm[str(model_id)] = model_entry
        nm[str(dataset_id)] = dataset_entry

    # -------- Add rows to embedding arrays --------
    voyage = np.vstack([voyage, new_voyage])  # +2 rows
    random_embs = np.vstack([random_embs, new_random])  # +2 rows

    # -------- Add edge to TEST split only --------
    edge_key = f"{model_id},{dataset_id}"
    em_te[edge_key] = {
        "accuracy": EDGE_ACCURACY,
        "metrics": {"accuracy": EDGE_ACCURACY},
        "source": "mnli_added_post_hoc",
    }
    # Append to pos_edges (test) — shape is (2, N) with row 0 = src, row 1 = dst
    new_edge = np.array([[model_id], [dataset_id]], dtype=pe_te.dtype)
    pe_te = np.concatenate([pe_te, new_edge], axis=1)

    # -------- Save everything --------
    print("\nSaving...")
    with open(tr_nm_path, "w") as f:
        json.dump(nm_tr, f)
    with open(te_nm_path, "w") as f:
        json.dump(nm_te, f)
    with open(te_em_path, "w") as f:
        json.dump(em_te, f)
    np.save(vo_path, voyage)
    np.save(rn_path, random_embs)
    np.savez(te_pe_path, edges=pe_te)
    print(f"  Appended node {model_id} ({NEW_MODEL})")
    print(f"  Appended node {dataset_id} ({NEW_DATASET})")
    print(f"  Appended test edge ({model_id}, {dataset_id}) with accuracy={EDGE_ACCURACY}")

    print_stats("AFTER")

    # Verify
    print("\n===== VERIFICATION =====")
    nm_check = json.load(open(te_nm_path))
    assert str(model_id) in nm_check and str(dataset_id) in nm_check
    em_check = json.load(open(te_em_path))
    assert edge_key in em_check
    emb_check = np.load(vo_path)
    assert emb_check.shape == (14052, 1024), f"Expected (14052, 1024), got {emb_check.shape}"
    print("  node_metadata: new IDs present ✓")
    print("  edge_metadata: new edge present ✓")
    print(f"  voyage embeddings: {emb_check.shape} ✓")


if __name__ == "__main__":
    main()
