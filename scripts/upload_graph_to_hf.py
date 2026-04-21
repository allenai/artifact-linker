#!/usr/bin/env python3
"""Upload artifact-graph splits (transductive + inductive) to HuggingFace Hub.

Target: https://huggingface.co/datasets/lwaekfjlk/artifact-graph

Requires HF_TOKEN env var.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ID = "lwaekfjlk/artifact-graph"
LOCAL_ROOT = Path("/home/haofeiy2/artifact-linker/data")

FOLDERS = {
    "full":         LOCAL_ROOT / "artifact_graph_data_v3_0314",
    "transductive": LOCAL_ROOT / "artifact_graph_splits_v3_0314_transductive",
    "inductive":    LOCAL_ROOT / "artifact_graph_splits_v3_0314_inductive",
}


def write_readme(api: HfApi, token: str):
    readme = """---
license: cc-by-4.0
language:
- en
tags:
- graph
- link-prediction
- benchmark
- model-dataset
size_categories:
- 10K<n<100K
---

# Artifact Graph

A heterogeneous graph of HuggingFace model/dataset/paper/codebase nodes with
observed (model, dataset, performance-metric) evaluation edges, used to
benchmark link prediction and attribute regression.

## Contents

| path                              | description |
|-----------------------------------|-------------|
| `full/`                           | Full unsplit graph: all nodes + all edges (by type) |
| `transductive/`                   | All nodes visible in both train and test; edges split |
| `inductive/`                      | Disjoint node partition: some nodes train-only, others test-only |

## Full graph (`full/`)

| file                                  | description |
|---------------------------------------|-------------|
| `node_metadata.json`                  | Per-node `{type, name, downloads, info}` for all 14K nodes |
| `node_mappings.json`                  | Integer ID ↔ HuggingFace ID mapping |
| `node_embeddings_voyage.npy`          | Voyage-3 embeddings, `(N, 1024)` |
| `node_embeddings_random.npy`          | L2-normalised random embeddings |
| `edges.npz`                           | All edges combined, `(2, E)` |
| `edges_eval.npz`                      | model × dataset evaluation edges |
| `edges_base_model.npz`                | model → base_model edges |
| `edges_resource.npz`                  | model/dataset → paper/codebase edges |
| `edge_metadata.json`                  | Raw (model, dataset, metric) edge records |
| `edge_metadata_normalized.json`       | Eval edges with metrics normalised to `[0, 1]` |
| `edge_metadata_eval.json`             | Eval-edge metadata only |
| `edge_metadata_base_model.json`       | base-model edge metadata |
| `edge_metadata_resource.json`         | paper / codebase resource edge metadata |

Each split directory contains:

| file                            | description |
|---------------------------------|-------------|
| `node_embeddings_voyage.npy`    | Voyage-3 embeddings, shape `(N, 1024)` |
| `node_embeddings_random.npy`    | L2-normalised random embeddings, same shape |
| `split_info.json`               | Split metadata (seed, counts, dates) |
| `node_split.json` (inductive)   | Per-node train/test assignment |
| `train_split/`                  | Training subgraph (see below) |
| `test_split/`                   | Test subgraph (held-out eval edges) |

Each `{train,test}_split/` holds:

| file                              | description |
|-----------------------------------|-------------|
| `node_metadata.json`              | Per-node `{type, name, downloads, info}` |
| `edge_metadata_normalized.json`   | Normalized `(u,v) → metric: value` map |
| `edges.npz`                       | Message-passing edges, `edges` key, shape `(2, E)` |
| `pos_edges.npz`                   | Positive eval edges (model × dataset with metric) |

## Node types

- `model`: HuggingFace models (e.g., `sileod/deberta-v3-large-tasksource-nli`)
- `dataset`: HuggingFace datasets (e.g., `nyu-mll/multi_nli`)
- `paper`: referenced papers (arXiv IDs)
- `codebase`: linked repositories

## Edge types

- `model ↔ dataset` (eval): accuracy / F1 / BLEU / etc. (normalized to `[0, 1]`)
- `model ↔ paper`, `model ↔ codebase`, `dataset ↔ paper`, `dataset ↔ codebase`: resource links
- `model ↔ model`: base-model / fine-tune relations

## Usage

```python
from huggingface_hub import snapshot_download
path = snapshot_download("lwaekfjlk/artifact-graph", repo_type="dataset")
import numpy as np, json
emb = np.load(f"{path}/transductive/node_embeddings_voyage.npy")
nm = json.load(open(f"{path}/transductive/train_split/node_metadata.json"))
pe = np.load(f"{path}/transductive/train_split/pos_edges.npz")["edges"]
print(emb.shape, len(nm), pe.shape)
```
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    print(f"Creating / getting repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="dataset", token=token, exist_ok=True)

    for sub, folder in FOLDERS.items():
        if not folder.exists():
            print(f"  ! skip {sub}: {folder} not found")
            continue
        print(f"\nUploading {folder} → {REPO_ID}/{sub} ...")
        upload_folder(
            folder_path=str(folder),
            path_in_repo=sub,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message=f"Add {sub} split",
        )

    print("\nWriting README...")
    write_readme(api, token)

    print(f"\nDone. Browse at https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
