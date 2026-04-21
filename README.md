# artifact-linker

Graph-based prediction of (model, dataset) performance on HuggingFace
artifacts. Jointly trains a GNN over a heterogeneous graph of models,
datasets, papers, and codebases to (a) predict whether an evaluation
edge exists (**link prediction**) and (b) regress its performance metric
(**attribute prediction**). NLI is shipped as a case study of the
framework.

Dataset on HF: **[lwaekfjlk/artifact-graph](https://huggingface.co/datasets/lwaekfjlk/artifact-graph)**

---

## 1. Setup

```bash
# Clone
git clone https://github.com/lwaekfjlk/artifact-linker.git
cd artifact-linker

# Python env (conda recommended; tested with CUDA 12.1 / A100)
conda create -n artifact-linker python=3.10 -y
conda activate artifact-linker
pip install -r setup-requirements.txt
pip install -e .
```

Required env vars (only for new HF uploads / Voyage embeddings):

```bash
export HF_TOKEN=...        # for pushing to HF Hub
export VOYAGE_API_KEY=...  # for recomputing text embeddings
```

---

## 2. Download the graph from HuggingFace

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "lwaekfjlk/artifact-graph",
    repo_type="dataset",
    local_dir="data/hf_graph",
)
print("Downloaded to:", path)
EOF
```

This fetches ~1 GB:

| subfolder | description |
|---|---|
| `full/` | Unsplit 14,050-node graph with all edge types |
| `transductive/` | Transductive train/test split (14,053 nodes, augmented) |
| `inductive/` | Inductive split with disjoint node partition |
| `case_study_nli/` | 576 raw NLI evals + fixed aggregate + figures |

To map the HF layout to the paths used by the scripts below:

```bash
ln -s data/hf_graph/transductive data/artifact_graph_splits_v3_0314_transductive
ln -s data/hf_graph/inductive    data/artifact_graph_splits_v3_0314_inductive
ln -s data/hf_graph/full         data/artifact_graph_data_v3_0314
```

---

## 3. Reproduce main experiments

### 3.1 All GNN + baseline results (Tables 1--3)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_reproduce.sh
```

Runs 6 GNN backbones (GATv2, GCN, NCN, NCNC, NeoGNN, BUDDY) × 2 embedding
modes (Voyage, random) × 2 tasks (link, attribute) × 2 splits
(transductive, inductive), plus 13 baseline methods (downloads, Katz,
common neighbors, matrix factorization, etc.). Output under
`data/final_results_reproduce/`. Use `--gnn-only` or `--baseline-only` to
subset. `--dry-run` prints commands without executing.

### 3.2 Layer-count ablation (Table 4)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablation_layers.sh
```

Re-runs the joint GATv2 trainer with `num_layers ∈ {1, 2, 3, 4}`. Outputs
live under `data/final_results_ablation_layers/L{1..4}/`.

### 3.3 Joint GNN training (single configuration)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_gnn.py \
    --split-dir data/artifact_graph_splits_v3_0314_transductive \
    --output-dir data/final_results_joint \
    --backbone gatv2 --num-layers 3 --hidden 128 --heads 8 \
    --epochs 1500 --lr 0.002 --attr-weight 5.0 --neg-ratio 2
```

Default hyperparameters match the main paper experiments.

### 3.4 Novel-edge prediction (sileod × MNLI case study)

```bash
# Graph augmentation (already applied to the shipped HF data)
VOYAGE_API_KEY=... python scripts/add_nodes_to_graph.py
VOYAGE_API_KEY=... python scripts/add_base_model_edge.py

# Inference with trained joint GATv2
CUDA_VISIBLE_DEVICES=0 python scripts/predict_new_edge.py
```

---

## 4. NLI case study

### 4.1 Download the 576 raw evaluations

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "lwaekfjlk/artifact-graph",
    repo_type="dataset",
    allow_patterns="case_study_nli/**",
    local_dir=".",
)
EOF
```

### 4.2 Rebuild the aggregate (applies 9 bug-fixes + 3-way masks)

```bash
python scripts/rebuild_nli_summary.py \
    --src case_study_nli/raw_evals \
    --out case_study_nli/all_results_summary_fixed.json
```

### 4.3 Regenerate figures

```bash
python scripts/plot_nli_heatmap.py \
    --input case_study_nli/all_results_summary_fixed.json \
    --out-dir case_study_nli/figures
python scripts/plot_nli_matrix_scree.py \
    --input case_study_nli/all_results_summary_fixed.json \
    --out-dir case_study_nli/figures
```

Produces the main heatmap (45 models × 12 datasets, masked cells in
yellow) and the double-centered SVD scree plot
(`data/figures/nli_matrix_scree.png`, rank-5 captures 91% of interaction
variance).

---

## 5. Project layout

```
artifact-linker/
├── artifact_graph/              Main Python package
│   ├── models/                  GATv2 / GCN / NCN / NCNC / NeoGNN / BUDDY predictors
│   ├── training/                Joint / link / attribute trainers
│   ├── runners/                 Train / eval orchestration
│   └── utils/                   Graph, metric, embedding helpers
├── scripts/
│   ├── run_reproduce.sh         All GNN + baseline experiments
│   ├── run_ablation_layers.sh   Layer-count ablation
│   ├── run_joint_gnn.py         Single joint-trainer run
│   ├── rebuild_nli_summary.py   576 raw → cleaned aggregate
│   ├── plot_nli_heatmap.py      Heatmap figure
│   ├── plot_nli_matrix_scree.py Scree figure
│   ├── predict_new_edge.py      GNN inference on held-out edge
│   ├── add_nodes_to_graph.py    Graph augmentation
│   └── add_base_model_edge.py   Graph augmentation
├── data/                        Data directories (symlink HF downloads here)
│   ├── artifact_graph_data_v3_0314/
│   ├── artifact_graph_splits_v3_0314_transductive/
│   ├── artifact_graph_splits_v3_0314_inductive/
│   └── figures/                 Paper figures
├── _archive/
│   └── old_scripts/             Standalone link / attr / ranking scripts
│                                (invoked by run_reproduce.sh)
└── all_results_summary_fixed.json   Fixed NLI aggregate (root copy)
```

---

## 6. Data notes

- **Node embeddings**: Voyage-3 (dim 1024), computed once per node from
  GPT-summarized README / paper abstract. Random L2-normalized
  embeddings are shipped alongside as a controlled baseline.
- **Accuracy normalization**: metric values above 1 are rescaled to
  `[0, 1]`; the `edge_metadata_normalized.json` files reflect this.
- **NLI 3-way caveat**: three zero-shot classifiers can only emit 2
  labels; their cells on MNLI / SNLI / ANLI / NLI_FEVER are masked in
  the aggregate and cannot be compared directly to 3-way models.
- **Reproducibility caveat**: the 576 per-cell NLI eval scripts were
  not uniformly persisted to disk (only 119 of 576 cells retained
  `run_eval.py`). The shipped `raw_evals/` contain predictions and
  results only; re-running requires either the HF eval pipeline used
  in the paper or a custom loader.

---

## 7. Citation

```bibtex
@article{artifact-linker,
  title   = {...},
  author  = {...},
  year    = {2026},
}
```
