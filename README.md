![Artifact Linker](assets/artifact-linker-bar.png)

<h1 align="center">
  Artifact-Linker: Linking Scientific Artifacts for Automatic SOTA Discovery
</h1>


<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/lwaekfjlk/artifact-linker/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://huggingface.co/datasets/lwaekfjlk/artifact-bench"><img src="https://img.shields.io/badge/🤗%20Dataset-artifact--graph-yellow" alt="HF Dataset"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" alt="PyTorch"></a>
</p>

**Artifact Linker** is a two-stage framework: (1) ranking promising unobserved model--dataset links using Graph Neural Networks (GNNs) or graph-augmented Large Language Models (LLMs), and (2) verifying top-ranked links via coding experiments with LLM-based agents. We further introduce a benchmark named ArtifactBench with 14,053 artifacts and 51,337 relations to evaluate the performance of both stages. Results show that (1) graph structures between existing artifacts are effective for missing link prediction; (2) end-to-end ranking and verification with ArtifactLinker help discover potential SOTA results and research insights.

## Installation

```bash
git clone https://github.com/lwaekfjlk/artifact-linker.git
cd artifact-linker

# Python env (conda recommended; tested with CUDA 12.1 / A100)
conda create -n artifact-linker python=3.10 -y
conda activate artifact-linker
pip install -r requirements.txt
pip install -e .
```

## Quick Start

#### 1) Download — fetch the heterogeneous artifact graph

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "lwaekfjlk/artifact-bench",
    repo_type="dataset",
    local_dir="data/hf_graph",
)
EOF

ln -s data/hf_graph/transductive data/artifact_graph_splits_v3_0314_transductive
ln -s data/hf_graph/inductive    data/artifact_graph_splits_v3_0314_inductive
ln -s data/hf_graph/full         data/artifact_graph_data_v3_0314
```

Fetches ~1 GB across `full/` (unsplit 14,050-node graph), `transductive/` (14,053 nodes, augmented), and `inductive/` (disjoint node partition).

#### 2) Reproduce — all GNN + baseline results

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_reproduce.sh
```

Runs 6 GNN backbones (GATv2, GCN, NCN, NCNC, NeoGNN, BUDDY) × 2 embedding modes (Voyage, random) × 2 tasks (link, attribute) × 2 splits (transductive, inductive), plus 13 baseline methods. Output under `data/final_results_reproduce/`. Use `--gnn-only` / `--baseline-only` / `--dry-run` to subset.

#### 3) Train — single joint GNN configuration

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_gnn.py \
    --split-dir data/artifact_graph_splits_v3_0314_transductive \
    --output-dir data/final_results_joint \
    --backbone gatv2 --num-layers 3 --hidden 128 --heads 8 \
    --epochs 1500 --lr 0.002 --attr-weight 5.0 --neg-ratio 2
```

Default hyperparameters match the main paper experiments.

#### 4) Verify — run the skills-multiagent coding agent

Stage 2 of the framework: top-ranked (model, dataset, metric) triples are handed to an LLM agent that writes and executes real evaluation code inside Docker. The agent's behaviour comes from two sources: (a) hardcoded `PLANNER_INSTRUCTIONS` / `EXECUTOR_INSTRUCTIONS` in `artifact_graph/evaluation_coder_skills_multiagent.py` (task-type cheat sheet, cross-cutting rules, task-specific patterns for vLLM / NER / multilabel / extractive QA), and (b) four HuggingFace skills under `skills/` (`hugging-face-datasets`, `hugging-face-dataset-viewer`, `hugging-face-evaluation`, `eval-templates`) that are packaged into the `ShellTool` at runtime.

**Build the sandbox image first** (one-time, requires Docker + NVIDIA Container Toolkit):

```bash
bash build_docker.sh   # builds artifact-linker-verification:latest (CUDA 12.8)
```

Then run the agent:

```bash
export OPENAI_API_KEY="..."
export HF_TOKEN="..."

CUDA_VISIBLE_DEVICES=0 python scripts/run_evaluation_coder.py \
    --backend skills_multiagent \
    --mode multiturn_cachefiletool \
    --llm-model openai/codex-5.2 \
    --json-file results/perfect_model_dataset_metrics_v3_0120_coding_agent_filtered_hard_both_successful.json \
    --max-samples 1000 \
    --gpu-id 0
```

Architecture: a **PlanningAgent** inspects the model/dataset via `ShellTool` (armed with the HF skills) and emits a JSON plan; an **ExecutionAgent** writes and runs evaluation code inside the `artifact-linker-verification` container via `run_code_in_docker`; a programmatic validator checks the output and feeds errors back for the plan → execute → validate retry loop. Per-triple outputs (`agent_response.json`, `run.log`) land in the auto-named `skills_multiagent_results_*` directory; a top-level `batch_summary.json` aggregates success rates.

Modes (`--mode`): `oneturn_onetool` (single turn, docker only), `multiturn_onetool` (multi-turn, docker only), `multiturn_metadatatool` (+ HF metadata tools), `multiturn_cachefiletool` (+ cached dataset/model loaders — recommended).

To parallelise across GPUs, add `--num-splits 4 --gpu-ids 0,1,2,3`; the script auto-spawns one subprocess per shard.

## Architecture Overview

The pipeline has four stages:

```
HuggingFace artifacts (models, datasets, papers, codebases)
    → [1. Graph construction]  — heterogeneous graph + Voyage-3 text embeddings
    → [2. Joint GNN training]  — link prediction + attribute regression, shared encoder
    → [3. Evaluation]          — transductive / inductive splits, 13 baselines
    → [4. Inference]           — score unseen (model, dataset) pairs
```

Core runtime components:

```
JointTrainer
  ├── Graph              (Heterogeneous: model / dataset / paper / code nodes)
  ├── Encoder            (GATv2 / GCN / NCN / NCNC / NeoGNN / BUDDY)
  ├── LinkHead           (Binary: does an evaluation edge exist?)
  └── AttrHead           (Regression: what metric value on that edge?)
```

Each backbone is swappable via a single `--backbone` flag; the shared encoder feeds both the link and attribute heads so gradients from metric regression regularize the link-prediction representation.

## Data Notes

- **Node embeddings**: Voyage-3 (dim 1024), computed once per node from GPT-summarized README / paper abstract. Random L2-normalized embeddings are shipped alongside as a controlled baseline.
- **Accuracy normalization**: metric values above 1 are rescaled to `[0, 1]`; the `edge_metadata_normalized.json` files reflect this.

## Environment Variables

```bash
export HF_TOKEN="..."          # for pushing to HF Hub
export VOYAGE_API_KEY="..."    # for recomputing text embeddings
export OPENAI_API_KEY="..."    # for the stage-2 coding agent
```

## Citation

If you find Artifact Linker useful in your research or work, please cite:

```bibtex
@software{artifact_linker2026,
  title   = {Artifact Linker: Graph-Based Prediction of Model--Dataset Performance on HuggingFace},
  author  = {...},
  year    = {2026},
  url     = {https://github.com/lwaekfjlk/artifact-linker}
}
```

## License

[Apache 2.0](https://github.com/lwaekfjlk/artifact-linker/blob/main/LICENSE)

## Disclaimer

This software is for educational and research use. Performance predictions are statistical estimates and should not be treated as guarantees for downstream deployment decisions.