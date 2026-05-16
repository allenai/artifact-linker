![Artifact Linker](assets/artifact-linker-bar.png)

<h1 align="center">
  Artifact-Linker: Linking Scientific Artifacts for Automatic SOTA Discovery
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/lwaekfjlk/artifact-linker/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://huggingface.co/datasets/lwaekfjlk/artifact-bench"><img src="https://img.shields.io/badge/🤗%20Dataset-artifact--bench-yellow" alt="HF Dataset"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" alt="PyTorch"></a>
</p>

**Artifact-Linker** is a two-stage framework for automatic SOTA discovery on HuggingFace: (1) **rank** promising unobserved (model, dataset) links with a GNN or graph-augmented LLM, and (2) **verify** top-ranked links by letting an LLM agent write and run real evaluation code. Ships with **ArtifactBench** (14,053 artifacts, 51,337 relations).

## Installation

```bash
git clone https://github.com/lwaekfjlk/artifact-linker.git
cd artifact-linker
conda create -n artifact-linker python=3.10 -y && conda activate artifact-linker
pip install -r requirements.txt && pip install -e .

export HF_TOKEN="..."          # required: dataset download + Stage-2 sandbox
export OPENAI_API_KEY="..."    # required for Stage-1 LLM rows + Stage-2 agent
```

Voyage text embeddings ship pre-computed in the dataset
(`node_embeddings_voyage.npy`), so no `VOYAGE_API_KEY` is needed to reproduce.

## Quick Start

**1) Download the graph**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lwaekfjlk/artifact-bench', repo_type='dataset', local_dir='data/hf_graph')"
ln -s hf_graph/transductive       data/artifact_graph_splits_v3_0314_transductive
ln -s hf_graph/inductive          data/artifact_graph_splits_v3_0314_inductive
ln -s hf_graph/full               data/artifact_graph_data_v3_0314
ln -s hf_graph/verification_bench data/verification_bench   # Stage-2 input + ref
```

`verification_bench/bench.json` is the canonical Stage-2 input (263 triples);
`verification_bench/agent_results/` holds the paper's per-triple reference agent
outputs (265 triples) to compare your Stage-2 runs against.

**2) Stage 1 — link & attribute prediction/ranking**

Three method families, one entry script each (results in `data/`):

```bash
# GNN: joint training (shared encoder + link & attr heads) then 4 task evals.
# This is the pipeline behind the paper Table-1/2 GNN rows.
GPUS=0 bash scripts/run_gnn.sh                       # all 5 backbones x 2 splits
BACKBONES=gatv2 SPLITS=trans bash scripts/run_gnn.sh # single cell

# Deterministic heuristic baselines (Adamic-Adar / MF / Katz; attr means).
bash scripts/run_baseline.sh

# LLM rows (litellm model id; HOPS=1 = "+graph", HOPS=0 = no graph context).
OPENAI_API_KEY=... LLM_MODEL=openai/gpt-5.2 HOPS=1 bash scripts/run_llm.sh
```

`run_gnn.sh` trains the joint model once per (backbone, split) and reuses the
checkpoint for all four evals (`train_joint_gnn.py` → `{predict,rank}_{link,attribute}_gnn.py`).
Pass `GPUS="0,1,2,3,4"` to put one cell per GPU.

**3) Stage 2 — verify links (coding agent)**

Input = `data/verification_bench/bench.json` (263 (model, dataset, metric)
triples), downloaded from the HF dataset in Quick Start step 1. The agent pulls
each HF model + dataset **at runtime inside the Docker sandbox**, so `HF_TOKEN`
is required (in addition to `OPENAI_API_KEY` for the agent LLM). Compare your
runs against the paper's reference outputs in
`data/verification_bench/agent_results/`.

Build the Docker sandbox once (requires Docker + NVIDIA Container Toolkit):

```bash
bash build_docker.sh
```

Run the agent on the bundled triples:

```bash
HF_TOKEN=... OPENAI_API_KEY=... GPU=0 LLM_MODEL=openai/gpt-5.2 \
    bash scripts/run_verification.sh

# quick single-triple smoke test
HF_TOKEN=... OPENAI_API_KEY=... GPU=0 LIMIT=1 bash scripts/run_verification.sh
```

Or call the agent directly:

```bash
HF_TOKEN=... OPENAI_API_KEY=... CUDA_VISIBLE_DEVICES=0 \
python scripts/verify_attribute_agent.py \
    --mode multiturn_cachefiletool \
    --llm-model openai/gpt-5.2 \
    --json-file data/verification_triples.json
```

Parallelise with `--num-splits 4 --gpu-ids 0,1,2,3`.

## Citation

```bibtex
@software{artifact_linker2026,
  title = {Artifact-Linker: Linking Scientific Artifacts for Automatic SOTA Discovery},
  author = {...},
  year = {2026},
  url = {https://github.com/lwaekfjlk/artifact-linker}
}
```

## License

[Apache 2.0](https://github.com/lwaekfjlk/artifact-linker/blob/main/LICENSE)
