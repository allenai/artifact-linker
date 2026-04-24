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

export HF_TOKEN="..."          # HF Hub
export VOYAGE_API_KEY="..."    # text embeddings
export OPENAI_API_KEY="..."    # stage-2 agent
```

## Quick Start

**1) Download the graph**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lwaekfjlk/artifact-bench', repo_type='dataset', local_dir='data/hf_graph')"
ln -s data/hf_graph/transductive data/artifact_graph_splits_v3_0314_transductive
ln -s data/hf_graph/inductive    data/artifact_graph_splits_v3_0314_inductive
ln -s data/hf_graph/full         data/artifact_graph_data_v3_0314
```

**2) Stage 1 — rank links (GNN + baselines)**

```bash
# Full reproduction: 6 backbones × 2 embeddings × 2 tasks × 2 splits + 13 baselines
CUDA_VISIBLE_DEVICES=0 bash scripts/run_reproduce.sh

# Or a single joint GATv2 run
CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_gnn.py \
    --split-dir data/artifact_graph_splits_v3_0314_transductive \
    --output-dir data/final_results_joint \
    --backbone gatv2 --num-layers 3 --hidden 128 --heads 8
```

**3) Stage 2 — verify links (coding agent)**

Build the Docker sandbox once (requires Docker + NVIDIA Container Toolkit):

```bash
bash build_docker.sh
```

Run the agent on top-ranked triples:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_evaluation_coder.py \
    --backend skills_multiagent \
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
