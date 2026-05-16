#!/bin/bash
# Stage-1 GNN: joint training (shared encoder + link & attr heads) then the
# 4 task evals (link/attr x predict/rank). This IS the pipeline behind the
# paper Table-1/2 GNN rows. Training runs ONCE per (backbone,split); the 4
# evals reuse that checkpoint.
#
# Usage:
#   bash scripts/run_gnn.sh
#   BACKBONES="gatv2" SPLITS="trans" bash scripts/run_gnn.sh
#   GPUS="0,1,2,3,4" bash scripts/run_gnn.sh        # one (backbone,split) per GPU
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/home/haofeiy2/.conda/envs/artifact-linker/bin/python}"
OUT="${OUT_DIR:-$ROOT/data/joint_sweep}"
mkdir -p "$OUT"

TRANS="$ROOT/data/artifact_graph_splits_v3_0314_transductive"
INDUC="$ROOT/data/artifact_graph_splits_v3_0314_inductive"

BACKBONES="${BACKBONES:-buddy neognn ncn ncnc gatv2}"
SPLITS="${SPLITS:-trans induc}"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-1500}"

# canonical joint config (matches archived run_ablation_layers.sh JOINT_*; verified vs paper)
TRAIN_CFG="--embedding-mode embedding --num-layers 3 --hidden 128 --heads 8 \
           --epochs $EPOCHS --lr 0.002 --attr-weight 5.0 --neg-ratio 2 --seed 42"

# one (backbone,split): train once -> 4 task evals on the saved checkpoint
run_cell() {
    local gpu="$1" sd="$2" tag="$3" bb="$4"
    local ckpt="$OUT/${tag}_joint_${bb}_model_emb.pth"
    export CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [[ ! -s "$ckpt" ]]; then
        "$PY" "$ROOT/scripts/train_joint_gnn.py" --split-dir "$sd" --output-dir "$OUT" \
            --backbone "$bb" --model-path "$ckpt" $TRAIN_CFG
    else
        echo "[skip train] $ckpt exists"
    fi
    for task in predict_link rank_link predict_attribute rank_attribute; do
        "$PY" "$ROOT/scripts/${task}_gnn.py" --split-dir "$sd" --output-dir "$OUT" \
            --model-path "$ckpt" --backbone "$bb" --embedding-mode embedding
    done
}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
i=0; pids=()
for split in $SPLITS; do
    case "$split" in
        trans) SD="$TRANS" ;; induc) SD="$INDUC" ;;
        *) echo "unknown split: $split"; exit 1 ;;
    esac
    for bb in $BACKBONES; do
        gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
        echo "[gpu=$gpu] $split/$bb"
        ( run_cell "$gpu" "$SD" "$split" "$bb" ) &
        pids+=($!); i=$((i+1))
        if (( i % ${#GPU_ARR[@]} == 0 )); then wait "${pids[@]}"; pids=(); fi
    done
done
[[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
echo "DONE. results: $OUT"
