#!/bin/bash
# Ablation: GNN layer number = 1, 2, 3, 4
# Runs:
#   - Separate link prediction + ranking (predict_link_gnn / rank_link_gnn)
#   - Joint training (link + attr) with full eval (run_joint_gnn.py)
# Each layer count gets its own subdirectory to avoid filename collisions.
#
# Output structure:
#   data/final_results_ablation_layers/
#     L1/  trans_gnn_gatv2_link_predictions_emb.json
#          trans_gnn_gatv2_link_rankings_emb.json
#          trans_joint_gatv2_results_emb.json       <- joint (all metrics)
#     L2/  ...
#     L3/  ...
#     L4/  ...
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_ablation_layers.sh
#   bash scripts/run_ablation_layers.sh --dry-run
#   MODELS="gatv2 gcn" bash scripts/run_ablation_layers.sh   # subset of models
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/haofeiy2/anaconda3/envs/artifact-linker/bin/python}"
SCRIPT_DIR="$ROOT/scripts"

TRANS_DIR="$ROOT/data/artifact_graph_splits_v3_0314_transductive"
INDUC_DIR="$ROOT/data/artifact_graph_splits_v3_0314_inductive"
BASE_OUT_DIR="$ROOT/data/final_results_ablation_layers"

GNN_MODELS="${MODELS:-gatv2 gcn ncn ncnc neognn buddy}"
LAYERS="1 2 3 4"
EMB="embedding"           # use real embeddings for ablation

# Link prediction hyperparams
LINK_EPOCHS=1000
LINK_PATIENCE=40
LINK_THRESHOLD=0.9
LINK_NEG_RATIO=10

# Joint training hyperparams
JOINT_EPOCHS=1500
JOINT_LR=0.002
JOINT_ATTR_WEIGHT=5.0
JOINT_NEG_RATIO=2

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_cmd() {
    echo "[CMD] $*"
    if $DRY_RUN; then return 0; fi
    "$@" 2>&1 | tail -20
    echo ""
}

skip_if_exists() {
    local dir="$1"
    local name="$2"
    local f="$dir/$name.json"
    if [ -f "$f" ]; then
        echo "  SKIP (exists): $name"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
run_layer_ablation() {
    local split_dir="$1"
    local prefix="$2"   # "trans" or "induc"

    for nlayers in $LAYERS; do
        local out_dir="$BASE_OUT_DIR/L${nlayers}"
        mkdir -p "$out_dir"

        for model in $GNN_MODELS; do
            local emb_tag="emb"
            local model_link="$out_dir/${prefix}_gnn_${model}_link_prediction_model_${emb_tag}.pth"
            local model_attr="$out_dir/${prefix}_gnn_${model}_attribute_prediction_model_${emb_tag}.pth"

            # 1. Link Prediction
            local lp="${prefix}_gnn_${model}_link_predictions_${emb_tag}"
            if ! skip_if_exists "$out_dir" "$lp"; then
                echo "=== [$prefix] $model link_prediction (layers=$nlayers) ==="
                run_cmd "$PYTHON" "$SCRIPT_DIR/predict_link_gnn.py" \
                    --split-dir "$split_dir" \
                    --output-dir "$out_dir" \
                    --gnn-model "$model" \
                    --embedding-mode "$EMB" \
                    --num-layers "$nlayers" \
                    --epochs $LINK_EPOCHS \
                    --patience $LINK_PATIENCE \
                    --threshold $LINK_THRESHOLD \
                    --neg-ratio $LINK_NEG_RATIO \
                    --save-model-path "$model_link"
            fi

            # 2. Link Ranking
            local lr="${prefix}_gnn_${model}_link_rankings_${emb_tag}"
            if ! skip_if_exists "$out_dir" "$lr"; then
                if [ -f "$model_link" ]; then
                    echo "=== [$prefix] $model link_ranking (layers=$nlayers) ==="
                    run_cmd "$PYTHON" "$SCRIPT_DIR/rank_link_gnn.py" \
                        --split-dir "$split_dir" \
                        --output-dir "$out_dir" \
                        --gnn-model "$model" \
                        --embedding-mode "$EMB" \
                        --model-path "$model_link"
                else
                    echo "  SKIP (no model): $lr"
                fi
            fi

            # 3. Joint training (link + attr) — replaces separate attr pred/rank
            local joint="${prefix}_joint_${model}_results_${emb_tag}"
            if ! skip_if_exists "$out_dir" "$joint"; then
                echo "=== [$prefix] $model joint (layers=$nlayers) ==="
                run_cmd "$PYTHON" "$ROOT/scripts/run_joint_gnn.py" \
                    --split-dir "$split_dir" \
                    --output-dir "$out_dir" \
                    --backbone "$model" \
                    --embedding-mode "$EMB" \
                    --num-layers "$nlayers" \
                    --hidden 128 \
                    --heads 8 \
                    --epochs $JOINT_EPOCHS \
                    --lr $JOINT_LR \
                    --attr-weight $JOINT_ATTR_WEIGHT \
                    --neg-ratio $JOINT_NEG_RATIO
            fi
        done
    done
}

echo "============================================="
echo "  Layer Ablation - $(date)"
echo "  Output: $BASE_OUT_DIR/L{1,2,3,4}/"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "  Models: $GNN_MODELS"
echo "  Layers: $LAYERS"
echo "============================================="
echo ""

echo "===== TRANSDUCTIVE ====="
run_layer_ablation "$TRANS_DIR" "trans"

echo ""
echo "===== INDUCTIVE ====="
run_layer_ablation "$INDUC_DIR" "induc"

echo ""
echo "============================================="
echo "  DONE - $(date)"
total=0
for L in $LAYERS; do
    n=$(ls "$BASE_OUT_DIR/L${L}"/*.json 2>/dev/null | wc -l)
    echo "  L${L}: $n JSON files"
    total=$((total + n))
done
echo "  Total: $total JSON files"
echo "  Size:  $(du -sh "$BASE_OUT_DIR" 2>/dev/null | cut -f1)"
echo "============================================="
