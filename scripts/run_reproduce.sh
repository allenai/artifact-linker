#!/bin/bash
# Reproduce ALL experiments: GNN + Baseline (link + attribute, prediction + ranking).
#
# Usage:
#   bash scripts/run_reproduce.sh                    # run everything
#   bash scripts/run_reproduce.sh --dry-run           # print commands without running
#   bash scripts/run_reproduce.sh --gnn-only          # GNN only
#   bash scripts/run_reproduce.sh --baseline-only     # baseline only
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_reproduce.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/haofeiy2/anaconda3/envs/artifact-linker/bin/python}"
SCRIPT_DIR="$ROOT/scripts"

DATA_DIR="$ROOT/data/artifact_graph_data_v3_0314"
TRANS_DIR="$ROOT/data/artifact_graph_splits_v3_0314_transductive"
INDUC_DIR="$ROOT/data/artifact_graph_splits_v3_0314_inductive"
OUT_DIR="$ROOT/data/final_results_reproduce"

# GNN config
GNN_MODELS="gatv2 gcn ncn ncnc neognn buddy"
EMBED_MODES="random embedding"
LINK_EPOCHS=1000
LINK_PATIENCE=40
LINK_THRESHOLD=0.9
LINK_NEG_RATIO=10
ATTR_EPOCHS=1000

# Baseline config
LINK_BASELINE_MODES="downloads random connectivity common_neighbors jaccard adamic_adar preferential_attachment resource_allocation katz matrix_factorization"
ATTR_BASELINE_MODES="dataset_average global_average model_average"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
DRY_RUN=false
RUN_GNN=true
RUN_BASELINE=true

for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --gnn-only)       RUN_BASELINE=false ;;
        --baseline-only)  RUN_GNN=false ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"

run_cmd() {
    echo "[CMD] $*"
    if $DRY_RUN; then return 0; fi
    "$@" 2>&1 | tail -20
    echo ""
}

skip_if_exists() {
    local f="$OUT_DIR/$1.json"
    if [ -f "$f" ]; then
        echo "  SKIP (exists): $1"
        return 0
    fi
    return 1
}

emb_tag() {
    if [ "$1" = "random" ]; then echo "random"; else echo "emb"; fi
}

# ===========================================================================
# GNN experiments
# ===========================================================================
run_gnn_suite() {
    local split_dir="$1"
    local prefix="$2"   # "trans" or "induc"

    for model in $GNN_MODELS; do
        for emb in $EMBED_MODES; do
            local tag
            tag=$(emb_tag "$emb")
            local model_link="$OUT_DIR/${prefix}_gnn_${model}_link_prediction_model_${tag}.pth"
            local model_attr="$OUT_DIR/${prefix}_gnn_${model}_attribute_prediction_model_${tag}.pth"

            # 1. Link Prediction (trains model)
            local lp="${prefix}_gnn_${model}_link_predictions_${tag}"
            if ! skip_if_exists "$lp"; then
                echo "=== [$prefix] GNN $model link_prediction ($tag) ==="
                run_cmd "$PYTHON" "$SCRIPT_DIR/predict_link_gnn.py" \
                    --split-dir "$split_dir" \
                    --output-dir "$OUT_DIR" \
                    --gnn-model "$model" \
                    --embedding-mode "$emb" \
                    --epochs $LINK_EPOCHS \
                    --patience $LINK_PATIENCE \
                    --threshold $LINK_THRESHOLD \
                    --neg-ratio $LINK_NEG_RATIO \
                    --save-model-path "$model_link"
            fi

            # 2. Link Ranking (uses trained model)
            local lr="${prefix}_gnn_${model}_link_rankings_${tag}"
            if ! skip_if_exists "$lr"; then
                if [ -f "$model_link" ]; then
                    echo "=== [$prefix] GNN $model link_ranking ($tag) ==="
                    run_cmd "$PYTHON" "$SCRIPT_DIR/rank_link_gnn.py" \
                        --split-dir "$split_dir" \
                        --output-dir "$OUT_DIR" \
                        --gnn-model "$model" \
                        --embedding-mode "$emb" \
                        --model-path "$model_link"
                else
                    echo "  SKIP (no model): $lr"
                fi
            fi

            # 3. Attribute Prediction (trains model)
            local ap="${prefix}_gnn_${model}_attr_predictions_${tag}"
            if ! skip_if_exists "$ap"; then
                echo "=== [$prefix] GNN $model attr_prediction ($tag) ==="
                run_cmd "$PYTHON" "$SCRIPT_DIR/predict_attribute_gnn.py" \
                    --split-dir "$split_dir" \
                    --output-dir "$OUT_DIR" \
                    --gnn-model "$model" \
                    --embedding-mode "$emb" \
                    --epochs $ATTR_EPOCHS \
                    --save-model-path "$model_attr"
            fi

            # 4. Attribute Ranking (uses trained model)
            local ar="${prefix}_gnn_${model}_attr_rankings_${tag}"
            if ! skip_if_exists "$ar"; then
                if [ -f "$model_attr" ]; then
                    echo "=== [$prefix] GNN $model attr_ranking ($tag) ==="
                    run_cmd "$PYTHON" "$SCRIPT_DIR/rank_attribute_gnn.py" \
                        --split-dir "$split_dir" \
                        --output-dir "$OUT_DIR" \
                        --gnn-model "$model" \
                        --embedding-mode "$emb" \
                        --model-path "$model_attr"
                else
                    echo "  SKIP (no model): $ar"
                fi
            fi
        done
    done
}

# ===========================================================================
# Baseline experiments
# ===========================================================================
run_baseline_suite() {
    local split_dir="$1"
    local prefix="$2"   # "trans" or "induc"

    # -- Link Prediction --
    for mode in $LINK_BASELINE_MODES; do
        local name="${prefix}_baseline_link_predictions_${mode}"
        if ! skip_if_exists "$name"; then
            echo "=== [$prefix] baseline link_prediction ($mode) ==="
            run_cmd "$PYTHON" "$SCRIPT_DIR/predict_link_baseline.py" \
                --data-dir "$DATA_DIR" \
                --split-dir "$split_dir" \
                --output-dir "$OUT_DIR" \
                --mode "$mode" \
                --threshold $LINK_THRESHOLD
        fi
    done

    # -- Link Ranking --
    for mode in $LINK_BASELINE_MODES; do
        local name="${prefix}_baseline_link_rankings_${mode}"
        if ! skip_if_exists "$name"; then
            echo "=== [$prefix] baseline link_ranking ($mode) ==="
            run_cmd "$PYTHON" "$SCRIPT_DIR/rank_link_baseline.py" \
                --data-dir "$DATA_DIR" \
                --split-dir "$split_dir" \
                --output-dir "$OUT_DIR" \
                --mode "$mode"
        fi
    done

    # -- Attribute Prediction --
    for mode in $ATTR_BASELINE_MODES; do
        local name="${prefix}_baseline_attr_predictions_${mode}"
        if ! skip_if_exists "$name"; then
            echo "=== [$prefix] baseline attr_prediction ($mode) ==="
            run_cmd "$PYTHON" "$SCRIPT_DIR/predict_attribute_baseline.py" \
                --data-dir "$DATA_DIR" \
                --split-dir "$split_dir" \
                --output-dir "$OUT_DIR" \
                --mode "$mode"
        fi
    done

    # -- Attribute Ranking --
    for mode in $ATTR_BASELINE_MODES; do
        local name="${prefix}_baseline_attr_rankings_${mode}"
        if ! skip_if_exists "$name"; then
            echo "=== [$prefix] baseline attr_ranking ($mode) ==="
            run_cmd "$PYTHON" "$SCRIPT_DIR/rank_attribute_baseline.py" \
                --data-dir "$DATA_DIR" \
                --split-dir "$split_dir" \
                --output-dir "$OUT_DIR" \
                --mode "$mode"
        fi
    done
}

# ===========================================================================
# Run
# ===========================================================================
echo "============================================="
echo "  Reproduce Experiments - $(date)"
echo "  Output: $OUT_DIR"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "  GNN: $RUN_GNN | Baseline: $RUN_BASELINE"
echo "============================================="
echo ""

if $RUN_GNN; then
    echo "########## GNN EXPERIMENTS ##########"
    echo ""
    echo "===== GNN TRANSDUCTIVE ====="
    run_gnn_suite "$TRANS_DIR" "trans"
    echo ""
    echo "===== GNN INDUCTIVE ====="
    run_gnn_suite "$INDUC_DIR" "induc"
    echo ""
fi

if $RUN_BASELINE; then
    echo "########## BASELINE EXPERIMENTS ##########"
    echo ""
    echo "===== BASELINE TRANSDUCTIVE ====="
    run_baseline_suite "$TRANS_DIR" "trans"
    echo ""
    echo "===== BASELINE INDUCTIVE ====="
    run_baseline_suite "$INDUC_DIR" "induc"
    echo ""
fi

echo "============================================="
echo "  DONE - $(date)"
echo "  Results: $(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l) JSON files"
echo "  Size:    $(du -sh "$OUT_DIR" | cut -f1)"
echo "============================================="
