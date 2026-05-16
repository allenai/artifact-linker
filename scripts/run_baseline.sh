#!/bin/bash
# Stage-1 baselines (deterministic): link heuristics + attribute means.
# Reproduces paper Table-1 Heuristic rows and Table-2 Heuristic rows exactly.
#
# Usage:
#   bash scripts/run_baseline.sh
#   LINK_MODES="katz" ATTR_MODES="model_average" bash scripts/run_baseline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/home/haofeiy2/.conda/envs/artifact-linker/bin/python}"
OUT="${OUT_DIR:-$ROOT/data/baseline_sweep}"
mkdir -p "$OUT"

DATA="$ROOT/data/artifact_graph_data_v3_0314"
TRANS="$ROOT/data/artifact_graph_splits_v3_0314_transductive"
INDUC="$ROOT/data/artifact_graph_splits_v3_0314_inductive"

# paper Table-1 Heuristic: adamic_adar, matrix_factorization, katz
LINK_MODES="${LINK_MODES:-adamic_adar matrix_factorization katz}"
# paper Table-2 Heuristic: global/model/dataset mean
ATTR_MODES="${ATTR_MODES:-global_average model_average dataset_average}"
SPLITS="${SPLITS:-trans induc}"

pids=()
for split in $SPLITS; do
    case "$split" in
        trans) SD="$TRANS" ;; induc) SD="$INDUC" ;;
        *) echo "unknown split: $split"; exit 1 ;;
    esac
    for m in $LINK_MODES; do
        "$PY" "$ROOT/scripts/predict_link_baseline.py" --data-dir "$DATA" --split-dir "$SD" \
            --output-dir "$OUT" --mode "$m" --threshold 0.9 & pids+=($!)
        "$PY" "$ROOT/scripts/rank_link_baseline.py"    --data-dir "$DATA" --split-dir "$SD" \
            --output-dir "$OUT" --mode "$m" & pids+=($!)
    done
    for m in $ATTR_MODES; do
        "$PY" "$ROOT/scripts/predict_attribute_baseline.py" --data-dir "$DATA" --split-dir "$SD" \
            --output-dir "$OUT" --mode "$m" & pids+=($!)
        "$PY" "$ROOT/scripts/rank_attribute_baseline.py"    --data-dir "$DATA" --split-dir "$SD" \
            --output-dir "$OUT" --mode "$m" & pids+=($!)
    done
done
wait "${pids[@]}"
echo "DONE. results: $OUT"
