#!/bin/bash
# Stage-1 LLM: link + attribute prediction & ranking via an LLM (litellm format).
# Produces the paper Table-2 LLM rows (and LLM link rows if reported).
# +graph variant = HOPS=1 (1-hop neighbour context); plain = HOPS=0.
#
# Requires: OPENAI_API_KEY (or the provider key your --llm-model needs).
#
# Usage:
#   OPENAI_API_KEY=... bash scripts/run_llm.sh
#   LLM_MODEL=openai/gpt-5.2 HOPS=1 SPLITS=trans bash scripts/run_llm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/home/haofeiy2/.conda/envs/artifact-linker/bin/python}"
OUT="${OUT_DIR:-$ROOT/data/llm_sweep}"
mkdir -p "$OUT"

DATA="$ROOT/data/artifact_graph_data_v3_0314"
TRANS="$ROOT/data/artifact_graph_splits_v3_0314_transductive"
INDUC="$ROOT/data/artifact_graph_splits_v3_0314_inductive"

LLM_MODEL="${LLM_MODEL:-openai/gpt-5.2}"
HOPS="${HOPS:-1}"                 # 1 = "+graph", 0 = no graph context
SPLITS="${SPLITS:-trans induc}"
TASKS="${TASKS:-link_pred link_rank attr_pred attr_rank}"

if [[ -z "${OPENAI_API_KEY:-}" && "$LLM_MODEL" == openai/* ]]; then
    echo "ERROR: OPENAI_API_KEY not set (needed for $LLM_MODEL)"; exit 1
fi

for split in $SPLITS; do
    case "$split" in
        trans) SD="$TRANS" ;; induc) SD="$INDUC" ;;
        *) echo "unknown split: $split"; exit 1 ;;
    esac
    for t in $TASKS; do
        case "$t" in
            link_pred) S=predict_link_llm.py ;;
            link_rank) S=rank_link_llm.py ;;
            attr_pred) S=predict_attribute_llm.py ;;
            attr_rank) S=rank_attribute_llm.py ;;
            *) echo "unknown task: $t"; exit 1 ;;
        esac
        echo "=== $split / $t / $LLM_MODEL / hops=$HOPS ==="
        "$PY" "$ROOT/scripts/$S" --data-dir "$DATA" --split-dir "$SD" \
            --output-dir "$OUT" --llm-model "$LLM_MODEL" --hops "$HOPS"
    done
done
echo "DONE. results: $OUT"
