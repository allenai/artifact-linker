#!/bin/bash
# Stage-2 verification: LLM coding agent writes & runs real evaluation code in a
# Docker sandbox to verify top-ranked (model, dataset, metric) triples.
#
# Requires:
#   - OPENAI_API_KEY (or provider key for --llm-model)
#   - Docker image artifact-linker-verification:latest  (build: bash build_docker.sh)
#
# Usage:
#   OPENAI_API_KEY=... bash scripts/run_verification.sh
#   LIMIT=1 GPU=0 bash scripts/run_verification.sh          # quick single-triple test
#   LLM_MODEL=openai/gpt-5.2 MODE=multiturn_cachefiletool bash scripts/run_verification.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/home/haofeiy2/.conda/envs/artifact-linker/bin/python}"

# Canonical Stage-2 input: verification_bench/bench.json from the HF dataset
# (download in Quick Start step 1).
JSON="${JSON_FILE:-$ROOT/data/verification_bench/bench.json}"
LLM_MODEL="${LLM_MODEL:-openai/gpt-5.2}"
MODE="${MODE:-multiturn_cachefiletool}"   # oneturn_onetool|multiturn_onetool|multiturn_metadatatool|multiturn_cachefiletool
GPU="${GPU:-0}"
LIMIT="${LIMIT:-0}"                       # 0 = all triples
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
OUT_DIR="${OUT_DIR:-}"

if [[ -z "${OPENAI_API_KEY:-}" && "$LLM_MODEL" == openai/* ]]; then
    echo "ERROR: OPENAI_API_KEY not set (needed for $LLM_MODEL)"; exit 1
fi
if ! docker image inspect artifact-linker-verification:latest >/dev/null 2>&1; then
    echo "ERROR: docker image artifact-linker-verification:latest missing."
    echo "       build it first:  bash build_docker.sh"
    exit 1
fi

ARGS=(--json-file "$JSON" --mode "$MODE" --llm-model "$LLM_MODEL"
      --gpu-id "$GPU" --limit "$LIMIT" --max-samples "$MAX_SAMPLES")
[[ -n "$OUT_DIR" ]] && ARGS+=(--output-dir "$OUT_DIR")

echo "=== Stage-2 verification | model=$LLM_MODEL mode=$MODE gpu=$GPU limit=$LIMIT ==="
exec "$PY" "$ROOT/scripts/verify_attribute_agent.py" "${ARGS[@]}"
