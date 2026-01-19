#!/bin/bash
# Re-run failed NLI SOTA experiments with larger max_steps
# 22 experiments without results.json

cd /data/haofeiy2/artifact-graph/scripts

echo "=========================================="
echo "Re-running Failed NLI SOTA Experiments"
echo "=========================================="
echo "Total: 22 experiments"
echo "Max steps: 15 (increased from 10)"
echo ""

# Run directly with increased max-steps
python run_failed_nli_sota.py \
    --max-steps 20 \
    --gpu-id 6 \
    --llm-model gpt-5.2

echo "=========================================="
echo "Re-run completed!"
echo "=========================================="

# Re-analyze results
echo ""
echo "Re-analyzing results..."
python analyze_nli_sota.py

