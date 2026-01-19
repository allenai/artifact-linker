#!/bin/bash
# Run ONE-SHOT coding agent evaluations in parallel
# 3 parts = 3 jobs, max_steps=1 (no retry)

cd /data/haofeiy2/artifact-graph/scripts

echo "=========================================="
echo "ONE-SHOT Coding Agent Evaluation"
echo "=========================================="
echo "Mode: One-shot (max_steps=1, no retry)"
echo "Part 1: 43 entries"
echo "Part 2: 43 entries"  
echo "Part 3: 45 entries"
echo "=========================================="

# Part 1 (GPU 5)
python run_smolagent_oneshot_coder.py \
    --json-file perfect_model_dataset_metrics_v2_1125_coding_agent_part1.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_part1_oneshot \
    --gpu-id 5 &
PID1=$!
echo "Started Part 1 (PID: $PID1, GPU 5)"

# Part 2 (GPU 6)
python run_smolagent_oneshot_coder.py \
    --json-file perfect_model_dataset_metrics_v2_1125_coding_agent_part2.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_part2_oneshot \
    --gpu-id 6 &
PID2=$!
echo "Started Part 2 (PID: $PID2, GPU 6)"

# Part 3 (GPU 7)
python run_smolagent_oneshot_coder.py \
    --json-file perfect_model_dataset_metrics_v2_1125_coding_agent_part3.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_part3_oneshot \
    --gpu-id 7 &
PID3=$!
echo "Started Part 3 (PID: $PID3, GPU 7)"

echo ""
echo "=========================================="
echo "3 jobs running in parallel (ONE-SHOT)"
echo "GPUs: 5, 6, 7"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f smolagent_results_coding_agent_part*_oneshot/run_log_*.log"
echo ""

# Wait for all jobs
wait $PID1 $PID2 $PID3

echo "=========================================="
echo "All ONE-SHOT evaluations completed!"
echo "=========================================="
