#!/bin/bash
# Run ADVANCED coding agent evaluations in parallel
# Uses enhanced tools: model/dataset metadata, readme, pip install
# 3 parts = 3 jobs

cd /data/haofeiy2/artifact-graph/scripts

echo "=========================================="
echo "ADVANCED Coding Agent Evaluation"
echo "=========================================="
echo "Script: run_smolagent_advanced_coder.py"
echo "Tools: get_model_readme, get_dataset_readme,"
echo "       get_model_metadata, get_dataset_metadata,"
echo "       run_code_in_docker, read_file, save_file"
echo "=========================================="
echo "Part 1: 29 entries"
echo "Part 2: 29 entries"  
echo "Part 3: 36 entries"
echo "=========================================="

# Part 1 (GPU 5)
python run_smolagent_advanced_coder.py \
    --json-file perfect_model_dataset_metrics_v2_1125_coding_agent_part1.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_part1_advanced \
    --gpu-id 5 &
PID1=$!
echo "Started Part 1 (PID: $PID1, GPU 5)"

# Part 2 (GPU 6)
python run_smolagent_advanced_coder.py \
    --json-file perfect_model_dataset_metrics_v2_1125_coding_agent_part2.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_part2_advanced \
    --gpu-id 6 &
PID2=$!
echo "Started Part 2 (PID: $PID2, GPU 6)"

# Part 3 (GPU 7)
python run_smolagent_advanced_coder.py \
    --json-file perfect_model_dataset_metrics_v2_1125_coding_agent_part3.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_part3_advanced \
    --gpu-id 7 &
PID3=$!
echo "Started Part 3 (PID: $PID3, GPU 7)"

echo ""
echo "=========================================="
echo "3 jobs running in parallel"
echo "GPUs: 5, 6, 7"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f smolagent_results_coding_agent_part*_advanced/run_log_*.log"
echo ""

# Wait for all jobs
wait $PID1 $PID2 $PID3

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

