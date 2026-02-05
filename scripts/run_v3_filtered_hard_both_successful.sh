#!/bin/bash
# Run evaluations on v3 filtered hard dataset (263 tuples where both model and dataset load successfully)
# Four modes using the new EvaluationCoder class:
#   - oneturn_onetool: Single turn, only run_code_in_docker (max_steps=1)
#   - multiturn_onetool: Multi-turn, only run_code_in_docker
#   - multiturn_metadatatool: Multi-turn, with metadata tools + base_tools
#   - multiturn_cachefiletool: Multi-turn, with all tools including cached loaders

cd /data/haofeiy2/artifact-linker/scripts

JSON_FILE="perfect_model_dataset_metrics_v3_0120_coding_agent_filtered_hard_both_successful.json"
LLM_MODEL="gpt-5.2"

echo "=========================================="
echo "V3 Filtered Hard - Both Successful (263 tuples)"
echo "=========================================="
echo "JSON: $JSON_FILE"
echo "LLM: $LLM_MODEL"
echo "=========================================="

# ============== MODE 1: ONETURN_ONETOOL ==============
run_oneturn_onetool() {
    echo "Mode: ONETURN_ONETOOL (single turn, only docker run)"
    python run_smolagent_evaluation_coder.py \
        --json-file $JSON_FILE \
        --mode oneturn_onetool \
        --llm-model $LLM_MODEL \
        --output-dir smolagent_results_v3_hard_oneturn_onetool \
        --gpu-id 5
    echo "ONETURN_ONETOOL completed!"
}

# ============== MODE 2: MULTITURN_ONETOOL ==============
run_multiturn_onetool() {
    echo "Mode: MULTITURN_ONETOOL (multi-turn, only docker run)"
    python run_smolagent_evaluation_coder.py \
        --json-file $JSON_FILE \
        --mode multiturn_onetool \
        --llm-model $LLM_MODEL \
        --output-dir smolagent_results_v3_hard_multiturn_onetool \
        --gpu-id 5 
    echo "MULTITURN_ONETOOL completed!"
}

# ============== MODE 3: MULTITURN_METADATATOOL ==============
run_multiturn_metadatatool() {
    echo "Mode: MULTITURN_METADATATOOL (multi-turn, metadata tools + base_tools)"
    python run_smolagent_evaluation_coder.py \
        --json-file $JSON_FILE \
        --mode multiturn_metadatatool \
        --llm-model $LLM_MODEL \
        --output-dir smolagent_results_v3_hard_multiturn_metadatatool \
        --gpu-id 5
    echo "MULTITURN_METADATATOOL completed!"
}

# ============== MODE 4: MULTITURN_CACHEFILETOOL ==============
run_multiturn_cachefiletool() {
    echo "Mode: MULTITURN_CACHEFILETOOL (multi-turn, all tools + cached loaders)"
    python run_smolagent_evaluation_coder.py \
        --json-file $JSON_FILE \
        --mode multiturn_cachefiletool \
        --llm-model $LLM_MODEL \
        --output-dir smolagent_results_v3_hard_multiturn_cachefiletool \
        --gpu-id 5
    echo "MULTITURN_CACHEFILETOOL completed!"
}

# ============== RUN SELECTED MODE ==============
# Usage: ./run_v3_filtered_hard_both_successful.sh [mode|all]

MODE=${1:-"all"}

case $MODE in
    oneturn_onetool)
        run_oneturn_onetool
        ;;
    multiturn_onetool)
        run_multiturn_onetool
        ;;
    multiturn_metadatatool)
        run_multiturn_metadatatool
        ;;
    multiturn_cachefiletool)
        run_multiturn_cachefiletool
        ;;
    all)
        echo "Running ALL modes in parallel on different GPUs..."
        run_oneturn_onetool &
        PID1=$!
        run_multiturn_onetool &
        PID2=$!
        run_multiturn_metadatatool &
        PID3=$!
        run_multiturn_cachefiletool &
        PID4=$!
        wait $PID1 $PID2 $PID3 $PID4
        ;;
    *)
        echo "Usage: $0 [oneturn_onetool|multiturn_onetool|multiturn_metadatatool|multiturn_cachefiletool|all]"
        echo ""
        echo "Modes:"
        echo "  oneturn_onetool       - Single turn, only run_code_in_docker (GPU 2)"
        echo "  multiturn_onetool     - Multi-turn, only run_code_in_docker (GPU 6)"
        echo "  multiturn_metadatatool - Multi-turn, metadata tools + base_tools (GPU 5)"
        echo "  multiturn_cachefiletool - Multi-turn, all tools + cached loaders (GPU 7)"
        echo "  all                   - Run all modes in parallel"
        exit 1
        ;;
esac

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
