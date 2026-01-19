#!/bin/bash
# Run NLI SOTA exploration - split into two GPU groups
cd /data/haofeiy2/artifact-graph/scripts

# ============== GPU 8: Group 1 (multi_nli, snli, scitail, mednli) ==============
python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name sick \
    --gpu-id 8 \
    --max-steps 10 ; \
python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name multi_nli \
    --gpu-id 7 \
    --max-steps 10

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name sick \
    --gpu-id 8 \
    --max-steps 10

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name scitail \
    --gpu-id 6 \
    --max-steps 10

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name mednli \
    --gpu-id 7 \
    --max-steps 10


# ============== GPU 2: Group 2 (wnli, qnli, rte, doc-nli) ==============
python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name wnli \
    --gpu-id 2 \
    --max-steps 15

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name qnli \
    --gpu-id 2 \
    --max-steps 15

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name rte \
    --gpu-id 2 \
    --max-steps 10

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name doc-nli \
    --gpu-id 7 \
    --max-steps 15

python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader             \
    --dataset-name babi_nli \
    --gpu-id 6 \
    --max-steps 10;\
python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name defeasible-nli \
    --gpu-id 6 \
    --max-steps 10 ;\
python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name nli_fever \
    --gpu-id 6 \
    --max-steps 10 ;\
python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name snli \
    --gpu-id 6 \
    --max-steps 10


python run_smolagent_advanced_coder.py \
    --json-file nli_sota_exploration.json \
    --llm-model gpt-5.2 \
    --output-dir smolagent_results_coding_agent_nli_sota_0112_full_shared_loader \
    --dataset-name contract-nli \
    --gpu-id 8 \
    --max-steps 10
