---
name: eval-templates
description: Runnable evaluation template scripts for ML tasks. Match task_type to template, adapt CONFIG, run.
---

# Evaluation Templates

Match the plan's task_type to a template in `scripts/`, adapt CONFIG variables, run via run_code_in_docker.

| task_type | Script | Notes |
|-----------|--------|-------|
| extractive_qa | eval_qa.py | Pipeline + no-answer threshold for SQuAD v2 |
| nli, classification | eval_nli.py | Label remapping, filter label=-1 |
| zero_shot | eval_zero_shot.py | Pipeline + candidate_labels |
| multilabel | eval_multilabel.py | Single-label argmax for accuracy metric |
| summarization | eval_summarization.py | Seq2Seq + ROUGE |
| ner | eval_ner.py | Token classification + seqeval, parquet fallback |
| vision | eval_vision.py | AutoImageProcessor, grayscale→RGB |
| sentiment | eval_sentiment.py | Standard classification |
| multiple_choice | eval_multiple_choice.py | vLLM + conditional log-likelihood |

## How to use
1. Read the matching template: `cat scripts/eval_<type>.py`
2. Adapt the CONFIG section at top (MODEL_ID, DATASET_ID, SPLIT, etc.)
3. Run via run_code_in_docker
