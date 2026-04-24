#!/usr/bin/env python3
"""Multiple choice evaluation script using vLLM conditional log-likelihood."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch", "vllm", "compressed-tensors",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-2-7b-hf"
DATASET_ID = "Rowan/hellaswag"
DATASET_CONFIG = None
SPLIT = "validation"
METRIC_NAME = "accuracy"
SEED = 42
MAX_SAMPLES = 1000
GPU_MEMORY_UTILIZATION = 0.5
IS_INSTRUCT_MODEL = False  # Set True for instruct/chat models
# ─────────────────────────────────────────────────────────────────────────────

from datasets import load_dataset
from transformers import AutoTokenizer


def build_prompts_hellaswag(example, tokenizer, is_instruct):
    """Build base and full prompts for each choice in a HellaSwag example."""
    ctx = example["ctx_a"] + " " + example["ctx_b"]
    ctx = ctx.strip()
    endings = example["endings"]
    label = int(example["label"])

    base_prompts = []
    full_prompts = []

    for ending in endings:
        if is_instruct:
            base_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": ctx}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": ctx},
                 {"role": "assistant", "content": ending}],
                tokenize=False,
            )
        else:
            base_text = ctx
            full_text = ctx + " " + ending

        base_prompts.append(base_text)
        full_prompts.append(full_text)

    return base_prompts, full_prompts, label, endings


def score_choices(llm, sampling_params, tokenizer, base_prompts, full_prompts):
    """Score each choice by conditional log-likelihood."""
    scores = []

    for base_prompt, full_prompt in zip(base_prompts, full_prompts):
        base_token_ids = tokenizer.encode(base_prompt)
        full_token_ids = tokenizer.encode(full_prompt)

        # Get logprobs for the full prompt
        outputs = llm.generate(
            prompt_token_ids=[full_token_ids],
            sampling_params=sampling_params,
        )

        # Sum logprobs for continuation tokens (those beyond base)
        prompt_logprobs = outputs[0].prompt_logprobs
        n_base = len(base_token_ids)
        total_logprob = 0.0

        if prompt_logprobs is not None:
            for idx in range(n_base, len(full_token_ids)):
                if idx < len(prompt_logprobs) and prompt_logprobs[idx] is not None:
                    token_id = full_token_ids[idx]
                    if token_id in prompt_logprobs[idx]:
                        total_logprob += prompt_logprobs[idx][token_id].logprob

        scores.append(total_logprob)

    return scores


def main():
    from vllm import LLM, SamplingParams

    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Initialize vLLM
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    correct = 0
    total = 0
    predictions = []

    for i, ex in enumerate(ds):
        base_prompts, full_prompts, label, endings = build_prompts_hellaswag(
            ex, tokenizer, IS_INSTRUCT_MODEL
        )

        scores = score_choices(llm, sampling_params, tokenizer, base_prompts, full_prompts)
        pred = max(range(len(scores)), key=lambda j: scores[j])

        is_correct = pred == label
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "input": {"ctx": (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip(), "endings": endings},
            "prediction": pred,
            "ground_truth": label,
        })

        # Debug: print first 5
        if i < 5:
            ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
            print(f"[{i}] Context: {ctx[:80]}")
            print(f"     Endings: {[e[:40] for e in endings]}")
            print(f"     Scores: {[round(s, 4) for s in scores]}")
            print(f"     Gold: {label}, Pred: {pred}, Correct: {is_correct}")
            print()

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{METRIC_NAME}: {accuracy:.4f} ({correct}/{total})")
    print("First 5 (prediction, ground_truth) pairs:")
    for p in predictions[:5]:
        print((p["prediction"], p["ground_truth"]))

    with open("results.json", "w") as f:
        json.dump({METRIC_NAME: round(accuracy, 4)}, f)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)

    print("Saved results.json and predictions.json")


if __name__ == "__main__":
    main()
