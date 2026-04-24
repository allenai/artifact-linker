#!/usr/bin/env python3
"""Extractive QA evaluation script (SQuAD v1/v2)."""

import subprocess, sys, json, re, string

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "deepset/roberta-base-squad2"
DATASET_ID = "rajpurkar/squad_v2"
DATASET_CONFIG = None
SPLIT = "validation"
METRIC_NAME = "exact_match"
SEED = 42
MAX_SAMPLES = 1000
NO_ANSWER_THRESHOLD = 0.20
# ─────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import pipeline


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Setup pipeline
    device = 0 if torch.cuda.is_available() else -1
    qa_pipe = pipeline("question-answering", model=MODEL_ID, device=device)

    correct = 0
    total = 0
    predictions = []

    for i, ex in enumerate(ds):
        question = ex["question"]
        context = ex["context"]
        gold_answers = ex["answers"]["text"]  # list of acceptable answers
        ground_truth = gold_answers[0] if gold_answers else ""

        # Run pipeline
        result = qa_pipe(question=question, context=context)
        raw_pred = result["answer"]
        score = result["score"]

        # Apply no-answer threshold for SQuAD v2
        if score < NO_ANSWER_THRESHOLD:
            pred = ""
        else:
            pred = raw_pred

        # Check exact match after normalization
        norm_pred = normalize_answer(pred)
        is_correct = False
        if not gold_answers:
            is_correct = norm_pred == ""
        else:
            is_correct = any(
                normalize_answer(gold) == norm_pred for gold in gold_answers
            )

        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "id": ex.get("id", str(i)),
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred,
            "score": score,
            "correct": is_correct,
        })

        # Debug: print first 5
        if i < 5:
            print(f"[{i}] Q: {question[:80]}")
            print(f"     Gold: {ground_truth[:80]}")
            print(f"     Pred: {pred[:80]} (score={score:.4f}, correct={is_correct})")
            print()

    accuracy = correct / total if total > 0 else 0.0
    results = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "metric_name": METRIC_NAME,
        "metric_value": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "no_answer_threshold": NO_ANSWER_THRESHOLD,
    }

    print(f"\n{METRIC_NAME}: {accuracy:.4f} ({correct}/{total})")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("Saved results.json and predictions.json")


if __name__ == "__main__":
    main()
