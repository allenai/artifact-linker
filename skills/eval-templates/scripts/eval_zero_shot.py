#!/usr/bin/env python3
"""Zero-shot classification evaluation script."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/bart-large-mnli"
DATASET_ID = "ag_news"
DATASET_CONFIG = None
SPLIT = "test"
METRIC_NAME = "accuracy"
SEED = 42
MAX_SAMPLES = 1000
TEXT_COL = "text"
LABEL_COL = "label"
CANDIDATE_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
HYPOTHESIS_TEMPLATE = "This text is about {}."
MULTI_LABEL = False
# ─────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import pipeline


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
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_ID,
        device=device,
    )

    # Map dataset integer labels to candidate label strings
    # Adjust this mapping based on your dataset
    label_int_to_str = {i: label for i, label in enumerate(CANDIDATE_LABELS)}

    correct = 0
    total = 0
    predictions = []

    for i, ex in enumerate(ds):
        text = ex[TEXT_COL]
        gold_label_int = ex[LABEL_COL]
        gold_label_str = label_int_to_str.get(gold_label_int, str(gold_label_int))

        # Run zero-shot classification
        result = classifier(
            text,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=MULTI_LABEL,
        )

        # Top predicted label
        pred_label_str = result["labels"][0]
        pred_score = result["scores"][0]

        is_correct = pred_label_str == gold_label_str
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "index": i,
            "text": text[:200],
            "gold_label": gold_label_str,
            "predicted_label": pred_label_str,
            "score": round(pred_score, 4),
            "all_scores": {
                label: round(score, 4)
                for label, score in zip(result["labels"], result["scores"])
            },
            "correct": is_correct,
        })

        # Debug: print first 5
        if i < 5:
            print(f"[{i}] Text: {text[:80]}")
            print(f"     Gold: {gold_label_str}")
            print(f"     Pred: {pred_label_str} (score={pred_score:.4f})")
            print(f"     Correct: {is_correct}")
            print()

    accuracy = correct / total if total > 0 else 0.0
    results = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "metric_name": METRIC_NAME,
        "metric_value": round(accuracy, 4),
        "total": total,
        "correct": correct,
    }

    print(f"\n{METRIC_NAME}: {accuracy:.4f} ({correct}/{total})")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("Saved results.json and predictions.json")


if __name__ == "__main__":
    main()
