#!/usr/bin/env python3
"""Text classification / sentiment evaluation script."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DATASET_ID = "sst2"
DATASET_CONFIG = None
SPLIT = "validation"
METRIC_NAME = "accuracy"
SEED = 42
MAX_SAMPLES = 1000
BATCH_SIZE = 32
TEXT_COL = "sentence"  # Check ds.column_names — may be "text", "verse_text", "content", etc.
LABEL_COL = "label"
# ─────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )

    print(f"Dataset columns: {ds.column_names}")
    print(f"Dataset size: {len(ds)}")

    # Use all examples if dataset is smaller than MAX_SAMPLES
    if len(ds) > MAX_SAMPLES:
        ds = ds.shuffle(seed=SEED).select(range(MAX_SAMPLES))

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    print(f"Model id2label: {model.config.id2label}")

    correct = 0
    total = 0
    predictions = []

    for batch_start in range(0, len(ds), BATCH_SIZE):
        batch = ds[batch_start : batch_start + BATCH_SIZE]
        texts = batch[TEXT_COL]
        labels = batch[LABEL_COL]

        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()

        for i, (pred, gold) in enumerate(zip(preds, labels)):
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1

            idx = batch_start + i
            pred_entry = {
                "index": idx,
                "text": texts[i][:200],
                "gold_label": gold,
                "predicted_label": pred,
                "gold_name": model.config.id2label.get(gold, str(gold)),
                "predicted_name": model.config.id2label.get(pred, str(pred)),
                "correct": is_correct,
            }
            predictions.append(pred_entry)

            # Debug: print first 5
            if idx < 5:
                print(f"[{idx}] Text: {texts[i][:80]}")
                print(f"     Gold: {pred_entry['gold_name']} ({gold})")
                print(f"     Pred: {pred_entry['predicted_name']} ({pred})")
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
