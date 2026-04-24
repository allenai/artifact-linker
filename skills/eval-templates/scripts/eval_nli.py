#!/usr/bin/env python3
"""NLI / Sequence classification evaluation script."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch", "sentencepiece",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "cross-encoder/nli-deberta-v3-base"
DATASET_ID = "nli"
DATASET_CONFIG = "multi_nli"
SPLIT = "validation_matched"
METRIC_NAME = "accuracy"
SEED = 42
MAX_SAMPLES = 1000
BATCH_SIZE = 32
PREMISE_COL = "premise"
HYPOTHESIS_COL = "hypothesis"
LABEL_COL = "label"
# Dataset label names in index order
DATASET_LABEL_NAMES = ["entailment", "neutral", "contradiction"]
# ─────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def build_label_remap(model_id2label, dataset_label_names):
    """Build mapping from model label index to dataset label index."""
    model_name_to_idx = {}
    for idx, name in model_id2label.items():
        model_name_to_idx[name.lower()] = int(idx)

    dataset_name_to_idx = {name.lower(): i for i, name in enumerate(dataset_label_names)}

    remap = {}
    for name, model_idx in model_name_to_idx.items():
        if name in dataset_name_to_idx:
            remap[model_idx] = dataset_name_to_idx[name]
    return remap


def main():
    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )

    # Filter out label == -1
    ds = ds.filter(lambda ex: ex[LABEL_COL] != -1)
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # Build label remapping
    remap = build_label_remap(model.config.id2label, DATASET_LABEL_NAMES)
    print(f"Model id2label: {model.config.id2label}")
    print(f"Label remap (model_idx -> dataset_idx): {remap}")

    correct = 0
    total = 0
    predictions = []

    for batch_start in range(0, len(ds), BATCH_SIZE):
        batch = ds[batch_start : batch_start + BATCH_SIZE]
        premises = batch[PREMISE_COL]
        hypotheses = batch[HYPOTHESIS_COL]
        labels = batch[LABEL_COL]

        inputs = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            model_preds = logits.argmax(dim=-1).cpu().tolist()

        for i, (model_pred, gold_label) in enumerate(zip(model_preds, labels)):
            # Remap model prediction to dataset label space
            dataset_pred = remap.get(model_pred, model_pred)
            is_correct = dataset_pred == gold_label
            if is_correct:
                correct += 1
            total += 1

            idx = batch_start + i
            pred_entry = {
                "index": idx,
                "premise": premises[i][:200],
                "hypothesis": hypotheses[i][:200],
                "gold_label": gold_label,
                "predicted_label": dataset_pred,
                "correct": is_correct,
            }
            predictions.append(pred_entry)

            # Debug: print first 5
            if idx < 5:
                print(f"[{idx}] P: {premises[i][:80]}")
                print(f"     H: {hypotheses[i][:80]}")
                print(f"     Gold: {gold_label}, Pred: {dataset_pred}, Correct: {is_correct}")
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
