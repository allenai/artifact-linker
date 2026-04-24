#!/usr/bin/env python3
"""Image classification evaluation script."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch", "torchvision", "Pillow",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "google/vit-base-patch16-224"
DATASET_ID = "imagenet-1k"
DATASET_CONFIG = None
SPLIT = "validation"
METRIC_NAME = "accuracy"
SEED = 42
MAX_SAMPLES = 1000
IMAGE_COL = "image"
LABEL_COL = "label"
CONVERT_RGB = True  # Set True for grayscale datasets (MNIST, etc.)
# ─────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    print(f"Model labels: {len(model.config.id2label)} classes")

    correct = 0
    total = 0
    predictions = []

    for i, ex in enumerate(ds):
        image = ex[IMAGE_COL]
        gold_label = ex[LABEL_COL]

        # Convert grayscale to RGB if needed
        if CONVERT_RGB:
            image = image.convert("RGB")

        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_label = logits.argmax(dim=-1).item()

        is_correct = pred_label == gold_label
        if is_correct:
            correct += 1
        total += 1

        pred_name = model.config.id2label.get(pred_label, str(pred_label))
        gold_name = model.config.id2label.get(gold_label, str(gold_label))

        predictions.append({
            "index": i,
            "gold_label": gold_label,
            "gold_name": gold_name,
            "predicted_label": pred_label,
            "predicted_name": pred_name,
            "correct": is_correct,
        })

        # Debug: print first 5
        if i < 5:
            print(f"[{i}] Gold: {gold_name} ({gold_label})")
            print(f"     Pred: {pred_name} ({pred_label})")
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
