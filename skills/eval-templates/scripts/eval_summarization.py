#!/usr/bin/env python3
"""Summarization evaluation script with ROUGE metric."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch", "rouge-score", "nltk",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
SPLIT = "test"
METRIC_NAME = "rouge2"
SEED = 42
MAX_SAMPLES = 200
INPUT_COL = "article"
REF_COL = "highlights"
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 142
MIN_OUTPUT_LENGTH = 56
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0
NO_REPEAT_NGRAM_SIZE = 3
BATCH_SIZE = 4
# ─────────────────────────────────────────────────────────────────────────────

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # Load ROUGE metric
    rouge = evaluate.load("rouge")

    all_preds = []
    all_refs = []
    predictions = []

    for batch_start in range(0, len(ds), BATCH_SIZE):
        batch = ds[batch_start : batch_start + BATCH_SIZE]
        articles = batch[INPUT_COL]
        references = batch[REF_COL]

        # Tokenize inputs
        inputs = tokenizer(
            articles,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate summaries
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                max_length=MAX_OUTPUT_LENGTH,
                min_length=MIN_OUTPUT_LENGTH,
                length_penalty=LENGTH_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            )

        # Decode
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]

        all_preds.extend(decoded_preds)
        all_refs.extend(references)

        for i, (pred, ref) in enumerate(zip(decoded_preds, references)):
            idx = batch_start + i
            predictions.append({
                "index": idx,
                "reference": ref[:300],
                "prediction": pred[:300],
            })

            # Debug: print first 5
            if idx < 5:
                print(f"[{idx}] Ref: {ref[:100]}")
                print(f"     Pred: {pred[:100]}")
                print()

    # Compute ROUGE scores
    metric_results = rouge.compute(
        predictions=all_preds,
        references=all_refs,
        use_stemmer=True,
    )

    metric_value = metric_results.get(METRIC_NAME, 0.0)

    results = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "metric_name": METRIC_NAME,
        "metric_value": round(metric_value, 4),
        "total": len(ds),
        "rouge1": round(metric_results.get("rouge1", 0.0), 4),
        "rouge2": round(metric_results.get("rouge2", 0.0), 4),
        "rougeL": round(metric_results.get("rougeL", 0.0), 4),
        "rougeLsum": round(metric_results.get("rougeLsum", 0.0), 4),
    }

    print(f"\nROUGE scores:")
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        print(f"  {k}: {metric_results.get(k, 0.0):.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("Saved results.json and predictions.json")


if __name__ == "__main__":
    main()
