#!/usr/bin/env python3
"""Multilabel classification evaluation script (GoEmotions-style)."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "SamLowe/roberta-base-go_emotions"
DATASET_ID = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"
SPLIT = "test"
METRIC_NAME = "accuracy"
SEED = 42
MAX_SAMPLES = 1000
BATCH_SIZE = 32
TEXT_COL = "text"
# Set to "accuracy" for single-label approach, "f1" for multilabel threshold
EVAL_MODE = "accuracy"  # "accuracy" or "f1"
SIGMOID_THRESHOLD = 0.5
EXCLUDE_NEUTRAL = True
# ─────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_emotion_columns(column_names):
    """Identify emotion label columns (binary 0/1 columns)."""
    skip = {"text", "id", "comment_text", "example_very_unclear"}
    return [c for c in column_names if c not in skip]


def main():
    # Load dataset
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SPLIT,
        trust_remote_code=True,
    )

    emotion_cols = get_emotion_columns(ds.column_names)
    print(f"Detected {len(emotion_cols)} emotion columns: {emotion_cols[:5]}...")

    if EVAL_MODE == "accuracy":
        # Single-label approach: filter to exactly one active label
        ds = ds.filter(lambda ex: sum(ex[c] for c in emotion_cols) == 1)
        print(f"After single-label filter: {len(ds)} examples")

        if EXCLUDE_NEUTRAL and "neutral" in emotion_cols:
            ds = ds.filter(lambda ex: ex["neutral"] == 0)
            eval_cols = [c for c in emotion_cols if c != "neutral"]
            print(f"After excluding neutral: {len(ds)} examples")
        else:
            eval_cols = emotion_cols
    else:
        eval_cols = emotion_cols

    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # Build label mapping: model label name -> dataset column index
    model_id2label = model.config.id2label
    model_label_to_col = {}
    for model_idx, label_name in model_id2label.items():
        clean_name = label_name.lower().strip()
        if clean_name in eval_cols:
            model_label_to_col[int(model_idx)] = eval_cols.index(clean_name)

    print(f"Model labels: {len(model_id2label)}, Mapped: {len(model_label_to_col)}")

    correct = 0
    total = 0
    predictions = []

    for batch_start in range(0, len(ds), BATCH_SIZE):
        batch = ds[batch_start : batch_start + BATCH_SIZE]
        texts = batch[TEXT_COL]

        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        for i in range(len(texts)):
            idx = batch_start + i

            if EVAL_MODE == "accuracy":
                # Single-label: argmax
                model_pred_idx = logits[i].argmax().item()

                # Find gold label
                gold_label_idx = None
                for col_idx, col_name in enumerate(eval_cols):
                    if batch[col_name][i] == 1:
                        gold_label_idx = col_idx
                        break

                # Remap model prediction to eval_cols space
                pred_col_idx = model_label_to_col.get(model_pred_idx, -1)
                is_correct = pred_col_idx == gold_label_idx

                if is_correct:
                    correct += 1
                total += 1

                pred_entry = {
                    "index": idx,
                    "text": texts[i][:200],
                    "gold_label": eval_cols[gold_label_idx] if gold_label_idx is not None else "unknown",
                    "predicted_label": model_id2label.get(model_pred_idx, "unknown"),
                    "correct": is_correct,
                }
            else:
                # Multilabel: sigmoid threshold
                probs = torch.sigmoid(logits[i]).cpu().tolist()
                pred_binary = [1 if p >= SIGMOID_THRESHOLD else 0 for p in probs]

                gold_binary = [batch[c][i] for c in eval_cols]

                # Map model preds to eval_cols space
                mapped_preds = [0] * len(eval_cols)
                for model_idx, col_idx in model_label_to_col.items():
                    if model_idx < len(pred_binary):
                        mapped_preds[col_idx] = pred_binary[model_idx]

                is_correct = mapped_preds == gold_binary
                if is_correct:
                    correct += 1
                total += 1

                pred_entry = {
                    "index": idx,
                    "text": texts[i][:200],
                    "gold": gold_binary,
                    "predicted": mapped_preds,
                    "correct": is_correct,
                }

            predictions.append(pred_entry)

            # Debug: print first 5
            if idx < 5:
                print(f"[{idx}] Text: {texts[i][:80]}")
                print(f"     Gold: {pred_entry.get('gold_label', pred_entry.get('gold', ''))}")
                print(f"     Pred: {pred_entry.get('predicted_label', pred_entry.get('predicted', ''))}")
                print(f"     Correct: {is_correct}")
                print()

    metric_value = correct / total if total > 0 else 0.0
    results = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "metric_name": METRIC_NAME,
        "metric_value": round(metric_value, 4),
        "total": total,
        "correct": correct,
        "eval_mode": EVAL_MODE,
    }

    print(f"\n{METRIC_NAME}: {metric_value:.4f} ({correct}/{total})")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("Saved results.json and predictions.json")


if __name__ == "__main__":
    main()
