#!/usr/bin/env python3
"""NER token classification evaluation script."""

import subprocess, sys, json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "evaluate", "huggingface_hub<1.0",
    "pyarrow", "fsspec", "torch", "seqeval",
])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID = "dslim/bert-base-NER"
DATASET_ID = "eriktks/conll2003"
DATASET_CONFIG = None
SPLIT = "test"
METRIC_NAME = "overall_f1"
SEED = 42
MAX_SAMPLES = 1000
BATCH_SIZE = 16
TOKENS_COL = "tokens"
TAGS_COL = "ner_tags"
PARQUET_FALLBACK_REVISION = "refs/convert/parquet"
# ─────────────────────────────────────────────────────────────────────────────

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_dataset_with_fallback(dataset_id, config, split):
    """Load dataset with parquet fallback for CoNLL-2003."""
    try:
        return load_dataset(dataset_id, config, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Primary load failed: {e}")
        print(f"Trying parquet fallback revision: {PARQUET_FALLBACK_REVISION}")
        return load_dataset(
            dataset_id, config, split=split,
            trust_remote_code=True,
            revision=PARQUET_FALLBACK_REVISION,
        )


def align_predictions(token_preds, word_ids):
    """Align sub-word token predictions to word-level using first sub-word."""
    word_preds = []
    prev_word_id = None
    for pred, word_id in zip(token_preds, word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            word_preds.append(pred)
        prev_word_id = word_id
    return word_preds


def main():
    # Load dataset
    ds = load_dataset_with_fallback(DATASET_ID, DATASET_CONFIG, SPLIT)
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

    # Get dataset tag names
    tag_feature = ds.features[TAGS_COL].feature
    dataset_tag_names = tag_feature.names
    dataset_id2label = {i: name for i, name in enumerate(dataset_tag_names)}
    print(f"Dataset tags: {dataset_tag_names}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    model_id2label = model.config.id2label
    print(f"Model id2label: {model_id2label}")

    # Build model-label to dataset-label remapping
    model_label_to_dataset_label = {}
    for model_idx, model_name in model_id2label.items():
        model_label_to_dataset_label[int(model_idx)] = model_name

    # Load seqeval metric
    seqeval = evaluate.load("seqeval")

    all_true_labels = []
    all_pred_labels = []
    predictions = []

    for i, ex in enumerate(ds):
        tokens = ex[TOKENS_COL]
        gold_tag_ids = ex[TAGS_COL]
        gold_labels = [dataset_id2label[t] for t in gold_tag_ids]

        # Tokenize
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        word_ids = encoded.word_ids()

        with torch.no_grad():
            logits = model(**encoded).logits
            token_preds = logits.argmax(dim=-1)[0].cpu().tolist()

        # Convert model predictions to label strings
        token_pred_labels = [
            model_label_to_dataset_label.get(p, "O") for p in token_preds
        ]

        # Align to word level
        word_pred_labels = align_predictions(token_pred_labels, word_ids)

        # Ensure same length
        min_len = min(len(gold_labels), len(word_pred_labels))
        gold_labels = gold_labels[:min_len]
        word_pred_labels = word_pred_labels[:min_len]

        all_true_labels.append(gold_labels)
        all_pred_labels.append(word_pred_labels)

        predictions.append({
            "index": i,
            "tokens": tokens[:20],
            "gold_labels": gold_labels[:20],
            "pred_labels": word_pred_labels[:20],
        })

        # Debug: print first 5
        if i < 5:
            print(f"[{i}] Tokens: {tokens[:10]}")
            print(f"     Gold:  {gold_labels[:10]}")
            print(f"     Pred:  {word_pred_labels[:10]}")
            print()

    # Compute seqeval metrics
    metric_results = seqeval.compute(
        predictions=all_pred_labels,
        references=all_true_labels,
    )

    metric_value = metric_results.get(METRIC_NAME, 0.0)

    results = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "metric_name": METRIC_NAME,
        "metric_value": round(metric_value, 4),
        "total": len(ds),
        "overall_f1": round(metric_results.get("overall_f1", 0.0), 4),
        "overall_precision": round(metric_results.get("overall_precision", 0.0), 4),
        "overall_recall": round(metric_results.get("overall_recall", 0.0), 4),
    }

    print(f"\n{METRIC_NAME}: {metric_value:.4f}")
    print(f"Precision: {metric_results.get('overall_precision', 0):.4f}")
    print(f"Recall: {metric_results.get('overall_recall', 0):.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("Saved results.json and predictions.json")


if __name__ == "__main__":
    main()
