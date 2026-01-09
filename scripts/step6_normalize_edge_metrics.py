#!/usr/bin/env python3
"""
Step 6: Normalize edge metrics using GPT to classify metric names.

This script:
1. Reads edge_metadata.json
2. Uses GPT to classify each unique metric name to standard categories
3. Filters to only keep: accuracy, bleu, chrf, f1, rouge-2, rouge-l, top-k_accuracy, wer
4. Normalizes values to 0-1 range
5. Outputs a filtered edge_metadata.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Standard metric names we want to keep
STANDARD_METRICS = {
    "accuracy",
    "bleu", 
    "chrf",
    "f1",
    "rouge-2",
    "rouge-l",
    "top-k_accuracy",
    "wer",
}

# Simple rule-based mapping for common variations (fallback if no GPT)
SIMPLE_METRIC_MAPPING = {
    # Accuracy variants
    "accuracy": "accuracy",
    "acc": "accuracy",
    "Accuracy": "accuracy",
    "ACC": "accuracy",
    "top1": "accuracy",
    "top-1": "accuracy",
    "top_1": "accuracy",
    "top1_accuracy": "accuracy",
    "exact_match": "accuracy",
    "em": "accuracy",
    "EM": "accuracy",
    
    # Top-k accuracy variants
    "top-k_accuracy": "top-k_accuracy",
    "top5": "top-k_accuracy",
    "top-5": "top-k_accuracy",
    "top_5": "top-k_accuracy",
    "top5_accuracy": "top-k_accuracy",
    "top10": "top-k_accuracy",
    "top-10": "top-k_accuracy",
    
    # BLEU variants
    "bleu": "bleu",
    "BLEU": "bleu",
    "Bleu": "bleu",
    "bleu-4": "bleu",
    "BLEU-4": "bleu",
    "bleu_score": "bleu",
    "sacrebleu": "bleu",
    
    # chrF variants
    "chrf": "chrf",
    "chr-F": "chrf",
    "chrF": "chrf",
    "chr-f": "chrf",
    "chrf++": "chrf",
    "chrF++": "chrf",
    
    # F1 variants
    "f1": "f1",
    "F1": "f1",
    "f1_score": "f1",
    "f1-score": "f1",
    "f1_macro": "f1",
    "f1-macro": "f1",
    "f1_micro": "f1",
    "f1-micro": "f1",
    "macro_f1": "f1",
    "micro_f1": "f1",
    "macro-f1": "f1",
    
    # ROUGE-2 variants
    "rouge-2": "rouge-2",
    "ROUGE-2": "rouge-2",
    "rouge2": "rouge-2",
    "ROUGE2": "rouge-2",
    "rougeL": "rouge-l",
    
    # ROUGE-L variants
    "rouge-l": "rouge-l",
    "ROUGE-L": "rouge-l",
    "rougeL": "rouge-l",
    "rouge_l": "rouge-l",
    "ROUGE_L": "rouge-l",
    "rougeLsum": "rouge-l",
    
    # WER variants
    "wer": "wer",
    "WER": "wer",
    "word_error_rate": "wer",
}


def get_gpt_metric_mapping(metric_names: List[str], model: str = "gpt-4o") -> Dict[str, Optional[str]]:
    """Use GPT to classify metric names to standard categories."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai not installed, using rule-based mapping only")
        return {}
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, using rule-based mapping only")
        return {}
    
    client = OpenAI(api_key=api_key)
    
    # Build prompt
    prompt = f"""You are a machine learning metrics expert. Given a list of metric names, classify each one into one of these standard categories:
- accuracy: metrics measuring prediction correctness (accuracy, acc, top1, exact_match, em, etc.)
- bleu: BLEU score variants for translation/generation
- chrf: chrF score variants
- f1: F1 score variants (f1, f1_macro, f1_micro, etc.)
- rouge-2: ROUGE-2 score
- rouge-l: ROUGE-L/ROUGE-Lsum score  
- top-k_accuracy: top-k accuracy metrics (top5, top10, etc.) - NOT top1 which is accuracy
- wer: Word Error Rate
- null: if the metric doesn't fit any category above (e.g., loss, perplexity, precision alone, recall alone, etc.)

Return a JSON object mapping each input metric name to its standard category (or null).

Metric names to classify:
{json.dumps(metric_names, indent=2)}

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
        )
        
        content = response.choices[0].message.content.strip()
        # Parse JSON from response
        content = content.replace("```json", "").replace("```", "").strip()
        mapping = json.loads(content)
        
        # Convert "null" strings to None
        return {k: (v if v != "null" else None) for k, v in mapping.items()}
    
    except Exception as e:
        print(f"GPT API error: {e}")
        return {}


def normalize_value(value: float, metric_name: str) -> float:
    """Normalize metric value to 0-1 range."""
    if value is None:
        return None
    
    # WER is typically 0-100 or 0-1, lower is better but we keep as-is for now
    # Most metrics are already 0-1 or 0-100
    
    if value > 1.0:
        # Likely a percentage, convert to 0-1
        return value / 100.0
    
    return value


def build_metric_mapping(unique_metrics: List[str], use_gpt: bool = True, gpt_model: str = "gpt-4o") -> Dict[str, Optional[str]]:
    """Build a mapping from original metric names to standard names."""
    mapping = {}
    
    # First apply simple rule-based mapping
    unmapped = []
    for metric in unique_metrics:
        if metric in SIMPLE_METRIC_MAPPING:
            mapping[metric] = SIMPLE_METRIC_MAPPING[metric]
        else:
            unmapped.append(metric)
    
    print(f"Rule-based mapping: {len(mapping)} metrics mapped, {len(unmapped)} remaining")
    
    # Use GPT for remaining metrics
    if use_gpt and unmapped:
        print(f"Using GPT to classify {len(unmapped)} unmapped metrics...")
        
        # Process in batches to avoid token limits
        batch_size = 100
        for i in tqdm(range(0, len(unmapped), batch_size), desc="GPT classification"):
            batch = unmapped[i:i + batch_size]
            gpt_mapping = get_gpt_metric_mapping(batch, gpt_model)
            mapping.update(gpt_mapping)
    
    # Mark any still unmapped as None
    for metric in unique_metrics:
        if metric not in mapping:
            mapping[metric] = None
    
    return mapping


def filter_and_normalize_edge_metadata(
    input_path: Path,
    output_path: Path,
    use_gpt: bool = True,
    gpt_model: str = "gpt-4o",
) -> Tuple[int, int, Dict[str, int]]:
    """Filter and normalize edge metadata."""
    
    # Load edge metadata
    print(f"Loading {input_path}...")
    with open(input_path, "r") as f:
        edge_metadata = json.load(f)
    
    print(f"Loaded {len(edge_metadata)} edges")
    
    # Collect all unique metric names
    all_metrics = set()
    for edge_data in edge_metadata.values():
        if "metrics" in edge_data:
            all_metrics.update(edge_data["metrics"].keys())
    
    print(f"Found {len(all_metrics)} unique metric names")
    
    # Build metric mapping
    metric_mapping = build_metric_mapping(list(all_metrics), use_gpt, gpt_model)
    
    # Save mapping for reference
    mapping_path = output_path.parent / "metric_name_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(metric_mapping, f, indent=2)
    print(f"Saved metric mapping to {mapping_path}")
    
    # Count mapped metrics
    mapped_counts = {}
    for orig, std in metric_mapping.items():
        if std:
            mapped_counts[std] = mapped_counts.get(std, 0) + 1
    
    print("\nMetric mapping summary:")
    for std_name in STANDARD_METRICS:
        count = mapped_counts.get(std_name, 0)
        print(f"  {std_name}: {count} source metrics")
    unmapped_count = sum(1 for v in metric_mapping.values() if v is None)
    print(f"  (unmapped/filtered out): {unmapped_count} metrics")
    
    # Filter and normalize
    filtered_metadata = {}
    edges_with_metrics = 0
    edges_without_metrics = 0
    final_metric_counts = {m: 0 for m in STANDARD_METRICS}
    
    for edge_key, edge_data in tqdm(edge_metadata.items(), desc="Filtering edges"):
        new_metrics = {}
        
        if "metrics" in edge_data:
            for metric_name, metric_value in edge_data["metrics"].items():
                std_name = metric_mapping.get(metric_name)
                
                if std_name and std_name in STANDARD_METRICS:
                    # Normalize value
                    try:
                        norm_value = normalize_value(float(metric_value), std_name)
                        if norm_value is not None:
                            # If multiple source metrics map to same standard name, take the first
                            if std_name not in new_metrics:
                                new_metrics[std_name] = norm_value
                                final_metric_counts[std_name] += 1
                    except (ValueError, TypeError):
                        continue
        
        # Create filtered edge data
        filtered_edge = {
            "model_id": edge_data.get("model_id"),
            "dataset_id": edge_data.get("dataset_id"),
            "metrics": new_metrics,
        }
        filtered_metadata[edge_key] = filtered_edge
        
        if new_metrics:
            edges_with_metrics += 1
        else:
            edges_without_metrics += 1
    
    # Save filtered metadata
    print(f"\nSaving filtered metadata to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(filtered_metadata, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Edges with metrics: {edges_with_metrics}")
    print(f"  Edges without metrics: {edges_without_metrics}")
    print("\nFinal metric counts:")
    for metric, count in sorted(final_metric_counts.items(), key=lambda x: -x[1]):
        print(f"  {metric}: {count}")
    
    return edges_with_metrics, edges_without_metrics, final_metric_counts


def main():
    parser = argparse.ArgumentParser(description="Normalize edge metrics using GPT classification")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/artifact_graph_data_v2_1125/edge_metadata.json"),
        help="Input edge_metadata.json path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/artifact_graph_data_v2_1125/edge_metadata_normalized.json"),
        help="Output normalized edge_metadata.json path",
    )
    parser.add_argument(
        "--no-gpt",
        action="store_true",
        help="Disable GPT classification, use rule-based mapping only",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-4o",
        help="GPT model to use for classification",
    )
    args = parser.parse_args()
    
    filter_and_normalize_edge_metadata(
        args.input,
        args.output,
        use_gpt=not args.no_gpt,
        gpt_model=args.gpt_model,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

