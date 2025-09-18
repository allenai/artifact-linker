#!/usr/bin/env python3
"""
Calculate F1 score and other evaluation metrics for LLM binary predictions.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import os

def load_predictions(file_path: str) -> list:
    """Load predictions from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def calculate_binary_metrics(predictions: list) -> dict:
    """Calculate various binary classification metrics."""
    true_labels = []
    predicted_labels = []

    for pred in predictions:
        if pred.get('status') == 'Success' and 'true_label' in pred and 'predicted_label' in pred:
            true_labels.append(int(pred['true_label']))
            predicted_labels.append(int(pred['predicted_label']))

    if len(true_labels) == 0:
        print("Error: No valid predictions found")
        return None

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    cm = confusion_matrix(true_labels, predicted_labels)

    return {
        'n_predictions': len(true_labels),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'true_labels': true_labels.tolist(),
        'predicted_labels': predicted_labels.tolist()
    }

def print_metrics(metrics: dict):
    """Print evaluation metrics in a formatted way."""
    print("="*50)
    print("LLM BINARY PREDICTION EVALUATION METRICS")
    print("="*50)
    print(f"Number of predictions: {metrics['n_predictions']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(metrics['true_labels'], metrics['predicted_labels']))
    print("\nCONFUSION MATRIX:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"      \tPredicted 0\tPredicted 1")
    print(f"Actual 0\t{cm[0][0]}\t\t{cm[0][1]}")
    print(f"Actual 1\t{cm[1][0]}\t\t{cm[1][1]}")
    print("-" * 50)
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TP: {cm[1][1]}")
    print("="*50)

def create_confusion_matrix_plot(metrics: dict, output_dir: str = None):
    """Create a heatmap plot of the confusion matrix."""
    cm = np.array(metrics['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    plt.show()

def save_metrics_to_file(metrics: dict, output_file: str):
    """Save metrics to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate F1 and other metrics for LLM binary predictions')
    parser.add_argument('--input', '-i', type=str, default='./output/llm_binary_predictions_neighborhood.json',
                        help='Input JSON file with binary predictions')
    parser.add_argument('--output-dir', '-o', type=str, default='./output/evaluation_binary',
                        help='Output directory for plots and metrics')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save metrics to JSON file')
    args = parser.parse_args()

    print(f"Loading predictions from {args.input}...")
    predictions = load_predictions(args.input)
    
    if predictions is None:
        return
        
    print("Calculating evaluation metrics...")
    metrics = calculate_binary_metrics(predictions)
    
    if metrics is None:
        return
        
    print_metrics(metrics)
    
    if args.save_metrics:
        os.makedirs(args.output_dir, exist_ok=True)
        save_metrics_to_file(metrics, os.path.join(args.output_dir, 'binary_evaluation_metrics.json'))
        
    if not args.no_plots:
        print("\nGenerating plots...")
        create_confusion_matrix_plot(metrics, args.output_dir)

if __name__ == '__main__':
    main()
