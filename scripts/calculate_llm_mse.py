#!/usr/bin/env python3
"""
Calculate MSE and other evaluation metrics for LLM predictions.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import os

def load_predictions(file_path: str) -> dict:
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

def calculate_metrics(predictions: list) -> dict:
    """Calculate various evaluation metrics."""
    # Extract true and predicted values
    true_values = []
    predicted_values = []
    
    for pred in predictions:
        if pred.get('status') == 'Success' and 'true_metric' in pred and 'predicted_metric' in pred:
            if pred['true_metric'] > 1:
                pred['true_metric'] /= 100  # Normalize percentage values
            if pred['predicted_metric'] > 1:
                pred['predicted_metric'] /= 100  # Normalize percentage values
            true_values.append(float(pred['true_metric']))
            predicted_values.append(float(pred['predicted_metric']))
    
    if len(true_values) == 0:
        print("Error: No valid predictions found")
        return None
    
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    # Calculate baseline (average of all ground-truth values)
    baseline_prediction = np.mean(true_values)
    baseline_predictions = np.full_like(true_values, baseline_prediction)
    baseline_mse = mean_squared_error(true_values, baseline_predictions)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(true_values, baseline_predictions)
    
    # Calculate metrics
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    
    # Calculate correlation coefficients
    pearson_corr, pearson_p = pearsonr(true_values, predicted_values)
    spearman_corr, spearman_p = spearmanr(true_values, predicted_values)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    bias = np.mean(predicted_values - true_values)
    
    return {
        'n_predictions': len(true_values),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mape': mape,
        'bias': bias,
        'baseline_mse': baseline_mse,
        'baseline_rmse': baseline_rmse,
        'baseline_mae': baseline_mae,
        'baseline_prediction': baseline_prediction,
        'true_values': true_values,
        'predicted_values': predicted_values
    }

def print_metrics(metrics: dict):
    """Print evaluation metrics in a formatted way."""
    print("="*50)
    print("LLM PREDICTION EVALUATION METRICS")
    print("="*50)
    print(f"Number of predictions: {metrics['n_predictions']}")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2_score']:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print(f"Bias (mean prediction - true): {metrics['bias']:.6f}")
    print()
    print("BASELINE COMPARISON:")
    print(f"Baseline prediction (average): {metrics['baseline_prediction']:.6f}")
    print(f"Baseline MSE: {metrics['baseline_mse']:.6f}")
    print(f"Baseline RMSE: {metrics['baseline_rmse']:.6f}")
    print(f"Baseline MAE: {metrics['baseline_mae']:.6f}")
    improvement = ((metrics['baseline_mse'] - metrics['mse']) / metrics['baseline_mse']) * 100
    improvement_rmse = ((metrics['baseline_rmse'] - metrics['rmse']) / metrics['baseline_rmse']) * 100
    improvement_mae = ((metrics['baseline_mae'] - metrics['mae']) / metrics['baseline_mae']) * 100
    print(f"MSE improvement over baseline: {improvement:.2f}%")
    print(f"RMSE improvement over baseline: {improvement_rmse:.2f}%")
    print(f"MAE improvement over baseline: {improvement_mae:.2f}%")
    if improvement > 0:
        print("✓ Model performs better than baseline")
    else:
        print("✗ Model performs worse than baseline")
    print()
    print("CORRELATION ANALYSIS:")
    print(f"Pearson Correlation: {metrics['pearson_correlation']:.6f} (p={metrics['pearson_p_value']:.6f})")
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.6f} (p={metrics['spearman_p_value']:.6f})")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    if metrics['pearson_correlation'] > 0.7:
        print("✓ Strong positive correlation between predictions and true values")
    elif metrics['pearson_correlation'] > 0.5:
        print("✓ Moderate positive correlation between predictions and true values")
    elif metrics['pearson_correlation'] > 0.3:
        print("⚠ Weak positive correlation between predictions and true values")
    else:
        print("✗ Poor correlation between predictions and true values")
    
    if metrics['bias'] > 0.05:
        print("⚠ Model tends to overestimate performance")
    elif metrics['bias'] < -0.05:
        print("⚠ Model tends to underestimate performance")
    else:
        print("✓ Model predictions are relatively unbiased")

def create_scatter_plot(metrics: dict, output_dir: str | None = None):
    """Create a scatter plot of true vs predicted values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(metrics['true_values'], metrics['predicted_values'], alpha=0.6, s=50, label='LLM Predictions')
    
    # Perfect prediction line (y=x)
    min_val = min(min(metrics['true_values']), min(metrics['predicted_values']))
    max_val = max(max(metrics['true_values']), max(metrics['predicted_values']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    # Baseline prediction line (horizontal line at baseline value)
    baseline = metrics['baseline_prediction']
    plt.axhline(y=baseline, color='orange', linestyle=':', label=f'Baseline (avg={baseline:.3f})', linewidth=2)
    
    # Best fit line
    z = np.polyfit(metrics['true_values'], metrics['predicted_values'], 1)
    p = np.poly1d(z)
    plt.plot(metrics['true_values'], p(metrics['true_values']), 'b-', alpha=0.8, label='Best Fit Line')
    
    plt.xlabel('True Metric Values', fontsize=24)
    plt.ylabel('Predicted Metric Values', fontsize=24)
    plt.title('LLM Predictions vs True Values', fontsize=28)
    plt.legend(fontsize=18, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Add metrics info to plot
    improvement = ((metrics['baseline_mse'] - metrics['mse']) / metrics['baseline_mse']) * 100
    #plt.text(0.05, 0.95, f'R² = {metrics["r2_score"]:.3f}\nPearson r = {metrics["pearson_correlation"]:.3f}\nRMSE = {metrics["rmse"]:.3f}\nBaseline MSE = {metrics["baseline_mse"]:.3f}\nImprovement = {improvement:.1f}%',
    #         transform=plt.gca().transAxes, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'prediction_scatter_plot.png'), dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {os.path.join(output_dir, 'prediction_scatter_plot.png')}")
    
    plt.show()

def create_residual_plot(metrics: dict, output_dir: str | None = None):
    """Create a residual plot to analyze prediction errors."""
    residuals = metrics['predicted_values'] - metrics['true_values']
    
    plt.figure(figsize=(12, 5))
    
    # Residual vs predicted plot
    plt.subplot(1, 2, 1)
    plt.scatter(metrics['predicted_values'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values', fontsize=24)
    plt.ylabel('Residuals (Predicted - True)', fontsize=24)
    plt.title('Residuals vs Predicted Values', fontsize=28)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Histogram of residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Predicted - True)', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.title('Distribution of Residuals', fontsize=28)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {os.path.join(output_dir, 'residual_analysis.png')}")
    
    plt.show()

def save_metrics_to_file(metrics: dict, output_file: str):
    """Save metrics to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    metrics_to_save = metrics.copy()
    metrics_to_save['true_values'] = metrics['true_values'].tolist()
    metrics_to_save['predicted_values'] = metrics['predicted_values'].tolist()
    
    with open(output_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Metrics saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate MSE and other metrics for LLM predictions')
    parser.add_argument('--input', '-i', type=str, default='./output/neighborhood_predictions_accuracy.json',
                        help='Input JSON file with predictions')
    parser.add_argument('--output-dir', '-o', type=str, default='./output/evaluation',
                        help='Output directory for plots and metrics')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save metrics to JSON file')
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.input}...")
    predictions_data = load_predictions(args.input)
    
    if predictions_data is None:
        return
    
    # Extract predictions list from the loaded data
    if isinstance(predictions_data, list):
        predictions = predictions_data
    elif isinstance(predictions_data, dict) and 'predictions' in predictions_data:
        predictions = predictions_data['predictions']
    elif isinstance(predictions_data, dict):
        # Assume the loaded data is a list of predictions directly
        predictions = list(predictions_data) if predictions_data else []
    else:
        predictions = predictions_data
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(predictions)
    
    if metrics is None:
        return
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics if requested
    if args.save_metrics:
        os.makedirs(args.output_dir, exist_ok=True)
        save_metrics_to_file(metrics, os.path.join(args.output_dir, 'evaluation_metrics.json'))
    
    # Create plots if requested
    if not args.no_plots:
        print("\nGenerating plots...")
        create_scatter_plot(metrics, args.output_dir)
        create_residual_plot(metrics, args.output_dir)

if __name__ == '__main__':
    main()
