#!/usr/bin/env python3
import sys
import json
import random
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.utils.evaluation_utils import (
    calculate_ndcg_standard, 
    calculate_ranking_correlation,
    calculate_map_continuous,
    calculate_recall_at_k,
    calculate_precision_at_k
)
from artifact_graph.utils.graph_builder import load_nx_graph
from artifact_graph.models.gnn_link_predictor import GNNLinkPredictor


def load_model_and_data(model_path: str, data_dir: str):
    """Load trained attribute prediction model and graph data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError("Invalid model file: missing model_state_dict")
    if "model_config" not in ckpt:
        raise ValueError("Invalid model file: missing model_config")
    
    cfg = ckpt["model_config"]
    
    model = GNNLinkPredictor(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Load graph data
    droot = Path(data_dir)
    emb_path = droot / "node_embeddings.npy"
    edges_path = droot.parent / "artifact_graph_splits" / "train_split" / "edges.npz"
    if not edges_path.exists():
        edges_path = droot / "edges.npz"
    
    # Load embeddings robustly
    arr = np.load(emb_path, allow_pickle=False)
    if hasattr(arr, 'dtype') and hasattr(arr.dtype, 'names') and 'embedding' in arr.dtype.names:
        x = torch.from_numpy(arr['embedding']).float().to(device)
    else:
        x = torch.from_numpy(arr).float().to(device)

    # TODO: random embedding ablation
    x = torch.randn(x.size(0), x.size(1))
    
    # Load edges
    edge_index = torch.from_numpy(np.load(edges_path)["edges"]).long().to(device)
    
    data = SimpleNamespace(x=x, edge_index=edge_index)
    print(f"[model] loaded: {cfg}")
    print(f"[data] nodes={x.shape[0]}, edges={edge_index.shape[1]}, feat_dim={x.shape[1]}")
    
    return model, data, device


@torch.no_grad()
def predict_scores(model, data, device, dataset_id: int, model_ids: list) -> dict:
    """Predict attribute scores for dataset-model pairs"""
    if not model_ids:
        return {}
    
    # Create edge pairs: (dataset_id, model_id)
    pairs = torch.tensor([[dataset_id, mid] for mid in model_ids], dtype=torch.long, device=device).t()
    
    # Get predictions
    z = model.encode(data.x, data.edge_index)
    logits = model.decode(z, pairs)
    scores = torch.sigmoid(logits).cpu().numpy().flatten()
    
    return {mid: float(score) for mid, score in zip(model_ids, scores)}


def normalize_edge_metadata_keys(edge_metadata: dict) -> dict:
    """Converts all string keys in edge_metadata (e.g., "u,v") to integer tuples (u, v)."""
    normalized_meta = {}
    for k, v in edge_metadata.items():
        try:
            # Handle keys like "2,1444" or "(2, 1444)"
            parts = str(k).replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
            if len(parts) == 2:
                u, v_int = int(parts[0]), int(parts[1])
                normalized_meta[(u, v_int)] = v
            else:
                # Keep original key if parsing fails
                normalized_meta[k] = v
        except (ValueError, TypeError):
            # Keep original key if conversion fails
            normalized_meta[k] = v
    return normalized_meta

def prepare_ranking_tasks(G, edge_metadata: dict, test_pos_edges: np.ndarray, max_datasets: int = 100, max_ranking_candidates: int = 20):
    """Create ranking tasks from test set edges, ensuring all models use the same metric for each dataset"""
    dataset_models_by_metric = {}
    
    # Group test set dataset-model pairs by metric type
    for i in range(test_pos_edges.shape[1]):
        u, v = int(test_pos_edges[0, i]), int(test_pos_edges[1, i])
        
        # Use tuple keys for lookup
        edge_data = edge_metadata.get((u, v), edge_metadata.get((v, u)))
        
        if not edge_data or "metrics" not in edge_data:
            continue
            
        u_type = G.nodes.get(u, {}).get("type")
        v_type = G.nodes.get(v, {}).get("type")
        
        if u_type == "dataset" and v_type == "model":
            dataset_id, model_id = u, v
        elif u_type == "model" and v_type == "dataset":
            dataset_id, model_id = v, u
        else:
            continue
            
        # Check all available metrics for this edge
        available_metrics = edge_data["metrics"]
        
        for metric_name, metric_value in available_metrics.items():
            try:
                score = float(metric_value)
                if score > 1.0:  # Convert percentage to [0,1]
                    score /= 100.0
                
                # Group by dataset and metric
                key = (dataset_id, metric_name)
                if key not in dataset_models_by_metric:
                    dataset_models_by_metric[key] = []
                dataset_models_by_metric[key].append((model_id, score))
            except (ValueError, TypeError):
                # Skip invalid metric values
                continue

    # Select the best metric for each dataset (prioritize accuracy, then most models)
    dataset_best_metric = {}
    for (dataset_id, metric_name), models in dataset_models_by_metric.items():
        if len(models) < 3:  # Restore threshold to 3 for meaningful ranking
            continue
            
        if dataset_id not in dataset_best_metric:
            dataset_best_metric[dataset_id] = (metric_name, models, len(models))
        else:
            current_metric, current_models, current_count = dataset_best_metric[dataset_id]
            
            # Prioritize accuracy, then count
            if (metric_name == "accuracy" and current_metric != "accuracy") or \
               (metric_name == current_metric and len(models) > current_count) or \
               (current_metric != "accuracy" and metric_name != "accuracy" and len(models) > current_count):
                dataset_best_metric[dataset_id] = (metric_name, models, len(models))
    
    # Create tasks from best metrics
    valid_datasets = list(dataset_best_metric.items())
    if len(valid_datasets) > max_datasets:
        valid_datasets = random.sample(valid_datasets, max_datasets)
    
    tasks = []
    for dataset_id, (metric_name, models, _) in valid_datasets:
        # Ensure all models have unique IDs (remove duplicates)
        unique_models = {}
        for model_id, score in models:
            if model_id not in unique_models:
                unique_models[model_id] = score
            else:
                # Keep the higher score if duplicate
                unique_models[model_id] = max(unique_models[model_id], score)
        
        models = list(unique_models.items())
        
        # Limit number of ranking candidates per dataset
        if len(models) > max_ranking_candidates:
            # Sample the top models + some random ones to keep diversity
            models_sorted = sorted(models, key=lambda x: x[1], reverse=True)
            top_models = models_sorted[:max_ranking_candidates//2]  # Keep top half
            remaining = models_sorted[max_ranking_candidates//2:]
            if remaining:
                random_models = random.sample(remaining, min(max_ranking_candidates - len(top_models), len(remaining)))
                models = top_models + random_models
            else:
                models = top_models
        
        # Sort by true score (ground truth ranking)
        ground_truth = sorted(models, key=lambda x: x[1], reverse=True)
        tasks.append({
            "dataset_id": dataset_id,
            "models": [mid for mid, _ in models],
            "ground_truth_scores": {mid: score for mid, score in ground_truth},
            "num_candidates": len(models),
            "metric_used": metric_name  # Track which metric was used
        })
    
    return tasks


def evaluate_ranking(predicted_scores: dict, ground_truth_scores: dict, dataset_id: int, metric_name: str):
    """Evaluate ranking and return detailed results with all metrics"""
    if not predicted_scores or not ground_truth_scores:
        # Return zeros for all metrics
        zero_metrics = {}
        for k in [1, 3, 5]:
            zero_metrics.update({
                f"ndcg@{k}": 0.0,
                f"map@{k}": 0.0,
                f"hit@{k}": 0.0,
                f"recall@{k}": 0.0
            })
        zero_metrics.update({
            "ndcg_full": 0.0,
            "map_full": 0.0,
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "pearson_r": 0.0,
            "dataset_id": dataset_id,
            "metric_used": metric_name,
            "predicted_ranking": [],
            "ground_truth_ranking": [],
            "num_models": 0
        })
        return zero_metrics
    
    # Create predicted ranking (sorted by predicted scores)
    predicted_ranking = [(mid, score) for mid, score in predicted_scores.items()]
    predicted_ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Create ground truth ranking (sorted by true scores)
    ground_truth_ranking = [(mid, score) for mid, score in ground_truth_scores.items()]
    ground_truth_ranking.sort(key=lambda x: x[1], reverse=True)
    
    try:
        metrics = {}
        k_values = [1, 3, 5]
        
        # Calculate NDCG@k and NDCG full
        for k in k_values:
            metrics[f"ndcg@{k}"] = calculate_ndcg_standard(predicted_ranking, ground_truth_scores, k=k)
        metrics["ndcg_full"] = calculate_ndcg_standard(predicted_ranking, ground_truth_scores)
        
        # Calculate MAP@k and MAP full
        for k in k_values:
            metrics[f"map@{k}"] = calculate_map_continuous(predicted_ranking, ground_truth_scores, k=k)
        metrics["map_full"] = calculate_map_continuous(predicted_ranking, ground_truth_scores)
        
        # Calculate correlation metrics
        corr_metrics = calculate_ranking_correlation(predicted_ranking, ground_truth_scores)
        metrics["kendall_tau"] = corr_metrics.get("kendall_tau", 0.0)
        metrics["spearman_rho"] = corr_metrics.get("spearman_rho", 0.0)
        metrics["pearson_r"] = corr_metrics.get("pearson_r", 0.0)
        
        # Calculate Hit@k and Recall@k
        # Define relevant items as top 50% by ground truth score
        ground_truth_scores_list = [score for _, score in ground_truth_ranking]
        median_score = sorted(ground_truth_scores_list)[len(ground_truth_scores_list) // 2]
        relevant_items = {mid for mid, score in ground_truth_scores.items() if score >= median_score}
        
        # Get predicted ranking order (just the model IDs)
        predicted_order = [mid for mid, _ in predicted_ranking]
        
        for k in k_values:
            metrics[f"hit@{k}"] = 1.0 if any(mid in relevant_items for mid in predicted_order[:k]) else 0.0
            metrics[f"recall@{k}"] = calculate_recall_at_k(predicted_order, relevant_items, k)
        
        # Add detailed ranking information
        metrics.update({
            "dataset_id": dataset_id,
            "metric_used": metric_name,
            "predicted_ranking": [
                {
                    "rank": i + 1,
                    "model_id": mid,
                    "predicted_score": float(score),
                    "ground_truth_score": float(ground_truth_scores.get(mid, 0)),
                    "metric_name": metric_name
                } for i, (mid, score) in enumerate(predicted_ranking)
            ],
            "ground_truth_ranking": [
                {
                    "rank": i + 1,
                    "model_id": mid,
                    "ground_truth_score": float(score),
                    "predicted_score": float(predicted_scores.get(mid, 0)),
                    "metric_name": metric_name
                } for i, (mid, score) in enumerate(ground_truth_ranking)
            ],
            "num_models": len(predicted_ranking)
        })
        
        return metrics
        
    except Exception as e:
        print(f"Evaluation failed for dataset {dataset_id}: {e}")
        # Return zeros for all metrics with error info
        zero_metrics = {}
        for k in [1, 3, 5]:
            zero_metrics.update({
                f"ndcg@{k}": 0.0,
                f"map@{k}": 0.0,
                f"hit@{k}": 0.0,
                f"recall@{k}": 0.0
            })
        zero_metrics.update({
            "ndcg_full": 0.0,
            "map_full": 0.0,
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "pearson_r": 0.0,
            "dataset_id": dataset_id,
            "metric_used": metric_name,
            "predicted_ranking": [],
            "ground_truth_ranking": [],
            "num_models": len(predicted_ranking) if 'predicted_ranking' in locals() and predicted_ranking else 0,
            "error": str(e)
        })
        return zero_metrics


def main():
    p = argparse.ArgumentParser(description="Simple GNN attribute ranking")
    p.add_argument("--model_path", type=str, required=True, help="Trained model path")
    p.add_argument("--data_dir", type=str, default="scripts/output/artifact_graph_data", help="Graph data dir")
    p.add_argument("--split_dir", type=str, default="scripts/output/artifact_graph_splits", help="Directory with train/val/test splits")
    p.add_argument("--max_datasets", type=int, default=10000000, help="Max datasets to test")
    p.add_argument("--max_ranking_candidates", type=int, default=20, help="Max models to rank per dataset")
    p.add_argument("--output_file", type=str, default="scripts/output/final_results/gnn_attribute_rankings.json", help="Output file")
    args = p.parse_args()
    
    # Load graph and metadata
    G, _, edge_metadata = load_nx_graph(args.data_dir)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Normalize metadata keys to ensure consistent lookup
    edge_metadata = normalize_edge_metadata_keys(edge_metadata)
    
    # Load test set positive edges
    test_pos_edges_path = Path(args.split_dir) / "test_split" / "pos_edges.npz"
    if not test_pos_edges_path.exists():
        raise FileNotFoundError(f"Test split positive edges not found at: {test_pos_edges_path}")
    
    test_pos_edges = np.load(test_pos_edges_path)["edges"]
    print(f"Loaded {test_pos_edges.shape[1]} positive edges from the test split.")
    
    # Prepare tasks exclusively from the test set
    tasks = prepare_ranking_tasks(G, edge_metadata, test_pos_edges, args.max_datasets, args.max_ranking_candidates)
    print(f"Ranking tasks created from test set: {len(tasks)}")
    
    # Try to load model
    try:
        model, data, device = load_model_and_data(args.model_path, args.data_dir)
        use_gnn = True
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("🔄 Using random ranking")
        use_gnn = False
    
    # Run tasks
    results = []
    for i, task in enumerate(tasks):
        dataset_id = task['dataset_id']
        num_models = task.get('num_candidates', len(task['models']))
        metric_used = task.get('metric_used', 'unknown')
        print(f"Task {i+1}/{len(tasks)}: dataset {dataset_id} (models: {num_models}, metric: {metric_used})")
        
        if use_gnn:
            # Use GNN to predict scores
            predicted_scores = predict_scores(model, data, device, dataset_id, task["models"])
        else:
            # Random scores
            predicted_scores = {mid: random.random() for mid in task["models"]}
        
        # Evaluate with detailed ranking information
        result = evaluate_ranking(predicted_scores, task["ground_truth_scores"], dataset_id, metric_used)
        results.append(result)
        
        print(f"  NDCG@1: {result.get('ndcg@1', 0):.3f}, NDCG@3: {result.get('ndcg@3', 0):.3f}, NDCG@5: {result.get('ndcg@5', 0):.3f}")
        print(f"  MAP@1: {result.get('map@1', 0):.3f}, MAP@3: {result.get('map@3', 0):.3f}, MAP@5: {result.get('map@5', 0):.3f}")
        print(f"  Spearman: {result.get('spearman_rho', 0):.3f}, Kendall: {result.get('kendall_tau', 0):.3f}")
        
        # Show top-3 predicted vs ground truth
        pred_ranking = result.get("predicted_ranking", [])
        if pred_ranking:
            pred_top3 = [f"M{r['model_id']}({r['predicted_score']:.2f})" for r in pred_ranking[:3]]
            print(f"  Top-3 predicted: {pred_top3}")
            gt_ranking = result.get("ground_truth_ranking", [])
            gt_top3 = [f"M{r['model_id']}({r['ground_truth_score']:.2f})" for r in gt_ranking[:3]]
            print(f"  Top-3 actual:    {gt_top3}")
    
    # Summary
    if results:
        # Calculate all average metrics
        metric_names = [
            "ndcg@1", "ndcg@3", "ndcg@5", "ndcg_full",
            "map@1", "map@3", "map@5", "map_full", 
            "hit@1", "hit@3", "hit@5",
            "recall@1", "recall@3", "recall@5",
            "kendall_tau", "spearman_rho", "pearson_r"
        ]
        
        avg_metrics = {}
        for metric in metric_names:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                avg_metrics[f"avg_{metric}"] = float(np.mean(values))
            else:
                avg_metrics[f"avg_{metric}"] = 0.0
        
        print(f"\n=== Attribute Ranking Results (Test Set) ===")
        print("NDCG Metrics:")
        print(f"  NDCG@1: {avg_metrics.get('avg_ndcg@1', 0):.4f}")
        print(f"  NDCG@3: {avg_metrics.get('avg_ndcg@3', 0):.4f}")
        print(f"  NDCG@5: {avg_metrics.get('avg_ndcg@5', 0):.4f}")
        print(f"  NDCG (full): {avg_metrics.get('avg_ndcg_full', 0):.4f}")
        
        print("MAP Metrics:")
        print(f"  MAP@1: {avg_metrics.get('avg_map@1', 0):.4f}")
        print(f"  MAP@3: {avg_metrics.get('avg_map@3', 0):.4f}")
        print(f"  MAP@5: {avg_metrics.get('avg_map@5', 0):.4f}")
        print(f"  MAP (full): {avg_metrics.get('avg_map_full', 0):.4f}")
        
        print("Hit@k and Recall@k:")
        for k in [1, 3, 5]:
            print(f"  Hit@{k}: {avg_metrics.get(f'avg_hit@{k}', 0):.4f}")
            print(f"  Recall@{k}: {avg_metrics.get(f'avg_recall@{k}', 0):.4f}")
        
        print("Correlation Metrics:")
        print(f"  Kendall's Tau: {avg_metrics.get('avg_kendall_tau', 0):.4f}")
        print(f"  Spearman's Rho: {avg_metrics.get('avg_spearman_rho', 0):.4f}")
        print(f"  Pearson R: {avg_metrics.get('avg_pearson_r', 0):.4f}")
        
        print(f"Valid rankings: {len(results)}/{len(tasks)}")
        
        # Show metric usage statistics
        metric_usage = {}
        for result in results:
            metric = result.get("metric_used", "unknown")
            metric_usage[metric] = metric_usage.get(metric, 0) + 1
        
        print("Metric usage:")
        for metric, count in metric_usage.items():
            print(f"  {metric}: {count} datasets")
        
        # Save
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as f:
            json.dump({
                "model_path": args.model_path,
                "model_used": use_gnn,
                "num_tasks": len(tasks),
                "max_datasets": args.max_datasets,
                "max_ranking_candidates": args.max_ranking_candidates,
                "metric_usage": metric_usage,  # Add metric usage statistics
                **avg_metrics,  # Include all average metrics
                "results": results,
            }, f, indent=2)
        
        print(f"Saved: {output_path}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()