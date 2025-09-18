#!/usr/bin/env python3
"""
Simple Model Performance Prediction Script using LLMLinkPredictor

Logic:
1. Build a graph with ALL metrics to get maximum coverage
2. Find the target dataset node in the graph
3. Get ALL model nodes from the graph
4. Create prediction pairs: (every model, target dataset)
5. Use LLMLinkPredictor with graph neighborhood for LLM prediction context

Updated to use LLMLinkPredictor following the pattern from predict_links_llm.py
"""

import json
import argparse
import sys
import os
from typing import List, Tuple, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifact_graph.utils.graph_builder import load_artifact_graph_from_json
from artifact_graph.models.llm_link_predictor import LLMLinkPredictor


def build_comprehensive_graph(graph_file: str) -> Any:
    """
    Build the most comprehensive graph possible using all available metrics.
    """
    print("🔨 Building comprehensive graph with all available metrics...")
    
    # Common metrics to try
    metrics_to_try = [
        "accuracy", "exact_match", "f1", "bleu", "rouge1", "rouge2", "rougeL", 
        "bertscore", "meteor", "sacrebleu", "squad", "em"
    ]
    
    graphs = []
    for metric in metrics_to_try:
        try:
            G = load_artifact_graph_from_json(
                json_file=graph_file,
                min_downloads=1,
                metric_key=None,
            )
            graphs.append((metric, G))
            print(f"  ✅ {metric}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            print(f"  ❌ {metric}: failed")
    
    if not graphs:
        raise Exception("Could not build graph with any metric!")
    
    # Use the graph with the most nodes (most comprehensive)
    best_metric, best_graph = max(graphs, key=lambda x: x[1].number_of_nodes())
    print(f"🏆 Selected '{best_metric}' graph: {best_graph.number_of_nodes()} nodes, {best_graph.number_of_edges()} edges")
    
    return best_graph


def find_dataset_node(G: Any, dataset_name: str) -> Optional[str]:
    """
    Find the actual dataset node name in the graph.
    """
    print(f"🔍 Looking for dataset '{dataset_name}' in graph...")
    
    # Direct match first
    if dataset_name in G.nodes():
        print(f"✅ Found exact match: '{dataset_name}'")
        return dataset_name
    
    # Try common variations
    variations = [
        dataset_name.lower(),
        dataset_name.upper(),
        f"rajpurkar/{dataset_name}",
        f"squad/{dataset_name}",
        dataset_name.replace("_", "-"),
        dataset_name.replace("-", "_"),
    ]
    
    for variation in variations:
        if variation in G.nodes():
            print(f"✅ Found variation: '{variation}'")
            return variation
    
    # Partial match
    dataset_nodes = []
    for node in G.nodes():
        if dataset_name.lower() in node.lower():
            dataset_nodes.append(node)
    
    if dataset_nodes:
        print(f"🎯 Found {len(dataset_nodes)} partial matches:")
        for i, node in enumerate(dataset_nodes[:5]):
            print(f"  {i+1}. {node}")
        
        # Return the first one for now
        selected = dataset_nodes[0]
        print(f"✅ Using: '{selected}'")
        return selected
    
    print(f"❌ Dataset '{dataset_name}' not found in graph!")
    print("Available dataset-like nodes (first 10):")
    all_nodes = sorted(G.nodes())
    dataset_like = [n for n in all_nodes if any(word in n.lower() for word in ['squad', 'glue', 'super', 'dataset'])]
    for i, node in enumerate(dataset_like[:10]):
        print(f"  {i+1}. {node}")
    
    return None


def get_all_model_nodes(G: Any) -> List[str]:
    """
    Get ALL model nodes from the graph.
    """
    print("🤖 Extracting all model nodes from graph...")
    
    models = []
    datasets = []
    unknown = []
    
    for node in G.nodes():
        node_data = G.nodes.get(node, {})
        node_type = node_data.get('type', 'unknown')
        
        if node_type == 'model':
            models.append(node)
        elif node_type == 'dataset':
            datasets.append(node)
        else:
            # Heuristic: models usually have format "org/model-name"
            if '/' in node and len(node.split('/')) == 2:
                # Exclude obvious dataset patterns
                if not any(word in node.lower() for word in ['dataset', 'benchmark', 'test', 'train', 'val']):
                    models.append(node)
                else:
                    datasets.append(node)
            else:
                unknown.append(node)
    
    print(f"📊 Node classification:")
    print(f"  Models: {len(models)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Unknown: {len(unknown)}")
    
    if len(models) == 0:
        print("⚠️ No models found! Trying fallback approach...")
        # Fallback: assume nodes with '/' are models
        models = [node for node in G.nodes() if '/' in node]
        print(f"🔄 Fallback found {len(models)} potential models")
    
    return sorted(models)


def predict_all_models_on_dataset(
    dataset_name: str,
    metric_name: str = "exact_match",
    model_name: str = "gpt-4o",
    mode: str = "simple",
    graph_file: str = "output/perfect_model_dataset_metrics.json",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simple main function: predict ALL models on one dataset.
    """
    print(f"🎯 PREDICTING ALL MODELS ON DATASET: '{dataset_name}'")
    print(f"📏 Metric: {metric_name}")
    print(f"🤖 LLM: {model_name} ({mode} mode)")
    print("=" * 60)
    
    # 1. Build comprehensive graph
    try:
        G = build_comprehensive_graph(graph_file)
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        return {"error": "Graph building failed", "details": str(e)}
    
    # 2. Find target dataset in graph
    target_dataset = find_dataset_node(G, dataset_name)
    if target_dataset is None:
        return {"error": "Dataset not found in graph", "requested": dataset_name}
    
    # 3. Get ALL model nodes from graph
    all_models = get_all_model_nodes(G)
    if not all_models:
        return {"error": "No models found in graph"}
    
    # 4. Create prediction pairs: (model, dataset) for ALL models
    edges_to_predict = [(model, target_dataset) for model in all_models]
    
    # 5. Apply limit if specified
    if limit:
        edges_to_predict = edges_to_predict[:limit]
        print(f"🔍 Limited to first {limit} models for testing")
    
    print(f"\n📋 PREDICTION PAIRS ({len(edges_to_predict)} total):")
    for i, (model, dataset) in enumerate(edges_to_predict[:10]):
        print(f"  {i+1:3d}. {model} → {dataset}")
    if len(edges_to_predict) > 10:
        print(f"  ... and {len(edges_to_predict) - 10} more")
    print()
    
    # 6. Validate all edges exist in graph
    valid_edges = []
    invalid_edges = []
    
    for model, dataset in edges_to_predict:
        if model in G.nodes() and dataset in G.nodes():
            valid_edges.append((model, dataset))
        else:
            invalid_edges.append((model, dataset))
    
    if invalid_edges:
        print(f"⚠️ Filtered out {len(invalid_edges)} invalid edges")
    
    edges_to_predict = valid_edges
    print(f"✅ Proceeding with {len(edges_to_predict)} valid prediction pairs")
    
    if not edges_to_predict:
        return {"error": "No valid edges for prediction"}
    
    # 7. Initialize predictor and predict
    try:
        print("🚀 Starting LLM predictions...")
        predictor = LLMLinkPredictor(model_name=model_name)
        
        predicted_metrics = predictor.predict(
            edges_to_predict,
            G,
            model_dir="output/models",
            dataset_dir="output/datasets",
            mode=mode,
            metric_name=metric_name,
        )
        
        print(f"✅ Completed {len(predicted_metrics)} predictions")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return {"error": "Prediction failed", "details": str(e)}
    
    # 8. Process results
    results = []
    for i, (model, dataset) in enumerate(edges_to_predict):
        prediction_result = predicted_metrics[i] if i < len(predicted_metrics) else None
        
        result_item = {
            "model": model,
            "dataset": dataset,
            "metric": metric_name,
            "predicted_score": None,
            "reasoning": "",
            "status": "Failed"
        }
        
        # Check if the prediction was valid for this pair (following predict_links_llm.py pattern)
        if prediction_result and prediction_result.get("prediction") is not None:
            try:
                score = float(prediction_result["prediction"])
                reasoning = prediction_result.get("reason", "")
                
                result_item.update({
                    "predicted_score": score,
                    "reasoning": reasoning,
                    "status": "Success"
                })
                
            except (ValueError, TypeError) as e:
                print(f"⚠️ Failed to parse prediction for {model}: {prediction_result}")
                result_item.update({
                    "error": f"Parse error: {e}",
                    "raw_response": prediction_result
                })
        else:
            print(f"⚠️ Prediction failed for {model} -> {dataset}")
            if prediction_result:
                result_item["raw_response"] = prediction_result
        
        results.append(result_item)
    
    # 9. Sort by predicted score and analyze results
    valid_results = [r for r in results if r.get("predicted_score") is not None]
    failed_results = [r for r in results if r.get("predicted_score") is None]
    valid_results.sort(key=lambda x: x["predicted_score"], reverse=True)
    
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"  Total models: {len(all_models)}")
    print(f"  Valid predictions: {len(valid_results)}")
    print(f"  Failed predictions: {len(failed_results)}")
    
    if valid_results:
        scores = [r["predicted_score"] for r in valid_results]
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Mean score: {sum(scores)/len(scores):.3f}")

    print(f"\n🏆 TOP PREDICTIONS:")
    for i, result in enumerate(valid_results[:10]):
        score = result["predicted_score"]
        model = result["model"]
        print(f"  {i+1:2d}. {model:<40} → {score:.3f}")
    
    return {
        "dataset": target_dataset,
        "metric": metric_name,
        "total_models": len(all_models),
        "valid_predictions": len(valid_results),
        "failed_predictions": len(failed_results),
        "predictions": results,
        "top_models": valid_results[:20]  # Top 20
    }


def main():
    parser = argparse.ArgumentParser(description="Predict all models on a dataset")
    parser.add_argument("--dataset", required=True, help="Target dataset name")
    parser.add_argument("--metric", default="exact_match", help="Metric to predict")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--mode", default="neighborhood", help="Prediction mode (simple, neighborhood, zero-shot)")
    parser.add_argument("--graph-file", default="output/perfect_model_dataset_metrics.json", help="Graph data file")
    parser.add_argument("--limit", type=int, help="Limit number of models (for testing)")
    parser.add_argument("--output", help="Output JSON file (default: auto-generated based on dataset and mode)")
    
    args = parser.parse_args()
    
    # Run prediction
    results = predict_all_models_on_dataset(
        dataset_name=args.dataset,
        metric_name=args.metric,
        model_name=args.model,
        mode=args.mode,
        graph_file=args.graph_file,
        limit=args.limit
    )
    
    # Save results (always save, even if there were errors)
    if args.output:
        output_file = args.output
    else:
        # Create default output filename based on dataset and mode
        safe_dataset_name = args.dataset.replace("/", "_").replace(" ", "_")
        output_file = f"output/model_predictions_{safe_dataset_name}_{args.mode}_{args.metric}.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {output_file}")
    
    # Print summary
    if "error" in results:
        print(f"\n❌ ERROR: {results['error']}")
        if "details" in results:
            print(f"Details: {results['details']}")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: Predicted {results['valid_predictions']} models on '{results['dataset']}'")
        if results.get('top_models'):
            best_model = results['top_models'][0]
            print(f"🥇 Best model: {best_model['model']} (score: {best_model['predicted_score']:.3f})")


if __name__ == "__main__":
    main()
