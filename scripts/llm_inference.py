import argparse
import torch
import os
import json
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from artifact_graph.data.collectors import ModelCollector, DatasetCollector, AccuracyCollector
from artifact_graph.data.processors import GraphBuilder, CardProcessor
from artifact_graph.models.llm_link_predictor import OpenAIGPTLinkPredictor


def evaluate_llm(predictor, data, split, node_names, G, model_cards, dataset_cards, mode="simple"):
    pos_e = getattr(data, f"{split}_pos_edge_index")
    pred_acc = predictor.predict(pos_e, node_names, G=G, model_cards=model_cards, dataset_cards=dataset_cards, mode=mode)
    # Ensure consistent length, fill None with nan
    pred_acc = [s if s is not None else float('nan') for s in pred_acc]

    gt_acc = getattr(data, f"{split}_pos_edge_attr", None)
    mse = None
    if gt_acc is not None:
        gt_acc = gt_acc.cpu().numpy()
        if len(gt_acc) == len(pred_acc) and len(gt_acc) > 0:
            gt_acc = torch.tensor(gt_acc, dtype=torch.float)
            pred_acc = torch.tensor(pred_acc, dtype=torch.float)
            mask = torch.isfinite(pred_acc)
            if mask.sum() > 0:
                mse = torch.mean((gt_acc[mask] - pred_acc[mask]) ** 2).item()
    return mse


def main(args):
    # 1) Use new modular data collection
    print("Initializing data collectors...")
    model_collector = ModelCollector(args.metadata_dir)
    dataset_collector = DatasetCollector(args.dataset_json)
    accuracy_collector = AccuracyCollector(args.data_dir)
    
    print("Building graph...")
    graph_builder = GraphBuilder(model_collector, dataset_collector, accuracy_collector)
    G = graph_builder.build_bipartite_graph(args.min_downloads)
    
    print("Converting to PyG data...")
    data = graph_builder.nx_to_pyg_data(G)
    data = graph_builder.prepare_link_pred_splits(data, args.val_ratio, args.test_ratio)

    # Ensure edge_index exists
    data.edge_index = data.train_pos_edge_index

    # Ensure all edge indices are [2, E] long tensors
    for name in [
        "edge_index", "train_pos_edge_index",
        "val_pos_edge_index", "val_neg_edge_index",
        "test_pos_edge_index", "test_neg_edge_index"
    ]:
        print(name)
        e = getattr(data, name)
        if e.dim() == 2 and e.size(1) == 2:
            e = e.t().contiguous()
        setattr(data, name, e.to(torch.long))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Prepare node texts if available, else use empty string
    if hasattr(data, 'node_names'):
        node_texts = [str(n) for n in data.node_names]
    else:
        node_texts = ["" for _ in range(data.x.size(0))]

    # Prepare card information
    print("Processing card information...")
    card_processor = CardProcessor(args.data_dir, args.metadata_dir, args.dataset_json)
    model_cards, dataset_cards = card_processor.prepare_cards()

    # 2) LLM Predictor
    predictor = OpenAIGPTLinkPredictor(model_name=args.llm_model)

    # 3) Evaluation
    print("Evaluating on validation set...")
    val_mse = evaluate_llm(predictor, data, 'val', data.node_names, G, model_cards, dataset_cards, mode="simple")
    print(f"Val MSE: {val_mse:.4f}")

    print("\nEvaluating on test set...")
    test_mse = evaluate_llm(predictor, data, 'test', data.node_names, G, model_cards, dataset_cards, mode="simple")
    print(f"Test MSE: {test_mse:.4f}")

    # 4) Top-50 negative-test edges
    neg_e = data.test_neg_edge_index
    neg_scores = predictor.predict(neg_e, data.node_names, G=G, model_cards=model_cards, dataset_cards=dataset_cards, mode="simple")
    neg_scores = [s if s is not None else 0.0 for s in neg_scores]
    edges = neg_e.cpu().numpy().T
    import numpy as np
    scores = np.array(neg_scores)
    topk_idx = scores.argsort()[::-1][:50]
    topk_edges = edges[topk_idx]
    topk_scores = scores[topk_idx]
    names = data.node_names if hasattr(data, 'node_names') else [str(i) for i in range(data.x.size(0))]
    print("\nTop-50 negative-test edges by LLM score:")
    print(f"{'u_idx':>6} {'u_name':>20}   {'v_idx':>6} {'v_name':>20}   {'score':>6}")
    for (u, v), score in zip(topk_edges, topk_scores):
        print(f"{u:6d} {names[u]:>20s}   {v:6d} {names[v]:>20s}   {score:6.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model', default='gpt-3.5-turbo', help="OpenAI model name")
    parser.add_argument('--data_dir',      default='../data/eval_datasets_json_download_ranks')
    parser.add_argument('--dataset_json',  default='../data/dataset_info.json')
    parser.add_argument('--metadata_dir',  default='../data/model_metadata_download_ranks')
    parser.add_argument('--min_downloads', type=int,   default=1000)
    parser.add_argument('--val_ratio',     type=float, default=0.1)
    parser.add_argument('--test_ratio',    type=float, default=0.8)
    args = parser.parse_args()
    main(args) 