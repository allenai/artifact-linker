#!/usr/bin/env python3
"""
Unified experiment runner for link/attribute prediction and ranking.

Usage:
    python run_experiments.py predict link gnn --epochs 300
    python run_experiments.py predict link llm --model openai/gpt-4o
    python run_experiments.py predict link baseline --mode downloads
    python run_experiments.py predict attr gnn --epochs 500
    python run_experiments.py rank link gnn --model-path path/to/model.pth
    python run_experiments.py rank attr llm --hops 1
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from artifact_graph.runners import (
    run_link_prediction,
    run_link_ranking,
    run_attribute_prediction,
    run_attribute_ranking,
)
from artifact_graph.runners.link_runner import LinkConfig
from artifact_graph.runners.attribute_runner import AttributeConfig


def add_common_args(p):
    """Add common arguments to parser."""
    p.add_argument("--data-dir", default="output/artifact_graph_data")
    p.add_argument("--split-dir", default="output/artifact_graph_splits")
    p.add_argument("--output-dir", default="output/final_results")
    p.add_argument("--seed", type=int, default=42)


def add_gnn_args(p, is_link=True):
    """Add GNN-specific arguments."""
    if is_link:
        p.add_argument("--epochs", type=int, default=300)
        p.add_argument("--patience", type=int, default=40)
        p.add_argument("--lr", type=float, default=5e-3)
        p.add_argument("--hidden", type=int, default=64)
        p.add_argument("--num-layers", type=int, default=3)
        p.add_argument("--heads", type=int, default=3)
        p.add_argument("--dropout", type=float, default=0.2)
    else:
        p.add_argument("--epochs", type=int, default=500)
        p.add_argument("--lr", type=float, default=0.005)
        p.add_argument("--hidden", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=3)
        p.add_argument("--heads", type=int, default=8)
        p.add_argument("--dropout", type=float, default=0.2)


def add_llm_args(p):
    """Add LLM-specific arguments."""
    p.add_argument("--model", default="openai/gpt-4o")
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--no-info", action="store_false", dest="use_info")
    p.add_argument("--workers", type=int, default=4)


def add_baseline_args(p, is_link=True):
    """Add baseline-specific arguments."""
    if is_link:
        p.add_argument("--mode", default="downloads",
                       choices=["downloads", "common_neighbors", "jaccard", "adamic_adar"])
        p.add_argument("--threshold", type=float, default=None)
    else:
        p.add_argument("--mode", default="dataset_average",
                       choices=["global_average", "dataset_average", "model_average"])


def build_link_config(args, method: str) -> LinkConfig:
    """Build LinkConfig from args."""
    kwargs = dict(
        method=method,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    if method == "gnn":
        kwargs.update(
            split_dir=args.split_dir,
            epochs=args.epochs,
            patience=getattr(args, "patience", 40),
            lr=args.lr,
            hidden=args.hidden,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            model_path=getattr(args, "model_path", ""),
        )
    elif method == "llm":
        kwargs.update(
            llm_model=args.model,
            hops=args.hops,
            use_info=args.use_info,
            workers=args.workers,
        )
    else:  # baseline
        kwargs.update(
            baseline_mode=args.mode,
            threshold=getattr(args, "threshold", None),
        )
    
    return LinkConfig(**kwargs)


def build_attr_config(args, method: str) -> AttributeConfig:
    """Build AttributeConfig from args."""
    kwargs = dict(
        method=method,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        metric_name=getattr(args, "metric", None),
    )
    
    if method == "gnn":
        kwargs.update(
            split_dir=args.split_dir,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            model_path=getattr(args, "model_path", ""),
        )
    elif method == "llm":
        kwargs.update(
            llm_model=args.model,
            hops=args.hops,
            use_info=args.use_info,
            workers=args.workers,
        )
    else:  # baseline
        kwargs.update(baseline_mode=args.mode)
    
    return AttributeConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Runner", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="task", required=True, help="Task type")
    
    # predict subparser
    predict = subparsers.add_parser("predict", help="Run prediction")
    predict_sub = predict.add_subparsers(dest="target", required=True)
    
    # predict link
    pred_link = predict_sub.add_parser("link", help="Link prediction")
    pred_link_sub = pred_link.add_subparsers(dest="method", required=True)
    
    for method in ["gnn", "llm", "baseline"]:
        p = pred_link_sub.add_parser(method)
        add_common_args(p)
        if method == "gnn":
            add_gnn_args(p, is_link=True)
        elif method == "llm":
            add_llm_args(p)
        else:
            add_baseline_args(p, is_link=True)
    
    # predict attr
    pred_attr = predict_sub.add_parser("attr", help="Attribute prediction")
    pred_attr_sub = pred_attr.add_subparsers(dest="method", required=True)
    
    for method in ["gnn", "llm", "baseline"]:
        p = pred_attr_sub.add_parser(method)
        add_common_args(p)
        p.add_argument("--metric", default=None)
        if method == "gnn":
            add_gnn_args(p, is_link=False)
        elif method == "llm":
            add_llm_args(p)
        else:
            add_baseline_args(p, is_link=False)
    
    # rank subparser
    rank = subparsers.add_parser("rank", help="Run ranking")
    rank_sub = rank.add_subparsers(dest="target", required=True)
    
    # rank link
    rank_link = rank_sub.add_parser("link", help="Link ranking")
    rank_link_sub = rank_link.add_subparsers(dest="method", required=True)
    
    for method in ["gnn", "llm", "baseline"]:
        p = rank_link_sub.add_parser(method)
        add_common_args(p)
        if method == "gnn":
            p.add_argument("--split-dir", default="output/artifact_graph_splits")
            p.add_argument("--model-path", required=True)
        elif method == "llm":
            add_llm_args(p)
        else:
            add_baseline_args(p, is_link=True)
    
    # rank attr
    rank_attr = rank_sub.add_parser("attr", help="Attribute ranking")
    rank_attr_sub = rank_attr.add_subparsers(dest="method", required=True)
    
    for method in ["gnn", "llm", "baseline"]:
        p = rank_attr_sub.add_parser(method)
        add_common_args(p)
        p.add_argument("--metric", default=None)
        if method == "gnn":
            p.add_argument("--split-dir", default="output/artifact_graph_splits")
            p.add_argument("--model-path", required=True)
        elif method == "llm":
            add_llm_args(p)
        else:
            add_baseline_args(p, is_link=False)
    
    args = parser.parse_args()
    
    # Route to appropriate runner
    if args.task == "predict":
        if args.target == "link":
            config = build_link_config(args, args.method)
            run_link_prediction(config)
        else:  # attr
            config = build_attr_config(args, args.method)
            run_attribute_prediction(config)
    else:  # rank
        if args.target == "link":
            config = build_link_config(args, args.method)
            run_link_ranking(config)
        else:  # attr
            config = build_attr_config(args, args.method)
            run_attribute_ranking(config)


if __name__ == "__main__":
    main()
