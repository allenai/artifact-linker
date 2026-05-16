#!/usr/bin/env python3
"""GNN attribute ranking — loads a trained joint checkpoint, ranks models/dataset.

Train the checkpoint first with scripts/train_joint_gnn.py.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from artifact_graph.runners.joint_gnn_runner import load_joint_model, eval_attr_ranking
from artifact_graph.runners.runner_utils import detect_split_type


def main():
    p = argparse.ArgumentParser(description="GNN Attribute Ranking (joint checkpoint)")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True, help="joint checkpoint .pth")
    p.add_argument("--backbone", default="gatv2", help="for output naming only")
    p.add_argument("--embedding-mode", default="embedding", choices=["random", "embedding"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prefix = detect_split_type(args.split_dir)
    emb = "random" if args.embedding_mode == "random" else "emb"

    model = load_joint_model(args.model_path, device)
    metrics = eval_attr_ranking(model, Path(args.split_dir), args.embedding_mode, device)
    print(f"[{prefix}/{args.backbone}] attr ranking:",
          {k: round(v, 4) for k, v in metrics.items()})

    out = out_dir / f"{prefix}_joint_{args.backbone}_attr_rankings_{emb}.json"
    out.write_text(json.dumps({"test_metrics": metrics}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
