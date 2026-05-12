#!/usr/bin/env python3
"""
Use the trained joint GATv2 (transductive, Voyage embeddings) to predict
the accuracy of the newly-added held-out edge:
    14050 (sileod/deberta-v3-large-tasksource-nli) × 14051 (nyu-mll/multi_nli)

Ground truth: 0.9300
Predicted:    ?  (printed below)

Also report link-prob prediction for the same pair as sanity check.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/haofeiy2/artifact-linker")
sys.path.append(str(ROOT))

from artifact_graph.training.gnn_joint_trainer import (
    JointModelConfig, build_joint_model,
)
from artifact_graph.training.gnn_attribute_trainer import load_attribute_split
from artifact_graph.runners.runner_utils import load_node_embeddings

SPLIT_DIR = ROOT / "data" / "artifact_graph_splits_v3_0314_transductive"
MODEL_PATH = ROOT / "data" / "final_results_0314" / "trans_joint_gatv2_model_emb.pth"

SILEOD_ID = 14050
MNLI_ID = 14051
BACKBONE_ID = 14052
GT_ACCURACY = 0.9300


def load_joint_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = JointModelConfig(
        in_channels=cfg_dict.get("in_channels", 1024),
        hidden_channels=cfg_dict.get("hidden_channels", 128),
        num_layers=cfg_dict.get("num_layers", 3),
        heads=cfg_dict.get("heads", 8),
        dropout=cfg_dict.get("dropout", 0.2),
        backbone="gatv2",
        model_type="gatv2",
    )
    model = build_joint_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model: layers={cfg.num_layers}, hidden={cfg.hidden_channels}, params={sum(p.numel() for p in model.parameters()):,}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load current augmented graph (train split is used for encoding context)
    print("\nLoading augmented graph...")
    forced_x = load_node_embeddings(str(SPLIT_DIR), "embedding")
    print(f"  Node embeddings: {forced_x.shape}")  # should be (14053, 1024)

    # Use test split (has the held-out edge in pos_edges)
    G_te, S_te = load_attribute_split(str(SPLIT_DIR / "test_split"), forced_x)
    print(f"  G_te.x: {G_te.x.shape}")
    print(f"  G_te.edge_index: {G_te.edge_index.shape}")
    G_te.x = G_te.x.to(device)
    G_te.edge_index = G_te.edge_index.to(device)

    # Load trained model
    model = load_joint_model(device)

    # Encode (message passing over train graph's edges -- in transductive, this
    # is the full train + test structural edges except held-out positives)
    with torch.no_grad():
        z = model.encode(G_te.x, G_te.edge_index)
    print(f"\nEncoded node representations: {z.shape}")

    # Predict the new edge
    pair = torch.tensor([[SILEOD_ID, MNLI_ID], [MNLI_ID, SILEOD_ID]],
                         dtype=torch.long, device=device).t()
    # Actually decode_attr expects shape (2, N) with row 0 = model, row 1 = dataset
    pair = torch.tensor([[SILEOD_ID], [MNLI_ID]], dtype=torch.long, device=device)

    with torch.no_grad():
        attr_logit = model.decode_attr(z, pair).squeeze()
        attr_pred = torch.sigmoid(torch.clamp(attr_logit, -10, 10)).item()
        link_logit = model.decode_link(z, pair).squeeze()
        link_prob = torch.sigmoid(link_logit).item()

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"  Cell: sileod/deberta-v3-large-tasksource-nli × nyu-mll/multi_nli")
    print(f"  Ground truth accuracy:  {GT_ACCURACY:.4f}")
    print(f"  Predicted accuracy:     {attr_pred:.4f}   (abs error: {abs(attr_pred - GT_ACCURACY):.4f})")
    print(f"  Predicted link prob:    {link_prob:.4f}")
    print("=" * 60)

    # Also show sileod's structural context
    sileod_neighbors = []
    ei = G_te.edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        if ei[0, i] == SILEOD_ID or ei[1, i] == SILEOD_ID:
            other = int(ei[1, i] if ei[0, i] == SILEOD_ID else ei[0, i])
            sileod_neighbors.append(other)
    print(f"\nsileod neighbors in message-passing graph: {sorted(set(sileod_neighbors))}")
    import json
    nm = json.load(open(SPLIT_DIR / "test_split" / "node_metadata.json"))
    for n in sorted(set(sileod_neighbors)):
        info = nm.get(str(n), {})
        print(f"  id={n}  type={info.get('type')}  name={info.get('name')}")


if __name__ == "__main__":
    main()
