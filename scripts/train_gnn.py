import argparse

import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.utils import negative_sampling

from artifact_graph.gnn import LinkPredictionGNN
from artifact_graph.utils import (
    build_bipartite_graph,
    nx_to_pyg_data,
    prepare_link_pred_splits,
)


def bipartite_negative_sampling(edge_index, model_mask, num_neg_samples, over_sample=3):
    """
    edge_index:       [2, E_pos] long tensor of positive (model⇄data) edges
    model_mask:       [N] bool tensor, True for model-nodes, False for data-nodes
    num_neg_samples:  how many negatives you actually need
    over_sample:      sample this many times more and then filter
    """
    neg = negative_sampling(
        edge_index=edge_index,
        num_nodes=model_mask.size(0),
        num_neg_samples=num_neg_samples * over_sample,
    )  # [2, E_neg_pool]

    u, v = neg
    cross_mask = (model_mask[u] & ~model_mask[v]) | (~model_mask[u] & model_mask[v])
    u, v = u[cross_mask], v[cross_mask]
    u, v = u[:num_neg_samples], v[:num_neg_samples]
    return torch.stack([u, v], dim=0)


def train(model, data, optimizer, bce_loss):
    model.train()
    optimizer.zero_grad()

    # Build model_mask tensor and attach to data
    if isinstance(data.type, list):
        model_mask = torch.tensor([t == "model" for t in data.type], dtype=torch.bool)
    else:
        model_mask = data.type == 0
    data.is_model = model_mask.to(data.x.device)

    # Sample bipartite negatives for training
    neg_edge_index = bipartite_negative_sampling(
        edge_index=data.train_pos_edge_index,
        model_mask=data.is_model,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )

    pos_pred, neg_pred = model(data.x, data.edge_index, data.train_pos_edge_index, neg_edge_index)
    pos_label = torch.ones(pos_pred.size(0), device=pos_pred.device)
    neg_label = torch.zeros(neg_pred.size(0), device=neg_pred.device)
    loss = bce_loss(pos_pred, pos_label) + bce_loss(neg_pred, neg_label)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, split, threshold):
    model.eval()
    pos_e = getattr(data, f"{split}_pos_edge_index")
    neg_e = getattr(data, f"{split}_neg_edge_index")
    with torch.no_grad():
        pos_pred, neg_pred = model(data.x, data.edge_index, pos_e, neg_e)
    y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).cpu().numpy()
    y_score = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    y_pred = (y_score > threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    return auc, ap, f1


def get_topk_negative_edges(model, data, split, k=50):
    model.eval()
    pos_e = getattr(data, f"{split}_pos_edge_index")
    neg_e = getattr(data, f"{split}_neg_edge_index")
    with torch.no_grad():
        _, neg_pred = model(data.x, data.edge_index, pos_e, neg_e)
    scores = neg_pred.cpu().numpy()
    edges = neg_e.cpu().numpy().T
    topk_idx = scores.argsort()[::-1][:k]
    topk_edges = edges[topk_idx]
    topk_scores = scores[topk_idx]
    return [(int(u), int(v), float(s)) for (u, v), s in zip(topk_edges, topk_scores)]


def main(args):
    # 1) Data preparation
    G = build_bipartite_graph(
        args.data_dir, args.dataset_json, args.metadata_dir, args.min_downloads
    )
    data = prepare_link_pred_splits(nx_to_pyg_data(G), args.val_ratio, args.test_ratio)

    # ─── Override val/test negatives to respect bipartite structure ───────────
    if isinstance(data.type, list):
        model_mask_cpu = torch.tensor([t == "model" for t in data.type], dtype=torch.bool)
    else:
        model_mask_cpu = data.type == 0

    data.val_neg_edge_index = bipartite_negative_sampling(
        edge_index=data.val_pos_edge_index,
        model_mask=model_mask_cpu,
        num_neg_samples=data.val_pos_edge_index.size(1),
    )
    data.test_neg_edge_index = bipartite_negative_sampling(
        edge_index=data.test_pos_edge_index,
        model_mask=model_mask_cpu,
        num_neg_samples=data.test_pos_edge_index.size(1) * 10000,
    )
    # ─────────────────────────────────────────────────────────────────────────
    breakpoint()

    # Re-attach training graph for encoder
    data.edge_index = data.train_pos_edge_index

    # Ensure all edge indices are [2, E] long tensors
    for name in [
        "edge_index",
        "train_pos_edge_index",
        "val_pos_edge_index",
        "val_neg_edge_index",
        "test_pos_edge_index",
        "test_neg_edge_index",
    ]:
        e = getattr(data, name)
        if e.dim() == 2 and e.size(1) == 2:
            e = e.t().contiguous()
        setattr(data, name, e.to(torch.long))

    # 2) Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = LinkPredictionGNN(
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = torch.nn.BCELoss()

    # 3) Training loop + checkpointing
    best_val_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, bce_loss)

        if epoch % args.log_every == 0:
            val_auc, val_ap, val_f1 = evaluate(model, data, "val", args.f1_threshold)
            print(
                f"Epoch {epoch:03d} │ Loss {loss:.4f} │ "
                f"Val AUC {val_auc:.4f} │ Val AP {val_ap:.4f} │ Val F1 {val_f1:.4f}"
            )

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), args.checkpoint_path)
                print(
                    f"→ New best val-F1 {best_val_auc:.4f}, checkpoint saved to {args.checkpoint_path}"
                )

    # 4) Load best checkpoint and evaluate on test set
    print(f"\nLoading best checkpoint (val-F1={best_val_auc:.4f}) from {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    test_auc, test_ap, test_f1 = evaluate(model, data, "test", args.f1_threshold)
    print(f"Test AUC {test_auc:.4f} │ Test AP {test_ap:.4f} │ Test F1 {test_f1:.4f}")

    # 5) Top-50 negative-test edges
    top50 = get_topk_negative_edges(model, data, "test", k=50)
    names = data.node_names  # list of node labels
    print("\nTop-50 negative-test edges by model score:")
    print(f"{'u_idx':>6} {'u_name':>20}   {'v_idx':>6} {'v_name':>20}   {'score':>6}")
    for u, v, score in top50:
        print(f"{u:6d} {names[u]:>20s}   {v:6d} {names[v]:>20s}   {score:6.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--f1_threshold",
        type=float,
        default=0.2,
        help="Threshold to binarize scores when computing F1",
    )
    parser.add_argument("--data_dir", default="../data/eval_datasets_json_download_ranks")
    parser.add_argument("--dataset_json", default="../data/dataset_info.json")
    parser.add_argument("--metadata_dir", default="../data/model_metadata_download_ranks")
    parser.add_argument("--min_downloads", type=int, default=1000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--checkpoint_path", default="best_model.pth", help="Path to save the best model checkpoint"
    )
    args = parser.parse_args()
    main(args)
