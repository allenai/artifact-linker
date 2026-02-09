#!/usr/bin/env python3
"""
Visualize transductive and inductive graph splits.

For each split type, produces a figure with 3 panels:
  - Left:   Training graph only (message-passing edges)
  - Center: Full graph with edges colored by train/val/test
  - Right:  Zoomed stats & structure summary

Key visual distinction:
  - SOLID lines  = message-passing edges (known at training time)
  - DASHED lines = target edges (to be predicted, NOT in training graph)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
EDGE_COLORS = {"train": "#3B82F6", "val": "#22C55E", "test": "#EF4444"}  # blue / green / red
NODE_COLORS = {
    "model_train": "#93C5FD",     # light blue
    "model_val": "#86EFAC",       # light green
    "model_test": "#FCA5A5",      # light red
    "dataset": "#FBBF24",         # amber / yellow for all datasets
    "model": "#93C5FD",           # default
    "node_dimmed": "#E5E7EB",     # light gray for unseen nodes
}
NODE_EDGE_COLORS = {
    "model_train": "#1E40AF",
    "model_val": "#15803D",
    "model_test": "#B91C1C",
    "dataset": "#92400E",
    "model": "#1E40AF",
    "node_dimmed": "#D1D5DB",
}


def load_data(split_dir: Path):
    """Load all data needed for visualization."""
    data = {}
    for name in ["train_split", "val_split", "test_split"]:
        d = {}
        p_edges = split_dir / name / "edges.npz"
        p_pos = split_dir / name / "pos_edges.npz"
        if p_edges.exists():
            e = np.load(p_edges)["edges"]
            if e.shape[0] != 2 and e.ndim == 2:
                e = e.T
            d["msg_edges"] = e
        if p_pos.exists():
            e = np.load(p_pos)["edges"]
            if e.shape[0] != 2 and e.ndim == 2:
                e = e.T
            d["pos_edges"] = e
        data[name.replace("_split", "")] = d

    # Node metadata (same across splits)
    for name in ["train_split", "val_split", "test_split"]:
        p = split_dir / name / "node_metadata.json"
        if p.exists():
            with open(p) as f:
                data["node_meta"] = {int(k): v for k, v in json.load(f).items()}
            break

    # Node split info (inductive only)
    ns_path = split_dir / "node_split.json"
    if ns_path.exists():
        with open(ns_path) as f:
            data["node_split"] = json.load(f)
    return data


def compute_layout(node_meta, seed=42):
    """
    Deterministic layout: datasets on top row, models on bottom row.
    Nodes with higher degree get placed more centrally.
    """
    rng = np.random.RandomState(seed)
    models = sorted([n for n, v in node_meta.items() if v.get("type") == "model"])
    datasets = sorted([n for n, v in node_meta.items() if v.get("type") == "dataset"])

    pos = {}
    # Spread datasets evenly across top
    for i, d in enumerate(datasets):
        x = (i + 0.5) / max(len(datasets), 1)
        y = 1.0 + rng.uniform(-0.02, 0.02)
        pos[d] = (x, y)
    # Spread models evenly across bottom
    for i, m in enumerate(models):
        x = (i + 0.5) / max(len(models), 1)
        y = 0.0 + rng.uniform(-0.02, 0.02)
        pos[m] = (x, y)
    return pos


def _draw_edges(ax, pos, edges, color, alpha=0.3, linewidth=0.4, linestyle="-"):
    """Draw edges efficiently using LineCollection-like approach."""
    if edges.shape[1] == 0:
        return
    # Deduplicate: for undirected, (u,v) and (v,u) are the same visual edge
    seen = set()
    xs, ys = [], []
    for i in range(edges.shape[1]):
        u, v = int(edges[0, i]), int(edges[1, i])
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        if u in pos and v in pos:
            xs.extend([pos[u][0], pos[v][0], None])
            ys.extend([pos[u][1], pos[v][1], None])
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)


def _draw_nodes(ax, pos, node_ids, color, edge_color, size=8, marker="o", zorder=3):
    """Draw a set of nodes."""
    if not node_ids:
        return
    xs = [pos[n][0] for n in node_ids if n in pos]
    ys = [pos[n][1] for n in node_ids if n in pos]
    ax.scatter(xs, ys, c=color, edgecolors=edge_color, s=size, marker=marker,
               linewidths=0.3, zorder=zorder)


# ═══════════════════════════════════════════════════════════════════════════
# Transductive
# ═══════════════════════════════════════════════════════════════════════════
def plot_transductive(data, output_path: Path):
    """
    Transductive: all nodes shared, only edges split.
    Panel 1: Training graph (msg edges only)
    Panel 2: Full graph with edges colored by split
    """
    meta = data["node_meta"]
    pos = compute_layout(meta)

    models = [n for n, v in meta.items() if v.get("type") == "model"]
    datasets = [n for n, v in meta.items() if v.get("type") == "dataset"]

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # ── Panel 1: Training graph only ──
    ax = axes[0]
    train_msg = data["train"]["msg_edges"]
    _draw_edges(ax, pos, train_msg, EDGE_COLORS["train"], alpha=0.3, linewidth=0.4)
    _draw_nodes(ax, pos, datasets, NODE_COLORS["dataset"], NODE_EDGE_COLORS["dataset"], size=15, marker="s")
    _draw_nodes(ax, pos, models, NODE_COLORS["model"], NODE_EDGE_COLORS["model"], size=6)
    ax.set_title("Training Graph\n(message-passing edges only)", fontsize=13, fontweight="bold")
    n_msg = train_msg.shape[1] // 2  # undirected counted both ways
    ax.text(0.02, 0.02, f"Nodes: {len(meta)}\nMsg edges: {n_msg}\n(undirected, both directions stored)",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.axis("off")

    # ── Panel 2: Full graph – edges colored by split ──
    ax = axes[1]
    # Draw train edges (solid)
    train_pos = data["train"]["pos_edges"]
    _draw_edges(ax, pos, train_pos, EDGE_COLORS["train"], alpha=0.25, linewidth=0.3)
    # Draw val edges (dashed) — these are targets
    val_pos = data["val"]["pos_edges"]
    _draw_edges(ax, pos, val_pos, EDGE_COLORS["val"], alpha=0.8, linewidth=1.0, linestyle="--")
    # Draw test edges (dashed) — these are targets
    test_pos = data["test"]["pos_edges"]
    _draw_edges(ax, pos, test_pos, EDGE_COLORS["test"], alpha=0.8, linewidth=1.0, linestyle="--")

    _draw_nodes(ax, pos, datasets, NODE_COLORS["dataset"], NODE_EDGE_COLORS["dataset"], size=15, marker="s")
    _draw_nodes(ax, pos, models, NODE_COLORS["model"], NODE_EDGE_COLORS["model"], size=6)

    ax.set_title("Full Graph with Edge Split\n(dashed = prediction targets)", fontsize=13, fontweight="bold")

    stats_text = (
        f"Train edges: {train_pos.shape[1]} (solid blue)\n"
        f"Val edges: {val_pos.shape[1]} (dashed green)\n"
        f"Test edges: {test_pos.shape[1]} (dashed red)\n"
        f"All {len(meta)} nodes shared across splits"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    handles = [
        mlines.Line2D([], [], color=EDGE_COLORS["train"], linewidth=2, label="Train edges (in graph)"),
        mlines.Line2D([], [], color=EDGE_COLORS["val"], linewidth=2, linestyle="--", label="Val edges (targets)"),
        mlines.Line2D([], [], color=EDGE_COLORS["test"], linewidth=2, linestyle="--", label="Test edges (targets)"),
        mlines.Line2D([], [], marker="s", color="w", markerfacecolor=NODE_COLORS["dataset"],
                       markersize=8, label="Datasets"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=NODE_COLORS["model"],
                       markersize=6, label="Models"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.axis("off")

    fig.suptitle("Transductive Split – All Nodes Shared, Only Edges Split",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Inductive
# ═══════════════════════════════════════════════════════════════════════════
def plot_inductive(data, output_path: Path):
    """
    Inductive (new_models): held-out model nodes unseen at train time.
    Panel 1: Training graph (only train nodes + train edges)
    Panel 2: Full graph with held-out nodes highlighted
    """
    meta = data["node_meta"]
    ns = data.get("node_split", {})
    pos = compute_layout(meta)

    # Build node split maps
    train_models = set(ns.get("models", {}).get("train", []))
    val_models = set(ns.get("models", {}).get("val", []))
    test_models = set(ns.get("models", {}).get("test", []))
    datasets = [n for n, v in meta.items() if v.get("type") == "dataset"]

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # ── Panel 1: Training graph only (no held-out nodes) ──
    ax = axes[0]
    train_msg = data["train"]["msg_edges"]
    _draw_edges(ax, pos, train_msg, EDGE_COLORS["train"], alpha=0.3, linewidth=0.4)

    # Draw held-out nodes as dimmed (they exist but have NO edges in training)
    _draw_nodes(ax, pos, list(val_models), NODE_COLORS["node_dimmed"], NODE_EDGE_COLORS["node_dimmed"], size=4)
    _draw_nodes(ax, pos, list(test_models), NODE_COLORS["node_dimmed"], NODE_EDGE_COLORS["node_dimmed"], size=4)
    # Draw train nodes normally
    _draw_nodes(ax, pos, datasets, NODE_COLORS["dataset"], NODE_EDGE_COLORS["dataset"], size=15, marker="s")
    _draw_nodes(ax, pos, list(train_models), NODE_COLORS["model_train"], NODE_EDGE_COLORS["model_train"], size=6)

    ax.set_title("Training Graph\n(held-out model nodes dimmed, NO edges to them)",
                 fontsize=13, fontweight="bold")
    n_msg = train_msg.shape[1] // 2
    stats = (
        f"Train model nodes: {len(train_models)} (blue)\n"
        f"Val model nodes: {len(val_models)} (gray, unseen)\n"
        f"Test model nodes: {len(test_models)} (gray, unseen)\n"
        f"Datasets: {len(datasets)} (shared)\n"
        f"Msg edges: {n_msg}"
    )
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.axis("off")

    # ── Panel 2: Full graph – held-out nodes + target edges shown ──
    ax = axes[1]

    # Train edges (solid) — in the graph
    train_pos = data["train"]["pos_edges"]
    _draw_edges(ax, pos, train_pos, EDGE_COLORS["train"], alpha=0.2, linewidth=0.3)
    # Val edges (dashed) — targets, connect to held-out val models
    val_pos = data["val"]["pos_edges"]
    _draw_edges(ax, pos, val_pos, EDGE_COLORS["val"], alpha=0.8, linewidth=1.2, linestyle="--")
    # Test edges (dashed) — targets, connect to held-out test models
    test_pos = data["test"]["pos_edges"]
    _draw_edges(ax, pos, test_pos, EDGE_COLORS["test"], alpha=0.8, linewidth=1.2, linestyle="--")

    # Nodes
    _draw_nodes(ax, pos, datasets, NODE_COLORS["dataset"], NODE_EDGE_COLORS["dataset"], size=15, marker="s")
    _draw_nodes(ax, pos, list(train_models), NODE_COLORS["model_train"], NODE_EDGE_COLORS["model_train"], size=6)
    _draw_nodes(ax, pos, list(val_models), NODE_COLORS["model_val"], NODE_EDGE_COLORS["model_val"], size=10)
    _draw_nodes(ax, pos, list(test_models), NODE_COLORS["model_test"], NODE_EDGE_COLORS["model_test"], size=10)

    ax.set_title("Full Graph with Held-out Nodes\n(dashed edges = prediction targets to unseen models)",
                 fontsize=13, fontweight="bold")

    stats_text = (
        f"Train edges: {train_pos.shape[1]} (solid blue, between train nodes)\n"
        f"Val edges: {val_pos.shape[1]} (dashed green, to val models)\n"
        f"Test edges: {test_pos.shape[1]} (dashed red, to test models)\n"
        f"Val/Test edges are ABSENT from training graph"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    handles = [
        mlines.Line2D([], [], color=EDGE_COLORS["train"], linewidth=2, label="Train edges (in graph)"),
        mlines.Line2D([], [], color=EDGE_COLORS["val"], linewidth=2, linestyle="--", label="Val target edges"),
        mlines.Line2D([], [], color=EDGE_COLORS["test"], linewidth=2, linestyle="--", label="Test target edges"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=NODE_COLORS["model_train"],
                       markersize=6, label=f"Train models ({len(train_models)})"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=NODE_COLORS["model_val"],
                       markersize=8, label=f"Val models ({len(val_models)}, held-out)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=NODE_COLORS["model_test"],
                       markersize=8, label=f"Test models ({len(test_models)}, held-out)"),
        mlines.Line2D([], [], marker="s", color="w", markerfacecolor=NODE_COLORS["dataset"],
                       markersize=8, label=f"Datasets ({len(datasets)}, shared)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=NODE_COLORS["node_dimmed"],
                       markersize=6, label="Unseen at train time"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    ax.axis("off")

    fig.suptitle("Inductive Split (new_models) – Held-out Model Nodes Unseen During Training",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    data_root = Path(__file__).parent.parent / "data"
    trans_dir = data_root / "artifact_graph_splits_v2_1125_transductive"
    induc_dir = data_root / "artifact_graph_splits_v2_1125_inductive" / "new_models"

    output_dir = data_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading transductive data...")
    trans_data = load_data(trans_dir)
    print("Loading inductive data...")
    induc_data = load_data(induc_dir)

    print("\nGenerating figures...")
    plot_transductive(trans_data, output_dir / "graph_split_transductive.png")
    plot_inductive(induc_data, output_dir / "graph_split_inductive.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
