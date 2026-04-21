#!/usr/bin/env python3
"""
Scree plot of the NLI accuracy matrix (singular values + cumulative energy).

Uses all_results_summary_fixed.json (45 NLI models x 12 datasets after dropping
models with any cell < min_cell).

Matches ablation_layers_link.png in figsize/fontsize.

Usage:
    python scripts/plot_nli_matrix_scree.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_matrix(results):
    models = sorted({r["model_id"] for r in results})
    datasets = sorted({r["dataset_id"] for r in results})
    m_idx = {m: i for i, m in enumerate(models)}
    d_idx = {d: i for i, d in enumerate(datasets)}
    M = np.full((len(datasets), len(models)), np.nan, dtype=float)
    for r in results:
        if r.get("masked"):
            continue
        M[d_idx[r["dataset_id"]], m_idx[r["model_id"]]] = r["accuracy"]
    return M, datasets, models


def drop_low_cell_models(M, models, threshold):
    # Ignore NaN (masked) cells in the threshold check
    bad = (M < threshold) & ~np.isnan(M)
    keep = ~bad.any(axis=0)
    return M[:, keep], [m for m, k in zip(models, keep) if k]


def plot_scree(M, out_png, out_pdf):
    # Impute masked cells with per-row mean before SVD (keeps them from shifting the mean)
    row_mean = np.nanmean(M, axis=1, keepdims=True)
    M_imp = np.where(np.isnan(M), row_mean, M)
    # Double-centering: remove additive row (dataset-difficulty) and column
    # (model-strength) effects; what remains is the interaction matrix.
    row_m = M_imp.mean(axis=1, keepdims=True)
    col_m = M_imp.mean(axis=0, keepdims=True)
    grand = M_imp.mean()
    Mc = M_imp - row_m - col_m + grand
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    # Drop the last (near-zero) singular value — double-centering makes
    # the matrix rank-deficient by 1 so the 12th component is meaningless.
    S = S[:-1]
    energy = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(energy)

    # Match ablation_layers_link.png style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2.0

    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.spines["top"].set_visible(False)

    bar_color = "#E74C3C"
    line_color = "#2C3E50"
    ks = np.arange(1, len(S) + 1)

    ax1.bar(ks, S, color=bar_color, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Rank $k$", fontsize=22, labelpad=8)
    ax1.set_ylabel(r"Singular value  $\sigma_k$", fontsize=22, color=bar_color, labelpad=8)
    ax1.set_yticks([0.1, 0.5, 1.0])
    ax1.set_yticklabels(["0.1", "0.5", "1"])
    ax1.yaxis.set_minor_locator(plt.NullLocator())
    ax1.tick_params(axis="y", labelcolor=bar_color, labelsize=14)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ks, cum, "o-", color=line_color, linewidth=2.2, markersize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative energy", fontsize=22, color=line_color, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=line_color, labelsize=14)

    # σ_1 annotation
    ax1.annotate(rf"$\sigma_1 = {S[0]:.2f}$", xy=(1, S[0]),
                 xytext=(2.2, S[0] * 1.02), fontsize=18, color=bar_color)

    # 90% line
    ax2.axhline(0.9, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.text(len(S) - 0.2, 0.91, "90%", fontsize=14, color="#777777",
             ha="right", va="bottom")

    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.14, top=0.96)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)

    print(f"Singular values: {S}")
    print(f"Cumulative energy: {cum}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="all_results_summary_fixed.json")
    p.add_argument("--out-dir", default="data/figures")
    p.add_argument("--stem", default="nli_matrix_scree")
    p.add_argument("--min-cell", type=float, default=0.05,
                   help="Drop models with any cell below this threshold")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    in_path = (root / args.input).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text())
    M, datasets, models = build_matrix(data["results"])
    if args.min_cell > 0:
        M, models = drop_low_cell_models(M, models, args.min_cell)
    print(f"Matrix shape: {M.shape} ({len(datasets)} datasets x {len(models)} models)")

    png = out_dir / f"{args.stem}.png"
    pdf = out_dir / f"{args.stem}.pdf"
    plot_scree(M, png, pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
