#!/usr/bin/env python3
"""Generate the NLI results heatmap (datasets x models) from all_results_summary.json.

Reproduces: data/figures/nli_results_heatmap.{png,pdf}
- 12 datasets (rows) x 45 models (columns; models with any 0-accuracy cell are dropped)
- Rows and columns sorted by mean accuracy (descending)
- Cells annotated as .xx (leading zero stripped), uniform black text
- Colormap: RdYlGn over [0, 1]
- Landscape figsize=(26, 8) to match ranking_cost_curve font metrics

Usage:
    python scripts/plot_nli_heatmap.py
    python scripts/plot_nli_heatmap.py --input all_results_summary.json --out-dir data/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASET_SHORT = {
    "stanfordnlp/snli": "SNLI",
    "nyu-mll/multi_nli": "MNLI",
    "SetFit/rte": "RTE",
    "SetFit/qnli": "QNLI",
    "facebook/anli": "ANLI",
    "allenai/scitail": "SciTail",
    "alisawuffles/WANLI": "WANLI",
    "araag2/MedNLI": "MedNLI",
    "tasksource/babi_nli": "bAbI-NLI",
    "tasksource/defeasible-nli": "δ-NLI",
    "pietrolesci/nli_fever": "NLI-FEVER",
    "pietrolesci/robust_nli_ST_SE": "RobustNLI",
}

MODEL_SHORT = {
    # Zero-shot NLI (MoritzLaurer)
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": "DeBERTa-L-ZS",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c": "DeBERTa-B-ZS",
    "MoritzLaurer/roberta-large-zeroshot-v2.0-c": "RoBERTa-L-ZS",
    "MoritzLaurer/ModernBERT-base-zeroshot-v2.0": "ModernBERT-B-ZS",
    "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33": "XtremeDistil",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": "mDeBERTa-XNLI",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": "DeBERTa-L-MFAW",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "DeBERTa-B-MFA",
    # Multi-task NLI
    "sileod/deberta-v3-large-tasksource-nli": "DeBERTa-L-Task",
    "sileod/deberta-v3-base-tasksource-nli": "DeBERTa-B-Task",
    "tasksource/ModernBERT-large-nli": "ModernBERT-L",
    "tasksource/ModernBERT-base-nli": "ModernBERT-B",
    "tasksource/deberta-small-long-nli": "DeBERTa-S-Long",
    "dleemiller/finecat-nli-l": "FineCat-L",
    "dleemiller/ModernCE-large-nli": "ModernCE-L",
    # SNLI+MNLI+FEVER+ANLI R3
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli": "ALBERT-XXL-R3",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli": "RoBERTa-L-R3",
    "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli": "BART-L-R3",
    "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli": "ELECTRA-L-R3",
    "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli": "DeBERTa-L-R3",
    "NDugar/debertav3-mnli-snli-anli": "DeBERTa-MSA",
    # Cross-encoder NLI
    "cross-encoder/nli-deberta-v3-large": "CE-DeBERTa-L",
    "cross-encoder/nli-deberta-v3-base": "CE-DeBERTa-B",
    "cross-encoder/nli-deberta-v3-small": "CE-DeBERTa-S",
    "cross-encoder/nli-roberta-base": "CE-RoBERTa-B",
    # DeBERTa-MNLI family
    "microsoft/deberta-v3-base": "DeBERTa-B",
    "microsoft/deberta-base-mnli": "DeBERTa-B-MNLI",
    "microsoft/deberta-large-mnli": "DeBERTa-L-MNLI",
    "microsoft/deberta-xlarge-mnli": "DeBERTa-XL-MNLI",
    "microsoft/deberta-v2-xlarge-mnli": "DeBERTa-v2-XL",
    "microsoft/deberta-v2-xxlarge-mnli": "DeBERTa-v2-XXL",
    "khalidalt/DeBERTa-v3-large-mnli": "DeBERTa-L-MNLI-K",
    "pepa/deberta-v3-large-snli": "DeBERTa-L-SNLI-P",
    "utahnlp/snli_microsoft_deberta-v3-large_seed-1": "DeBERTa-L-SNLI-U",
    # BART / RoBERTa
    "facebook/bart-large-mnli": "BART-L-MNLI",
    "joeddav/bart-large-mnli-yahoo-answers": "BART-Yahoo",
    "roberta-large-mnli": "RoBERTa-L-MNLI",
    "alisawuffles/roberta-large-wanli": "RoBERTa-L-WANLI",
    # Domain-specific / smaller
    "pritamdeka/PubMedBERT-MNLI-MedNLI": "PubMedBERT-NLI",
    "IDEA-CCNL/Erlangshen-Roberta-330M-NLI": "Erlangshen-RoBERTa",
    "cmarkea/distilcamembert-base-nli": "DistilCamemBERT",
    "typeform/distilbert-base-uncased-mnli": "DistilBERT",
    "prajjwal1/albert-base-v2-mnli": "ALBERT-B-MNLI",
    "textattack/bert-base-uncased-snli": "BERT-B-SNLI",
    # Generative LLMs
    "EleutherAI/gpt-neo-1.3B": "GPT-Neo-1.3B",
    "google/gemma-2b": "Gemma-2B",
    "google/gemma-2-2b": "Gemma-2-2B",
    "google/gemma-3-1b-it": "Gemma-3-1B",
}


def short_model_name(full: str, max_len: int = 26) -> str:
    if full in MODEL_SHORT:
        return MODEL_SHORT[full]
    name = full.split("/", 1)[1] if "/" in full else full
    if len(name) > max_len:
        name = name[: max_len - 2] + ".."
    return name


def build_matrix(results):
    models = sorted({r["model_id"] for r in results})
    datasets = sorted({r["dataset_id"] for r in results})
    m_idx = {m: i for i, m in enumerate(models)}
    d_idx = {d: i for i, d in enumerate(datasets)}
    M = np.full((len(datasets), len(models)), np.nan, dtype=float)
    mask = np.zeros_like(M, dtype=bool)
    for r in results:
        i, j = d_idx[r["dataset_id"]], m_idx[r["model_id"]]
        M[i, j] = r["accuracy"]
        if r.get("masked"):
            mask[i, j] = True
    return M, mask, datasets, models


def drop_low_cell_models(M, models, threshold):
    keep = ~(M < threshold).any(axis=0)
    return M[:, keep], [m for m, k in zip(models, keep) if k], [m for m, k in zip(models, keep) if not k]


def sort_by_mean(M, mask, datasets, models):
    # Sort ignoring masked cells
    M_sort = np.where(mask, np.nan, M)
    d_order = np.argsort(-np.nanmean(M_sort, axis=1))
    m_order = np.argsort(-np.nanmean(M_sort, axis=0))
    return (
        M[np.ix_(d_order, m_order)],
        mask[np.ix_(d_order, m_order)],
        [datasets[i] for i in d_order],
        [models[i] for i in m_order],
    )


def fmt_cell(v: float) -> str:
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s


def plot_heatmap(M, mask, datasets, models, out_png, out_pdf):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 26
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2.0

    fig, ax = plt.subplots(figsize=(32, 8))
    # Draw masked cells as grey (use separate NaN-ed matrix for imshow)
    M_display = np.where(mask, np.nan, M)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#FFF4B8")  # light yellow for masked cells
    ax.imshow(M_display, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            if mask[i, j]:
                ax.text(j, i, "—", ha="center", va="center", fontsize=20, color="#8B6914")
            else:
                ax.text(j, i, fmt_cell(v), ha="center", va="center", fontsize=17, color="black")

    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels([short_model_name(m) for m in models], rotation=55, ha="right", fontsize=20)
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels([DATASET_SHORT.get(d, d) for d in datasets], fontsize=26)
    ax.tick_params(axis="both", length=3)

    # bbox_inches='tight' auto-fits long labels (ROBUST_NLI / DEFEASIBLE)
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.05, top=0.98)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="all_results_summary_fixed.json")
    p.add_argument("--out-dir", default="data/figures")
    p.add_argument("--stem", default="nli_results_heatmap")
    p.add_argument("--min-cell", type=float, default=0.05,
                   help="Drop models with any cell below this threshold (default 0.05)")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    in_path = (root / args.input).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text())
    M, mask, datasets, models = build_matrix(data["results"])
    dropped = []
    if args.min_cell > 0:
        # Ignore masked cells for the drop check
        M_chk = np.where(mask, np.nan, M)
        keep = ~(M_chk < args.min_cell).any(axis=0)
        dropped = [m for m, k in zip(models, keep) if not k]
        M = M[:, keep]
        mask = mask[:, keep]
        models = [m for m, k in zip(models, keep) if k]
    M, mask, datasets, models = sort_by_mean(M, mask, datasets, models)

    png = out_dir / f"{args.stem}.png"
    pdf = out_dir / f"{args.stem}.pdf"
    plot_heatmap(M, mask, datasets, models, png, pdf)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Datasets ({len(datasets)}, high→low mean): {[DATASET_SHORT.get(d, d) for d in datasets]}")
    print(f"Models shown: {len(models)}" + (f"   dropped: {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
