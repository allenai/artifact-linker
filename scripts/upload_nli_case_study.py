#!/usr/bin/env python3
"""Upload NLI case-study (576 evaluations + aggregated summary + scripts + figures)
as a subfolder of the artifact-graph dataset repo on HuggingFace Hub.

Target: https://huggingface.co/datasets/lwaekfjlk/artifact-graph/tree/main/case_study_nli
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

REPO_ID = "lwaekfjlk/artifact-graph"
STAGE = Path("/home/haofeiy2/artifact-linker/data/_hf_staging_nli")


README_APPEND = """

## Case study: NLI (`case_study_nli/`)

A frozen snapshot of 576 (model, dataset) evaluations used as the NLI case
study in our paper: 48 NLI models × 12 NLI datasets. Each cell was run by
an LLM-coder pipeline that produced per-example predictions and a
top-level accuracy.

### Layout

| path                                  | description |
|---------------------------------------|-------------|
| `case_study_nli/raw_evals/`           | 576 subdirs, one per (model, dataset) cell |
| `raw_evals/<model>_<dataset>_accuracy/predictions.json` | List of `{idx, prediction, ground_truth}` per example |
| `raw_evals/<model>_<dataset>_accuracy/results.json`     | `{accuracy: float}` (and `previous_accuracy` for 9 bug cells) |
| `raw_evals/<model>_<dataset>_accuracy/run_eval.py`      | LLM-written eval script (present in 119 cells) |
| `all_results_summary_fixed.json`      | Cleaned aggregate: 576 rows with 9 bug fixes and per-cell mask flags |
| `scripts/rebuild_nli_summary.py`      | Raw `raw_evals/` → `all_results_summary_fixed.json` |
| `scripts/plot_nli_heatmap.py`         | Renders 45-model heatmap (3 models with degenerate cells excluded) |
| `scripts/plot_nli_matrix_scree.py`    | Renders double-centered SVD scree plot |
| `figures/nli_results_heatmap.{png,pdf}` | Main heatmap figure |
| `figures/nli_matrix_scree.{png,pdf}`    | Scree plot showing rank-5 captures 91% of interaction variance |

### Known issues in raw eval pipeline (reproduced in `raw_evals/`)

1. **9 bug-fix cells**: top-level `accuracy=0` but `previous_accuracy>0`
   (overwrite bug in the LLM-coder's final results.json). The fixed summary
   uses `previous_accuracy`.
2. **Binary-output models on 3-way datasets**: three zero-shot
   classifiers (`MoritzLaurer/{deberta-v3-large,roberta-large,xtremedistil}-zeroshot-v2.0`)
   output only 2 labels. On MNLI / SNLI / ANLI / NLI_FEVER their accuracies
   are masked in the aggregate (not directly comparable to 3-way models).
3. **2 true failures**: `microsoft/deberta-v3-base` on `allenai/scitail` and
   `araag2/MedNLI` — model loaded but produced degenerate predictions.

### Reproducing the aggregate and figures

```bash
pip install datasets numpy matplotlib
python scripts/rebuild_nli_summary.py \\
    --src case_study_nli/raw_evals \\
    --out case_study_nli/all_results_summary_fixed.json
python scripts/plot_nli_heatmap.py \\
    --input case_study_nli/all_results_summary_fixed.json \\
    --out-dir case_study_nli/figures
python scripts/plot_nli_matrix_scree.py \\
    --input case_study_nli/all_results_summary_fixed.json \\
    --out-dir case_study_nli/figures
```
"""


def update_readme(api: HfApi, token: str):
    """Download current README, append case-study section, re-upload."""
    from huggingface_hub import hf_hub_download
    cur_path = hf_hub_download(REPO_ID, "README.md", repo_type="dataset", token=token)
    with open(cur_path) as f:
        cur = f.read()
    if "Case study: NLI" in cur:
        print("  README already has case-study section, skipping append")
        return
    new = cur + README_APPEND
    api.upload_file(
        path_or_fileobj=new.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        commit_message="Add NLI case study section",
    )
    print("  README updated")


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    print(f"Uploading {STAGE} → {REPO_ID}/case_study_nli ...")
    upload_folder(
        folder_path=str(STAGE),
        path_in_repo="case_study_nli",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        commit_message="Add NLI case study: 576 raw evals + aggregate + scripts + figures",
    )

    print("\nUpdating README...")
    update_readme(api, token)

    print(f"\nDone. Browse at https://huggingface.co/datasets/{REPO_ID}/tree/main/case_study_nli")


if __name__ == "__main__":
    main()
