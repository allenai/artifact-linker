#!/usr/bin/env python3
"""
Step 5: Summarize READMEs (removing info leakage) + normalize edge metrics.

Three parts:
  A) Summarize model/dataset/paper/codebase READMEs via LLM
  B) Compute text embeddings from summaries (Voyage / random)
  C) Normalize metric names to standard set and values to 0-1

Input:
  - data/artifact_raw_data/filtered_eval_pairs.json      (step 3)
  - data/artifact_raw_data/models/readmes/               (step 1)
  - data/artifact_raw_data/datasets/readmes/             (step 3)
  - data/artifact_raw_data/papers/metadata/              (step 4)
  - data/artifact_raw_data/codebases/readmes/            (step 4)
  - data/artifact_raw_data/resource_links.json           (step 4)

Output:
  - data/artifact_raw_data/readme_summaries.json
  - data/artifact_raw_data/node_embeddings_voyage.npy (or random)
  - data/artifact_raw_data/normalized_eval_pairs.json
  - data/artifact_raw_data/metric_name_mapping.json
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

STANDARD_METRICS = {
    "accuracy",
    "bleu",
    "chrf",
    "f1",
    "rouge-2",
    "rouge-l",
    "top-k_accuracy",
    "wer",
}

SIMPLE_METRIC_MAPPING = {
    # Accuracy variants
    "accuracy": "accuracy", "acc": "accuracy", "Accuracy": "accuracy",
    "ACC": "accuracy", "top1": "accuracy", "top-1": "accuracy",
    "top_1": "accuracy", "top1_accuracy": "accuracy",
    "exact_match": "accuracy", "em": "accuracy", "EM": "accuracy",
    "acc_norm": "accuracy", "normalized accuracy": "accuracy",
    "accuracy (0-shot)": "accuracy", "accuracy (5-shot)": "accuracy",
    "accuracy (10-shot)": "accuracy", "acc (5-shot)": "accuracy",
    "acc_norm (0-shot)": "accuracy", "acc_norm (5-shot)": "accuracy",
    "multiple_choice_grade": "accuracy", "mc2": "accuracy",
    "pass@1": "accuracy",
    "prompt_level_strict_acc": "accuracy", "inst_level_strict_acc": "accuracy",
    "prompt_level_loose_acc": "accuracy", "inst_level_loose_acc": "accuracy",
    # Top-k accuracy
    "top-k_accuracy": "top-k_accuracy",
    "top5": "top-k_accuracy", "top-5": "top-k_accuracy", "top_5": "top-k_accuracy",
    "top5_accuracy": "top-k_accuracy", "top5_accuracy": "top-k_accuracy",
    "top10": "top-k_accuracy", "top-10": "top-k_accuracy",
    "top1_accuracy": "accuracy",
    # BLEU
    "bleu": "bleu", "BLEU": "bleu", "Bleu": "bleu",
    "bleu-4": "bleu", "BLEU-4": "bleu", "bleu_score": "bleu", "sacrebleu": "bleu",
    # chrF
    "chrf": "chrf", "chr-F": "chrf", "chrF": "chrf",
    "chr-f": "chrf", "chrf++": "chrf", "chrF++": "chrf",
    # F1
    "f1": "f1", "F1": "f1", "f1_score": "f1", "f1-score": "f1",
    "f1_macro": "f1", "f1-macro": "f1", "f1_micro": "f1", "f1-micro": "f1",
    "macro_f1": "f1", "micro_f1": "f1", "macro-f1": "f1",
    # ROUGE-2
    "rouge-2": "rouge-2", "ROUGE-2": "rouge-2", "rouge2": "rouge-2", "ROUGE2": "rouge-2",
    # ROUGE-L
    "rouge-l": "rouge-l", "ROUGE-L": "rouge-l", "rougeL": "rouge-l",
    "rouge_l": "rouge-l", "ROUGE_L": "rouge-l", "rougeLsum": "rouge-l",
    # WER
    "wer": "wer", "WER": "wer", "word_error_rate": "wer",
}


# ──────────────────────────────────────────────
# Part A: README Summarization
# ──────────────────────────────────────────────

SUMMARIZE_SYSTEM = (
    "You are a strict JSON generator. Return ONLY a valid JSON object with the key:\n"
    '- "info": string (150-250 word summary)\n'
    "Use double quotes for keys/strings. No extra text."
)

MODEL_INSTRUCTION = (
    "Summarize the following model README. Focus on what the model does, its architecture, "
    "training data, and intended use cases. Keep 150-250 words.\n"
    "CRITICAL: Do NOT include any evaluation results, benchmark scores, metric values, "
    "or performance numbers. These must be completely excluded to prevent information leakage.\n"
    "Return only the JSON object."
)

DATASET_INSTRUCTION = (
    "Summarize the following dataset README. Focus on what the dataset contains, its format, "
    "size, domain, and intended use cases. Keep 150-250 words.\n"
    "CRITICAL: Do NOT include any evaluation results, benchmark scores, metric values, "
    "or performance numbers. These must be completely excluded to prevent information leakage.\n"
    "Return only the JSON object."
)

PAPER_INSTRUCTION = (
    "Summarize the following research paper (title + abstract). Focus on the problem being "
    "solved, the proposed method, key contributions, and the domain. Keep 100-200 words.\n"
    "CRITICAL: Do NOT include specific numeric results, benchmark scores, or metric values.\n"
    "Return only the JSON object."
)

CODEBASE_INSTRUCTION = (
    "Summarize the following GitHub repository README. Focus on what the project does, its "
    "main features, and the intended use cases. Keep 100-200 words.\n"
    "CRITICAL: Do NOT include benchmark scores or metric values.\n"
    "Return only the JSON object."
)


def _find_readme(readme_dir: Path, entity_id: str) -> Optional[Path]:
    """Find README file for a model/dataset ID."""
    # Try double-underscore encoding
    fname = entity_id.replace("/", "__") + ".md"
    path = readme_dir / fname
    if path.exists():
        return path
    # Try with _README suffix
    fname2 = entity_id.replace("/", "__") + "_README.md"
    path2 = readme_dir / fname2
    if path2.exists():
        return path2
    return None


def _parse_json_from_llm(raw: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, with fallback."""
    cleaned = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: try to find JSON block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(cleaned[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"info": cleaned[:500]}


def _load_paper_text(arxiv_id: str, papers_metadata_dir: Path) -> Optional[str]:
    """Load paper title + abstract from its metadata JSON."""
    path = papers_metadata_dir / (arxiv_id.replace("/", "__") + ".json")
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")
        return f"Title: {title}\n\nAbstract: {abstract}"
    except Exception:
        return None


async def _summarize_readmes(
    model_ids: Set[str],
    dataset_ids: Set[str],
    model_readme_dir: Path,
    dataset_readme_dir: Path,
    output_path: Path,
    llm_model: str = "gpt-4o",
    max_concurrency: int = 8,
    paper_ids: Optional[Set[str]] = None,
    papers_metadata_dir: Optional[Path] = None,
    codebase_ids: Optional[Set[str]] = None,
    codebases_readme_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """Summarize READMEs for all models, datasets, papers, and codebases."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for README summarization.")
    client = AsyncOpenAI(api_key=api_key)

    # Load existing summaries for resume support
    summaries: Dict[str, Dict[str, str]] = {
        "models": {}, "datasets": {}, "papers": {}, "codebases": {}
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        summaries["models"] = existing.get("models", {})
        summaries["datasets"] = existing.get("datasets", {})
        summaries["papers"] = existing.get("papers", {})
        summaries["codebases"] = existing.get("codebases", {})
        print(
            f"  Resuming: {len(summaries['models'])} models, "
            f"{len(summaries['datasets'])} datasets, "
            f"{len(summaries['papers'])} papers, "
            f"{len(summaries['codebases'])} codebases loaded"
        )

    sem = asyncio.Semaphore(max_concurrency)
    processed = [0]
    save_interval = 50

    def _save():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    async def _call_llm(text: str, instruction: str) -> str:
        async with sem:
            for attempt in range(4):
                try:
                    resp = await client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": SUMMARIZE_SYSTEM},
                            {"role": "user", "content": f"{instruction}\n\n{text[:12000]}"},
                        ],
                        temperature=0,
                        max_tokens=1024,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    if attempt == 3:
                        raise
                    delay = 0.8 * (2 ** attempt)
                    await asyncio.sleep(delay)
        return ""

    async def _summarize_one(entity_id: str, text: str, instruction: str, kind: str):
        raw = await _call_llm(text, instruction)
        return entity_id, kind, _parse_json_from_llm(raw)

    # Build tasks
    tasks = []

    for mid in sorted(model_ids):
        if mid not in summaries["models"]:
            path = _find_readme(model_readme_dir, mid)
            if path:
                text = path.read_text(encoding="utf-8", errors="ignore")
                tasks.append(_summarize_one(mid, text, MODEL_INSTRUCTION, "models"))

    for did in sorted(dataset_ids):
        if did not in summaries["datasets"]:
            path = _find_readme(dataset_readme_dir, did)
            if path:
                text = path.read_text(encoding="utf-8", errors="ignore")
                tasks.append(_summarize_one(did, text, DATASET_INSTRUCTION, "datasets"))

    if paper_ids and papers_metadata_dir:
        for pid in sorted(paper_ids):
            if pid not in summaries["papers"]:
                text = _load_paper_text(pid, papers_metadata_dir)
                if text:
                    tasks.append(_summarize_one(pid, text, PAPER_INSTRUCTION, "papers"))

    if codebase_ids and codebases_readme_dir:
        for cid in sorted(codebase_ids):
            if cid not in summaries["codebases"]:
                path = _find_readme(codebases_readme_dir, cid)
                if path:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    tasks.append(_summarize_one(cid, text, CODEBASE_INSTRUCTION, "codebases"))

    print(f"  {len(tasks)} items to summarize (skipping already done)")

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
        try:
            entity_id, kind, parsed = await coro
            summaries[kind][entity_id] = parsed
            processed[0] += 1
            if processed[0] % save_interval == 0:
                _save()
        except Exception as e:
            print(f"  Error: {e}")

    _save()
    print(
        f"  Done: {len(summaries['models'])} models, {len(summaries['datasets'])} datasets, "
        f"{len(summaries['papers'])} papers, {len(summaries['codebases'])} codebases"
    )
    return summaries


# ──────────────────────────────────────────────
# Part B: Embeddings
# ──────────────────────────────────────────────

def generate_random_embeddings(num_nodes: int, dim: int = 1024, seed: int = 42) -> np.ndarray:
    """Generate L2-normalized random embeddings."""
    rng = np.random.RandomState(seed)
    emb = rng.normal(0, 1, (num_nodes, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-8)


def generate_voyage_embeddings(
    texts: List[str],
    model_name: str = "voyage-3",
    batch_size: int = 128,
) -> np.ndarray:
    """Generate embeddings via Voyage AI API."""
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("VOYAGE_API_KEY is required.")
    client = voyageai.Client(api_key=api_key)

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Voyage embeddings"):
        batch = texts[i:i + batch_size]
        result = client.embed(batch, model=model_name, input_type="document")
        all_embs.extend(result.embeddings)
    return np.array(all_embs, dtype=np.float32)


def compute_embeddings(
    summaries: Dict[str, Dict[str, Any]],
    model_ids_ordered: List[str],
    dataset_ids_ordered: List[str],
    embedding_type: str = "random",
    embedding_dim: int = 1024,
    paper_ids_ordered: Optional[List[str]] = None,
    codebase_ids_ordered: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute embeddings for all nodes (models, datasets, papers, codebases)."""
    paper_ids_ordered = paper_ids_ordered or []
    codebase_ids_ordered = codebase_ids_ordered or []
    num_nodes = (
        len(model_ids_ordered) + len(dataset_ids_ordered)
        + len(paper_ids_ordered) + len(codebase_ids_ordered)
    )

    if embedding_type == "random":
        print(f"  Generating random embeddings: ({num_nodes}, {embedding_dim})")
        return generate_random_embeddings(num_nodes, embedding_dim)

    # Build text list: models → datasets → papers → codebases
    texts = []
    for mid in model_ids_ordered:
        info = summaries.get("models", {}).get(mid, {})
        text = info.get("info", "") if isinstance(info, dict) else str(info)
        texts.append(text[:8000] if text else mid)
    for did in dataset_ids_ordered:
        info = summaries.get("datasets", {}).get(did, {})
        text = info.get("info", "") if isinstance(info, dict) else str(info)
        texts.append(text[:8000] if text else did)
    for pid in paper_ids_ordered:
        info = summaries.get("papers", {}).get(pid, {})
        text = info.get("info", "") if isinstance(info, dict) else str(info)
        texts.append(text[:8000] if text else pid)
    for cid in codebase_ids_ordered:
        info = summaries.get("codebases", {}).get(cid, {})
        text = info.get("info", "") if isinstance(info, dict) else str(info)
        texts.append(text[:8000] if text else cid)

    if embedding_type == "voyage":
        return generate_voyage_embeddings(texts)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# ──────────────────────────────────────────────
# Part C: Metric Normalization
# ──────────────────────────────────────────────

def get_gpt_metric_mapping(metric_names: List[str], model: str = "gpt-4o") -> Dict[str, Optional[str]]:
    """Use GPT to classify unmapped metric names."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai not installed, using rule-based mapping only")
        return {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {}

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a machine learning metrics expert. Classify each metric name into one of:\n"
        "accuracy, bleu, chrf, f1, rouge-2, rouge-l, top-k_accuracy, wer, null\n"
        "(null if it doesn't fit any category)\n\n"
        f"Metrics: {json.dumps(metric_names)}\n\n"
        "Return ONLY a JSON object mapping each name to its category."
    )
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=4096,
        )
        content = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        mapping = json.loads(content)
        return {k: (v if v != "null" else None) for k, v in mapping.items()}
    except Exception as e:
        print(f"  GPT classification error: {e}")
        return {}


def normalize_value(value: float) -> float:
    """Normalize metric value to 0-1 range."""
    if value > 1.0:
        return value / 100.0
    return value


def normalize_metrics(
    pairs: List[Dict[str, Any]],
    use_gpt: bool = True,
    gpt_model: str = "gpt-4o",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[str]]]:
    """Normalize metric names and values, grouping by model-dataset edge.

    Returns:
        edges: dict keyed by "model_id|dataset_id" with {model_id, dataset_id, metrics: {std_name: value}}
        metric_mapping: full mapping from raw metric names to standard names
    """
    # Collect unique metric names
    raw_metrics = set()
    for p in pairs:
        raw_metrics.add(p["metric"])

    print(f"  {len(raw_metrics)} unique metric names found")

    # Build mapping: rule-based + optional GPT
    mapping: Dict[str, Optional[str]] = {}
    unmapped = []
    for m in raw_metrics:
        if m in SIMPLE_METRIC_MAPPING:
            mapping[m] = SIMPLE_METRIC_MAPPING[m]
        else:
            unmapped.append(m)

    print(f"  Rule-based: {len(mapping)} mapped, {len(unmapped)} remaining")

    if use_gpt and unmapped:
        print(f"  Using GPT to classify {len(unmapped)} unmapped metrics...")
        batch_size = 100
        for i in range(0, len(unmapped), batch_size):
            batch = unmapped[i:i + batch_size]
            gpt_map = get_gpt_metric_mapping(batch, gpt_model)
            mapping.update(gpt_map)

    for m in raw_metrics:
        if m not in mapping:
            mapping[m] = None

    # Group pairs into edges
    edges: Dict[str, Dict[str, Any]] = {}
    for p in pairs:
        key = f"{p['model_id']}|{p['dataset_id']}"
        std_name = mapping.get(p["metric"])
        if not std_name or std_name not in STANDARD_METRICS:
            continue

        value = normalize_value(p["result"])
        if value is None or not (0 <= value <= 1.0 or std_name == "wer"):
            continue

        if key not in edges:
            edges[key] = {
                "model_id": p["model_id"],
                "dataset_id": p["dataset_id"],
                "metrics": {},
            }
        # First occurrence wins for duplicate standard metric names
        if std_name not in edges[key]["metrics"]:
            edges[key]["metrics"][std_name] = round(value, 6)

    return edges, mapping


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: Summarize READMEs + normalize metrics."
    )
    parser.add_argument("--filtered-pairs", default="../data/artifact_raw_data/filtered_eval_pairs.json")
    parser.add_argument("--model-readme-dir", default="../data/artifact_raw_data/models/readmes")
    parser.add_argument("--dataset-readme-dir", default="../data/artifact_raw_data/datasets/readmes")
    parser.add_argument("--papers-metadata-dir", default="../data/artifact_raw_data/papers/metadata",
                        help="Directory of paper metadata JSON files (step 4 output).")
    parser.add_argument("--codebases-readme-dir", default="../data/artifact_raw_data/codebases/readmes",
                        help="Directory of codebase README files (step 4 output).")
    parser.add_argument("--resource-links", default="../data/artifact_raw_data/resource_links.json",
                        help="Artifact→resource mapping from step 4.")
    parser.add_argument("--output-dir", default="../data/artifact_raw_data")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model for summarization.")
    parser.add_argument("--embedding-type", choices=["random", "voyage"], default="random")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--no-gpt-metrics", action="store_true", help="Skip GPT for metric classification.")
    parser.add_argument("--skip-summarize", action="store_true", help="Skip README summarization.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding computation.")
    parser.add_argument("--max-concurrency", type=int, default=8, help="LLM concurrency.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    pairs_path = (script_dir / args.filtered_pairs).resolve()
    model_readme_dir = (script_dir / args.model_readme_dir).resolve()
    dataset_readme_dir = (script_dir / args.dataset_readme_dir).resolve()
    papers_metadata_dir = (script_dir / args.papers_metadata_dir).resolve()
    codebases_readme_dir = (script_dir / args.codebases_readme_dir).resolve()
    resource_links_path = (script_dir / args.resource_links).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered pairs
    print("Loading filtered eval pairs...")
    with open(pairs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data["results"]
    print(f"  {len(pairs)} pairs loaded")

    # Extract unique model/dataset IDs
    model_ids = sorted({p["model_id"] for p in pairs})
    dataset_ids = sorted({p["dataset_id"] for p in pairs})
    print(f"  {len(model_ids)} models, {len(dataset_ids)} datasets")

    # Load paper/codebase IDs from resource_links (step 4 output)
    paper_ids: Set[str] = set()
    codebase_ids: Set[str] = set()
    if resource_links_path.exists():
        rl = json.loads(resource_links_path.read_text(encoding="utf-8"))
        for links in list(rl.get("models", {}).values()) + list(rl.get("datasets", {}).values()):
            paper_ids.update(links.get("arxiv_ids", []))
            codebase_ids.update(links.get("github_repos", []))
        # Only include papers/codebases that were actually fetched
        if papers_metadata_dir.exists():
            fetched_papers = {f.stem.replace("__", "/") for f in papers_metadata_dir.glob("*.json")}
            paper_ids &= fetched_papers
        if codebases_readme_dir.exists():
            fetched_codebases = {f.stem.replace("__", "/") for f in codebases_readme_dir.glob("*.md")}
            codebase_ids &= fetched_codebases
    paper_ids_sorted = sorted(paper_ids)
    codebase_ids_sorted = sorted(codebase_ids)
    print(f"  {len(paper_ids_sorted)} papers, {len(codebase_ids_sorted)} codebases")

    # ── Part A: Summarize READMEs ──
    summaries_path = output_dir / "readme_summaries.json"
    if not args.skip_summarize:
        print("\n=== Part A: Summarizing READMEs ===")
        summaries = asyncio.run(_summarize_readmes(
            model_ids=set(model_ids),
            dataset_ids=set(dataset_ids),
            model_readme_dir=model_readme_dir,
            dataset_readme_dir=dataset_readme_dir,
            output_path=summaries_path,
            llm_model=args.llm_model,
            max_concurrency=args.max_concurrency,
            paper_ids=paper_ids,
            papers_metadata_dir=papers_metadata_dir,
            codebase_ids=codebase_ids,
            codebases_readme_dir=codebases_readme_dir,
        ))
    else:
        print("\nSkipping summarization, loading existing...")
        if summaries_path.exists():
            summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        else:
            summaries = {"models": {}, "datasets": {}, "papers": {}, "codebases": {}}

    # ── Part B: Embeddings ──
    if not args.skip_embeddings:
        print(f"\n=== Part B: Computing {args.embedding_type} embeddings ===")
        embeddings = compute_embeddings(
            summaries, model_ids, dataset_ids,
            args.embedding_type, args.embedding_dim,
            paper_ids_ordered=paper_ids_sorted,
            codebase_ids_ordered=codebase_ids_sorted,
        )
        emb_path = output_dir / f"node_embeddings_{args.embedding_type}.npy"
        np.save(emb_path, embeddings)
        print(f"  Saved embeddings to {emb_path} (shape: {embeddings.shape})")
    else:
        print("\nSkipping embedding computation.")

    # ── Part C: Normalize metrics ──
    print("\n=== Part C: Normalizing metrics ===")
    edges, metric_mapping = normalize_metrics(
        pairs, use_gpt=not args.no_gpt_metrics, gpt_model=args.llm_model,
    )

    # Save normalized pairs
    norm_path = output_dir / "normalized_eval_pairs.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump({"edges": edges, "metric_name_mapping": metric_mapping}, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(edges)} normalized edges to {norm_path}")

    # Save metric mapping separately
    mapping_path = output_dir / "metric_name_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(metric_mapping, f, indent=2, ensure_ascii=False)

    # Stats
    metric_counts = Counter()
    for e in edges.values():
        for m in e["metrics"]:
            metric_counts[m] += 1
    print("\n  Metric distribution:")
    for m, c in metric_counts.most_common():
        print(f"    {m}: {c}")

    edges_with = sum(1 for e in edges.values() if e["metrics"])
    edges_without = sum(1 for e in edges.values() if not e["metrics"])
    print(f"\n  Edges with metrics: {edges_with}")
    print(f"  Edges without metrics: {edges_without}")
    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
