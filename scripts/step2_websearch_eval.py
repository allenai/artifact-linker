#!/usr/bin/env python3
"""
Web search agent script to find evaluation datasets, metrics, results, and sources
for each model listed in /data/artifact_raw_data/models/metadata/.

Uses the OpenAI Agents SDK with web search to concurrently query each model
and extract structured evaluation information, requiring HuggingFace dataset links.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, WebSearchTool
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled
from pydantic import BaseModel

# Disable SDK tracing to suppress non-fatal SSL telemetry noise
set_tracing_disabled(True)


# ──────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────

class EvalEntry(BaseModel):
    """A single evaluation result for one HuggingFace dataset."""
    dataset_name: str         # Human-readable dataset name
    dataset_hf_url: str       # HuggingFace URL: https://huggingface.co/datasets/<owner>/<name>
    metric: str               # e.g. accuracy, F1, BLEU, WER, ROUGE-L ...
    result: str               # Numeric value or short description
    source: str               # URL where this result was reported


class ModelEvalResult(BaseModel):
    """Aggregated evaluation results for one model."""
    model_id: str
    evaluations: List[EvalEntry]
    search_summary: str


# ──────────────────────────────────────────────
# Agent definition
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant specializing in finding machine learning model evaluation results.

Given a Hugging Face model ID, you will:
1. Search the web to find what benchmark datasets this model has been evaluated on.
2. For every dataset found, you MUST identify its HuggingFace Datasets page URL in the format:
   https://huggingface.co/datasets/<owner>/<dataset_name>
   If the exact HuggingFace dataset page cannot be confirmed, skip that dataset.
3. For each confirmed HuggingFace dataset, extract:
   - dataset_name: the HuggingFace dataset ID in "<owner>/<name>" format (e.g. "rajpurkar/squad", "allenai/gpqa", "facebook/natural_questions"). Never use a short human-readable name like "SQuAD" or "GPQA".
   - dataset_hf_url: the full HuggingFace datasets URL (e.g. https://huggingface.co/datasets/rajpurkar/squad)
   - metric: evaluation metric used (e.g. accuracy, F1, BLEU, WER, ROUGE-L, exact_match)
   - result: the numeric score or value (e.g. "92.1", "56.3%")
   - source: URL of the paper, model card, or leaderboard where the result was reported

CRITICAL RULES — strictly follow these:
- Only include results from DIRECT evaluation of THIS exact model on the dataset as-is.
- EXCLUDE any results where the model was fine-tuned, instruction-tuned, adapted, or further trained on the target dataset before evaluation. Those are fine-tuning results, not raw evaluation results.
- EXCLUDE results from downstream task-specific variants of the model (e.g. model-finetuned-on-squad). Only report results for the base/pretrained model itself.
- EXCLUDE self-reported or estimated results unless backed by a citable paper or leaderboard.
- If you are unsure whether a result is from raw evaluation or fine-tuning, skip it.

Focus on authoritative sources: HuggingFace model cards, arXiv papers, Papers With Code leaderboards.

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{
  "evaluations": [
    {
      "dataset_name": "<owner>/<name>  (HuggingFace dataset ID, e.g. allenai/gpqa)",
      "dataset_hf_url": "https://huggingface.co/datasets/<owner>/<name>",
      "metric": "<metric name>",
      "result": "<score>",
      "source": "<URL>"
    }
  ],
  "search_summary": "<one-sentence summary of findings>"
}

If no evaluation results with confirmed HuggingFace dataset URLs are found, return:
{
  "evaluations": [],
  "search_summary": "No HuggingFace dataset evaluation results found for this model."
}
"""


def build_agent() -> Agent:
    """Create the web search agent."""
    return Agent(
        name="EvalSearchAgent",
        instructions=SYSTEM_PROMPT,
        tools=[WebSearchTool()],
        model="gpt-5.2",
        model_settings=ModelSettings(temperature=0.1),
    )


# ──────────────────────────────────────────────
# Model ID loading
# ──────────────────────────────────────────────

def load_model_ids(metadata_dir: str) -> List[str]:
    """Load model IDs from the metadata directory, sorted by downloads descending.

    File names use double-underscore to encode the HF owner/repo separator,
    e.g. 'openai__whisper-large-v3.json' -> 'openai/whisper-large-v3'.
    Models with missing/null download counts are placed at the end.
    """
    meta_path = Path(metadata_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    entries = []
    for json_file in meta_path.glob("*.json"):
        model_id = json_file.stem.replace("__", "/", 1)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            downloads = meta.get("downloads") or 0
        except Exception:
            downloads = 0
        entries.append((model_id, downloads))

    # Sort by downloads descending; highest-traffic models first
    entries.sort(key=lambda x: x[1], reverse=True)

    print(f"Loaded {len(entries)} models. "
          f"Top: {entries[0][0]} ({entries[0][1]:,} downloads), "
          f"Bottom: {entries[-1][0]} ({entries[-1][1]:,} downloads)")

    return [model_id for model_id, _ in entries]


# ──────────────────────────────────────────────
# Result helpers
# ──────────────────────────────────────────────

def load_existing_results(output_file: str) -> Dict[str, Any]:
    """Load previously saved results so we can resume interrupted runs."""
    if Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"results": [], "failed": []}


def save_results(output_file: str, data: Dict[str, Any]) -> None:
    """Persist current results to disk (atomic-ish via temp file)."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(out_path)


def parse_agent_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from the agent's text response."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back: find the outermost { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


# ──────────────────────────────────────────────
# Core processing (concurrent)
# ──────────────────────────────────────────────

async def search_model_evals(
    agent: Agent,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Run the agent to find HuggingFace dataset evaluation results for one model."""
    query = (
        f"Find all benchmark evaluation results for the Hugging Face model '{model_id}'. "
        f"For each dataset, find its HuggingFace Datasets page (huggingface.co/datasets/...). "
        f"Report metrics, scores, and source URLs. "
        f"Check the model card at https://huggingface.co/{model_id} and any associated papers."
    )

    async with semaphore:
        try:
            result = await Runner.run(agent, query)
            raw_text = result.final_output or ""
            parsed = parse_agent_output(raw_text)

            if parsed:
                return {
                    "model_id": model_id,
                    "evaluations": parsed.get("evaluations", []),
                    "search_summary": parsed.get("search_summary", ""),
                    "raw_output": raw_text,
                    "status": "success",
                }
            else:
                return {
                    "model_id": model_id,
                    "evaluations": [],
                    "search_summary": "",
                    "raw_output": raw_text,
                    "status": "parse_error",
                }
        except Exception as e:
            return {
                "model_id": model_id,
                "evaluations": [],
                "search_summary": "",
                "raw_output": str(e),
                "status": "error",
            }


async def process_models(
    model_ids: List[str],
    output_file: str,
    concurrency: int = 5,
    save_every: int = 10,
    verbose: bool = True,
) -> None:
    """Process all models concurrently and collect evaluation data."""
    agent = build_agent()
    data = load_existing_results(output_file)

    # Resume support: skip already-processed model IDs
    processed = {r["model_id"] for r in data["results"]}
    failed_ids = set(data.get("failed", []))
    pending = [mid for mid in model_ids if mid not in processed]

    total = len(pending)
    done_count = len(processed)
    print(f"Total models to process: {total}  (already done: {done_count})")
    print(f"Concurrency: {concurrency} parallel workers")

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()           # Protect shared state writes
    completed = 0

    async def run_and_collect(model_id: str) -> None:
        nonlocal completed
        entry = await search_model_evals(agent, model_id, semaphore)

        async with lock:
            completed += 1
            n_evals = len(entry.get("evaluations", []))
            status_icon = "✅" if entry["status"] == "success" else "⚠️"

            if verbose:
                print(
                    f"[{completed}/{total}] {status_icon} {model_id}  "
                    f"({n_evals} eval(s), {entry['status']})",
                    flush=True,
                )

            data["results"].append(entry)
            if entry["status"] != "success" or n_evals == 0:
                failed_ids.add(model_id)

            # Periodic checkpoint save
            if completed % save_every == 0:
                data["failed"] = list(failed_ids)
                save_results(output_file, data)
                print(f"  💾 Checkpoint saved  [{completed}/{total}]")

    # Launch all tasks; semaphore limits actual concurrency
    tasks = [asyncio.create_task(run_and_collect(mid)) for mid in pending]
    await asyncio.gather(*tasks)

    # Final save
    data["failed"] = list(failed_ids)
    save_results(output_file, data)

    success = sum(1 for r in data["results"] if r["status"] == "success")
    total_evals = sum(len(r.get("evaluations", [])) for r in data["results"])
    print(
        f"\n✅  Done: {success}/{len(data['results'])} models OK | "
        f"{total_evals} evaluation entries | "
        f"{len(failed_ids)} with no results"
    )


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use OpenAI Agents SDK to web-search HF dataset evaluation results for models."
    )
    parser.add_argument(
        "--metadata-dir",
        default="../data/artifact_raw_data/models/metadata",
        help="Directory containing model metadata JSON files.",
    )
    parser.add_argument(
        "--output-file",
        default="../data/artifact_raw_data/model_eval_websearch.json",
        help="Path to write results (supports resumable runs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N models (useful for testing).",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=None,
        help="Specific model IDs to search (overrides --metadata-dir scan).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Number of models to search in parallel (default: 5).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N completed models (default: 10).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-model progress output.",
    )
    args = parser.parse_args()

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    # Resolve model list
    if args.model_ids:
        model_ids = args.model_ids
    else:
        script_dir = Path(__file__).parent
        metadata_dir = (script_dir / args.metadata_dir).resolve()
        print(f"Loading model IDs from: {metadata_dir}")
        model_ids = load_model_ids(str(metadata_dir))

    if args.limit:
        model_ids = model_ids[: args.limit]

    print(f"Models to search: {len(model_ids)}")

    asyncio.run(
        process_models(
            model_ids=model_ids,
            output_file=args.output_file,
            concurrency=args.concurrency,
            save_every=args.save_every,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
