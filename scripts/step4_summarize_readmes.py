import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

JSONDict = Dict[str, object]


def _load_llm_helpers(
    repo_root: Path,
) -> Tuple[Optional[Callable[..., tuple]], Optional[Callable[..., tuple]]]:
    llm_py = repo_root / "artifact_graph" / "utils" / "llm.py"
    if not llm_py.exists():
        return None, None
    spec = importlib.util.spec_from_file_location("ag_llm", str(llm_py))
    if spec is None or spec.loader is None:
        return None, None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ag_llm"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "create_client", None), getattr(mod, "get_response_from_llm", None)


async def _fallback_openai_client(model: str):
    # Async client (OpenAI Python v1)
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'openai' >= 1.0 to use AsyncOpenAI.") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to use OpenAI fallback.")
    return AsyncOpenAI(api_key=api_key), model


@dataclass
class LLMCaller:
    acall: Callable[[str], Awaitable[str]]  # async call(text) -> str


def build_llm_caller(model: str, base_system: str) -> LLMCaller:
    """
    Returns an async LLM caller:
    - If repo helpers exist (likely sync), wrap in asyncio.to_thread for parallelism.
    - Else use AsyncOpenAI chat.completions.create() natively.
    """
    repo_root = Path(__file__).resolve().parents[1]
    create_client, get_response_from_llm = _load_llm_helpers(repo_root)

    if create_client and get_response_from_llm:
        # Likely synchronous helpers; we wrap them for async concurrency.
        client, resolved_model = create_client(model)

        async def _acall(text: str) -> str:
            def _run_sync() -> str:
                out, _ = get_response_from_llm(text, client, resolved_model, base_system)
                return out

            return await asyncio.to_thread(_run_sync)

        return LLMCaller(acall=_acall)

    # Async OpenAI fallback
    # (We create the client lazily inside the closure after awaiting in an inner helper)
    _client_holder: Dict[str, object] = {"client": None, "model": model}

    async def _ensure_client():
        if _client_holder["client"] is None:
            client, resolved_model = await _fallback_openai_client(model)
            _client_holder["client"] = client
            _client_holder["model"] = resolved_model

    async def _acall(text: str) -> str:
        await _ensure_client()
        client = _client_holder["client"]
        resolved_model = _client_holder["model"]
        # Try structured JSON first; if not supported, fall back quietly.
        try:
            resp = await client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=2048,
                seed=0,
            )
        except TypeError:
            resp = await client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=2048,
            )
        return resp.choices[0].message.content or ""

    return LLMCaller(acall=_acall)


def _owner_repo_to_filename(oid: str) -> str:
    safe = oid.replace("/", "__")
    return f"{safe}_README.md"


def _find_readme(readme_dir: Path, oid: str) -> Optional[Path]:
    exact = readme_dir / _owner_repo_to_filename(oid)
    if exact.exists():
        return exact
    prefix = oid.replace("/", "__")
    candidates = list(readme_dir.glob(f"{prefix}*.md"))
    if candidates:
        candidates.sort(key=lambda p: ("_README" not in p.name, len(p.name)))
        return candidates[0]
    return None


def _load_ids(metrics_json: Path) -> tuple[Set[str], Set[str]]:
    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    model_ids, dataset_ids = set(), set()
    for r in results:
        mid, did = r.get("model_id"), r.get("dataset_id")
        if isinstance(mid, str) and mid:
            model_ids.add(mid)
        if isinstance(did, str) and did:
            dataset_ids.add(did)
    return model_ids, dataset_ids


BASE_SYSTEM = (
    "You are a strict JSON generator. Return ONLY a valid JSON object with keys:\n"
    '- "model_info": string,\n'
    '- "evaluation_results": object,\n'
    '- "code_example": string.\n'
    "Use double quotes for keys/strings. No extra text."
)
MODEL_USER_INSTR = "Summarize the following model README. Keep ~150-250 words for model_info; include evaluation_results and code_example if present. Return only the JSON object."
DATASET_USER_INSTR = "Summarize the following dataset README. Keep ~150-250 words for model_info; include evaluation_results and code_example if present. Return only the JSON object."


def _assemble_user_prompt(
    instruction: str, readme_text: str, extra_prompt: str, max_chars: int = 12000
) -> str:
    return f"{instruction}{extra_prompt}\n\n{readme_text[:max_chars]}"


def _parse_json_from_llm(raw: str) -> JSONDict:
    cleaned = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"summary_raw": cleaned}


async def _with_retries(
    coro_factory: Callable[[], Awaitable[str]], *, retries: int = 4, base_delay: float = 0.8
) -> str:
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            # simple exponential backoff with jitter
            delay = base_delay * (2**attempt)
            delay = delay * (0.8 + 0.4 * os.urandom(1)[0] / 255.0)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore


async def summarize_ids_to_json_async(
    metrics_json: Path,
    model_readme_dir: Path,
    dataset_readme_dir: Path,
    output_json: Path,
    model: str,
    prompt_file: Optional[Path] = None,
    existing_json: Optional[Path] = None,
) -> None:
    extra_prompt = ""
    if prompt_file and prompt_file.exists():
        extra_prompt = "\n\nAdditional instructions:\n" + prompt_file.read_text(encoding="utf-8")

    llm = build_llm_caller(model, BASE_SYSTEM)
    model_ids, dataset_ids = _load_ids(metrics_json)

    # Load existing summaries to skip already processed items
    out: Dict[str, Dict[str, JSONDict]] = {"models": {}, "datasets": {}}
    if existing_json and existing_json.exists():
        existing = json.loads(existing_json.read_text(encoding="utf-8"))
        out["models"] = existing.get("models", {})
        out["datasets"] = existing.get("datasets", {})
        print(f"[info] Loaded {len(out['models'])} existing model summaries, {len(out['datasets'])} dataset summaries")

    # Concurrency control
    max_concurrency = int(os.environ.get("LLM_CONCURRENCY", "8"))
    sem = asyncio.Semaphore(max_concurrency)
    
    # Counter for incremental save
    processed_count = [0]
    save_interval = 50

    async def _summarize_one(
        kind: str, oid: str, path: Path, instruction: str
    ) -> Tuple[str, JSONDict]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        user_prompt = _assemble_user_prompt(instruction, text, extra_prompt)

        async def _do():
            async with sem:
                return await llm.acall(user_prompt)

        resp = await _with_retries(_do)
        return oid, _parse_json_from_llm(resp)

    # Build tasks - skip already processed items
    model_tasks: List[Awaitable[Tuple[str, JSONDict]]] = []
    skipped_models = 0
    for mid in sorted(model_ids):
        if mid in out["models"]:
            skipped_models += 1
            continue
        path = _find_readme(model_readme_dir, mid)
        if not path:
            continue
        model_tasks.append(_summarize_one("model", mid, path, MODEL_USER_INSTR))
    
    dataset_tasks: List[Awaitable[Tuple[str, JSONDict]]] = []
    skipped_datasets = 0
    for did in sorted(dataset_ids):
        if did in out["datasets"]:
            skipped_datasets += 1
            continue
        path = _find_readme(dataset_readme_dir, did)
        if not path:
            continue
        dataset_tasks.append(_summarize_one("dataset", did, path, DATASET_USER_INSTR))

    print(f"[info] Skipped {skipped_models} models (already in summaries), {len(model_tasks)} to process")
    print(f"[info] Skipped {skipped_datasets} datasets (already in summaries), {len(dataset_tasks)} to process")

    # Helper to save incrementally
    def _save_output():
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Execute with progress bars
    for f in tqdm(asyncio.as_completed(model_tasks), total=len(model_tasks), desc="Models"):
        try:
            mid, payload = await f
            out["models"][mid] = payload
            processed_count[0] += 1
            if processed_count[0] % save_interval == 0:
                _save_output()
                print(f"\n[save] Saved progress ({processed_count[0]} processed)")
        except Exception as e:
            print(f"Error summarizing model: {e}")

    for f in tqdm(asyncio.as_completed(dataset_tasks), total=len(dataset_tasks), desc="Datasets"):
        try:
            did, payload = await f
            out["datasets"][did] = payload
            processed_count[0] += 1
            if processed_count[0] % save_interval == 0:
                _save_output()
                print(f"\n[save] Saved progress ({processed_count[0]} processed)")
        except Exception as e:
            print(f"Error summarizing dataset: {e}")

    _save_output()
    print(f"[done] Final save complete. Total: {len(out['models'])} models, {len(out['datasets'])} datasets")


def summarize_ids_to_json(
    metrics_json: Path,
    model_readme_dir: Path,
    dataset_readme_dir: Path,
    output_json: Path,
    model: str,
    prompt_file: Optional[Path] = None,
    existing_json: Optional[Path] = None,
) -> None:
    # Keep a sync façade for existing callers
    asyncio.run(
        summarize_ids_to_json_async(
            metrics_json,
            model_readme_dir,
            dataset_readme_dir,
            output_json,
            model,
            prompt_file,
            existing_json,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize model/dataset READMEs based on IDs in metrics JSON"
    )
    ap.add_argument(
        "--metrics-json", type=Path, default=Path("output/artifact_graph_raw_data/perfect_model_dataset_metrics_v2_1125.json")
    )
    ap.add_argument("--model-readme-dir", type=Path, default=Path("output/models/readmes"))
    ap.add_argument("--dataset-readme-dir", type=Path, default=Path("output/datasets/readmes"))
    ap.add_argument("-o", "--output-json", type=Path, default=Path("output/artifact_graph_raw_data/readme_summaries_v2_1125.json"))
    ap.add_argument("-m", "--model", type=str, default="gpt-4o")
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--existing-json", type=Path, default=None,
                    help="Path to existing summaries JSON to resume from (skip already processed items)")
    args = ap.parse_args()
    
    # Auto-use output as existing if it exists and --existing-json not specified
    existing = args.existing_json
    if existing is None and args.output_json.exists():
        existing = args.output_json
        print(f"[info] Auto-resuming from existing output: {existing}")
    
    summarize_ids_to_json(
        args.metrics_json,
        args.model_readme_dir,
        args.dataset_readme_dir,
        args.output_json,
        args.model,
        args.prompt_file,
        existing,
    )


if __name__ == "__main__":
    main()
