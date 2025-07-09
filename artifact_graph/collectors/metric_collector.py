import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from litellm import completion

def ensure_directory(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

class MetricCollector:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._lock = threading.Lock()

    def _ask_gpt(self, text: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a helpful assistant that reads model evaluation results from a README. "
            "Return ONLY a JSON object mapping metric names to their numeric values, "
            "for example {\"F1\": 0.9}. No additional text or nesting."
        )
        try:
            resp = completion(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:12000]}
                ],
                temperature=0
            )
            content = resp["choices"][0]["message"]["content"]
            cleaned = content.strip().strip("```json").strip("```")
            return json.loads(cleaned)
        except Exception:
            return {}

    def process_readme(self, readme_path: Path) -> Dict[str, Any]:
        """
        Reads a README.md file and returns the extracted metrics as JSON.
        """
        text = readme_path.read_text(encoding="utf-8")
        return self._ask_gpt(text)

    def batch_process(self, readme_dir: Path, output_dir: Path, max_workers: int = 5) -> None:
        """Process all README.md files in a directory in parallel and save per-model JSON."""
        files = list(Path(readme_dir).glob("*.md"))
        print(f"Found {len(files)} README files. Writing to {output_dir}")
        ensure_directory(Path(output_dir))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(files), desc="Processing") as pbar:
                futures = [
                    executor.submit(self._process_and_save, p, Path(output_dir), pbar)
                    for p in files
                ]
                for f in as_completed(futures):
                    f.result()

    def _process_and_save(self, readme_path: Path, output_dir: Path, pbar: tqdm) -> None:
        model_id = readme_path.stem
        output_path = output_dir / f"{model_id}.json"
        try:
            metrics = self.process_readme(readme_path)
            output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            msg = f"Processed {model_id}"
        except Exception:
            output_path.write_text("{}", encoding="utf-8")
            msg = f"Error {model_id}"
        with self._lock:
            pbar.update(1)
            pbar.set_description(msg)