import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from litellm import completion
from tqdm import tqdm


def ensure_directory(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


class MetricCollector:
    """
    Collects and manages evaluation metrics extracted from model README files.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def _ask_gpt(self, text: str) -> Dict[str, Any]:
        """
        Extracts dataset metrics from README text using an LLM.
        """
        system_prompt = (
            "You are a helpful assistant that extracts evaluation dataset names and corresponding "
            "metrics from README content. Return a JSON dict with dataset names as keys and metric-value maps as values."
        )
        try:
            resp = completion(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:12000]},
                ],
                temperature=0,
            )
            content = resp["choices"][0]["message"]["content"]
            # Remove code fences if present
            cleaned = content.strip().strip("```json").strip("```")
            return json.loads(cleaned)
        except Exception:
            return {}

    def process_readme(
        self, readme_path: Path, output_dir: Path, pbar: Optional[tqdm] = None
    ) -> str:
        """
        Parses a single README.md file and saves extracted metrics to JSON.
        """
        model_id = readme_path.stem
        output_path = output_dir / f"{model_id}.json"
        if output_path.exists():
            msg = f"Skipped {model_id}"
        else:
            try:
                text = readme_path.read_text(encoding="utf-8")
                metrics = self._ask_gpt(text)
                ensure_directory(output_dir)
                output_path.write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                msg = f"Processed {model_id}"
            except Exception:
                output_path.write_text("{}", encoding="utf-8")
                msg = f"Error processing {model_id}"
        if pbar:
            with self._lock:
                pbar.update(1)
                pbar.set_description(msg)
        return msg

    def batch_process(self, readme_dir: Path, output_dir: Path, max_workers: int = 5) -> None:
        """
        Processes all README.md files in a directory in parallel.
        """
        readme_dir = Path(readme_dir)
        output_dir = Path(output_dir)
        files = list(readme_dir.glob("*.md"))
        print(f"Found {len(files)} README files. Output directory: {output_dir}")
        ensure_directory(output_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(files), desc="Processing") as pbar:
                futures = [
                    executor.submit(self.process_readme, path, output_dir, pbar) for path in files
                ]
                for future in as_completed(futures):
                    future.result()

    def collect_one(self, readme_path: Path) -> Dict[str, Any]:
        """
        Process a single README file and return metrics without saving.

        Args:
            readme_path: Path to the README.md file to process.

        Returns:
            Dict containing metrics and readme path.
        """
        try:
            text = readme_path.read_text(encoding="utf-8")
            metrics = self._ask_gpt(text)
            return {"metrics": metrics, "readme_path": str(readme_path)}
        except Exception as e:
            return {"metrics": {}, "readme_path": str(readme_path), "error": str(e)}

    def save_metrics_file(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        metrics_dir: str = "metrics",
    ) -> Path:
        """Save metrics dict to a JSON file."""
        out_dir = Path(metrics_dir)
        ensure_directory(out_dir)
        fname = model_id.replace("/", "__") + ".json"
        path = out_dir / fname
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def load_metrics(model_id: str, metrics_dir: str = "metrics") -> Optional[Dict[str, Any]]:
        """Load saved metrics JSON for a given model ID."""
        fname = model_id.replace("/", "__") + ".json"
        path = Path(metrics_dir) / fname
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_all_metrics(metrics_dir: str = "metrics") -> Dict[str, Dict[str, Any]]:
        """Load all metrics files from directory."""
        results = {}
        metrics_path = Path(metrics_dir)
        if not metrics_path.exists():
            return results
        for file in metrics_path.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                model_id = file.stem.replace("__", "/")
                results[model_id] = data
            except Exception:
                continue
        return results
