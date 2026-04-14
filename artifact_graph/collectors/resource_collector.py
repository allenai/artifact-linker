import base64
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


ARXIV_URL_RE = re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', re.I)
ARXIV_BARE_RE = re.compile(r'\barXiv[: ]+(\d{4}\.\d{4,5}(?:v\d+)?)\b', re.I)
ARXIV_HF_RE = re.compile(r'huggingface\.co/papers/(\d{4}\.\d{4,5}(?:v\d+)?)', re.I)
# Bare numbers like "2301.12345" — only valid new-style IDs (YY in 13..26, MM in 01..12)
ARXIV_BARE_NUM_RE = re.compile(r'(?<![/\d])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)', re.I)
GITHUB_REPO_RE = re.compile(
    r'github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+?)(?:[/\s"\')\]#]|\.git\b|$)', re.I
)

# Generic infrastructure repos unlikely to be a paper's codebase
_NOISE_REPOS = {
    "huggingface/datasets",
    "huggingface/huggingface_hub",
    "huggingface/transformers",
    "huggingface/accelerate",
    "huggingface/peft",
    "huggingface/trl",
    "huggingface/evaluate",
    "openai/openai-python",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
}

ARXIV_API = "http://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_name(resource_id: str) -> str:
    """Convert an ID with slashes to a safe filename stem."""
    return resource_id.replace("/", "__")


class ResourceCollector:
    """
    Extracts arxiv paper IDs and GitHub repo names from model/dataset READMEs,
    then fetches their metadata and content from the respective APIs.
    """

    def __init__(self, github_token: Optional[str] = None) -> None:
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

    # ── Link extraction ────────────────────────────────────────────────────

    def extract_links(self, text: str) -> Tuple[Set[str], Set[str]]:
        """Return (arxiv_ids, github_repos) found in README text."""
        arxiv_ids: Set[str] = set()
        for pat in (ARXIV_URL_RE, ARXIV_BARE_RE, ARXIV_HF_RE):
            for m in pat.finditer(text):
                # Strip version suffix and .pdf for a canonical ID
                aid = re.sub(r'\.pdf$', '', re.sub(r'v\d+$', '', m.group(1)))
                arxiv_ids.add(aid)

        # Bare numbers like "2301.12345" — validate as new-style arxiv ID
        # YY >= 07 (new format started Apr 2007), MM in 01-12, seq > 0
        for m in ARXIV_BARE_NUM_RE.finditer(text):
            aid = m.group(1)
            yymm, seq = aid.split(".")
            yy, mm = int(yymm[:2]), int(yymm[2:])
            if 7 <= yy <= 26 and 1 <= mm <= 12 and int(seq) > 0:
                arxiv_ids.add(aid)

        github_repos: Set[str] = set()
        for m in GITHUB_REPO_RE.finditer(text):
            repo = re.sub(r'\.git$', '', m.group(1).rstrip('.'))
            if repo not in _NOISE_REPOS and '.' not in repo.split('/')[-1]:
                github_repos.add(repo)

        return arxiv_ids, github_repos

    def extract_links_from_dir(
        self, readme_dir: Path
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan every *.md file in readme_dir and return per-artifact link maps.
        Returns: {artifact_id: {"arxiv_ids": [...], "github_repos": [...]}}
        Only artifacts with at least one link are included.
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        for readme_file in sorted(readme_dir.glob("*.md")):
            artifact_id = readme_file.stem.replace("__", "/")
            try:
                text = readme_file.read_text(errors="ignore")
            except Exception:
                continue
            arxiv_ids, github_repos = self.extract_links(text)
            if arxiv_ids or github_repos:
                result[artifact_id] = {
                    "arxiv_ids": sorted(arxiv_ids),
                    "github_repos": sorted(github_repos),
                }
        return result

    def extract_links_from_metadata_dir(
        self, metadata_dir: Path
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract arxiv IDs from HuggingFace metadata JSON files (tags field).
        Tags like "arxiv:2301.12345" are parsed.
        Returns: {artifact_id: {"arxiv_ids": [...], "github_repos": []}}
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        if not metadata_dir.exists():
            return result
        for meta_file in sorted(metadata_dir.glob("*.json")):
            artifact_id = meta_file.stem.replace("__", "/")
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            tags = data.get("tags", [])
            if not isinstance(tags, list):
                continue
            arxiv_ids: Set[str] = set()
            for tag in tags:
                m = re.match(r"arxiv:(\d{4}\.\d{4,5}(?:v\d+)?)", str(tag))
                if m:
                    aid = re.sub(r"v\d+$", "", m.group(1))
                    arxiv_ids.add(aid)
            if arxiv_ids:
                result[artifact_id] = {
                    "arxiv_ids": sorted(arxiv_ids),
                    "github_repos": [],
                }
        return result

    # ── Arxiv ──────────────────────────────────────────────────────────────

    def fetch_arxiv_metadata(
        self, arxiv_id: str, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        url = f"{ARXIV_API}?id_list={arxiv_id}&max_results=1"
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    return self._parse_arxiv_xml(resp.read().decode("utf-8"), arxiv_id)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
        return None

    def fetch_arxiv_metadata_batch(
        self, arxiv_ids: List[str], batch_size: int = 50, max_retries: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch metadata for multiple arxiv IDs in batches. Returns {arxiv_id: metadata}."""
        results: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            id_list = ",".join(batch)
            url = f"{ARXIV_API}?id_list={id_list}&max_results={len(batch)}"
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(url, timeout=30) as resp:
                        xml_text = resp.read().decode("utf-8")
                    parsed = self._parse_arxiv_xml_multi(xml_text, batch)
                    results.update(parsed)
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(2.0 * (attempt + 1))
            # Rate limit: arXiv asks for 3s between requests
            time.sleep(3.0)
        return results

    def _parse_arxiv_xml_multi(
        self, xml_text: str, requested_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse arXiv API response with multiple entries."""
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        results: Dict[str, Dict[str, Any]] = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return results
        for entry in root.findall("atom:entry", ns):
            # Extract arxiv_id from the entry's <id> tag
            entry_id_text = (entry.findtext("atom:id", "", ns) or "").strip()
            # e.g. "http://arxiv.org/abs/2301.12345v1"
            aid = entry_id_text.rsplit("/", 1)[-1] if "/" in entry_id_text else ""
            aid = re.sub(r"v\d+$", "", aid)
            if not aid:
                continue
            title = (entry.findtext("atom:title", "", ns) or "").strip()
            # Skip entries that are "Error" responses from arXiv
            if not title or title.lower() == "error":
                continue
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
            published = (entry.findtext("atom:published", "", ns) or "")[:10]
            updated = (entry.findtext("atom:updated", "", ns) or "")[:10]
            authors = [
                a.findtext("atom:name", "", ns)
                for a in entry.findall("atom:author", ns)
            ]
            cats: List[str] = []
            for c in entry.findall("arxiv:primary_category", ns):
                cats.append(c.get("term", ""))
            for c in entry.findall("atom:category", ns):
                cats.append(c.get("term", ""))
            categories = list(dict.fromkeys(filter(None, cats)))
            pdf_url = next(
                (lnk.get("href", "") for lnk in entry.findall("atom:link", ns)
                 if lnk.get("type") == "application/pdf"),
                "",
            )
            results[aid] = {
                "arxiv_id": aid,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": published,
                "updated": updated,
                "categories": categories,
                "pdf_url": pdf_url,
            }
        return results

    def _parse_arxiv_xml(self, xml_text: str, arxiv_id: str) -> Optional[Dict[str, Any]]:
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title = (entry.findtext("atom:title", "", ns) or "").strip()
        abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
        published = (entry.findtext("atom:published", "", ns) or "")[:10]
        updated = (entry.findtext("atom:updated", "", ns) or "")[:10]
        authors = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ]
        # Collect categories (primary first, then secondary)
        cats: List[str] = []
        for c in entry.findall("arxiv:primary_category", ns):
            cats.append(c.get("term", ""))
        for c in entry.findall("atom:category", ns):
            cats.append(c.get("term", ""))
        categories = list(dict.fromkeys(filter(None, cats)))
        pdf_url = next(
            (lnk.get("href", "") for lnk in entry.findall("atom:link", ns)
             if lnk.get("type") == "application/pdf"),
            "",
        )
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published,
            "updated": updated,
            "categories": categories,
            "pdf_url": pdf_url,
        }

    # ── GitHub ─────────────────────────────────────────────────────────────

    def _github_get(self, path: str) -> Optional[Any]:
        req = urllib.request.Request(f"{GITHUB_API}{path}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.github_token:
            req.add_header("Authorization", f"Bearer {self.github_token}")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def fetch_github_metadata(
        self, repo: str, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        for attempt in range(max_retries):
            data = self._github_get(f"/repos/{repo}")
            if data and "full_name" in data:
                return {
                    "full_name": data["full_name"],
                    "description": data.get("description") or "",
                    "stars": data.get("stargazers_count", 0),
                    "forks": data.get("forks_count", 0),
                    "language": data.get("language") or "",
                    "topics": data.get("topics", []),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "license": (data.get("license") or {}).get("spdx_id") or "",
                }
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
        return None

    def fetch_github_readme(
        self, repo: str, max_retries: int = 3
    ) -> Optional[str]:
        for attempt in range(max_retries):
            data = self._github_get(f"/repos/{repo}/readme")
            if data and "content" in data:
                try:
                    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception:
                    pass
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
        return None

    # ── Save helpers ───────────────────────────────────────────────────────

    def save_paper(self, metadata: Dict[str, Any], papers_dir: Path) -> None:
        _ensure_dir(papers_dir)
        path = papers_dir / f"{_safe_name(metadata['arxiv_id'])}.json"
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_codebase(
        self,
        repo: str,
        metadata: Optional[Dict[str, Any]],
        readme: Optional[str],
        metadata_dir: Path,
        readme_dir: Path,
    ) -> None:
        if metadata:
            _ensure_dir(metadata_dir)
            (metadata_dir / f"{_safe_name(repo)}.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if readme:
            _ensure_dir(readme_dir)
            (readme_dir / f"{_safe_name(repo)}.md").write_text(readme, encoding="utf-8")

    # ── Batch collection ───────────────────────────────────────────────────

    def _fetch_paper_item(
        self, arxiv_id: str, papers_dir: Path
    ) -> Dict[str, str]:
        if (papers_dir / f"{_safe_name(arxiv_id)}.json").exists():
            return {"arxiv_id": arxiv_id, "status": "skipped"}
        meta = self.fetch_arxiv_metadata(arxiv_id)
        if meta:
            self.save_paper(meta, papers_dir)
            return {"arxiv_id": arxiv_id, "status": "success"}
        return {"arxiv_id": arxiv_id, "status": "error"}

    def _fetch_codebase_item(
        self, repo: str, metadata_dir: Path, readme_dir: Path
    ) -> Dict[str, str]:
        if (metadata_dir / f"{_safe_name(repo)}.json").exists():
            return {"repo": repo, "status": "skipped"}
        meta = self.fetch_github_metadata(repo)
        readme = self.fetch_github_readme(repo)
        if meta or readme:
            self.save_codebase(repo, meta, readme, metadata_dir, readme_dir)
            return {"repo": repo, "status": "success"}
        return {"repo": repo, "status": "error"}

    def collect_all_papers(
        self,
        arxiv_ids: List[str],
        papers_dir: Path,
        max_concurrent: int = 5,
    ) -> List[Dict[str, str]]:
        _ensure_dir(papers_dir)
        fn = partial(self._fetch_paper_item, papers_dir=papers_dir)
        results: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            with tqdm(total=len(arxiv_ids), desc="Fetching papers") as pbar:
                for r in executor.map(fn, arxiv_ids):
                    results.append(r)
                    pbar.update(1)
        return results

    def collect_all_codebases(
        self,
        repos: List[str],
        metadata_dir: Path,
        readme_dir: Path,
        max_concurrent: int = 5,
    ) -> List[Dict[str, str]]:
        _ensure_dir(metadata_dir)
        _ensure_dir(readme_dir)
        fn = partial(self._fetch_codebase_item, metadata_dir=metadata_dir, readme_dir=readme_dir)
        results: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            with tqdm(total=len(repos), desc="Fetching codebases") as pbar:
                for r in executor.map(fn, repos):
                    results.append(r)
                    pbar.update(1)
        return results
