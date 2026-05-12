"""Unified reranker client.

Supports two local backends:
* **jina**    – Jina open-source cross-encoder (local, needs ``trust_remote_code``).
* **local**   – Any HuggingFace CrossEncoder model (local, e.g. BGE reranker).

Usage::

    # Jina (needs trust_remote_code + transformers compat patch)
    client = RerankerClient.create("jina/jinaai/jina-reranker-v2-base-multilingual")

    # BGE reranker (standard HF model, no custom code)
    client = RerankerClient.create("local/BAAI/bge-reranker-v2-m3")

    scores = client.score(query_text, [doc1, doc2, doc3])
    # scores: list[float], same length as documents

The ``score()`` method returns a relevance score for each document.  Higher
is more relevant.
"""
from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RerankerClient:
    """Abstract reranker scorer."""

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return a relevance score for each document (higher = more relevant)."""
        raise NotImplementedError

    def score_single(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        return self.score(query, [document])[0]

    def score_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 256) -> List[float]:
        """Score a batch of (query, document) pairs.

        Default implementation falls back to calling ``score_single`` in a loop.
        Subclasses that support efficient batching should override this.
        """
        return [self.score_single(q, d) for q, d in pairs]

    # Factory -----------------------------------------------------------------

    @staticmethod
    def create(model: str) -> "RerankerClient":
        """Create a reranker client from a model spec string.

        Supported formats:
        * ``jina/<hf_model>`` → JinaReranker   (trust_remote_code + compat patch)
        * ``local/<hf_model>``→ LocalReranker   (standard HF CrossEncoder)
        """
        if "/" not in model:
            raise ValueError(
                f"Reranker model must be prefixed with a provider, e.g. "
                f"'jina/jinaai/jina-reranker-v2-base-multilingual' or "
                f"'local/BAAI/bge-reranker-v2-m3'. Got: {model}"
            )

        prefix, rest = model.split("/", 1)
        pfx = prefix.strip().lower()

        if pfx == "jina":
            return JinaReranker(model_name=rest.strip())
        if pfx == "local":
            return LocalReranker(model_name=rest.strip())

        raise ValueError(
            f"Unknown reranker provider '{pfx}'. Use 'jina' or 'local'."
        )


# ---------------------------------------------------------------------------
# Generic local HuggingFace CrossEncoder reranker
# ---------------------------------------------------------------------------

class LocalReranker(RerankerClient):
    """Any HuggingFace CrossEncoder reranker running locally.

    Works with standard models like ``BAAI/bge-reranker-v2-m3``,
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``, etc.

    The model is loaded once on first call and cached for subsequent calls.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        default_batch_size: int = 256,
        max_length: int = 1024,
    ):
        self.model_name = model_name
        self.default_batch_size = default_batch_size
        self.max_length = max_length
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        print(f"Loading local reranker: {self.model_name} ...")
        self._model = CrossEncoder(
            self.model_name,
            model_kwargs={"torch_dtype": "auto"},
        )
        # Cap tokenizer max_length to avoid OOM on long texts
        if self.max_length and hasattr(self._model, "tokenizer"):
            self._model.tokenizer.model_max_length = self.max_length
        print(f"Local reranker loaded (max_length={self.max_length}, batch_size={self.default_batch_size}).")

    def score(self, query: str, documents: List[str]) -> List[float]:
        self._ensure_model()
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, batch_size=self.default_batch_size)
        return [float(s) for s in scores]

    def score_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 256) -> List[float]:
        """Efficiently score a batch of (query, document) pairs on GPU."""
        self._ensure_model()
        bs = batch_size
        scores = self._model.predict(pairs, batch_size=bs, show_progress_bar=True)
        return [float(s) for s in scores]


# ---------------------------------------------------------------------------
# Jina open-source cross-encoder reranker (local, needs custom code)
# ---------------------------------------------------------------------------

class JinaReranker(LocalReranker):
    """Jina cross-encoder reranker running locally.

    Extends ``LocalReranker`` with:
    * ``trust_remote_code=True`` (Jina uses custom model code)
    * Monkey-patch for transformers >= 5.x compat
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        default_batch_size: int = 256,
        max_length: int = 1024,
    ):
        super().__init__(model_name=model_name, default_batch_size=default_batch_size, max_length=max_length)

    @staticmethod
    def _patch_transformers_compat():
        """Monkey-patch ``create_position_ids_from_input_ids`` back into
        ``transformers.models.xlm_roberta.modeling_xlm_roberta`` if it was
        removed (transformers >= 5.x).  Jina's custom model code imports it.
        """
        import transformers.models.xlm_roberta.modeling_xlm_roberta as _mod
        if hasattr(_mod, "create_position_ids_from_input_ids"):
            return  # already present, nothing to do

        import torch

        def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = (
                torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
            ) * mask
            return incremental_indices.long() + padding_idx

        _mod.create_position_ids_from_input_ids = create_position_ids_from_input_ids

    def _ensure_model(self):
        if self._model is not None:
            return

        # Patch transformers compat before loading custom Jina code
        self._patch_transformers_compat()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import CrossEncoder

        print(f"Loading Jina reranker: {self.model_name} ...")
        self._model = CrossEncoder(
            self.model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "auto"},
        )
        # Cap tokenizer max_length to avoid OOM on long texts
        if self.max_length and hasattr(self._model, "tokenizer"):
            self._model.tokenizer.model_max_length = self.max_length
        print(f"Jina reranker loaded (max_length={self.max_length}, batch_size={self.default_batch_size}).")
