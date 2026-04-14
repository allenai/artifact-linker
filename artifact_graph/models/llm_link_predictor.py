from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from artifact_graph.utils.llm_client import call_llm
from artifact_graph.utils.llm_response_parser import parse_llm_response_to_json

Edge = Tuple[int, int]


class LLMLinkPredictor:
    """Predict whether a (model, dataset) link exists, optionally using RAG pre-filtering."""

    def __init__(
        self,
        model_name: str = "openai/gpt-3.5-turbo",
        hop_number: int = 1,
        use_info: bool = True,
        # RAG options
        use_rag: bool = False,
        rag_top_k: int = 100,
        rag_strategy: str = "embedding",
        data_dir: Optional[str] = None,
        retriever=None,
    ):
        self.model_name = model_name
        self.hop_number = hop_number
        self.use_info = use_info
        # RAG
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.rag_strategy = rag_strategy
        self.data_dir = data_dir
        self.retriever = retriever
        # Per-batch RAG state (populated by prepare_prediction_batch)
        self._rag_scores: Dict[Edge, float] = {}
        self._rag_selected: Set[Edge] = set()

    # ------------------------------------------------------------------
    # RAG helpers
    # ------------------------------------------------------------------

    def _ensure_retriever(self):
        """Lazily initialise the embedding retriever."""
        if self.retriever is not None or not self.data_dir:
            return
        from artifact_graph.utils.retriever import CandidateRetriever

        self.retriever = CandidateRetriever.from_data_dir(
            self.data_dir, strategy=self.rag_strategy, top_k=self.rag_top_k,
        )

    def prepare_prediction_batch(
        self, edges: List[Edge], G: nx.Graph,
    ) -> Dict[str, int]:
        """Score all candidate pairs via RAG and select the top-k per dataset.

        Must be called once before iterating ``predict()`` over the same edges.
        """
        self._rag_scores, self._rag_selected = {}, set()
        if not self.use_rag:
            return {"scored_pairs": 0, "selected_pairs": 0}

        self._ensure_retriever()
        if self.retriever is None:
            return {"scored_pairs": 0, "selected_pairs": 0}

        # Group model ids by dataset
        by_dataset: Dict[int, List[int]] = {}
        for m, d in edges:
            by_dataset.setdefault(d, []).append(m)

        for did, mids in by_dataset.items():
            scored = self.retriever.score_all(did, mids, G)
            for mid, score in scored:
                self._rag_scores[(mid, did)] = score
            for mid, _ in scored[: self.rag_top_k]:
                self._rag_selected.add((mid, did))

        return {
            "scored_pairs": len(self._rag_scores),
            "selected_pairs": len(self._rag_selected),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        model_id: int,
        dataset_id: int,
        G: nx.Graph,
        node_metadata: dict,
    ) -> Optional[dict]:
        """Predict a single (model, dataset) pair.

        Returns dict with ``prediction`` (bool), ``reason``, ``score``, ``retrieval_score``,
        or *None* on failure.
        """
        meta = node_metadata or {}
        key = (model_id, dataset_id)
        rag_score = self._rag_scores.get(key)

        # RAG filter: pairs not selected are predicted as negative without calling LLM.
        if self._rag_scores and key not in self._rag_selected:
            return {
                "prediction": False,
                "reason": "Filtered out by RAG top-k",
                "score": rag_score or 0.0,
                "retrieval_score": rag_score,
            }

        try:
            prompt = self._build_prompt(model_id, dataset_id, G, meta)
            response = call_llm(
                [{"role": "user", "content": prompt}],
                model=self.model_name,
                agent_name="link_predictor",
            )

            m_name = self._node_name(meta, model_id)
            d_name = self._node_name(meta, dataset_id)

            if not response["success"]:
                print(f"Warning: LLM call failed for ({m_name}, {d_name}). "
                      f"Error: {response.get('error')}")
                return None

            parsed = self._parse_response(response["content"].strip(), m_name, d_name)
            if parsed is not None:
                parsed["retrieval_score"] = rag_score
            return parsed

        except Exception as e:
            print(f"Error predicting for {key}: {e}")
            return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _node_name(meta: dict, nid: int) -> str:
        return meta.get(nid, {}).get("name", str(nid))

    @staticmethod
    def _node_info(meta: dict, nid: int) -> Optional[str]:
        return meta.get(nid, {}).get("info")

    def _collect_neighbors(
        self, G: nx.Graph, meta: dict, model_id: int, dataset_id: int,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Return dict of neighbor lists by type: {type: [(name, info), ...]}."""
        if self.hop_number <= 0 or not G:
            return {}

        UG = G.to_undirected()
        nearby = set()
        for src in (model_id, dataset_id):
            nearby.update(
                nx.single_source_shortest_path_length(UG, src, cutoff=self.hop_number)
            )
        nearby -= {model_id, dataset_id}

        by_type: Dict[str, List[Tuple[str, str]]] = {}
        for nid in nearby:
            ntype = meta.get(nid, {}).get("type")
            if ntype:
                entry = (self._node_name(meta, nid), self._node_info(meta, nid) or "")
                by_type.setdefault(ntype, []).append(entry)
        return by_type

    def _build_prompt(
        self, model_id: int, dataset_id: int, G: nx.Graph, meta: dict,
    ) -> str:
        m_name = self._node_name(meta, model_id)
        d_name = self._node_name(meta, dataset_id)

        parts = [f"Given a machine learning model named '{m_name}' "
                 f"and a dataset named '{d_name}'."]

        if self.use_info:
            m_info = self._node_info(meta, model_id)
            d_info = self._node_info(meta, dataset_id)
            if m_info:
                parts.append(f"More information about this model: {m_info}")
            if d_info:
                parts.append(f"More information about this dataset: {d_info}")

        nbrs = self._collect_neighbors(G, meta, model_id, dataset_id)
        if nbrs.get("model"):
            lines = "\n".join(f"- {n}: {i}" for n, i in nbrs["model"])
            parts.append(
                "There are other models that are evaluated on the dataset "
                "to judge whether the model and dataset are connected:\n" + lines
            )
        if nbrs.get("dataset"):
            lines = "\n".join(f"- {n}: {i}" for n, i in nbrs["dataset"])
            parts.append(
                "There are other datasets that are evaluated on the model "
                "to judge whether the model and dataset are connected:\n" + lines
            )
        if nbrs.get("paper"):
            lines = "\n".join(f"- {n}: {i}" for n, i in nbrs["paper"])
            parts.append(
                "Related papers that may provide context about the model or dataset:\n" + lines
            )
        if nbrs.get("codebase"):
            lines = "\n".join(f"- {n}: {i}" for n, i in nbrs["codebase"])
            parts.append(
                "Related code repositories associated with the model or dataset:\n" + lines
            )

        parts.append(
            "Please predict how likely this model is to be evaluated on this dataset. "
            "Provide your answer as a JSON object with three keys: "
            "'score' (a float between 0.0 and 1.0 representing the probability that "
            "this model-dataset pair is connected, where 0.0 means very unlikely and "
            "1.0 means almost certain), "
            "'prediction' (a boolean, true if score >= 0.5, false otherwise), and "
            "'reason' (a brief explanation of your reasoning)."
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(answer: str, m_name: str, d_name: str) -> Optional[dict]:
        result = parse_llm_response_to_json(answer)
        if not result:
            print(f"Warning: Could not parse LLM output for ({m_name}, {d_name}). "
                  f"Output: {answer}")
            return None

        # Extract probability score (preferred) or fall back to boolean prediction
        score = result.get("score")
        prediction = result.get("prediction")

        if score is not None:
            try:
                score = float(score)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                prediction = score >= 0.5
            except (ValueError, TypeError):
                score = None

        if score is None:
            # Fallback: if LLM only returned boolean, convert to 0/1 score
            if isinstance(prediction, bool):
                score = 1.0 if prediction else 0.0
            else:
                return None

        return {
            "prediction": prediction,
            "score": score,
            "reason": result.get("reason", ""),
        }
