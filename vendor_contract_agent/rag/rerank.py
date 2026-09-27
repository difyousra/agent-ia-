from __future__ import annotations

import logging
import re
import threading
from typing import Sequence

from .config import RAGConfig
from .schemas import RerankedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerService:
    """Lazy-loaded BGE reranker with lexical fallback for offline/tests."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig.from_env()
        self._model = None
        self._lock = threading.Lock()
        self._backend = self.config.reranker_backend

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RerankedChunk]:
        if not documents:
            return []
        limit = top_n or self.config.rerank_top_n
        backend = self._resolve_backend()
        if backend == "lexical":
            scored = [_lexical_score(query, doc) for doc in documents]
        else:
            scored = self._bge_score(query, documents)

        ranked = sorted(scored, key=lambda item: item.rerank_score, reverse=True)
        return ranked[:limit]

    def _resolve_backend(self) -> str:
        if self._backend != "auto":
            return self._backend
        try:
            import FlagEmbedding  # noqa: F401

            return "bge"
        except Exception:
            try:
                from sentence_transformers import CrossEncoder  # noqa: F401

                return "bge"
            except Exception:
                logger.warning("BGE reranker unavailable; using lexical reranker")
                return "lexical"

    def _bge_score(self, query: str, documents: Sequence[RetrievedChunk]) -> list[RerankedChunk]:
        model = self._load_model()
        pairs = [[query, doc.content] for doc in documents]
        if hasattr(model, "compute_score"):
            scores = model.compute_score(pairs, normalize=True)
            if isinstance(scores, (float, int)):
                scores = [float(scores)]
            else:
                scores = [float(s) for s in scores]
        else:
            scores = [float(s) for s in model.predict(pairs)]
        output: list[RerankedChunk] = []
        for doc, score in zip(documents, scores):
            output.append(
                RerankedChunk(
                    content=doc.content,
                    rerank_score=score,
                    metadata={
                        "id": doc.id,
                        "contract_id": doc.contract_id,
                        "document_name": doc.document_name,
                        "clause_number": doc.clause_number,
                        "clause_type": doc.clause_type,
                        "section": doc.section,
                        "page": doc.page,
                        "retrieval_score": doc.score,
                        "source": doc.source,
                    },
                )
            )
        return output

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            logger.info("Loading reranker model %s", self.config.reranker_model)
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.config.reranker_model, use_fp16=True)
            except Exception:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.config.reranker_model)
            return self._model


def _lexical_score(query: str, doc: RetrievedChunk) -> RerankedChunk:
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    d_tokens = set(re.findall(r"[a-z0-9]+", doc.content.lower()))
    if not q_tokens:
        score = 0.0
    else:
        overlap = len(q_tokens & d_tokens) / len(q_tokens)
        phrase = 0.25 if query.strip().lower() in doc.content.lower() else 0.0
        score = min(1.0, overlap + phrase)
    return RerankedChunk(
        content=doc.content,
        rerank_score=score,
        metadata={
            "id": doc.id,
            "contract_id": doc.contract_id,
            "document_name": doc.document_name,
            "clause_number": doc.clause_number,
            "clause_type": doc.clause_type,
            "section": doc.section,
            "page": doc.page,
            "retrieval_score": doc.score,
            "source": doc.source,
        },
    )
