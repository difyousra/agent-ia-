from __future__ import annotations

import logging
from typing import Any

from .config import RAGConfig
from .context import build_context, sources_from_results
from .db import create_store
from .embeddings import EmbeddingService
from .hybrid import hybrid_search
from .rerank import RerankerService
from .schemas import ContractQueryResponse, FactWithSource, RetrievalDebug, SourceRef

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Hybrid retrieval → RRF → rerank → context builder."""

    def __init__(
        self,
        store=None,
        embedder: EmbeddingService | None = None,
        reranker: RerankerService | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or RAGConfig.from_env()
        self.store = store or create_store(self.config)
        self.embedder = embedder or EmbeddingService(self.config)
        self.reranker = reranker or RerankerService(self.config)

    def retrieve(
        self,
        query: str,
        *,
        contract_id: str | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Empty query")

        if contract_id and not self.store.contract_exists(contract_id):
            raise ValueError(f"Invalid contract_id: {contract_id}")

        semantic_results, keyword_results, rrf_results = hybrid_search(
            query,
            self.store,
            self.embedder,
            top_k=top_k or self.config.hybrid_top_k,
            contract_id=contract_id,
            config=self.config,
        )
        reranked = self.reranker.rerank(
            query,
            rrf_results,
            top_n=top_n or self.config.rerank_top_n,
        )
        context = build_context(reranked)
        sources = sources_from_results(reranked)
        return {
            "semantic_results": semantic_results,
            "keyword_results": keyword_results,
            "rrf_results": rrf_results,
            "reranked_results": reranked,
            "context": context,
            "sources": sources,
            "insufficient_information": len(reranked) == 0,
        }
