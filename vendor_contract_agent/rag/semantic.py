from __future__ import annotations

import logging

from .config import RAGConfig
from .embeddings import EmbeddingService
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def semantic_search(
    query: str,
    store,
    embedder: EmbeddingService,
    *,
    top_k: int = 10,
    contract_id: str | None = None,
) -> list[RetrievedChunk]:
    if not query or not query.strip():
        return []
    logger.debug("semantic_search top_k=%s contract_id=%s", top_k, contract_id)
    query_embedding = embedder.embed_query(query.strip())
    return store.semantic_search(query_embedding, top_k=top_k, contract_id=contract_id)
