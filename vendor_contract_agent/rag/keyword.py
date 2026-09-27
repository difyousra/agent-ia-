from __future__ import annotations

import logging

from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def keyword_search(
    query: str,
    store,
    *,
    top_k: int = 10,
    contract_id: str | None = None,
) -> list[RetrievedChunk]:
    if not query or not query.strip():
        return []
    logger.debug("keyword_search top_k=%s contract_id=%s", top_k, contract_id)
    return store.keyword_search(query.strip(), top_k=top_k, contract_id=contract_id)
