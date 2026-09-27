from __future__ import annotations

import logging

from .config import RAGConfig
from .keyword import keyword_search
from .rrf import reciprocal_rank_fusion
from .schemas import RetrievedChunk
from .semantic import semantic_search

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    store,
    embedder,
    *,
    top_k: int | None = None,
    contract_id: str | None = None,
    config: RAGConfig | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
    """Run semantic + keyword search and fuse with RRF.

    Returns (semantic_results, keyword_results, fused_results).
    """
    cfg = config or RAGConfig.from_env()
    fused_top_k = top_k or cfg.hybrid_top_k
    semantic_results = semantic_search(
        query,
        store,
        embedder,
        top_k=cfg.semantic_top_k,
        contract_id=contract_id,
    )
    keyword_results = keyword_search(
        query,
        store,
        top_k=cfg.keyword_top_k,
        contract_id=contract_id,
    )
    fused = reciprocal_rank_fusion(
        [semantic_results, keyword_results],
        k=cfg.rrf_k,
        top_k=fused_top_k,
    )
    logger.info(
        "hybrid_search semantic=%s keyword=%s fused=%s",
        len(semantic_results),
        len(keyword_results),
        len(fused),
    )
    return semantic_results, keyword_results, fused
