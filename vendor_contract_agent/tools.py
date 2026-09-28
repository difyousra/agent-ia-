from __future__ import annotations

import logging
from typing import Any

from .config import AgentConfig
from .legal.provider import create_legal_provider
from .legal.schemas import LegalSearchResponse
from .legal.search import legal_search as run_legal_search
from .rag.pipeline import RAGPipeline
from .rag.schemas import RerankedChunk

logger = logging.getLogger(__name__)


def contract_search(
    query: str,
    *,
    contract_id: str | None = None,
    top_k: int = 5,
    rag: RAGPipeline | None = None,
) -> dict[str, Any]:
    """Search the internal ChromaDB contract knowledge base."""
    pipeline = rag or RAGPipeline()
    try:
        retrieval = pipeline.retrieve(
            query,
            contract_id=contract_id,
            top_n=top_k,
        )
    except ValueError as exc:
        return {
            "source_type": "contract",
            "results": [],
            "error": str(exc),
            "available": False,
            "insufficient_information": True,
            "raw": None,
        }
    except Exception as exc:
        logger.exception("contract_search failed")
        return {
            "source_type": "contract",
            "results": [],
            "error": f"chromadb_unavailable: {exc}",
            "available": False,
            "insufficient_information": True,
            "raw": None,
        }

    results = []
    for item in retrieval["reranked_results"]:
        if isinstance(item, RerankedChunk):
            meta = item.metadata
            results.append(
                {
                    "document": meta.get("document_name"),
                    "contract_id": meta.get("contract_id"),
                    "clause_number": meta.get("clause_number"),
                    "clause_type": meta.get("clause_type"),
                    "page": meta.get("page"),
                    "content": item.content,
                    "score": float(item.rerank_score),
                }
            )
        else:
            results.append(
                {
                    "document": item.document_name,
                    "contract_id": item.contract_id,
                    "clause_number": item.clause_number,
                    "clause_type": item.clause_type,
                    "page": item.page,
                    "content": item.content,
                    "score": float(item.score),
                }
            )

    return {
        "source_type": "contract",
        "results": results,
        "error": None,
        "available": True,
        "insufficient_information": bool(retrieval["insufficient_information"]),
        "raw": retrieval,
    }


def legal_search(
    query: str,
    *,
    jurisdiction: str = "DZ",
    top_k: int = 5,
    config: AgentConfig | None = None,
    provider=None,
) -> dict[str, Any]:
    """Search external legal sources via the configured provider adapter."""
    cfg = config or AgentConfig.from_env()
    try:
        response: LegalSearchResponse = run_legal_search(
            query,
            jurisdiction=jurisdiction or cfg.default_jurisdiction,
            top_k=top_k,
            config=cfg,
            provider=provider or create_legal_provider(cfg),
        )
    except Exception as exc:
        logger.exception("legal_search failed")
        return {
            "source_type": "legal",
            "jurisdiction": jurisdiction or cfg.default_jurisdiction,
            "results": [],
            "error": f"legal_api_unavailable: {exc}",
            "available": False,
        }

    return {
        "source_type": "legal",
        "jurisdiction": response.jurisdiction,
        "results": [item.model_dump() for item in response.results],
        "error": response.error,
        "available": response.available,
    }


def no_retrieval() -> dict[str, Any]:
    """Explicit no-op tool when retrieval is unnecessary."""
    return {"source_type": "none", "results": [], "available": True}
