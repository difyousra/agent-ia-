from __future__ import annotations

from ..config import AgentConfig
from .provider import create_legal_provider
from .schemas import LegalSearchResponse


def legal_search(
    query: str,
    *,
    jurisdiction: str | None = None,
    top_k: int = 5,
    config: AgentConfig | None = None,
    provider=None,
) -> LegalSearchResponse:
    cfg = config or AgentConfig.from_env()
    jur = (jurisdiction or cfg.default_jurisdiction or "DZ").upper()
    searcher = provider or create_legal_provider(cfg)
    return searcher.search(query=query, jurisdiction=jur, top_k=top_k)
