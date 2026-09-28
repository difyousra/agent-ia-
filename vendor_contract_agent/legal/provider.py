from __future__ import annotations

import logging
from typing import Any
from urllib import error, request
import json

from ..config import AgentConfig
from .base import LegalSearchProvider
from .schemas import LegalSearchResponse, LegalSearchResult

logger = logging.getLogger(__name__)


class NullLegalSearchProvider(LegalSearchProvider):
    """Used when LEGAL_SEARCH_ENABLED is false or credentials are missing."""

    def __init__(self, reason: str = "legal_search_disabled") -> None:
        self.reason = reason

    def search(
        self,
        query: str,
        jurisdiction: str = "DZ",
        top_k: int = 5,
    ) -> LegalSearchResponse:
        return LegalSearchResponse(
            source_type="legal",
            jurisdiction=jurisdiction,
            results=[],
            error=self.reason,
            available=False,
        )


class HttpLegalSearchProvider(LegalSearchProvider):
    """Generic HTTP adapter. Provider-specific parsing stays replaceable."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str = "",
        timeout: int = 20,
    ) -> None:
        if not api_url:
            raise ValueError("LEGAL_API_URL is required for HttpLegalSearchProvider")
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def search(
        self,
        query: str,
        jurisdiction: str = "DZ",
        top_k: int = 5,
    ) -> LegalSearchResponse:
        payload = {
            "query": query,
            "jurisdiction": jurisdiction,
            "top_k": top_k,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            logger.warning("Legal API HTTP error: %s", exc)
            return LegalSearchResponse(
                jurisdiction=jurisdiction,
                results=[],
                error=f"legal_api_http_error: {exc.code}",
                available=False,
            )
        except error.URLError as exc:
            logger.warning("Legal API unavailable: %s", exc)
            return LegalSearchResponse(
                jurisdiction=jurisdiction,
                results=[],
                error=f"legal_api_unavailable: {exc.reason}",
                available=False,
            )
        except Exception as exc:
            logger.exception("Legal API call failed")
            return LegalSearchResponse(
                jurisdiction=jurisdiction,
                results=[],
                error=f"legal_api_error: {exc}",
                available=False,
            )

        return _parse_legal_payload(data, jurisdiction=jurisdiction, top_k=top_k)


def _parse_legal_payload(
    data: Any,
    *,
    jurisdiction: str,
    top_k: int,
) -> LegalSearchResponse:
    """Accept a few common response shapes without inventing fields."""
    items: list[Any]
    if isinstance(data, dict):
        items = data.get("results") or data.get("data") or data.get("hits") or []
        jurisdiction = str(data.get("jurisdiction") or jurisdiction)
    elif isinstance(data, list):
        items = data
    else:
        items = []

    results: list[LegalSearchResult] = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("text") or item.get("excerpt")
        title = item.get("title")
        reference = item.get("reference") or item.get("citation")
        url = item.get("url") or item.get("link")
        # Never invent URLs/titles: keep only provided values.
        results.append(
            LegalSearchResult(
                title=str(title) if title else None,
                reference=str(reference) if reference else None,
                document_type=str(item["document_type"]) if item.get("document_type") else None,
                date=str(item["date"]) if item.get("date") else None,
                content=str(content) if content else None,
                url=str(url) if url else None,
            )
        )

    return LegalSearchResponse(
        source_type="legal",
        jurisdiction=jurisdiction,
        results=results,
        available=True,
    )


def create_legal_provider(config: AgentConfig | None = None) -> LegalSearchProvider:
    cfg = config or AgentConfig.from_env()
    if not cfg.legal_search_enabled:
        return NullLegalSearchProvider("legal_search_disabled")
    if not cfg.legal_api_url:
        return NullLegalSearchProvider("legal_api_url_missing")
    return HttpLegalSearchProvider(
        api_url=cfg.legal_api_url,
        api_key=cfg.legal_api_key,
        timeout=cfg.legal_api_timeout,
    )
