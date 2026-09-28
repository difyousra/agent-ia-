from __future__ import annotations

from vendor_contract_agent.config import AgentConfig
from vendor_contract_agent.legal.provider import NullLegalSearchProvider, create_legal_provider
from vendor_contract_agent.legal.schemas import LegalSearchResponse, LegalSearchResult
from vendor_contract_agent.tools import legal_search


class FakeLegalProvider:
    def search(self, query: str, jurisdiction: str = "DZ", top_k: int = 5) -> LegalSearchResponse:
        return LegalSearchResponse(
            jurisdiction=jurisdiction,
            results=[
                LegalSearchResult(
                    title="Sample Act",
                    reference="Art. 1",
                    content=f"Relevant to: {query}",
                    url=None,
                )
            ],
        )


def test_null_provider_when_disabled(monkeypatch):
    monkeypatch.setenv("LEGAL_SEARCH_ENABLED", "false")
    monkeypatch.delenv("LEGAL_API_URL", raising=False)
    provider = create_legal_provider(AgentConfig.from_env())
    assert isinstance(provider, NullLegalSearchProvider)
    result = provider.search("termination", jurisdiction="DZ", top_k=3)
    assert result.available is False
    assert result.results == []


def test_legal_search_adapter_does_not_invent_urls():
    payload = legal_search(
        "termination",
        jurisdiction="DZ",
        top_k=3,
        provider=FakeLegalProvider(),
    )
    assert payload["source_type"] == "legal"
    assert payload["available"] is True
    assert payload["results"][0]["url"] is None
    assert "termination" in (payload["results"][0]["content"] or "")
