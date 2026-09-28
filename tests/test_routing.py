from __future__ import annotations

from vendor_contract_agent.config import AgentConfig
from vendor_contract_agent.router import Route, classify_route_rules
from vendor_contract_agent.tools import no_retrieval


CASES = [
    (
        "What is the contract value?",
        Route.CONTRACT_ONLY,
        ["contract_search"],
    ),
    (
        "What does Algerian law say about termination?",
        Route.LEGAL_ONLY,
        ["legal_search"],
    ),
    (
        "Does the termination clause in CTR-001 comply with Algerian law?",
        Route.CONTRACT_AND_LEGAL,
        ["contract_search", "legal_search"],
    ),
    (
        "What is an SLA?",
        Route.NO_RETRIEVAL,
        ["no_retrieval"],
    ),
    (
        "What are the vendor's obligations under CTR-001?",
        Route.CONTRACT_ONLY,
        ["contract_search"],
    ),
    (
        "What are the legal requirements for data protection in Algeria?",
        Route.LEGAL_ONLY,
        ["legal_search"],
    ),
]


def test_routing_matrix():
    for query, expected_route, expected_tools in CASES:
        decision = classify_route_rules(query)
        assert decision.route == expected_route, f"{query!r} -> {decision.route}"
        assert decision.tools_used == expected_tools


def test_no_retrieval_tool():
    payload = no_retrieval()
    assert payload["source_type"] == "none"
    assert payload["results"] == []


def test_debug_flag_default_off():
    cfg = AgentConfig.from_env()
    assert cfg.debug_agent_routing is False or isinstance(cfg.debug_agent_routing, bool)
